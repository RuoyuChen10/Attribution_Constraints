import os
import random
from datetime import timedelta
import argparse
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

import math

from PIL import Image
from tqdm import tqdm

from dataloader import ImageNetSDataset, make_imagenet_s_dataloaders
# from interpretation.HUMAN_LIMA_Efficient import HumanLIMA
from utils import SubRegionDivision

from vit_model import VIT_IMAGE_MEAN, VIT_IMAGE_STD, build_vit_classifier

# 只保留 math kernel（最可能支持二阶导）
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# -------------------
# 分布式
# -------------------
def setup_distributed(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        args.rank, args.world_size, args.local_rank = 0, 1, 0

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    if args.world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=1800)
        )
        dist.barrier()

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

# -------------------
# 准确率
# -------------------
@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # [K, B]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))  # [K, B]
    res = []
    for k in topk:
        correct_k = correct[:k].any(dim=0).float().sum(0) if k == 1 \
                    else correct[:k].any(dim=0).float().sum(0)
        # 注意：Top-1 用 any 与 sum 结果一致；这里保持一致写法
        res.append((correct[:k].any(dim=0).float().sum().item()) / batch_size)
    return res

# -------------------
# 评测
# -------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    top1 = top2 = n = 0
    for images, _, labels, _, _ in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        out = model(pixel_values=images)   # HF ViT 输入参数名是 pixel_values
        logits = out.logits                # [B, C]
        # logits = model(images)  # [B, C]
        # Top-1
        pred1 = logits.argmax(dim=1)
        top1 += (pred1 == labels).sum().item()
        # Top-2
        top2 += (logits.topk(2, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
        n += labels.size(0)
    return top1 / n, top2 / n

def forward_attention(attention_layer, x, head_mask=None, output_attentions=False):
    num_attention_heads = 1
    attention_head_size = 768 // num_attention_heads
    
    def transpose_for_scores(x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    mixed_query_layer = attention_layer.query(x)

    key_layer = transpose_for_scores(attention_layer.key(x))
    value_layer = transpose_for_scores(attention_layer.value(x))
    query_layer = transpose_for_scores(mixed_query_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    attention_scores = attention_scores / math.sqrt(attention_layer.attention_head_size)

    # Normalize the attention scores to probabilities.
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    # attention_probs = attention_layer.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (attention_layer.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    return outputs, (query_layer, key_layer, value_layer)


def vit_classifier(x, model):
    # modified from ViT
    batch_size, num_channels, height, width = x.shape

    ## keep image size 
    x = model.vit.embeddings(x, interpolate_pos_encoding=True)

    for i, layer_module in enumerate(model.vit.encoder.layer[:-1]):
        layer_outputs = layer_module(x, None, False)
        x = layer_outputs[0]
    

    ### the last layer in ViT
    lastLY = model.vit.encoder.layer[-1]

    x = lastLY.layernorm_before(x)

    attention_layer = lastLY.attention.attention
    self_outputs, (q,k,v) = forward_attention(attention_layer, x)
    attention_output = lastLY.attention.output(self_outputs[0], x)

    # residual connection
    x = attention_output + x

    # in ViT, layernorm is also applied after self-attention
    layer_output = lastLY.layernorm_after(x)
    layer_output = lastLY.intermediate(layer_output)

    # second residual connection is done here
    layer_output = lastLY.output(layer_output, x)
    sequence_output = model.vit.layernorm(layer_output)

    logits = model.classifier(sequence_output[:, 0, :])
    return logits, self_outputs[0], (q, k, v), (int(height//16), int(width//16))

def grad_eclip(c, q, k, v, att_output, map_size, withksim=True):
    D = k.shape[-1]
    ## gradient on last attention output
    grad = torch.autograd.grad(
        c,
        att_output,
        retain_graph=True)[0]
    # grad = grad.detach()
    grad_cls = grad[0,:1,:]
    if withksim:
        q_cls = q[0,0,:1,:]
        k_patch = k[0,0,1:,:]
        q_cls = F.normalize(q_cls, dim=-1)
        k_patch = F.normalize(k_patch, dim=-1)
        cosine_qk = (q_cls * k_patch).sum(-1) 
        cosine_qk = (cosine_qk-cosine_qk.min()) / (cosine_qk.max()-cosine_qk.min())
        emap_lastv = F.relu_((grad_cls * v[0,0,1:,:] * cosine_qk[:,None]).sum(-1)) # 
    else:
        emap_lastv = F.relu_((grad_cls * v[0,0,1:,:]).sum(-1)) 
    return emap_lastv.reshape(*map_size)

# -------------------
# 训练
# -------------------
def train_one_epoch(model, optimizer, scaler, loader, device, args, epoch):
    model.train()
    ce = nn.CrossEntropyLoss()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main_process())
    
    optimizer.zero_grad(set_to_none=True)

    for images, masks, labels, img_paths, mask_paths in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)  # 期望 [B,1,H,W] 或 [B,H,W]，值 0/1

        # --- 让输入可求导（RRR input-gradient）---
        # images.requires_grad_(True)

        optimizer.zero_grad(set_to_none=True)

        # ========= 1) 主 CE 优化步 =========
        with torch.autocast("cuda", enabled=args.amp):
            # logits = model(images)  # [B, C]
            logits, last_att_outputs, (q, k, v), map_size = vit_classifier(images, model)
            loss_ce = ce(logits, labels)
        
        # logits_fp32 = logits.float()
        # target_score = logits_fp32.gather(1, labels.view(-1, 1)).sum()
        emaps = []
        for logit, label in zip(logits, labels):
            emap = grad_eclip(logit[label], q, k, v, last_att_outputs, map_size, withksim=True)
            emaps.append(emap)
        emaps = torch.stack(emaps)  # torch.Size([32, 14, 14])
        emaps = emaps.unsqueeze(1)
        
        mask_small = F.interpolate(masks.unsqueeze(1).float(), size=(14,14), mode="nearest")  # [B,1,16,16]
        
        # 归一化到 [0,1]（每张图单独 min-max）
        emin = emaps.amin(dim=(2,3), keepdim=True)
        emax = emaps.amax(dim=(2,3), keepdim=True)
        emap01 = (emaps - emin) / (emax - emin + 1e-6)
        
        loss_vis = F.l1_loss(emap01, mask_small)   # L1

        loss_all = loss_ce + 0.1 * loss_vis
        
        # ========= 3) backward + step =========
        scaler.scale(loss_all).backward()
        scaler.step(optimizer)
        scaler.update()

        # images.requires_grad_(False)

        if is_main_process():
            pbar.set_postfix(loss=f"{loss_all.item():.4f}")
        
        
# -------------------
# main_worker
# -------------------
def main_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", default="data_list/imagenet-s919/train.txt")
    parser.add_argument("--test_txt", default="data_list/imagenet-s919/test.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--align_steps", type=int, default=20)
    parser.add_argument("--division_number", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--output_dir", default="./ckpt_vision_imagenets/ckpts_vit_b16_imnet_MEGL/")
    parser.add_argument("--num_classes", type=int, default=918, help="默认训练 20 类；若与数据集不一致将以数据集为准")
    parser.add_argument("--train_scope", type=str, default="full",
                        choices=["full", "head"],
                        help="full=全量微调; head=仅分类头")
    args = parser.parse_args()

    setup_distributed(args)
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 数据
    train_loader, test_loader, train_ds, test_ds, label_to_idx = make_imagenet_s_dataloaders(
        args.train_txt,
        args.test_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(224,224),
        image_mean=VIT_IMAGE_MEAN,
        image_std=VIT_IMAGE_STD,
    )
    train_sampler = DistributedSampler(train_ds) if args.world_size > 1 else None
    test_sampler  = DistributedSampler(test_ds, shuffle=False) if args.world_size > 1 else None
    idx_to_label = {v:k for k,v in label_to_idx.items()}

    # 模型 & 预处理
    freeze_backbone = (args.train_scope == "head")
    model = build_vit_classifier(
        num_classes=len(label_to_idx),
        freeze_backbone=freeze_backbone,
    ).to(device)

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # 优化器
    params = model.parameters()
    if args.train_scope == "head":
        # 仅分类头
        params = (p for n, p in model.named_parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=args.amp)

    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        if train_sampler: train_sampler.set_epoch(epoch)

        train_one_epoch(model, optimizer, scaler, train_loader, device, args, epoch)

        # 评测
        eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        top1, top2 = evaluate(eval_model, test_loader, device)
        if is_main_process():
            print(f"[Eval] Epoch {epoch}: top1={top1:.4f}, top2={top2:.4f}")
            if top1 > best_acc:
                best_acc = top1
                torch.save(eval_model.state_dict(), os.path.join(args.output_dir, f"best_epoch{epoch}.pt"))

    cleanup_distributed()

if __name__ == "__main__":
    main_worker()
