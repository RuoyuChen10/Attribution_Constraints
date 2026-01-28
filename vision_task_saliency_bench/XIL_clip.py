import os
from datetime import timedelta
from contextlib import nullcontext
import math
import argparse
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import CLIPModel, AutoTokenizer
from tqdm import tqdm

# 只保留 math kernel（最可能支持二阶导）
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ✅ 换成 ImageNet-S 的 dataloader（你说“从dataloader里导入另外两个”）
from dataloader import ImageNetSDataset, make_dataloaders

# from interpretation.HUMAN_LIMA_Efficient import HumanLIMA
from utils import mkdir, SubRegionDivision

# 关闭 tokenizer 并行提示
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # for Chinese
os.environ["HF_HOME"] = "./model_checkpoint/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------
# 数据集
# -------------------
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def read_list_file(list_path: str):
    items = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_p, mask_p, label = line.split()
            items.append((img_p, mask_p, label))
    return items

def build_label_map(*list_files: str):
    labels = []
    for lf in list_files:
        for _, _, lab in read_list_file(lf):
            labels.append(lab)
    labels = sorted(list(set(labels)))
    return {lab: i for i, lab in enumerate(labels)}

def clamp_logit_scale(m):
    with torch.no_grad():
        max_val = math.log(100.0)   # OpenAI 官方限制
        if hasattr(m, "logit_scale"):
            m.logit_scale.data.clamp_(max=max_val)

# -------------------
# 分布式初始化
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
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

# -------------------
# 训练和评测
# -------------------
def build_texts(labels_idx, idx_to_label, templates):
    texts = []
    for i in labels_idx.tolist():
        name = idx_to_label[i].replace("_", " ")
        texts.append(templates[0].format(name=name))
    return texts

@torch.no_grad()
def evaluate(model, dataloader, device, idx_to_label, templates):
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    classnames = [idx_to_label[i] for i in range(len(idx_to_label))]
    feats = []
    for cname in classnames:
        texts = [tmp.format(name=cname) for tmp in templates]
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model.get_text_features(**tokens)
        out = out / out.norm(dim=-1, keepdim=True)
        feats.append(out.mean(dim=0))
    text_feats = torch.stack(feats, dim=1)  # [D,C]

    top1 = top2 = n = 0
    for images, _, labels, _, _ in dataloader:
        images, labels = images.to(device), labels.to(device)
        img_feats = model.get_image_features(pixel_values=images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        logits = (img_feats @ text_feats) * model.logit_scale.exp()

        pred = logits.argmax(dim=1)
        top1 += (pred == labels).sum().item()
        top2 += (logits.topk(2, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
        n += labels.size(0)

    return top1/n, top2/n

def grad_eclip_vit_tokens(
    model,
    pixel_values,
    text_feats,
    labels,
    grid_hw=(16, 16),
    use_logprob=True,
):
    out = model.vision_model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
    tokens = out.last_hidden_state                  # [B,1+N,C]
    cls   = tokens[:, :1, :]                        # [B,1,C]
    patch = tokens[:, 1:, :]                        # [B,N,C]

    # ---- 关键：让 image embedding 显式依赖 patch ----
    # 用 mean-pooled token 作为 pooled（你也可以用 cls + eps*patch_mean）
    patch_mean = patch.mean(dim=1, keepdim=True)    # [B,1,C]
    pooled = (cls + patch_mean).squeeze(1)          # [B,C]  显式依赖 patch

    img_emb = model.visual_projection(pooled)       # [B,D]
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    scale = model.logit_scale.exp()
    logits = (img_emb @ text_feats) * scale         # [B,C]

    if use_logprob:
        logp = F.log_softmax(logits.float(), dim=1)
        target = logp.gather(1, labels.view(-1, 1)).sum()
    else:
        target = logits.float().gather(1, labels.view(-1, 1)).sum()

    # grads wrt patch tokens
    G = torch.autograd.grad(
        outputs=target,
        inputs=patch,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]                                             # [B,N,C]

    alpha = G.mean(dim=1)                            # [B,C]
    E = torch.relu((patch * alpha.unsqueeze(1)).sum(dim=-1))  # [B,N]

    H, W = grid_hw
    E = E.view(E.size(0), 1, H, W)
    E = E / (E.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return E, logits

def train_one_epoch(model, optimizer, scaler, loader, device, tokenizer, idx_to_label, args, epoch, text_feats):
    model.train()
    ce = nn.CrossEntropyLoss()

    # —— 可选：限制 logit_scale 避免数值太大（按需保留/删除）——
    with torch.no_grad():
        if hasattr(model, "logit_scale"):
            model.logit_scale.data.clamp_(max=math.log(100.0))

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main_process())

    optimizer.zero_grad(set_to_none=True)

    for images, masks, labels, img_paths, mask_paths in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)  # 0/1 mask, same spatial size 
        
        # 关键：让输入可求导（input-level gradient）
        images.requires_grad_(True)

        optimizer.zero_grad(set_to_none=True)
            
        # ========= forward =========
        with torch.autocast("cuda", enabled=args.amp):
            img_feats = model.get_image_features(pixel_values=images)  # [B, D]
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            scale = model.logit_scale.exp()
            logits = (img_feats @ text_feats) * scale
            
            E, logits = grad_eclip_vit_tokens(model, images, text_feats, labels, grid_hw=(16,16))
            loss_ce = ce(logits, labels)
            
        mask_small = F.interpolate(masks.unsqueeze(1).float(), size=(16,16), mode="nearest")  # [B,1,16,16]

        forbidden = 1.0 - mask_small
        loss_xil = (forbidden * E).mean()   # 或 (forbidden * E**2).mean()

        loss_all = loss_ce + 0.1 * loss_xil
        
        # ========= backward + step（AMP） =========
        scaler.scale(loss_all).backward()
        scaler.step(optimizer)
        scaler.update()

        # 可选：为了避免下一轮梯度累积/显存占用
        images.requires_grad_(False)

# -------------------
# main
# -------------------
def main_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", default="data_list/saliency-bench/train.txt")
    parser.add_argument("--test_txt", default="data_list/saliency-bench/test.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--align_steps", type=int, default=50)
    parser.add_argument("--division_number", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--output_dir", default="./ckpt_vision_saliency_bench/ckpts_clip_L14_XIL")
    parser.add_argument("--train_scope", type=str, default="vision",
                        choices=["full", "vision", "proj"],
                        help="训练范围: full=全量; vision=只训练视觉塔; proj=只训练投影层和logit_scale")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    setup_distributed(args)
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

    # 数据
    train_loader, test_loader, train_ds, test_ds, label_to_idx = make_dataloaders(
        args.train_txt,
        args.test_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(224,224),
    )
    train_sampler = DistributedSampler(train_ds) if args.world_size > 1 else None
    test_sampler  = DistributedSampler(test_ds, shuffle=False) if args.world_size > 1 else None
    idx_to_label = {v:k for k,v in label_to_idx.items()}
    
    # 模型
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # 冻结文本 encoder
    # 根据 scope 冻结不同部分
    if args.train_scope == "vision":
        for name, param in model.named_parameters():
            if "text_model" in name:
                param.requires_grad = False
    elif args.train_scope == "proj":
        for name, param in model.named_parameters():
            if not ("visual_projection" in name or "text_projection" in name or "logit_scale" in name):
                param.requires_grad = False
    # full 就不用改
    model.logit_scale.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # 评测
    templates = [
        "a photo of a {name}.",
    ]
    # 构建文本特征
    classnames = [idx_to_label[i] for i in range(len(idx_to_label))]
    feats = []
    for cname in classnames:
        texts = [tmp.format(name=cname) for tmp in templates]
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model.get_text_features(**tokens)
        out = out / out.norm(dim=-1, keepdim=True)
        feats.append(out.mean(dim=0))
    text_feats = torch.stack(feats, dim=1).detach()  # [D,C] [768, 20]

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=args.amp)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs+1):
        if train_sampler: train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, scaler, train_loader, device, tokenizer, idx_to_label, args, epoch,  text_feats)

        # 评测
        eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        top1, top2 = evaluate(eval_model, test_loader, device, idx_to_label,
                              ["a photo of a {name}.", 
                            #    "a photo of the {name}."
                               ])
        if is_main_process():
            print(f"[Eval] Epoch {epoch}: top1={top1:.4f}, top2={top2:.4f}")
            if top1 > best_acc:
                best_acc = top1
                torch.save(eval_model.state_dict(), os.path.join(args.output_dir, f"best_epoch{epoch}.pt"))
            torch.save(eval_model.state_dict(), os.path.join(args.output_dir, f"epoch{epoch}.pt"))

    cleanup_distributed()

if __name__ == "__main__":
    main_worker()