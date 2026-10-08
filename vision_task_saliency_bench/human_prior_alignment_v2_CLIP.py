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

from dataloader import PascalSaliencyDataset, make_dataloaders
from interpretation.HUMAN_LIMA_Efficient import HumanLIMA
from utils import mkdir, SubRegionDivision

# 关闭 tokenizer 并行提示
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # for Chinese
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

class CLIPAdaptor():
    def __init__(self, model, text_feats):
        self.model = model
        self.text_feats = text_feats
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, x):
        with torch.no_grad():
            img_feats = self.model.get_image_features(pixel_values=x)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            
            logits = (img_feats @ self.text_feats) * self.model.logit_scale.exp()
            logits = self.softmax(logits)
            
        return logits

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

def train_one_epoch(model, optimizer, scaler, loader, device, tokenizer, idx_to_label, args, epoch, human_lima, text_feats):
    model.train()
    ce = nn.CrossEntropyLoss()

    # —— 一次性准备文本特征（通常不需要梯度）——
    text_feats = text_feats.to(device, non_blocking=True).detach()
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    # —— 可选：限制 logit_scale 避免数值太大（按需保留/删除）——
    with torch.no_grad():
        if hasattr(model, "logit_scale"):
            model.logit_scale.clamp_(max=math.log(100.0))

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main_process())

    aug_step_count = 1
    optimizer.zero_grad(set_to_none=True)

    for images, masks, labels, img_paths, mask_paths in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ========= 1) 主 CE 优化步 =========
        with torch.autocast("cuda", enabled=args.amp):
            img_feats = model.get_image_features(pixel_values=images)  # [B, D]
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            scale = model.logit_scale.exp()
            logits = (img_feats @ text_feats) * scale
            loss_ce = ce(logits, labels)
            
            loss_all = loss_ce

        # scaler.scale(loss_ce).backward()
        # scaler.step(optimizer)           # ⚠️ 放在 autocast 外
        # scaler.update()
        # optimizer.zero_grad(set_to_none=True)

        if is_main_process():
            pbar.set_postfix(loss_ce=f"{loss_ce.item():.4f}")

        # ========= 2) HUMAN-LIMA（每 align_steps+1 步触发） =========
        if aug_step_count % (args.align_steps + 1) == 0:
            # —— 将 eval_model 注入 adaptor（你已在 LIMA 内部规避梯度）——
            eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            human_lima.model = CLIPAdaptor(eval_model, text_feats)

            # —— 置信度筛选（可直接复用刚算的 logits）——
            with torch.inference_mode():
                scores = F.softmax(logits, dim=-1)
                preds = scores.argmax(dim=-1)
                confidences = scores[torch.arange(len(scores), device=labels.device), preds]
                mask_sel = (preds == labels) & (confidences > args.threshold)

                selected_indices = torch.nonzero(mask_sel, as_tuple=True)[0]
                selected_indices_list = selected_indices.detach().cpu().tolist()

                # 子集张量/路径
                selected_img_paths = [img_paths[i] for i in selected_indices_list]
                selected_masks     = masks[selected_indices.cpu()]                  # [K,H,W]
                selected_images    = images[selected_indices]                       # [K,C,H,W]
                selected_labels    = labels[selected_indices]                       # [K]

            # —— 生成训练对（你的人类一致性逻辑保持不变）——
            samples_first_suppress = []
            samples_first_suppress_reverse = []
            samples_first_suppress_labels = []

            samples_after_none_human = []
            samples_before_none_human = []
            samples_after_none_human_reverse = []
            samples_before_none_human_reverse = []
            samples_none_human_labels = []

            # human_lima 内部你已处理 no_grad；这里读取图像也无需建图
            for selected_img_path, selected_image, selected_mask, selected_label in zip(
                selected_img_paths, selected_images, selected_masks, selected_labels
            ):
                image_tmp = cv2.imread(selected_img_path)  # BGR; 若算法依赖颜色，必要时转 RGB
                # cv2.cvtColor(image_tmp, cv2.COLOR_BGR2RGB)

                # 子区域划分
                region_size = int((image_tmp.shape[0] * image_tmp.shape[1] / args.division_number) ** 0.5)
                V_set = SubRegionDivision(image_tmp, mode="slico", region_size=region_size)

                # 这里 human_lima 内部已 no_grad
                S_set, saved_json_file = human_lima(selected_image, V_set, selected_label)

                # S_tensor: [N,H,W] (bool)，selected_mask: [H,W] (bool)
                S_tensor = torch.from_numpy(np.stack([S.squeeze(-1) for S in S_set], axis=0)).to(device=device, dtype=torch.bool)  # [N,H,W]
                selected_mask = selected_mask.to(device=device, dtype=torch.bool).unsqueeze(0)  # [1,H,W]

                # —— 计算人类一致性 —— #
                total_ones = S_tensor.sum(dim=(1, 2))                               # [N]
                overlap = (S_tensor & selected_mask).sum(dim=(1, 2))               # [N]
                human_consistency_judge = (overlap.float() / (total_ones.float() + 1e-6)) > 0.15  # [N]

                if torch.all(human_consistency_judge):
                    pass
                elif human_consistency_judge[0] == False:
                    # 第一个都不合格，不做累加：只压第一个
                    m0 = S_tensor[0].unsqueeze(0)             # [1,H,W] bool
                    x_mask      = selected_image * m0.float()
                    x_mask_rev  = selected_image * (~m0).float()
                    samples_first_suppress.append(x_mask)
                    samples_first_suppress_reverse.append(x_mask_rev)
                    samples_first_suppress_labels.append(selected_label)
                else:
                    # 第一个合格，后续某些不合格
                    false_indices = torch.nonzero(~human_consistency_judge[1:], as_tuple=True)[0] + 1
                    for fi in false_indices:
                        # 前 fi（含）与前 fi（不含）
                        m_inc = S_tensor[:fi+1].any(dim=0)   # [H,W] bool
                        m_exc = S_tensor[:fi].any(dim=0)     # [H,W] bool
                        x_after      = selected_image * m_inc.unsqueeze(0).float()
                        x_after_rev  = selected_image * (~m_inc).unsqueeze(0).float()
                        x_before     = selected_image * m_exc.unsqueeze(0).float()
                        x_before_rev = selected_image * (~m_exc).unsqueeze(0).float()

                        samples_after_none_human.append(x_after)
                        samples_after_none_human_reverse.append(x_after_rev)
                        samples_before_none_human.append(x_before)
                        samples_before_none_human_reverse.append(x_before_rev)
                        samples_none_human_labels.append(selected_label)

            # —— HUMAN loss：单独一步（与你策略一致）——
            if len(samples_first_suppress) != 0:
                samples_first_suppress_labels = torch.stack(samples_first_suppress_labels).to(device, dtype=torch.long)
                s1 = torch.stack(samples_first_suppress, dim=0).to(device, non_blocking=True)            # [B,C,H,W]
                s2 = torch.stack(samples_first_suppress_reverse, dim=0).to(device, non_blocking=True)     # [B,C,H,W]

                with torch.autocast("cuda", enabled=args.amp):
                    img1 = model.get_image_features(pixel_values=s1)
                    img1 = img1 / img1.norm(dim=-1, keepdim=True)
                    img2 = model.get_image_features(pixel_values=s2)
                    img2 = img2 / img2.norm(dim=-1, keepdim=True)

                    scale = model.logit_scale.exp()
                    inserts = (img1 @ text_feats) * scale
                    dels    = (img2 @ text_feats) * scale

                    inserts = inserts.softmax(dim=-1)
                    dels    = dels.softmax(dim=-1)

                    B_local = inserts.size(0)
                    ar = torch.arange(B_local, device=device)
                    gt_ins = inserts[ar, samples_first_suppress_labels]
                    gt_del = dels[ar,   samples_first_suppress_labels]

                    loss_human = gt_ins.mean() + (1 - gt_del).mean()
                    loss_all = loss_all + 0.5 * loss_human
            
            print(" —— HUMAN loss —— ", loss_human.item())

            # ========= 3) 冗余 REDUNDANCY：分批 + AMP + 累积 =========
            # if len(samples_after_none_human) != 0:
            #     A   = torch.stack(samples_after_none_human, dim=0).to(device, non_blocking=True)
            #     Bfr = torch.stack(samples_before_none_human, dim=0).to(device, non_blocking=True)
            #     Arv = torch.stack(samples_after_none_human_reverse, dim=0).to(device, non_blocking=True)
            #     Brv = torch.stack(samples_before_none_human_reverse, dim=0).to(device, non_blocking=True)
            #     Lab = torch.stack(samples_none_human_labels).to(device, dtype=torch.long)

            #     bs = getattr(args, "redundancy_bs", 8)
            #     accum = getattr(args, "redundancy_accum", 2)

            #     n = A.size(0)
            #     nb = (n + bs - 1) // bs
                
            #     loss_redundancy_total = 0.0  # ✅ 累积用于日志

            #     for bi in range(nb):
            #         s = bi * bs
            #         e = min((bi + 1) * bs, n)

            #         # DDP 下非 step 小步关闭同步，减少通信
            #         if isinstance(model, torch.nn.parallel.DistributedDataParallel) and ((bi + 1) % accum != 0) and (bi != nb - 1):
            #             sync_cm = model.no_sync()
            #         else:
            #             sync_cm = nullcontext()

            #         with sync_cm:
            #             with torch.autocast("cuda", enabled=args.amp):
            #                 img_after   = model.get_image_features(pixel_values=A[s:e]);   img_after   = img_after   / img_after.norm(dim=-1, keepdim=True)
            #                 img_before  = model.get_image_features(pixel_values=Bfr[s:e]); img_before  = img_before  / img_before.norm(dim=-1, keepdim=True)
            #                 img_after_r = model.get_image_features(pixel_values=Arv[s:e]); img_after_r = img_after_r / img_after_r.norm(dim=-1, keepdim=True)
            #                 img_before_r= model.get_image_features(pixel_values=Brv[s:e]); img_before_r= img_before_r/ img_before_r.norm(dim=-1, keepdim=True)

            #                 scale = model.logit_scale.exp()
            #                 ins_a = (img_after   @ text_feats) * scale
            #                 ins_b = (img_before  @ text_feats) * scale
            #                 del_a = (img_after_r @ text_feats) * scale
            #                 del_b = (img_before_r@ text_feats) * scale

            #                 ins_a = ins_a.softmax(dim=-1)
            #                 ins_b = ins_b.softmax(dim=-1)
            #                 del_a = del_a.softmax(dim=-1)
            #                 del_b = del_b.softmax(dim=-1)

            #                 B_local = ins_a.size(0)
            #                 ar = torch.arange(B_local, device=device)
            #                 lab = Lab[s:e]

            #                 gt_ia = ins_a[ar, lab]; gt_ib = ins_b[ar, lab]
            #                 gt_da = del_a[ar, lab]; gt_db = del_b[ar, lab]

            #                 # 压 inserts（after-before>0 罚），提 dels（before-after>0 罚）
            #                 loss_b = F.relu(gt_ia - gt_ib).mean() + F.relu(gt_db - gt_da).mean()
            #                 loss_b = loss_b / accum

            #             scaler.scale(0.1 * loss_b).backward()
            #             loss_redundancy_total += loss_b.item() * accum  # ✅ 累积总 loss（乘回去还原）

            #         if ((bi + 1) % accum == 0) or (bi == nb - 1):
            #             # 可选：梯度裁剪
            #             # scaler.unscale_(optimizer)
            #             # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #             scaler.step(optimizer)
            #             scaler.update()
            #             optimizer.zero_grad(set_to_none=True)
            #     # ✅ 求平均并打印/记录日志
            #     loss_redundancy_avg = loss_redundancy_total / nb
                 
            #     print(" —— REDUNDANCY loss —— ", loss_redundancy_avg)
        scaler.scale(loss_all).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        aug_step_count += 1

# -------------------
# main_worker
# -------------------
def main_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", default="data_list/saliency-bench/train.txt")
    parser.add_argument("--test_txt", default="data_list/saliency-bench/test.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--align_steps", type=int, default=5)
    parser.add_argument("--division_number", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--output_dir", default="./ckpts_clip_L14_human_prior_v2")
    parser.add_argument("--train_scope", type=str, default="vision",
                    choices=["full", "vision", "proj"],
                    help="训练范围: full=全量; vision=只训练视觉塔; proj=只训练投影层和logit_scale")
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

    adaptor = CLIPAdaptor(model, text_feats)
    human_lima = HumanLIMA(adaptor, threshold=args.threshold)

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=args.amp)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs+1):
        if train_sampler: train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, scaler, train_loader, device, tokenizer, idx_to_label, args, epoch, human_lima, text_feats)

        # 评测
        eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        top1, top2 = evaluate(eval_model, test_loader, device, idx_to_label,
                              ["a photo of a {name}.", 
                            #    "a photo of the {name}."
                               ])
        if is_main_process():
            print(f"[Eval] Epoch {epoch}: top1={top1:.4f}, top2={top2:.4f}")
            if top2 > best_acc:
                best_acc = top2
                torch.save(eval_model.state_dict(), os.path.join(args.output_dir, f"best_epoch{epoch}.pt"))
            torch.save(eval_model.state_dict(), os.path.join(args.output_dir, f"epoch{epoch}.pt"))

    cleanup_distributed()

if __name__ == "__main__":
    main_worker()
