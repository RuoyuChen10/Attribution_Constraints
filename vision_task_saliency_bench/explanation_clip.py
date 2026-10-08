import os
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import CLIPModel, AutoTokenizer
from tqdm import tqdm

from interpretation.LIMA import BlackBoxSingleModalCounterfactualSubModularExplanation
from utils import mkdir, SubRegionDivision

import cv2

# 关闭 tokenizer 并行提示
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # for Chinese
os.environ["HF_HOME"] = "./model_checkpoint/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# OpenAI CLIP 预处理参数
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

img_tf = transforms.Compose([
            transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
        ])

# -------------------
# 数据集
# -------------------
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

def build_label_map(list_file: str):
    labels = [lab for _, _, lab in read_list_file(list_file)]
    labels = sorted(list(set(labels)))
    return {lab: i for i, lab in enumerate(labels)}

# -------------------
# 测试函数
# -------------------
class CLIPAdaptor():
    def __init__(self, model, text_feats):
        self.model = model
        self.text_feats = text_feats
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, x):
        img_feats = self.model.get_image_features(pixel_values=x)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        
        logits = (img_feats @ self.text_feats) * self.model.logit_scale.exp()
        logits = self.softmax(logits)
            
        return logits

# -------------------
# main
# -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_txt", type=str, default="data_list/saliency-bench/test.txt")
    parser.add_argument("--ckpt", default=None, type=str, help="训练好的模型权重 .pt")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument('--division-number', 
                        type=int, default=50,
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default='./interpretation_results/CLIP-baseline-50/',
                        help='output directory to save results')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集
    label_to_idx = build_label_map("data_list/saliency-bench/test.txt")
    idx_to_label = {v:k for k,v in label_to_idx.items()}

    # 模型
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    if args.ckpt:
        print("Load ckpt!")
        state = torch.load(args.ckpt, map_location="cpu")
        if "model" in state:  # 兼容保存的 dict
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
    model.to(device)

    # 评测
    templates = [
        "a photo of a {name}.",
        # "a photo of the {name}.",
        # "a close-up photo of a {name}.",
        # "a blurry photo of a {name}.",
        # "a photo of a small {name}.",
        # "a photo of a large {name}.",
    ]
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # 构建文本特征
    classnames = [idx_to_label[i] for i in range(len(idx_to_label))]
    feats = []
    for cname in classnames:
        texts = [tmp.format(name=cname) for tmp in templates]
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model.get_text_features(**tokens)
        out = out / out.norm(dim=-1, keepdim=True)
        feats.append(out.mean(dim=0))
    text_feats = torch.stack(feats, dim=1)  # [D,C]

    adaptor = CLIPAdaptor(model, text_feats)
    
    smdl = BlackBoxSingleModalCounterfactualSubModularExplanation(
        adaptor,
        lambda1=1,
        lambda2=1
    )
    
    save_dir = args.save_dir
    
    mkdir(save_dir)
    
    save_npy_root_path = os.path.join(save_dir, "npy")
    mkdir(save_npy_root_path)
    
    save_json_root_path = os.path.join(save_dir, "json")
    mkdir(save_json_root_path)
    
    with open(args.test_txt, "r") as f:
        items = [line.strip().split() for line in f if line.strip()]
        
    for item in tqdm(items):
        img_path, mask_path, label_name = item
        label = label_to_idx[label_name]

        image = cv2.imread(img_path)
        
        image_tensor = img_tf(Image.open(img_path).convert("RGB")).to(device)
    
        # Sub-region division
        region_size = int((image.shape[0] * image.shape[1] / args.division_number) ** 0.5)
        V_set = SubRegionDivision(image, mode="slico", region_size = region_size)
        
        S_set, saved_json_file = smdl(image_tensor, V_set, label)
        
        # Save npy file
        np.save(
            os.path.join(save_npy_root_path, img_path.split("/")[-1].replace(".png", ".npy")),
            np.array(S_set)
        )
        
        # Save json file
        with open(
            os.path.join(save_json_root_path, img_path.split("/")[-1].replace(".png", ".json")), "w") as f:
            f.write(json.dumps(saved_json_file, ensure_ascii=False, indent=4, separators=(',', ':')))
    

if __name__ == "__main__":
    main()
