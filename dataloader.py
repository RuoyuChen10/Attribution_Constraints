import os
from typing import Optional, Tuple, Dict, List
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode


OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HF_VIT_MEAN = (0.5, 0.5, 0.5)
HF_VIT_STD = (0.5, 0.5, 0.5)


def read_list_file(list_path: str) -> List[Tuple[str, str, str]]:
    """读取 txt，每行: image_path mask_path label_name"""
    triplets = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_p, mask_p, label = line.split()
            triplets.append((img_p, mask_p, label))
    return triplets


def build_label_map(*list_files: str) -> Dict[str, int]:
    """从多个 txt 里收集 label_name -> id 的全局映射"""
    labels = []
    for lf in list_files:
        for _, _, lab in read_list_file(lf):
            labels.append(lab)
    labels = sorted(list(set(labels)))
    return {lab: i for i, lab in enumerate(labels)}


class PascalSaliencyDataset(Dataset):
    """
    读取行格式: <image_path> <mask_path> <label_name>
    返回:
      image: FloatTensor [3,H,W]  (0-1 或标准化)
      mask:  FloatTensor [1,H,W]  (0-1)
      label: LongTensor ()        类别索引
    """
    def __init__(
        self,
        list_file: str,
        label_to_idx: Dict[str, int],
        target_size: Optional[Tuple[int, int]] = (448, 448),  # 设 None 保持原图
        image_mean: Tuple[float, float, float] = OPENAI_CLIP_MEAN,
        image_std: Tuple[float, float, float] = OPENAI_CLIP_STD,
        normalize_image: bool = True,
        mask_interp: str = "bilinear",  # "nearest" | "bilinear"
        image_interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        super().__init__()
        self.items = read_list_file(list_file)
        self.label_to_idx = label_to_idx
        self.target_size = target_size
        self.normalize_image = normalize_image
        self.mask_interp = mask_interp

        # 图像预处理
        tfs = []
        if target_size is not None:
            tfs.append(transforms.Resize(target_size, interpolation=image_interpolation))
        tfs.append(transforms.ToTensor())  # [0,1]
        if normalize_image:
            tfs.append(transforms.Normalize(mean=image_mean, std=image_std))
        self.img_tf = transforms.Compose(tfs)

    def __len__(self):
        return len(self.items)

    def _resize_mask(self, mask_t: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        # mask_t: [1,h,w] -> [1,H,W]
        mode = "bilinear" if self.mask_interp == "bilinear" else "nearest"
        mask_t = F.interpolate(mask_t.unsqueeze(0), size=size_hw, mode=mode, align_corners=False if mode=="bilinear" else None)
        return mask_t.squeeze(0)

    def __getitem__(self, idx):
        img_path, mask_path, label_name = self.items[idx]

        # --- image ---
        img = Image.open(img_path).convert("RGB")
        img_t = self.img_tf(img)  # [3,H,W]

        # --- mask ---
        mask_np = np.load(mask_path)  # 可能是 [H,W] / [H,W,1] / [1,H,W]
        if mask_np.ndim == 3:
            mask_np = np.squeeze(mask_np)
        mask_np = mask_np.astype(np.float32)

        # 归一化到 [0,1]（若原本是0/255或其他范围）
        # mmax = float(mask_np.max()) if mask_np.size > 0 else 1.0
        # if mmax > 0:
        #     # 常见情况：0/255、0/1、或任意正数范围
        #     if mmax > 1.5:  # 粗略判断 0-255 等
        #         mask_np = mask_np / mmax
        mask_t = torch.from_numpy(mask_np).float()
        # if mask_t.ndim == 2:
        #     mask_t = mask_t.unsqueeze(0)  # [1,h,w]

        # 尺寸对齐：与图像一致
        H, W = img_t.shape[1], img_t.shape[2]
        if mask_t.shape[-2:] != (H, W):
            mask_t = self._resize_mask(mask_t, (H, W))

        # --- label ---
        label = torch.tensor(self.label_to_idx[label_name], dtype=torch.long)

        return img_t, mask_t, label, img_path, mask_path  # 附带路径便于调试


def make_dataloaders(
    train_txt: str,
    test_txt: str,
    batch_size: int = 8,
    num_workers: int = 4,
    target_size: Optional[Tuple[int, int]] = (448, 448),
    normalize_image: bool = True,
    mask_interp: str = "bilinear",
    image_mean: Tuple[float, float, float] = OPENAI_CLIP_MEAN,
    image_std: Tuple[float, float, float] = OPENAI_CLIP_STD,
    image_interpolation: InterpolationMode = InterpolationMode.BICUBIC,
):
    label_to_idx = build_label_map(train_txt, test_txt)

    train_ds = PascalSaliencyDataset(
        list_file=train_txt,
        label_to_idx=label_to_idx,
        target_size=target_size,
        normalize_image=normalize_image,
        mask_interp=mask_interp,
        image_mean=image_mean,
        image_std=image_std,
        image_interpolation=image_interpolation,
    )
    test_ds = PascalSaliencyDataset(
        list_file=test_txt,
        label_to_idx=label_to_idx,
        target_size=target_size,
        normalize_image=normalize_image,
        mask_interp=mask_interp,
        image_mean=image_mean,
        image_std=image_std,
        image_interpolation=image_interpolation,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    return train_loader, test_loader, train_ds, test_ds, label_to_idx





class ImageNetSDataset(Dataset):
    """
    读取 ImageNet-S:
      txt 每行: <image_path> <mask_png_path> <label_name>

    返回:
      image: FloatTensor [3, H, W]
      mask:  FloatTensor [1, H, W]   (0/1)
      label: LongTensor ()
    """
    def __init__(
        self,
        list_file: str,
        label_to_idx: Dict[str, int],
        target_size: Tuple[int, int] = (224, 224),
        image_mean: Tuple[float, float, float] = OPENAI_CLIP_MEAN,
        image_std: Tuple[float, float, float] = OPENAI_CLIP_STD,
        normalize_image: bool = True,
        image_interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        super().__init__()
        self.items = read_list_file(list_file)
        self.label_to_idx = label_to_idx
        self.target_size = target_size

        # image transform
        tfs = [
            transforms.Resize(target_size, interpolation=image_interpolation),
            transforms.ToTensor(),  # [0,1]
        ]
        if normalize_image:
            tfs.append(transforms.Normalize(mean=image_mean, std=image_std))
        self.img_tf = transforms.Compose(tfs)

        # mask resize（只做 resize，不 ToTensor）
        self.mask_resize = transforms.Resize(
            target_size, interpolation=InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.items)

    def _load_mask(self, mask_path: str) -> torch.Tensor:
        """
        Load PNG mask and convert to [1, H, W] float tensor in {0,1}
        """
        # ImageNet-S mask 可能是 P / RGB / L
        mask = Image.open(mask_path)

        # 转成灰度，保证单通道
        mask = mask.convert("L")  # [H, W], 0-255

        # resize（nearest，保证离散性）
        mask = self.mask_resize(mask)

        # -> tensor
        mask_t = transforms.functional.to_tensor(mask)  # [1, H, W], 0-1

        # 二值化（非常重要）
        mask_t = (mask_t[0] > 0).float()

        return mask_t

    def __getitem__(self, idx):
        img_path, mask_path, label_name = self.items[idx]

        # --- image ---
        img = Image.open(img_path).convert("RGB")
        img_t = self.img_tf(img)  # [3,H,W]

        # --- mask ---
        mask_t = self._load_mask(mask_path)  # [H,W]

        # --- label ---
        label = torch.tensor(
            self.label_to_idx[label_name], dtype=torch.long
        )

        return img_t, mask_t, label, img_path, mask_path
    
def make_imagenet_s_dataloaders(
    train_txt: str,
    val_txt: str,
    batch_size: int = 32,
    num_workers: int = 8,
    target_size: Tuple[int, int] = (224, 224),
    normalize_image: bool = True,
    image_mean: Tuple[float, float, float] = OPENAI_CLIP_MEAN,
    image_std: Tuple[float, float, float] = OPENAI_CLIP_STD,
    image_interpolation: InterpolationMode = InterpolationMode.BICUBIC,
):
    label_to_idx = build_label_map(train_txt, val_txt)

    train_ds = ImageNetSDataset(
        list_file=train_txt,
        label_to_idx=label_to_idx,
        target_size=target_size,
        normalize_image=normalize_image,
        image_mean=image_mean,
        image_std=image_std,
        image_interpolation=image_interpolation,
    )
    val_ds = ImageNetSDataset(
        list_file=val_txt,
        label_to_idx=label_to_idx,
        target_size=target_size,
        normalize_image=normalize_image,
        image_mean=image_mean,
        image_std=image_std,
        image_interpolation=image_interpolation,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, train_ds, val_ds, label_to_idx
