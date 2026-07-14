# Where Not to Learn: Prior-Aligned Training with Subset-based Attribution Constraints for Reliable Decision-Making

Official PyTorch implementation of [*Where Not to Learn: Prior-Aligned Training with Subset-based Attribution Constraints for Reliable Decision-Making*](https://arxiv.org/abs/2602.07008).

[![arXiv](https://img.shields.io/badge/arXiv-2602.07008-b31b1b.svg)](https://arxiv.org/abs/2602.07008)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository studies visual attribution methods whose explanations align with human priors. It includes training, evaluation, Pointing Game, and visualization entrypoints for Saliency-Bench and ImageNet-S919.

## Setup

```bash
conda create -n vea python=3.10
conda activate vea
```

Install the PyTorch build appropriate for your CUDA version, followed by the project dependencies used by the selected training script.

## Datasets

The supplied lists use paths relative to the repository root.

| Dataset | List directory | Expected root directory |
| --- | --- | --- |
| Saliency-Bench Object-XAI (PASCAL) | `data_list/saliency-bench/` | `Saliency-Bench/pascal/` |
| ImageNet-S919 | `data_list/imagenet-s919/` | `ImageNetS919/` |

`Saliency-Bench/pascal/` contains `image/<class>/` and `attention_mask/<class>/`; each mask is a NumPy `.npy` file aligned with its image. The dataset files are intentionally excluded from version control.

ImageNet-S919 provides public segmentation annotations, but its train and validation images are derived from ImageNet-1K. Obtain an authorized local ImageNet-1K copy first, then follow the official ImageNet-S preparation workflow to populate `ImageNetS919/train-semi/` and `ImageNetS919/validation/`. The expected mask locations are `train-semi-segmentation/` and `validation-segmentation/`.

## Running experiments

Training entrypoints are grouped by dataset:

```bash
# Saliency-Bench
python vision_task_saliency_bench/human_prior_alignment_v2_CLIP.py

# ImageNet-S919
python vision_task_imagenet-s/human_prior_alignment_v3_resnet.py
```

Each script exposes `--train_txt` and `--test_txt`, defaulting to the corresponding lists under `data_list/`. Use `--help` on a specific entrypoint to inspect model, batch size, checkpoint, and distributed-training options.

For attribution evaluation, use the Pointing Game scripts:

```bash
python point_game_saliency_bench.py
python point_game_imagenets.py
```

## Results

| Model | Human prior | Method | Top-1 accuracy | Top-2 accuracy | Point Game@0.2 |
| --- | --- | --- | ---: | ---: | ---: |
| CLIP | Mask | Zero-shot | 0.6574 | 0.7847 | 0.5440 |
| CLIP | Mask | Fine-tuning | 0.6076 | 0.8495 | 0.5231 |
| CLIP | Mask | Ours v2 | 0.6551 | 0.8264 | 0.5648 |
| ViT-Base | Mask | Fine-tuning | 0.5440 | 0.7558 | 0.4363 |
| ViT-Base | Mask | Ours v2 | 0.5359 | 0.7512 | 0.4838 |
| ViT-Base | Mask | Ours v3 | 0.5787 | 0.7650 | 0.4988 |
| ResNet-101 | Mask | Fine-tuning | 0.3866 | 0.5880 | 0.4838 |
| ResNet-101 | Mask | Ours v3 | 0.4965 | 0.7014 | 0.5231 |

## Visualizations

| Model | Fine-tuning | Human alignment |
| --- | --- | --- |
| ViT | ![](examples/vit_ft_2007_001288.png) | ![](examples/vit_ours_2007_001288.png) |
| ViT | ![](examples/vit_ft_2007_004627.png) | ![](examples/vit_ours_2007_004627.png) |

## License

The source code is released under the [MIT License](LICENSE). Dataset licenses remain with their respective providers.

## Citation

If you use this code, please cite:

```bibtex
@article{chen2026where,
  title={Where Not to Learn: Prior-Aligned Training with Subset-based Attribution Constraints for Reliable Decision-Making},
  author={Chen, Ruoyu and Sun, Shangquan and Guo, Xiaoqing and Zhang, Sanyi and Liu, Kangwei and Liu, Shiming and Wang, Zhangcheng and Zhang, Qunli and Zhang, Hua and Cao, Xiaochun},
  journal={arXiv preprint arXiv:2602.07008},
  year={2026}
}
```
