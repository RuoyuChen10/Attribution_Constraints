# Attribution Alignment with Human Annotation

## 🛠️ Environment

```shell
conda create -n vea python=3.10
```

## Experiments

Experiments on `Saliency Bench` dataset

|Model|Human Prior|Method|Top-1 Accuracy|Top-2 Accuracy|Point Game@0.2|
|:--:|:--:|:--:|:--:|:--:|:--:|
|CLIP|Mask|Zero-shot|0.6574|0.7847|0.5440|
| |  | Fine-tuning | 0.6076 | 0.8495 |0.5231|
| |  | Ours v2 | 0.6551 | 0.8264 | 0.5648 |
|ViT base | Mask | Fine-tuning| 0.5440 | 0.7558 | 0.4363 |
| |  | Ours v2 | 0.5359 | 0.7512 | 0.4838 |
| |  | Ours v3 | 0.5787 | 0.7650 | 0.4988 |
| ResNet-101 | Mask | Fine-tuning| 0.3866 | 0.5880 | 0.4838 |
| |  | Ours v3 | 0.4965 | 0.7014 | 0.5231 |

torchrun --nproc_per_node=4 train_clip_ddp.py --train_scope proj