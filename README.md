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
| |  | Ours | 0.6551 | 0.8264 | 0.5648 |

torchrun --nproc_per_node=4 train_clip_ddp.py --train_scope proj