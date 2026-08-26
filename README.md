# RISTER

## 📌 Introduction

This repository provides a **PyTorch implementation** of:

> **RISTER** (TODO: 添加论文标题 / 会议信息)

RISTER is a scene text recognition (STR) model built upon the [OpenOCR](https://github.com/Topdu/OpenOCR) framework. It consists of two core components:

- **RELG** — a resolution-adaptive encoder that combines local convolution blocks with global attention blocks.
- **RITextDecoder** — an autoregressive transformer decoder for character sequence generation.

RISTER provides four model scales (Tiny / Small / Base / Large) to trade off accuracy and efficiency.

---

## 🛠️ Environment Setup

This project is built upon the OpenOCR framework.

Please follow the official guide for environment setup:

👉 https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md

---

## 📂 Dataset Preparation

Datasets should also be prepared according to the OpenOCR instructions:

👉 https://github.com/Topdu/OpenOCR/blob/main/docs/svtrv2.md

The models are trained on the **Union14M-L-Filtered** dataset (real-world scene text recognition benchmark).

---

## 📦 Pretrained Models

| Model | Depths | Heads | Params | Checkpoint |
|:-----:|:------:|:-----:|:------:|:----------:|
| RISTER-T | [1, 1, 1, 6] | 6 | 15.94M | TODO |
| RISTER-S | [3, 3, 3, 9] | 6 | 21.83M | TODO |
| RISTER-B | [4, 4, 4, 15] | 12 | 32.76M | TODO |
| RISTER-L | [6, 6, 6, 18] | 12 | 38.65M | TODO |

*(TODO: 将权重上传至 Google Drive / GitHub Release 并填入链接)*

---

## ▶️ Training

### 🔹 Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train_rec.py --c configs/rec/rister/rister_l.yml
```

### 🔹 Single-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_rec.py --c configs/rec/rister/rister_l.yml
```

Available configs:

| Model | Config |
|:-----:|:------:|
| RISTER-T | `configs/rec/rister/rister_t.yml` |
| RISTER-S | `configs/rec/rister/rister_s.yml` |
| RISTER-B | `configs/rec/rister/rister_b.yml` |
| RISTER-L | `configs/rec/rister/rister_l.yml` |

---

## ▶️ Evaluation

```bash
python tools/eval_rec.py --c configs/rec/rister/rister_l.yml
```

To evaluate on a specific test set, override the data path:

```bash
python tools/eval_rec.py --c configs/rec/rister/rister_l.yml -o "Eval.dataset.data_dir=/path/to/test/lmdb"
```

---

## 📌 Citation

If you find this work useful, please consider citing:

```bibtex
@article{rister,
  title={RISTER},
  author={},
  journal={},
  year={}
}
```

*(TODO: 论文正式发表后更新 BibTeX)*

---

## 🙏 Acknowledgement

This project is implemented based on the following open-source repository:

👉 https://github.com/Topdu/OpenOCR

We sincerely thank the authors for their excellent work.
