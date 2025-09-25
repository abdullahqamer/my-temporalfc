# TemporalFC — MLP for Temporal Range Prediction

Predict the **start** and **end** year of a fact given a triple **(subject, predicate, object)**.
This repo contains a lightweight MLP model that maps triple embeddings to a **time interval**.

> **TL;DR**
> Input: (s, p, o) → embeddings → MLP → outputs two numbers (start, end).
> Best run so far (Wikidata6): **MAE ≈ 3.6 / 5.2 years**, **IoU ≈ 0.75**.

---

## Highlights

* **Simple architecture**: 3×(Linear→GELU→Dropout) + 2-unit head (start & delta).
* **Interval guarantee**: `end = start + (1 − start) * delta` ensures `end ≥ start`.
* **Practical training tweaks** (data/opt regularization):

  * Tiny **label jitter** on time indices (train-only)
  * **Cosine LR** with a small floor
  * Slightly higher **end loss weight**
  * Tiny **Gaussian noise** on embeddings (train-only)

---

## Results (Wikidata6)

| Setting                  | Start MAE ↓ | End MAE ↓ | IoU ↑ |
| ------------------------ | ----------: | --------: | ----: |
| Baseline MLP             |       ~17.6 |     ~18.7 | ~0.53 |
| Improved MLP (this repo) |        ~3.6 |      ~5.2 | ~0.75 |

*(MAE in years; higher IoU is better.)*

---

## Model at a glance

```
(s, p, o)
   │
   ├─→ Embeddings: h = emb(s), r = emb(p), t = emb(o)
   │     (train-only) + tiny Gaussian noise on h,r,t
   │
   ├─→ Interaction features: |h−t|,  h⊙t
   │
   ├─→ Concat: x = [h, r, t, |h−t|, h⊙t]
   │
   ├─→ MLP trunk:
   │     Linear → GELU → Dropout
   │     Linear → GELU → Dropout
   │     Linear → GELU → Dropout
   │
   ├─→ Head (2 units): [s_raw, d_raw]
   │     start = σ(s_raw)
   │     delta = σ(d_raw)
   │     end   = start + (1 − start) * delta   (guarantees end ≥ start)
   │
   └─→ Output: [start_norm, end_norm]  (normalized to [0,1], later mapped to years)
```

---

## Installation

Tested with Python 3.9+ and PyTorch.

```bash
# 1) clone
git clone https://github.com/abdullahqamer/my-temporalfc.git
cd my-temporalfc

# 2) (recommended) create and activate a virtual env
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 3) install deps
pip install -r requirements.txt
```

---

## Dataset

Download the prepared data release from GitHub:

* **Wikidata6 release:**
  [https://github.com/abdullahqamer/my-temporalfc/releases/tag/v1.0](https://github.com/abdullahqamer/my-temporalfc/releases/tag/v1.0)

Unzip/place the contents so the repo can find:

```
dataset/
  wikidata6/
    train/...
    valid/...
    test/...
```

*(If your folder differs, adjust the `--path_dataset_folder` or related flags.)*

---

## Reproducing the improved result

The command below matches the improved configuration that produced **MAE ≈ 3.6 / 5.2** and **IoU ≈ 0.75** on Wikidata6. (Minor variation is normal due to random seeds.)

```bash
python main.py \
  --eval_dataset wikidata6 \
  --task range-prediction \
  --model range-mlp \
  --emb_type dihedron \
  --embedding_dim 100 \
  --batch_size 1024 \
  --val_batch_size 1000 \
  --use_interaction 1 \
  --use_prod 1 \
  --loss_type huber \
  --huber_beta 0.8970321391066037 \
  --end_weight 0.95 \
  --extra_order_pen 0.042741660857969106 \
  --lr 0.00184775182894049 \
  --num_workers 4 \
  --max_num_epochs 120 \
  --seed 42 \
  --emb_noise 0.01
```

Notes:

* **Cosine LR** schedule is configured inside the model (`CosineAnnealingLR` with a small floor).
* **Label jitter** is applied train-only (inside the training step), so you don’t need flags for it.
* **End loss weight** (`--end_weight 0.95`) slightly emphasizes the end boundary, which helped IoU.

---

## Quick evaluation

If you trained with checkpoints enabled, the script will automatically load the best checkpoint for testing/evaluation. Otherwise, specify `--resume_from_checkpoint` when needed.

Typical output includes:

* Start/End **MAE** (years)
* **IoU** of predicted interval vs. ground truth
* ±k-year accuracies

---

## Repo structure (short)

```
my-temporalfc/
├─ main.py                     # entry point
├─ executer_TP.py              # training/eval runner (Lightning)
├─ nn_models_TP/range_mlp_model.py   # MLP model for start/end
├─ utils_TP/                   # helpers
└─ dataset/                    # place Wikidata6 here (see Dataset section)
```

---

## How this work builds on prior tools

This project stands on the shoulders of open-source frameworks and ideas:

* **PyTorch** — tensor library and autograd
* **PyTorch Lightning** — structured training loop
* **Wikidata** — source of temporal facts (processed into the “Wikidata6” split)
* **Embedding ideas (e.g., Dihedron)** — entity/relation embeddings for triples

Thanks to the maintainers and communities behind these tools and datasets.

---

## License

This repository is released for academic/research use.
Please check licenses of dependencies and datasets accordingly.

---

## Contact

Questions or issues? Please open a GitHub issue or reach out via the repository.

---
