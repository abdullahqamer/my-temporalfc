# TemporalFC-RangeMLP (MoE Interval Prediction)

This repository provides a clean implementation of a **relation-aware MLP (“Range-MLP”) head** for **interval (start–end year) prediction** on knowledge-graph facts.  
It builds on the original **TemporalFC** codebase.

- **Task:** Given a triple `(s, p, o)`, predict a validity interval `[y_s, y_e]`.
- **Highlights:** coupled start–end head, overlap-aware training (soft-IoU auxiliary), frozen Dihedron embeddings, simple MLP trunk.
- **Default branch:** `MLP`

---

## 1) Setup

### Clone and environment (Conda)
```bash
git clone https://github.com/abdullahqamer/TemporalFC-RangeMLP.git
cd TemporalFC-RangeMLP

# create environment (Conda)
conda env create -f environment.yml
conda activate tfc
````

> If you prefer pip, export your own requirements from the environment after it’s created.

### Dataset & embeddings

Download **wikidata6** from Releases and unzip under `data_TP/`:

* **Release:** [https://github.com/abdullahqamer/TemporalFC-RangeMLP/releases/tag/v1.0](https://github.com/abdullahqamer/TemporalFC-RangeMLP/releases/tag/v1.0)
* **Asset:** `wikidata6.zip`

Expected layout (key parts):

```
data_TP/
  wikidata6/
    entities            # or entities_map.tsv (keep one style)
    relations           # or relations_map.tsv
    times               # or times_map.tsv
    embeddings/
      dihedron/
        entity.npy  relation.npy  time.npy    # or *.pkl (pick one format)
    train/train
    valid/valid
    test/test
```

---

## 2) Quick start (1-epoch smoke test)
Runs end-to-end quickly to verify data paths, loaders, and training loop.

```bash
python main.py \
  --path_dataset_folder "./data_TP" \
  --eval_dataset "wikidata6" \
  --task "range-prediction" \
  --model "range-mlp" \
  --emb_type "dihedron" \
  --embedding_dim 100 \
  -batch_size 256 \
  --val_batch_size 256 \
  --num_workers 2 \
  --max_num_epochs 1 \
  --min_num_epochs 1 \
  --check_val_every_n_epochs 1 \
  --seed 42
```

For full runs, increase `--batch_size`, `--val_batch_size`, and set `--max_num_epochs` (e.g., 120).

---

## 3) Best-known config
Use this when you want the strongest numbers.

```bash
python main.py \
  --path_dataset_folder "./data_TP" \
  --eval_dataset "wikidata6" \
  --task "range-prediction" \
  --model "range-mlp" \
  --emb_type "dihedron" \
  --embedding_dim 100 \
  --batch_size 1024 \
  --val_batch_size 1000 \
  --loss_type "huber" \
  --huber_beta 0.7366304701152739 \
  --end_weight 1.0013169655965504 \
  --extra_order_pen 0.06686696713024719 \
  --lr 0.0018887997194523478 \
  --use_interaction 1 \
  --use_prod 0 \
  --hidden_dim 1024 \
  --dropout 0.11699588659730913 \
  --gauss_sigma_idx 1.3507654810031005 \
  --emb_noise 0.011450934693677826 \
  --t_dim 128 \
  --num_experts 5 \
  --k_experts 3 \
  --gate_temp_start 1.4307208041278083 \
  --gate_balance 0.023266080281596692 \
  --use_bands 0 \
  --use_prior 0 \
  --num_workers 4 \
  --max_num_epochs 120 \
  --min_num_epochs 1 \
  --check_val_every_n_epochs 1 \
  --seed 42 \
  --storage_path "HYBRID_Storage"
```

---

## 4) How it works

**Inputs.** We use frozen **Dihedron** embeddings for entities and relations; simple interactions (e.g., `|h−t|`, `h⊙t`) are concatenated with the base features.

**Trunk + MoE.** A compact MLP trunk feeds a **relation-aware Mixture-of-Experts (MoE)** gate. The gate routes to a small set of experts to capture heterogeneous temporal regimes across relations.

**Calendar head (coupled start/end).** Over a **shared, trainable calendar embedding table** (years), we score **two distributions**: one for the start year and one for the end year. A **triangular coupling/mask** enforces the constraint **`end ≥ start`** during scoring/normalization, so the two predictions are coherent by construction (no post-hoc swapping).

**Training signals.** We combine **smoothed cross-entropy** on the year indices with an **overlap-aware auxiliary (soft-IoU)** so optimization tracks interval quality, not just endpoint sharpness. A light **order/length regularizer** stabilizes the head.

**Frozen vs. trainable.** Entity/relation embeddings are **frozen**; the trunk, MoE, and calendar head are trainable.

**Evaluation.** Primary metric is **interval IoU**, with **MAE** on start/end reported as diagnostics.

---

## 5) Project structure (minimal)

```
.
├── main.py                         # CLI runner
├── executer_TP.py                  # training/eval orchestration
├── data_TP.py                      # dataset utilities
├── utils_TP/
│   ├── static_funcs.py             # model registry, arg checks
│   └── dataset_classes.py          # DataModule/Dataset
├── nn_models_TP/
│   ├── base_model.py
│   └── range_mlp_model.py          # MLP interval head (new)
├── environment.yml
└── README.md
```

---

## 6) Repro tips

* Keep `--seed 42` for reproducibility.
* Use `--num_workers 4–8` to speed up loading (depends on CPU).
* Only `--model range-mlp` is exposed (legacy LSTM/time-point models removed).

---

## 7) Dataset (Releases)

* **wikidata6 (v1.0):** [https://github.com/abdullahqamer/TemporalFC-RangeMLP/releases/tag/v1.0](https://github.com/abdullahqamer/TemporalFC-RangeMLP/releases/tag/v1.0)
  Contains ID maps, splits, and Dihedron embeddings needed to run.

---

## 8) Acknowledgements

This work is derived from and inspired by **TemporalFC** (ISWC 2023).
Original project: [https://github.com/dice-group/TemporalFC](https://github.com/dice-group/TemporalFC)
Embeddings: **Dihedron** (ESWC 2022).

If you use this repo, please also cite the original works.

**TemporalFC**

```bibtex
@inproceedings{Qudus2023TemporalFC,
  title     = {TemporalFC: A Temporal Fact Checking Approach over Knowledge Graphs},
  booktitle = {ISWC},
  year      = {2023}
}
```

**Dihedron**

```bibtex
@inproceedings{Nayyeri2022Dihedron,
  title     = {Dihedron Algebraic Embeddings for Spatio-Temporal Knowledge Graph Completion},
  booktitle = {ESWC},
  year      = {2022}
}
```

---

## 9) License

This repository reuses parts of TemporalFC. Please refer to the upstream license and include attribution when publishing results based on this code.

