import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class ResBlock(nn.Module):
    def __init__(self, d, p_drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1  = nn.Linear(d, 4*d)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(4*d, d)
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):
        z = self.norm(x)
        z = self.fc2(self.act(self.fc1(z)))
        return x + self.drop(z)

class RangeMLPModel(pl.LightningModule):
    def __init__(self,
                 num_entities,
                 num_relations,
                 num_times,
                 embedding_dim=100,
                 hidden_dim=256,
                 dropout=0.3,
                 lr=1e-3,
                 weight_decay=0.0,
                 max_num_epochs=80,
                 idx_time_dict=None,
                 order_penalty_lambda=0.0,
                 use_interaction=False,
                 loss_type="l1",
                 huber_beta=0.5,
                 use_prod=False,
                 end_weight=1.0,
                 extra_order_pen=0.0,
                 emb_noise=0.0,
                 ):
        super().__init__()
        self.name = 'RangeMLP'
        self.save_hyperparameters()
        # --- canonical time map: always expose idx -> year (int) ---
        self.idx_time_dict = idx_time_dict or {}

        self.idx_to_year = None  # list[int] length = num_times
        self.year_idx_dict = {}  # dict[int idx] -> int year (same info as idx_to_year)

        def _is_int_like(x):
            try:
                int(str(x));
                return True
            except Exception:
                return False

        # Build a dense array idx_to_year using num_times
        T = int(self.hparams.num_times)
        arr = [None] * T

        if self.idx_time_dict:
            # self.idx_time_dict at this point is year->idx (per your file)
            for year, idx in self.idx_time_dict.items():
                if not (_is_int_like(year) and _is_int_like(idx)):
                    continue
                i = int(str(idx))
                if 0 <= i < T:
                    arr[i] = int(str(year))

        # Fill any holes by nearest-neighbour so every 0..T-1 is defined
        last = None
        for i in range(T):
            if arr[i] is None:
                nxt = None
                for j in range(i + 1, T):
                    if arr[j] is not None:
                        nxt = arr[j]
                        break
                arr[i] = last if nxt is None else (last if last is not None else nxt)
            last = arr[i]

        self.idx_to_year = arr
        self.year_idx_dict = {i: y for i, y in enumerate(arr)}

        print(f"[DEBUG] idx->year sample: {self.idx_to_year[:5]}  (num_times={T})")

        # --- year normalization helpers ---
        # Build a numeric list of YEARS from idx->year dict
        try:
            _years = [int(float(y)) for y in self.year_idx_dict.values()]
        except Exception:
            _years = []
        if len(_years) == 0:
            _years = [0, 1]  # fallback if empty

        min_year = min(_years)
        max_year = max(_years)
        year_span = max(1, max_year - min_year)

        # keep as buffers so they move to device automatically
        self.register_buffer("min_year", torch.tensor(float(min_year)))
        self.register_buffer("max_year", torch.tensor(float(max_year)))
        self.register_buffer("year_span", torch.tensor(float(year_span)))

        def _year_to_norm_fn(y: torch.Tensor) -> torch.Tensor:
            # y in YEARS (float) → [0,1]
            return (y - self.min_year) / self.year_span

        def _norm_to_year_fn(n: torch.Tensor) -> torch.Tensor:
            # n in [0,1] → YEARS (float)
            return self.min_year + n * self.year_span

        # bind as methods
        self._year_to_norm = _year_to_norm_fn
        self._norm_to_year = _norm_to_year_fn

        self.embedding_dim = embedding_dim

       # self.ent_emb = nn.Embedding(num_entities, embedding_dim)
       # self.rel_emb = nn.Embedding(num_relations, embedding_dim)

        # --- unfreeze plan ---
        self.unfreeze_epoch = -1
        self._did_freeze_once = False

        self.num_entities = num_entities
        self.num_relations = num_relations

        # --- duration prior buffers (learned online via EMA) ---
        self.register_buffer("rel_mu", torch.zeros(self.num_relations))
        self.register_buffer("rel_sigma", torch.ones(self.num_relations))
        self.register_buffer("rel_count", torch.zeros(self.num_relations))  # for debug/inspection
        self.prior_momentum = 0.05  # EMA step (try 0.02–0.1)
        self.prior_weight = 0.0  # loss weight (try 0.02–0.1)
        self._eps = 1e-6

        self.num_times = num_times
        self.lr = lr
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_num_epochs)
        # ---- model width & dropout (single change) ----
        self.hidden_dim = getattr(self, "hidden_dim", 1024)  # was 512; try 1024
        self.dropout_p = getattr(self, "dropout_p", 0.10)  # light regularization

        # --- SWA knobs ---
        self.use_swa = False  # flip to False to disable easily
        self.swa_start_epoch = max(5, int(0.9 * self.max_epochs))  # start near the end
        self.swa_update_freq = 1  # update every epoch
        self._swa_inited = False
        self._swa_model = None

        print(f"[DEBUG] RangeMLPModel: using lr={self.lr}")
        self.order_penalty_lambda = order_penalty_lambda
        self.use_interaction = use_interaction
        self.use_prod = use_prod
        self.end_weight = float(end_weight)
        self.extra_order_pen = float(extra_order_pen)
        self.emb_noise = float(emb_noise)  # 0.0 disables noise
        self.loss_type = loss_type
        self.huber_beta = huber_beta

        # Precompute scale for normalization <-> index space
        # idx in [0, num_times-1]  <->  norm in [0,1]
        self.register_buffer("_timescale", torch.tensor(float(max(1, num_times - 1))))

        # Embeddings (will be set from externally loaded tensors)
        self.ent_emb = nn.Embedding(num_entities, embedding_dim)
        self.rel_emb = nn.Embedding(num_relations, embedding_dim)

        # MLP layers
        #self.fc1 = nn.Linear(embedding_dim * 3, 512)
        # ---- wider trunk + 2-dim head ----
        parts = 3  # [h, r, t]
        if self.use_interaction:
            parts += 1  # |h - t|
        if self.use_prod:
            parts += 1  # h ⊙ r
        in_feats = embedding_dim * parts
        H = int(getattr(self, "hidden_dim", 1024))  # use the wider width you set above

        # trunk: 3x[Linear->GELU->Dropout]
        self.trunk = nn.Sequential(
            nn.Linear(in_feats, H),
            nn.GELU(),
            nn.Dropout(self.dropout_p),

            nn.Linear(H, H),
            nn.GELU(),
            nn.Dropout(self.dropout_p),

            nn.Linear(H, H),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )

        # 2-dim head: [start, delta] logits
        self.out = nn.Linear(H, 2)

        # init: Kaiming for trunk, zeros for small head
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        #self.dropout = nn.Dropout(dropout)

        if loss_type.lower() == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type.lower() == "huber":
            self.loss_fn = nn.SmoothL1Loss(beta=huber_beta)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Use Huber loss (SmoothL1Loss) for robustness and smoother convergence
        #self.loss_fn = nn.SmoothL1Loss(beta=0.5)
        #self.loss_fn = nn.L1Loss()

        # ---- helpers ----
    def _to_norm(self, idx_float: torch.Tensor) -> torch.Tensor:
        """Index -> normalized [0,1]."""
        return idx_float / self._timescale

    def _to_index(self, norm_float: torch.Tensor) -> torch.Tensor:
        """Normalized [0,1] -> index space (float)."""
        return norm_float * self._timescale

    def on_train_epoch_start(self):
        return  # no freezing/unfreezing in the baseline

    def forward(self, h_idx, r_idx, t_idx):
        h = self.ent_emb(h_idx)
        r = self.rel_emb(r_idx)
        t = self.ent_emb(t_idx)

        # tiny Gaussian noise on embeddings during training (regularization)
        if self.training and self.emb_noise > 0.0:
            n = float(self.emb_noise)
            h = h + n * torch.randn_like(h)
            r = r + n * torch.randn_like(r)
            t = t + n * torch.randn_like(t)

        feats = [h, r, t]
        if self.use_interaction:
            feats.append(torch.abs(h - t))  # |h - t|
        if self.use_prod:
            feats.append(h * t)  # h ⊙ t

        x = torch.cat(feats, dim=1)  # [B, in_feats]
        o = self.out(self.trunk(x))  # [B, 2]
        s_raw, d_raw = o.unbind(dim=-1)

        # standard sigmoid (no temperature/margins)
        start_norm = torch.sigmoid(s_raw)
        delta_norm = torch.sigmoid(d_raw)

        # end = start + (1 - start) * delta  → ensures start<=end in [0,1]
        end_norm = start_norm + (1.0 - start_norm) * delta_norm

        return torch.stack([start_norm, end_norm], dim=-1)  # [B, 2] in [0,1]

    def _loss_on_norm(self, preds_norm, y1_idx, y2_idx):
        """
        BASELINE: compute loss in INDEX space (not year-normalized).
        """
        start_norm, end_norm = preds_norm[:, 0], preds_norm[:, 1]

        # normalized -> float index
        start_idx_f = self._to_index(start_norm)
        end_idx_f = self._to_index(end_norm)

        # main loss directly on indices
        loss = self.loss_fn(start_idx_f, y1_idx.float()) + self.end_weight * self.loss_fn(end_idx_f, y2_idx.float())

        # return shapes compatible with training_step's logging (we won't use y*_norm now)
        # fabricate "norms" for logging consistency (not used for loss terms now)
        y1_norm = self._to_norm(y1_idx.float())
        y2_norm = self._to_norm(y2_idx.float())

        return loss, start_norm, end_norm, y1_norm, y2_norm

    def training_step(self, batch, batch_idx):
        h, r, t, y1_idx, y2_idx = batch

        preds_norm = self.forward(h, r, t)  # [B,2] in [0,1]

        # --- tiny label jitter in INDEX space (training only) ---
        # keep it *very* small; we regularize without changing labels meaningfully
        with torch.no_grad():
            jitter_std = 0.10  # ~5% of an index bin
            # create noise in float, same device
            eps1 = torch.randn_like(y1_idx, dtype=torch.float)
            eps2 = torch.randn_like(y2_idx, dtype=torch.float)
            y1_idx_j = (y1_idx.float() + jitter_std * eps1).clamp_(0, self.num_times - 1)
            y2_idx_j = (y2_idx.float() + jitter_std * eps2).clamp_(0, self.num_times - 1)

        # use jittered labels for the loss only
        loss, start_norm, end_norm, y1_norm, y2_norm = self._loss_on_norm(preds_norm, y1_idx_j, y2_idx_j)

        self.log("train_loss", loss, prog_bar=True)

        # log current LR once per epoch
        if self.trainer and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        else:
            current_lr = self.lr
        self.log("lr", current_lr, prog_bar=False, on_step=False, on_epoch=True)

        # --------- metrics in YEARS for logging only (no grad) ----------
        with torch.no_grad():
            # predictions to YEARS (continuous, no rounding for MAE)
            pred_start_years = self._norm_to_year(start_norm).clamp(self.min_year, self.max_year)
            pred_end_years = self._norm_to_year(end_norm).clamp(self.min_year, self.max_year)

            # targets: YEARS from idx
            idx2year = self.idx_to_year
            tgt_start_years = torch.tensor([float(idx2year[int(i)]) for i in y1_idx.tolist()], device=self.device)
            tgt_end_years = torch.tensor([float(idx2year[int(i)]) for i in y2_idx.tolist()], device=self.device)

            # enforce start<=end for reporting
            swap = pred_start_years > pred_end_years
            if swap.any():
                tmp = pred_start_years[swap].clone()
                pred_start_years[swap] = pred_end_years[swap]
                pred_end_years[swap] = tmp

            mae_start = torch.mean(torch.abs(pred_start_years - tgt_start_years))
            mae_end = torch.mean(torch.abs(pred_end_years - tgt_end_years))
            self.log("train_mae_year_start", mae_start, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train_mae_year_end", mae_end, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        h, r, t, y1_idx, y2_idx = batch
        preds_norm = self.forward(h, r, t)
        #loss = self.loss_fn(preds[:, 0], y1_idx.float()) + self.loss_fn(preds[:, 1], y2_idx.float())
        loss, _, _, _, _ = self._loss_on_norm(preds_norm, y1_idx, y2_idx)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

        ''' target = torch.stack([y1.float(), y2.float()], dim=1)
        output = self.forward(h, r, t)
        loss = self.loss_fn(output, target)
        self.log("val_loss", loss)
        return loss '''

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

        if not self.use_swa:
            return

        epoch = int(self.current_epoch)
        if epoch < self.swa_start_epoch:
            return

        # lazy init once we know device/dtype
        if not self._swa_inited:
            from torch.optim.swa_utils import AveragedModel
            self._swa_model = AveragedModel(self)
            self._swa_inited = True

        # update running average every epoch (or every k epochs)
        if (epoch - self.swa_start_epoch) % self.swa_update_freq == 0:
            self._swa_model.update_parameters(self)

    def on_fit_end(self):
        super().on_fit_end()
        # If SWA was used, copy averaged weights into the live model for test/export
        if self.use_swa and self._swa_inited and (self._swa_model is not None):
            for p_avg, p in zip(self._swa_model.parameters(), self.parameters()):
                p.data.copy_(p_avg.data)

    def test_step(self, batch, batch_idx):
        h, r, t, y1_idx, y2_idx = batch
        preds_norm = self.forward(h, r, t)
        #loss = self.loss_fn(preds[:, 0], y1_idx.float()) + self.loss_fn(preds[:, 1], y2_idx.float())
        loss, _, _, _, _ = self._loss_on_norm(preds_norm, y1_idx, y2_idx)
        self.log("test_loss", loss)
        return loss

        ''' h, r, t, y1, y2 = batch
        target = torch.stack([y1.float(), y2.float()], dim=1)
        output = self.forward(h, r, t)
        loss = self.loss_fn(output, target)
        self.log("test_loss", loss)
        return loss '''

    def configure_optimizers(self):
        # read values you already stored in __init__
        lr = float(getattr(self, "lr", 1e-3))
        wd = float(getattr(self, "weight_decay", 0.0))
        max_epochs = int(getattr(self, "max_epochs", 80))

        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        # Cosine decay from lr down to 10% of it by the final epoch
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(opt, T_max=max_epochs, eta_min=lr * 0.1)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",  # step each epoch
                "frequency": 1,
            },
        }

    def forward_triples(self, h, r, t, y1, y2, type=None):
        preds_norm = self.forward(h, r, t)  # [B, 2] in [0,1]
        start_idx = self._to_index(preds_norm[:, 0])  # float in [0, num_times-1]
        end_idx = self._to_index(preds_norm[:, 1])

        # clamp to valid index range (keep as float)
        start_idx = start_idx.clamp(0, self.num_times - 1)
        end_idx = end_idx.clamp(0, self.num_times - 1)

        # enforce start <= end
        swap = start_idx > end_idx
        if swap.any():
            tmp = start_idx[swap].clone()
            start_idx[swap] = end_idx[swap]
            end_idx[swap] = tmp

        return start_idx, end_idx  # floats; evaluation will round+cast


