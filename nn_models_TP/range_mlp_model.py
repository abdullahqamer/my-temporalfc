import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class RangeMLPModel(pl.LightningModule):
    def __init__(self,
                 num_entities,
                 num_relations,
                 num_times,
                 embedding_dim=100,
                 hidden_dim=256,
                 dropout=0.3,
                 lr=1e-3,
                 idx_time_dict=None,
                 order_penalty_lambda=0.0,
                 use_interaction=False,
                 loss_type="l1",
                 huber_beta=0.5,
                 ):
        super().__init__()
        self.name = 'RangeMLP'
        self.save_hyperparameters()
        self.idx_time_dict = idx_time_dict or {}  # Store the dictionary
        self.year_idx_dict = {v: k for k, v in
                              self.idx_time_dict.items()}  # Maps indices to year strings (e.g., 0: '1984')

        # DEBUG: show samples of both mappings
        print(" [DEBUG] time map (year->idx) sample:", list(self.idx_time_dict.items())[:5])
        print(" [DEBUG] time map (idx->year) sample:", list(self.year_idx_dict.items())[:5])

        self.embedding_dim = embedding_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_times = num_times
        self.lr = lr
        self.order_penalty_lambda = order_penalty_lambda
        self.use_interaction = use_interaction
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
        in_feats = embedding_dim * (4 if use_interaction else 3)
        self.fc1 = nn.Linear(in_feats, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 3)

        ''' self.fc1 = nn.Linear(embedding_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, 2) '''

        self.dropout = nn.Dropout(dropout)

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

    def forward(self, h_idx, r_idx, t_idx):
        h = self.ent_emb(h_idx)
        r = self.rel_emb(r_idx)
        t = self.ent_emb(t_idx)

        #x = torch.cat([h, r, t], dim=1)
        #x = torch.cat([h, r, t, torch.abs(h - t)], dim=1)
        if self.use_interaction:
            x = torch.cat([h, r, t, torch.abs(h - t)], dim=1)
        else:
            x = torch.cat([h, r, t], dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        o = self.out(x)  # [B,3]
        start_raw = o[:, 0]
        delta_raw = o[:, 2]
        #output = self.out(x)
        #return output  # shape: [batch_size, 2]

        # map to [0,1]
        start_norm = torch.sigmoid(start_raw)
        # monotonic end: start + remaining * sigmoid(delta)
        end_norm = start_norm + (1.0 - start_norm) * torch.sigmoid(delta_raw)

        return torch.stack([start_norm, end_norm], dim=1)  # [B,2] in normalized space

    def _loss_on_norm(self, preds_norm, y1_idx, y2_idx):
        """Compute loss in normalized space with optional order penalty."""
        start_norm, end_norm = preds_norm[:, 0], preds_norm[:, 1]
        y1_norm = self._to_norm(y1_idx.float())
        y2_norm = self._to_norm(y2_idx.float())

        loss = self.loss_fn(start_norm, y1_norm) + self.loss_fn(end_norm, y2_norm)

        # Extra order penalty (should be redundant thanks to monotonic end, but harmless)
        if self.order_penalty_lambda > 0:
            loss = loss + self.order_penalty_lambda * F.relu(start_norm - end_norm).mean()
        return loss, start_norm, end_norm, y1_norm, y2_norm

    def training_step(self, batch, batch_idx):
        h, r, t, y1_idx, y2_idx = batch
        preds_norm = self.forward(h, r, t)  # [B,2] normalized [0,1]
        loss, start_norm, end_norm, y1_norm, y2_norm = self._loss_on_norm(preds_norm, y1_idx, y2_idx)
        #preds = self.forward(h, r, t)  # [B,2], requires_grad=True
        #start_pred, end_pred = preds[:, 0], preds[:, 1]

        # loss in index space (differentiable)
        #loss = self.loss_fn(start_pred, y1_idx.float()) + self.loss_fn(end_pred, y2_idx.float())
        self.log("train_loss", loss, prog_bar=True)

        # Logging MAE in YEARS (no grad)
        with torch.no_grad():
            # denormalize to indices, round/clamp, then map idx->year
            start_idx_pred = self._to_index(start_norm).round().clamp_(0, self.num_times - 1).long()
            # end is already >= start by construction, still clamp to be safe
            end_idx_pred = self._to_index(end_norm).round().clamp_(0, self.num_times - 1).long()

            tgt_start_years = torch.tensor(
                [float(self.year_idx_dict[int(i.item())]) for i in y1_idx], device=self.device
            )
            tgt_end_years = torch.tensor(
                [float(self.year_idx_dict[int(i.item())]) for i in y2_idx], device=self.device
            )
            pred_start_years = torch.tensor(
                [float(self.year_idx_dict[int(i.item())]) for i in start_idx_pred], device=self.device
            )
            pred_end_years = torch.tensor(
                [float(self.year_idx_dict[int(i.item())]) for i in end_idx_pred], device=self.device
            )

            mae_start_years = torch.mean(torch.abs(pred_start_years - tgt_start_years))
            mae_end_years   = torch.mean(torch.abs(pred_end_years   - tgt_end_years))
            self.log("train_mae_year_start", mae_start_years, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train_mae_year_end",   mae_end_years,   prog_bar=False, on_step=False, on_epoch=True)

        return loss

        '''start_pred, end_pred = self.forward_triples(h, r, t, y1_idx, y2_idx)
        # Convert indices to years (assuming self.idx_time_dict is accessible)
        y1_years = torch.tensor([int(self.year_idx_dict.get(i.item(), '0')) for i in y1_idx],
                                device=self.device).float()
        y2_years = torch.tensor([int(self.year_idx_dict.get(i.item(), '0')) for i in y2_idx],
                                device=self.device).float()
        pred_start_idx = start_pred.round().long().clamp(min=0, max=self.num_times - 1)
        pred_end_idx = end_pred.round().long().clamp(min=0, max=self.num_times - 1)
        pred_start_years = torch.tensor([int(self.year_idx_dict.get(p.item(), '0')) for p in pred_start_idx],
                                        device=self.device).float()
        pred_end_years = torch.tensor([int(self.year_idx_dict.get(p.item(), '0')) for p in pred_end_idx],
                                      device=self.device).float()
        loss = self.loss_fn(pred_start_years, y1_years) + self.loss_fn(pred_end_years,
                                                                                     y2_years)
        self.log('train_loss', loss)
        return loss

         h, r, t, y1, y2 = batch
        target = torch.stack([y1.float(), y2.float()], dim=1)
        output = self.forward(h, r, t)
        loss = self.loss_fn(output, target)
        self.log("train_loss", loss)
        return loss '''

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "strict": False,
            },
        }
        '''return torch.optim.Adam(self.parameters(), lr=self.lr) '''

    def forward_triples(self, h, r, t, y1, y2, type=None):
        preds_norm = self.forward(h, r, t)
        start_idx = self._to_index(preds_norm[:, 0])
        end_idx = self._to_index(preds_norm[:, 1])
        return start_idx, end_idx
