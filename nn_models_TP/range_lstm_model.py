import torch
import torch.nn as nn
import pytorch_lightning as pl

class RangeLSTMModel(pl.LightningModule):
    """
    Given a pair ((h,r,t,y1), (h,r,t,y2)), predicts both start and end time buckets.
    """
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        num_times: int,
        embedding_dim: int = 100,
        lstm_hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.name = "range-lstm"

        # Embedding layers
        self.entity_embeddings   = nn.Embedding(num_entities,   embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations,  embedding_dim)
        self.time_embeddings = nn.Embedding(num_times, embedding_dim)

        # Shared LSTM encoder for a single (h, r, t)
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 4,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Fusion + two output heads (predict scalar start/end)
        # We'll concatenate the two LSTM outputs (start & end)
        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.start_reg = nn.Linear(lstm_hidden_dim, 1)
        self.end_reg = nn.Linear(lstm_hidden_dim, 1)
        #MSE Loss for continuous year prediction
        self.loss_fn = nn.MSELoss()
        #self.start_head = nn.Linear(lstm_hidden_dim, num_times)
        #self.end_head   = nn.Linear(lstm_hidden_dim, num_times)


        # Loss
        #self.loss_fn = nn.CrossEntropyLoss()

    def encode(self, h, r, t):
        # lookup embeddings
        he = self.entity_embeddings(h)   # (B, D)
        re = self.relation_embeddings(r) # (B, D)
        te = self.entity_embeddings(t)   # (B, D)
        # build a sequence of length 3
        seq = torch.stack([he, re, te], dim=1)  # (B, 3, D)
        out, (hn, cn) = self.lstm(seq)          # hn: (num_layers, B, H)
        return hn[-1]                           # (B, H)

    def forward(self, h, r, o, y1):
        #embed all inputs
        he = self.entity_embeddings(h)
        re = self.relation_embeddings(r)
        oe = self.entity_embeddings(o)
        y1e = self.time_embeddings(y1)

        #concatenate into one feature vector
        x = torch.cat([he, re, oe, y1e], dim=1)
        #print(f"[DEBUG] LSTM x shape (should be NX4D): {x.shape}")
        x = x.unsqueeze(1)

        #pass through LSTM & fusion
        out, (hn, cn) = self.lstm(x)
        fused = self.fusion(hn[-1])

        #regression outputs
        start_pred = self.start_reg(fused).squeeze(1)
        end_pred = self.end_reg(fused).squeeze(1)
        return start_pred, end_pred

        """
        v = self.encode(h, r, t)
        # fuse
        fused = self.fusion(v)
        # logits
        #start_logits = self.start_head(fused)
        #end_logits   = self.end_head(fused)
        return self.start_head(fused), self.end_head(fused)
        """

    def forward_triples(self, h, r, t, y1, y2, type="training"):
        #debug to confirm you are reaching this method
        print(f"[DEBUG] [RangeLSTMModel] forward_triples called with " f"h:{h.shape}, r:{r.shape}, t:{t.shape}, y1:{y1.shape}, y2:{y2.shape}")
        #delegate to your normal forward
        return self.forward(h, r, t, y1)

    def training_step(self, batch, batch_idx):
        """
        h, r, t, y1, y2 = batch
        start_logits, end_logits = self.forward(h, r, t)

        loss_start = self.loss_fn(start_logits, y1)
        loss_end   = self.loss_fn(end_logits,   y2)
        """
        h, r, o, y1, y2 = batch
        start_pred, end_pred = self.forward(h, r, o, y1)

        print(f"[DEBUG training_step] start_pred.dim()={start_pred.dim()}, end_pred.dim()={end_pred.dim()}")

        #MSE to true year-indices (floats)
        loss_start = self.loss_fn(start_pred, y1.float())
        loss_end = self.loss_fn(end_pred, y2.float())
        loss = loss_start + loss_end

        # optional ordering penalty: ensure start ≤ end
        #penalty = torch.relu(start_pred.argmax(dim=1).float() - end_pred.argmax(dim=1).float()).mean()
        penalty = torch.relu(start_pred - end_pred).mean()
        loss = loss + 0.1 * penalty

        self.log('train_loss', loss, prog_bar=True)
        return loss

        print(f"y1 min: {y1.min().item()}, y1 max: {y1.max().item()}")
        print(f"y2 min: {y2.min().item()}, y2 max: {y2.max().item()}")
        print(f"num_times: {self.hparams.num_times}")

    def validation_step(self, batch, batch_idx):
        """
        h, r, t, y1, y2 = batch
        start_logits, end_logits = self.forward(h, r, t)

        print(f"[DEBUG] y1 min={y1.min().item()}, y1 max={y1.max().item()}, num_times={self.hparams.num_times}")
        print(f"[DEBUG] y2 min={y2.min().item()}, y2 max={y2.max().item()}, num_times={self.hparams.num_times}")

        loss = self.loss_fn(start_logits, y1) + self.loss_fn(end_logits, y2)
        """
        h, r, o, y1, y2 = batch
        start_pred, end_pred = self.forward(h, r, o, y1)

        # MSE to true year-indices (floats)
        loss_start = self.loss_fn(start_pred, y1.float())
        loss_end = self.loss_fn(end_pred, y2.float())
        loss = loss_start + loss_end

        # accuracy: both must be correct
        #start_acc = (start_pred.argmax(dim=1) == y1).float().mean()
        #end_acc   = (end_pred.argmax(dim=1)   == y2).float().mean()
        #acc = (start_acc + end_acc) / 2.0

        #regression
        pred_start = start_pred.round().long().clamp(min=0, max=self.hparams.num_times - 1)
        pred_end = end_pred.round().long().clamp(min=0, max=self.hparams.num_times - 1)

        #endpoint wise accuracy
        start_acc = (pred_start == y1).float().mean()
        end_acc = (pred_end == y2).float().mean()
        acc = (start_acc + end_acc) / 2.0

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc',  acc,      prog_bar=True)

        print(f"y1 min: {y1.min().item()}, y1 max: {y1.max().item()}")
        print(f"y2 min: {y2.min().item()}, y2 max: {y2.max().item()}")
        print(f"num_times: {self.hparams.num_times}")

    def test_step(self, batch, batch_idx):
        h, r, t, y1, y2 = batch
        print(f"[DEBUG test_step] batch shapes: h={h.shape}, r={r.shape}, o={t.shape}, y1={y1.shape}, y2={y2.shape}")
        #if batch_idx == 0:
         #   print(f"[DEBUG TEST BATCH] y1 true:", y1[:10].tolist())
          #  print(f"[DEBUG TEST BATCH] y2 true:", y2[:10].tolist())
        start_pred, end_pred = self.forward(h, r, t, y1)
        #if batch_idx == 0:
        pred_start = start_pred.round().long().clamp(min=0, max=self.hparams.num_times - 1)
        pred_end = end_pred.round().long().clamp(min=0, max=self.hparams.num_times - 1)
         #   print(f"[DEBUG TEST BATCH] y1 pred:", pred_start[:10].tolist())
          #  print(f"[DEBUG TEST BATCH] y2 pred:", pred_end[:10].tolist())
        acc = ((pred_start == y1) & (pred_end == y2)).float().mean()
        self.log('test_acc', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
