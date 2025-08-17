import torch
import torch.nn as nn
import pytorch_lightning as pl

class TemporalLSTMModel(pl.LightningModule):

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
        # save hyperparameters for .ckpt and logging
        self.save_hyperparameters()
        # give the model a name attribute for logging
        self.name = "temporal-lstm"

        # Embedding layers for entities and relations
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # LSTM that processes the sequence [head, relation, tail]
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classifier head: projects last hidden state to time-label logits
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, num_times)
        )

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, head_idx, rel_idx, tail_idx):
        # Lookup embeddings
        head_emb = self.entity_embeddings(head_idx)  # (B, D)
        rel_emb  = self.relation_embeddings(rel_idx) # (B, D)
        tail_emb = self.entity_embeddings(tail_idx)  # (B, D)

        # Create sequence tensor: (B, seq_len=3, D)
        seq = torch.stack([head_emb, rel_emb, tail_emb], dim=1)

        # LSTM forward
        out, (hn, cn) = self.lstm(seq)
        # Use the last layer's hidden state: (B, H)
        last_h = hn[-1]

        # Classifier to predict time bucket logits
        logits = self.classifier(last_h)
        return logits

    def forward_triples(self, head_idx, rel_idx, tail_idx, time_idx=None, sent_idx=None, ver=None, type="valid"):

        logits = self(head_idx, rel_idx, tail_idx)
        # convert logits to probabilities
        prob = torch.softmax(logits, dim=1)
        return prob

    def training_step(self, batch, batch_idx):
        head, rel, tail, time_idx, *_ = batch
        logits = self(head, rel, tail)
        loss = self.loss_fn(logits, time_idx)
        acc = (logits.argmax(dim=1) == time_idx).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc',  acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        head, rel, tail, time_idx, *_ = batch
        logits = self(head, rel, tail)
        loss = self.loss_fn(logits, time_idx)
        acc = (logits.argmax(dim=1) == time_idx).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc',  acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        head, rel, tail, time_idx, *_ = batch
        logits = self(head, rel, tail)
        acc = (logits.argmax(dim=1) == time_idx).float().mean()
        self.log('test_acc', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
