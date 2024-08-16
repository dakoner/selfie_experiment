from torch import nn
import torch
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy


class VAEEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        pad_idx: int,
        q_dim: int = 512,
        q_layers: int = 1,
        latent_dim: int = 128,
        emb_dim: int = 256,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.n_vocab = n_vocab
        self.q_dim = q_dim
        self.q_layers = q_layers

        # Embeddings layer to avoid the need to one hot encode manually
        self.embed = nn.Embedding(n_vocab, emb_dim, self.pad_idx)
        self.encoder_rnn = nn.GRU(
            emb_dim,
            self.q_dim,
            num_layers=self.q_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.q_mu = nn.Linear(self.q_dim, latent_dim)
        self.q_logvar = nn.Linear(self.q_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_emb = self.embed(x)
        h_complete, hidden = self.encoder_rnn(x_emb)
        # batch first, and use only last layer
        hidden = hidden.permute(1, 0, 2)[:, :1, :]
        mu, log_var = self.q_mu(hidden), self.q_logvar(hidden)
        return mu, log_var, x_emb, hidden, h_complete


class VAEDecoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        d_dim: int = 512,
        d_layers: int = 1,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.d_dim = d_dim
        self.d_layers = d_layers

        self.decoder_rnn = nn.GRU(
            latent_dim,
            self.d_dim,
            num_layers=self.d_layers,
            batch_first=True,
        )
        self.decoder_fc = nn.Linear(self.d_dim, n_vocab)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None) -> torch.Tensor:
        output, hidden = self.decoder_rnn(x)
        return self.decoder_fc(output), hidden



class VAE(pl.LightningModule):
    def __init__(
        self,
        n_vocab: int,
        pad_idx: int,
        q_dim: int = 512,
        q_layers: int = 1,
        latent_dim: int = 128,
        emb_dim: int = 256,
        d_dim: int = 512,
        d_layers: int = 1,
        lr: float = 1e-4
    ):
        super().__init__()

        self.lr = lr
        self.encoder = VAEEncoder(n_vocab=n_vocab, pad_idx=pad_idx, emb_dim=emb_dim, q_dim=q_dim, q_layers=q_layers, latent_dim=latent_dim)
        self.decoder = VAEDecoder(
            n_vocab=n_vocab, latent_dim=latent_dim,
            d_dim=d_dim, d_layers=d_layers
        )
        self.accuracy = MulticlassAccuracy(
            n_vocab,
        )
        self.cross_entropy = nn.CrossEntropyLoss(
            reduction="mean",
        )

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(self.device)
        z = mu + noise * std
        return z

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x = batch

        mu, log_var, x_emb, hidden, h_complete = self.encoder(x)

        # reparameterization trick
        std = torch.exp(log_var / 2.)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # This is the same as doing a for loop and passing the hidden state
        x_dec, _ = self.decoder(z.repeat(1, x.size(1), 1))

        # reconstruction loss

        recon_loss = self.cross_entropy(x_dec.permute(0, 2, 1), x).mean()
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        ).mean()

        elbo = (kl_loss - recon_loss)

        # could add a weight for kl
        loss = (recon_loss + kl_loss)
        accuracy = self.accuracy(x_dec.argmax(-1), x)

        self.log_dict({
            'loss': loss,
            'elbo': elbo,
            'kl': kl_loss,
            'recon_loss': recon_loss,
            'accuracy': accuracy,
        })

        return loss
    