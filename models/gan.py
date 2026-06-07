"""
models/gan.py
=============
Baseline WGAN-GP (Wasserstein GAN con Gradient Penalty) per il phase space modeling.

Riproduce fedelmente l'architettura del paper:
    Sarrut et al. (2019) "GAN for Compact Beam Source Modelling in MC Simulations"
    DOI: 10.1088/1361-6560/ab3fc3

Estensioni:
    - Supporto a cGAN condizionato su vettore c = [E_nom, jaw_x, jaw_y]
    - Gradient penalty (WGAN-GP) per maggiore stabilità rispetto alla weight clipping

Riferimento matematico:
    min_G max_D  E_real[D(x)] - E_gen[D(G(z))]
                + lambda * E[(||grad_x D(x_hat)||_2 - 1)^2]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List


# ─── Blocchi architetturali ────────────────────────────────────────────────────

def _linear_block(
    in_features: int,
    out_features: int,
    norm: str = "layer",
    activation: str = "leaky_relu",
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    Blocco lineare con normalizzazione e attivazione opzionale.

    Parameters
    ----------
    norm       : "layer", "batch", o "none"
    activation : "leaky_relu", "relu", "tanh", o "none"
    """
    layers = [nn.Linear(in_features, out_features)]

    if norm == "layer":
        layers.append(nn.LayerNorm(out_features))
    elif norm == "batch":
        layers.append(nn.BatchNorm1d(out_features))

    if activation == "leaky_relu":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    elif activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif activation == "tanh":
        layers.append(nn.Tanh())

    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


# ─── Generatore ───────────────────────────────────────────────────────────────

class PhaseSpaceGenerator(nn.Module):
    """
    Generatore G: z (+ c) → s

    Input:
        z : (B, latent_dim)  rumore gaussiano
        c : (B, cond_dim)    condizioni [opzionale]

    Output:
        s : (B, 7)  vettore di phase space (nello spazio normalizzato)

    Post-processing:
        La direzione (dx, dy, dz) viene normalizzata a norma 1 nell'output
        per rispettare il vincolo fisico ||d||_2 = 1.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        cond_dim:   int = 0,
        hidden_dims: List[int] = [256, 512, 512, 256],
        output_dim: int = 7,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim   = cond_dim
        self.output_dim = output_dim

        in_dim = latent_dim + cond_dim
        layers = []
        for h in hidden_dims:
            layers.append(_linear_block(in_dim, h, norm="layer"))
            in_dim = h

        self.trunk  = nn.Sequential(*layers)
        self.head   = nn.Linear(in_dim, output_dim)

    def forward(
        self,
        z: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = z if c is None else torch.cat([z, c], dim=-1)
        x = self.trunk(x)
        s = self.head(x)

        # Normalizzazione fisica della direzione (canali 3, 4, 5)
        d     = s[:, 3:6]
        d_norm = d / (torch.norm(d, dim=-1, keepdim=True) + 1e-8)
        s     = torch.cat([s[:, :3], d_norm, s[:, 6:]], dim=-1)

        return s

    def sample(
        self,
        n_samples: int,
        c: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Campiona n_samples nuovi vettori di phase space."""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        if c is not None and c.device.type != device:
            c = c.to(device)
        with torch.no_grad():
            return self.forward(z, c)


# ─── Discriminatore / Critic ───────────────────────────────────────────────────

class PhaseSpaceCritic(nn.Module):
    """
    Critic D: s (+ c) → scalar score (NON una probabilità — WGAN)

    Input:
        s : (B, 7)           vettore di phase space
        c : (B, cond_dim)    condizioni [opzionale]

    Output:
        score : (B, 1)  score scalare (non bounded)

    Nota: in WGAN il critic NON usa sigmoid nell'output.
    LayerNorm al posto di BatchNorm (più stabile per WGAN-GP).
    """

    def __init__(
        self,
        input_dim:   int = 7,
        cond_dim:    int = 0,
        hidden_dims: List[int] = [256, 512, 512, 256],
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.cond_dim = cond_dim

        in_dim = input_dim + cond_dim
        layers = []
        for h in hidden_dims:
            layers.append(_linear_block(in_dim, h, norm="layer",
                                        activation="leaky_relu", dropout=dropout))
            in_dim = h

        self.trunk = nn.Sequential(*layers)
        self.head  = nn.Linear(in_dim, 1)

    def forward(
        self,
        s: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = s if c is None else torch.cat([s, c], dim=-1)
        return self.head(self.trunk(x))


# ─── Gradient Penalty ─────────────────────────────────────────────────────────

def gradient_penalty(
    critic: PhaseSpaceCritic,
    real:   torch.Tensor,
    fake:   torch.Tensor,
    c:      Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Calcola la gradient penalty per WGAN-GP:

        GP = E[(||grad_x D(x_hat)||_2 - 1)^2]

    dove x_hat = epsilon * real + (1 - epsilon) * fake  (interpolazione lineare)
    """
    B = real.size(0)
    eps = torch.rand(B, 1, device=device)
    eps = eps.expand_as(real)

    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)

    score = critic(x_hat, c)
    grads = torch.autograd.grad(
        outputs=score,
        inputs=x_hat,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
    )[0]

    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# ─── Trainer WGAN-GP ──────────────────────────────────────────────────────────

class WGANGPTrainer:
    """
    Trainer per WGAN-GP con supporto a condizionamento.

    Iperparametri chiave:
        n_critic     : step del critic per ogni step del generatore (default: 5)
        lambda_gp    : peso della gradient penalty (default: 10)
        lr           : learning rate per entrambi (default: 1e-4)
    """

    def __init__(
        self,
        generator:  PhaseSpaceGenerator,
        critic:     PhaseSpaceCritic,
        device:     str = "cpu",
        lr:         float = 1e-4,
        n_critic:   int   = 5,
        lambda_gp:  float = 10.0,
        betas:      Tuple[float, float] = (0.0, 0.9),  # Adam con beta1=0 per WGAN
    ):
        self.G        = generator.to(device)
        self.D        = critic.to(device)
        self.device   = device
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp

        self.opt_G = optim.Adam(generator.parameters(), lr=lr, betas=betas)
        self.opt_D = optim.Adam(critic.parameters(),    lr=lr, betas=betas)

        self.history = {"loss_G": [], "loss_D": [], "w_dist": [], "gp": []}

    def _to(self, x):
        return x.to(self.device) if isinstance(x, torch.Tensor) else x

    def train_step(
        self,
        real_batch: torch.Tensor,
        cond_batch: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Esegue un passo di training (n_critic step D + 1 step G).

        Returns
        -------
        dict con loss_D, loss_G, wasserstein_estimate, gp
        """
        real = self._to(real_batch)
        c    = self._to(cond_batch) if cond_batch is not None else None
        B    = real.size(0)

        # ── Critic update (n_critic volte) ──────────────────────────────────
        loss_D_total = 0.0
        gp_total     = 0.0
        for _ in range(self.n_critic):
            z    = torch.randn(B, self.G.latent_dim, device=self.device)
            fake = self.G(z, c).detach()

            score_real = self.D(real, c)
            score_fake = self.D(fake, c)

            gp = gradient_penalty(self.D, real, fake, c, self.device)

            # Wasserstein loss + GP
            loss_D = -(score_real.mean() - score_fake.mean()) + self.lambda_gp * gp

            self.opt_D.zero_grad()
            loss_D.backward()
            self.opt_D.step()

            loss_D_total += loss_D.item()
            gp_total     += gp.item()

        # ── Generator update (1 volta) ───────────────────────────────────────
        z    = torch.randn(B, self.G.latent_dim, device=self.device)
        fake = self.G(z, c)

        loss_G = -self.D(fake, c).mean()

        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

        # Stima distanza di Wasserstein (senza GP)
        w_dist = -(loss_D_total / self.n_critic) + (self.lambda_gp * gp_total / self.n_critic)

        metrics = {
            "loss_G": loss_G.item(),
            "loss_D": loss_D_total / self.n_critic,
            "w_dist": w_dist,
            "gp":     gp_total / self.n_critic,
        }

        for k, v in metrics.items():
            self.history[k].append(v)

        return metrics

    def save(self, path: str):
        torch.save({
            "generator":  self.G.state_dict(),
            "critic":     self.D.state_dict(),
            "opt_G":      self.opt_G.state_dict(),
            "opt_D":      self.opt_D.state_dict(),
            "history":    self.history,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["generator"])
        self.D.load_state_dict(ckpt["critic"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        self.history = ckpt.get("history", self.history)
