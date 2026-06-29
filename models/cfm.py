"""
models/cfm.py
=============
Conditional Flow Matching (CFM) per il phase space modeling.

Basato su:
    Lipman et al. (2022) "Flow Matching for Generative Modeling"
    ICLR 2023 — arXiv:2210.02747

    Farmer et al. (2025) "Generative Monte Carlo for Constant-Cost Particle Transport"
    arXiv:2512.13965 — Blueprint diretto per questo task

Il modello apprende un campo vettoriale v_theta(x_t, t, c) tale che
integrare l'ODE:
    dx/dt = v_theta(x_t, t, c)
trasporti campioni da N(0, I) alla distribuzione del phase space p(s | c).

Percorso OT (Optimal Transport):
    x_t = (1 - t) * x_0 + t * x_1
    u_t(x_t | x_1) = x_1 - x_0   (costante lungo la traiettoria)

Vantaggi rispetto ai NF discreti:
    - Architettura della rete completamente libera (nessun vincolo di biettività)
    - Training simulation-free (nessuna simulazione ODE durante il training)
    - Loss MSE diretta, convergenza veloce
    - Percorsi OT più corti → meno step di integrazione in inferenza
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("[WARNING] torchdiffeq non disponibile. Installare: pip install torchdiffeq")


# ─── Sinusoidal time embedding ────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Embedding sinusoidale per il tempo t ∈ [0, 1].

    Trasforma lo scalare t in un vettore di dimensione embed_dim tramite:
        [sin(2π f_1 t), cos(2π f_1 t), ..., sin(2π f_k t), cos(2π f_k t)]
    """

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        # Frequenze geometricamente spaziate [1, max_freq]
        freqs = torch.exp(
            torch.linspace(0, np.log(1000.0), embed_dim // 2)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t : (B,) o (B, 1)  scalare temporale in [0, 1]
        Returns: (B, embed_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)

        args = 2 * torch.pi * t * self.freqs.unsqueeze(0)  # (B, embed_dim/2)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, embed_dim)


# ─── Rete velocity (campo vettoriale) ─────────────────────────────────────────

class VelocityNet(nn.Module):
    """
    Rete neurale che parametrizza il campo vettoriale v_theta(x_t, t, c).

    Input:  concatenazione di [x_t (7D), time_embedding (32D), c_embedding (32D)]
    Output: velocità v ∈ R^7 (stessa dimensione di x)

    Architettura: MLP residuale con skip connections.
    La semplicità è deliberata: per d=7 un MLP è sufficiente e veloce.

    Parametri
    ---------
    dim        : dimensione del phase space (7)
    time_dim   : dimensione dell'embedding temporale (32)
    cond_dim   : dimensione del vettore di condizione (0 = non condizionato)
    context_dim: dimensione dell'embedding di condizione (32)
    hidden_dim : dimensione dei layer nascosti
    n_layers   : numero di layer residuali
    """

    def __init__(
        self,
        dim:         int = 7,
        time_dim:    int = 32,
        cond_dim:    int = 0,
        context_dim: int = 32,
        hidden_dim:  int = 256,
        n_layers:    int = 4,

    ):
        super().__init__()
        self.dim  = dim
        self.cond_dim = cond_dim

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        # Condition encoder (se condizionato)
        if cond_dim > 0:
            self.cond_embed = nn.Sequential(
                nn.Linear(cond_dim, context_dim),
                nn.SiLU(),
                nn.Linear(context_dim, context_dim),
            )
            in_dim = dim + time_dim + context_dim
        else:
            self.cond_embed = None
            in_dim = dim + time_dim

        # Proiezione iniziale
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Layer residuali
        self.res_layers = nn.ModuleList([
            _ResidualBlock(hidden_dim) for _ in range(n_layers)
        ])

        # Output
        self.output_proj = nn.Linear(hidden_dim, dim)

        # Inizializzazione: output vicino a zero inizialmente
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calcola il campo vettoriale v_theta(x, t, c).

        Parameters
        ----------
        x : (B, 7)   posizione corrente nel percorso OT
        t : (B,)     tempo in [0, 1]
        c : (B, cond_dim)  condizioni [opzionale]

        Returns
        -------
        v : (B, 7)  velocità
        """
        t_emb = self.time_embed(t)                 # (B, time_dim)

        parts = [x, t_emb]
        if c is not None and self.cond_embed is not None:
            c_emb = self.cond_embed(c)             # (B, context_dim)
            parts.append(c_emb)

        h = torch.cat(parts, dim=-1)               # (B, in_dim)
        h = self.input_proj(h)                     # (B, hidden_dim)

        for layer in self.res_layers:
            h = layer(h)

        return self.output_proj(h)                 # (B, 7)


class _ResidualBlock(nn.Module):
    """Blocco residuale: h → h + MLP(LayerNorm(h))"""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.net(h)


# ─── CFM Flow completo ────────────────────────────────────────────────────────

class PhaseSpaceCFM(nn.Module):
    """
    Conditional Flow Matching per il phase space 7D.

    Combina la rete velocity con la logica di training CFM e
    il campionamento tramite ODE solve.

    Training (metodo cfm_loss):
        1. Campiona x_0 ~ N(0, I) e x_1 ~ p_data(s | c)
        2. Interpola: x_t = (1-t)*x_0 + t*x_1  (percorso OT)
        3. Target velocity: u_t = x_1 - x_0  (costante)
        4. Loss: MSE(v_theta(x_t, t, c), u_t)

    Inferenza (metodo sample):
        Integra l'ODE dx/dt = v_theta(x, t, c) da t=0 a t=1
        usando DOPRI5 (Runge-Kutta adattivo ordine 4/5).
    """

    def __init__(
        self,
        dim:         int = 7,
        time_dim:    int = 32,
        cond_dim:    int = 0,
        context_dim: int = 32,
        hidden_dim:  int = 256,
        n_layers:    int = 4,
        sigma_min:   float = 1e-4,  # rumore minimo per regularizzazione
    ):
        super().__init__()
        self.dim       = dim
        self.cond_dim  = cond_dim
        self.sigma_min = sigma_min

        self.velocity_net = VelocityNet(
            dim=dim,
            time_dim=time_dim,
            cond_dim=cond_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

    def cfm_loss(
        self,
        x1: torch.Tensor,
        c:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calcola la Conditional Flow Matching loss:

        L_CFM(θ) = E_{t~U[0,1], x0~N(0,I), x1~p_data}
                    [ ||v_θ(x_t, t, c) - u_t(x_t|x_1)||^2 ]

        dove x_t = (1-t)*x_0 + t*x_1  e  u_t = x_1 - x_0

        Parameters
        ----------
        x1 : (B, 7)  campioni reali del phase space
        c  : (B, cond_dim)  condizioni [opzionale]
        """
        B = x1.shape[0]

        # Campiona tempo t uniformemente in [0, 1]
        t = torch.rand(B, device=x1.device)

        # Campiona punto iniziale dal prior N(0, I)
        x0 = torch.randn_like(x1)

        # Percorso OT con piccola regularizzazione sigma_min
        # x_t = (1 - (1 - σ_min) * t) * x_0 + t * x_1
        # (leggermente modificato rispetto all'interpolazione lineare pura
        #  per evitare collasso numerico a t=1)
        xt = (1 - (1 - self.sigma_min) * t.unsqueeze(-1)) * x0 \
             + t.unsqueeze(-1) * x1

        # Target velocity (costante per percorso OT)
        ut = x1 - (1 - self.sigma_min) * x0

        # Velocità predetta
        vt = self.velocity_net(xt, t, c)

        # MSE loss
        return ((vt - ut) ** 2).mean()

    def sample(
        self,
        n_samples: int,
        c: Optional[torch.Tensor] = None,
        n_steps: int = 100,
        method: str = "dopri5",
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> torch.Tensor:
        """
        Genera n_samples nuovi vettori di phase space integrando l'ODE.

        dx/dt = v_theta(x, t, c),  x(0) ~ N(0, I)

        Parameters
        ----------
        n_samples : numero di campioni da generare
        c         : (n_samples, cond_dim) condizioni
        method    : ODE solver ("dopri5" adattivo, "rk4" fisso)
        atol, rtol: tolleranze per il solver adattivo

        Returns
        -------
        x1 : (n_samples, 7) vettori generati
        """
        if not TORCHDIFFEQ_AVAILABLE:
            raise RuntimeError("Installare torchdiffeq: pip install torchdiffeq")

        device = next(self.parameters()).device
        x0 = torch.randn(n_samples, self.dim, device=device)

        # ODE function con context fisso (closure)
        def ode_func(t, x):
            # t è uno scalare; lo replichiamo per il batch
            t_batch = t.expand(x.shape[0])
            return self.velocity_net(x, t_batch, c)

        t_span = torch.tensor([0.0, 1.0], device=device)

        with torch.no_grad():
            trajectory = odeint(
                ode_func,
                x0,
                t_span,
                method=method,
                atol=atol,
                rtol=rtol,
            )
            # trajectory: (2, n_samples, 7) — prendiamo t=1
            x1 = trajectory[-1]

        return x1

    def sample_fast(
        self,
        n_samples: int,
        c: Optional[torch.Tensor] = None,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """
        Campionamento veloce con Euler discreto (n_steps step fissi).
        Meno accurato ma 10-100x più veloce di dopri5 per generazione massiva.

        Usare per: generazione di file PS da milioni di campioni
        Usare dopri5 per: confronti di qualità e validation
        """
        device = next(self.parameters()).device
        x = torch.randn(n_samples, self.dim, device=device)
        dt = 1.0 / n_steps

        with torch.no_grad():
            for i in range(n_steps):
                t_val = i / n_steps
                t_batch = torch.full((n_samples,), t_val, device=device)
                v = self.velocity_net(x, t_batch, c)
                x = x + dt * v

        return x


# ─── Trainer CFM ──────────────────────────────────────────────────────────────

class CFMTrainer:
    """
    Trainer per Conditional Flow Matching.

    Training diretto con loss MSE (nessun discriminatore, nessun ODE).
    Convergenza solitamente più veloce e stabile rispetto a GAN e DM.

    Iperparametri:
        lr    : learning rate (default: 1e-3, più alto rispetto a NF classici
                perché CFM non ha issues di overflow nel log-det)
    """

    def __init__(
        self,
        model:        PhaseSpaceCFM,
        device:       str = "cpu",
        lr:           float = 1e-3,
        weight_decay: float = 1e-5,
        clip_grad:    float = 1.0,
        scheduler_patience: int = 10,
        epochs:       int = 200,
    ):
        self.model   = model.to(device)
        self.device  = device
        self.clip_grad = clip_grad

        self.opt = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=epochs,
            eta_min=1e-6,
        )

        self.history = {
            "train_loss": [], "val_loss": [], "lr": []
        }

    def _to(self, x):
        return x.to(self.device) if isinstance(x, torch.Tensor) else x

    def train_step(
        self,
        x1_batch: torch.Tensor,
        c_batch: Optional[torch.Tensor] = None,
    ) -> float:
        """Singolo step di training. Returns: CFM loss (float)"""
        x1 = self._to(x1_batch)
        c  = self._to(c_batch) if c_batch is not None else None

        self.model.train()
        self.opt.zero_grad()

        loss = self.model.cfm_loss(x1, c)
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.opt.step()

        return loss.item()

    @torch.no_grad()
    def val_step(
        self,
        x1_val: torch.Tensor,
        c_val: Optional[torch.Tensor] = None,
        batch_size: int = 4096,
    ) -> float:
        """Calcola la loss sul validation set."""
        self.model.eval()
        x1_val = self._to(x1_val)
        c_val  = self._to(c_val) if c_val is not None else None

        total = 0.0
        n = 0
        for i in range(0, len(x1_val), batch_size):
            xb = x1_val[i:i + batch_size]
            cb = c_val[i:i + batch_size] if c_val is not None else None
            total += self.model.cfm_loss(xb, cb).item()
            n += 1

        return total / n

    def save(self, path: str):
        torch.save({
            "model":   self.model.state_dict(),
            "opt":     self.opt.state_dict(),
            "history": self.history,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["opt"])
        self.history = ckpt.get("history", self.history)
