"""
models/nsf.py
=============
Neural Spline Flow (NSF) per il phase space modeling.

Architettura basata su:
    Durkan et al. (2019) "Neural Spline Flows"
    NeurIPS 2019 — arXiv:1906.04032

Implementazione tramite la libreria `nflows` (PyTorch).

Il modello impara la distribuzione congiunta p(s | c) tramite:
    1. Composizione di K Rational-Quadratic NSF coupling layers
    2. Il vettore di condizione c viene concatenato in ogni coupling net
    3. La log-likelihood esatta è disponibile per ogni campione

Vantaggi rispetto alle GAN (dal punto di vista del phase space):
    - Likelihood esatta: log p(s) calcolabile per ogni campione
    - Campionamento in un singolo forward pass (veloce)
    - Training stabile con loss NLL (nessun bilanciamento G-D)
    - No mode collapse: la NLL penalizza uniformemente tutte le regioni
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, List, Tuple, Dict
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.nn.nets import ResidualNet 

   

# ─── Rete di condizionamento (context encoder) ────────────────────────────────

class ConditionEncoder(nn.Module):
    """
    Codifica il vettore di condizione c in un embedding di dimensione fissa.

    Input:  c = [E_nom, jaw_x, jaw_y]  (3D dopo normalizzazione)
    Output: embedding di dimensione context_dim
    """

    def __init__(
        self,
        cond_dim:    int = 3,
        context_dim: int = 32,
        hidden_dim:  int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, context_dim),
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


# ─── NSF coupling layer helper ────────────────────────────────────────────────

def _build_coupling_net(
    in_features:  int,
    out_features: int,
    context_features: Optional[int] = None,
    hidden_dim:   int = 128,
    n_layers:     int = 2,
) -> nn.Module:
    """
    Rete residuale usata all'interno di ogni coupling layer.
    Gestisce l'input condizionato concatenando il context.
    """
    return ResidualNet(
        in_features=in_features,
        out_features=out_features,
        hidden_features=hidden_dim,
        context_features=context_features,
        num_blocks=n_layers,
        activation=torch.nn.functional.silu,
        use_batch_norm=False,
    )


# ─── Modello NSF completo ─────────────────────────────────────────────────────

class PhaseSpaceNSF(nn.Module):
    """
    Neural Spline Flow per il phase space 7D.

    Architettura:
        - K coupling layers con Rational-Quadratic spline
        - Random permutation tra ogni layer (mixing delle dimensioni)
        - Context encoder per il condizionamento opzionale
        - Prior: N(0, I_7)

    Parametri chiave
    ----------------
    dim            : dimensione del phase space (default: 7)
    n_transforms   : numero di coupling layer (default: 6)
    hidden_dim     : dimensione dei layer interni della coupling net
    n_bins         : numero di bin della spline (default: 8)
    tail_bound     : limite della spline (campioni fuori range → linear tail)
    cond_dim       : dimensione del vettore di condizione (0 = non condizionato)
    context_dim    : dimensione dell'embedding di condizione
    """

    def __init__(
        self,
        dim:           int = 7,
        n_transforms:  int = 6,
        hidden_dim:    int = 128,
        n_bins:        int = 8,
        tail_bound:    float = 5.0,
        cond_dim:      int = 0,
        context_dim:   int = 32,
    ):
        super().__init__()

        

        self.dim        = dim
        self.cond_dim   = cond_dim
        self.context_dim = context_dim if cond_dim > 0 else None

        # Condition encoder (se condizionato)
        self.cond_encoder = (
            ConditionEncoder(cond_dim, context_dim) if cond_dim > 0 else None
        )

        # Costruzione dei coupling transforms
        transforms = []
        for k in range(n_transforms):
            # Permutazione casuale per mixare le dimensioni tra i layer
            transforms.append(RandomPermutation(features=dim))

            # Rational-Quadratic Spline coupling
            # Ogni layer trasforma metà delle dimensioni condizionatamente all'altra metà
            transforms.append(
                PiecewiseRationalQuadraticCouplingTransform(
                    mask=_alternating_mask(dim, k),
                    transform_net_create_fn=lambda in_f, out_f: _build_coupling_net(
                        in_features=in_f,
                        out_features=out_f,
                        context_features=context_dim if cond_dim > 0 else None,
                        hidden_dim=hidden_dim,
                    ),
                    num_bins=n_bins,
                    tails="linear",
                    tail_bound=tail_bound,
                )
            )

        # Prior gaussiano standard
        prior = StandardNormal([dim])

        self.flow = Flow(
            transform=CompositeTransform(transforms),
            distribution=prior,
        )

    def log_prob(
        self,
        s: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calcola la log-likelihood esatta per ogni campione.

        log p(s | c) = log p_z(f^{-1}(s)) + log|det J_{f^{-1}}(s)|

        Parameters
        ----------
        s : (B, 7)  vettore di phase space
        c : (B, cond_dim)  condizioni [opzionale]

        Returns
        -------
        log_prob : (B,) log-likelihood per campione
        """
        context = self._encode_context(c)
        return self.flow.log_prob(s, context=context)

    def sample(
        self,
        n_samples: int,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Genera n_samples nuovi vettori di phase space.

        Generazione in un singolo forward pass (ODE solve con NFow).
        """
        context = self._encode_context(c)
        with torch.no_grad():
            samples = self.flow.sample(n_samples, context=context)
            # nflows restituisce (1, n_samples, dim) quando context è fornito
            # oppure (n_samples, dim) senza context → squeeze per uniformare
            if samples.dim() == 3:
                samples = samples.squeeze(0)
            return samples

    def _encode_context(self, c: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if c is None or self.cond_encoder is None:
            return None
        return self.cond_encoder(c)

    def nll_loss(
        self,
        s: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Negative Log-Likelihood (da minimizzare durante il training).

        L(theta) = -E_{s ~ p_data}[log p_theta(s | c)]
        """
        return -self.log_prob(s, c).mean()


def _alternating_mask(dim: int, layer_idx: int) -> torch.Tensor:
    """
    Crea una maschera alternata per i coupling layers.
    Layer pari: trasforma le dimensioni dispari
    Layer dispari: trasforma le dimensioni pari
    """
    mask = torch.zeros(dim, dtype=torch.bool)
    mask[layer_idx % 2::2] = True
    return mask


# ─── Trainer NSF ──────────────────────────────────────────────────────────────

class NSFTrainer:
    """
    Trainer per Neural Spline Flow.

    Training con NLL (Negative Log-Likelihood) tramite Adam.
    Convergenza stabile (loss monotona decrescente su validation set).

    Iperparametri:
        lr           : learning rate (default: 1e-3)
        weight_decay : L2 regularization (default: 1e-5)
        clip_grad    : gradient clipping (default: 1.0)
    """

    def __init__(
        self,
        model:        PhaseSpaceNSF,
        device:       str = "cpu",
        lr:           float = 1e-3,
        weight_decay: float = 1e-5,
        clip_grad:    float = 1.0,
        scheduler_patience: int = 20,
    ):
        self.model   = model.to(device)
        self.device  = device
        self.clip_grad = clip_grad

        self.opt = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        #Learning rate scheduler: riduce lr se la val loss smette di migliorare
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            patience=scheduler_patience,
            factor=0.5,
            min_lr=1e-6,
        )
       

        self.history = {
            "train_nll": [], "val_nll": [], "lr": []
        }

    def _to(self, x):
        return x.to(self.device) if isinstance(x, torch.Tensor) else x

    def train_step(
        self,
        s_batch: torch.Tensor,
        c_batch: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Singolo step di training su un batch.

        Returns: NLL loss (float)
        """
        s = self._to(s_batch)
        c = self._to(c_batch) if c_batch is not None else None

        self.model.train()
        self.opt.zero_grad()

        loss = self.model.nll_loss(s, c)
        loss.backward()

        # Gradient clipping per stabilità
        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

        self.opt.step()
        #self.scheduler.step()        # cosine step per ogni batch
        return loss.item()

    @torch.no_grad()
    def val_step(
        self,
        s_val: torch.Tensor,
        c_val: Optional[torch.Tensor] = None,
        batch_size: int = 4096,
    ) -> float:
        """
        Calcola la NLL sul validation set (in batches per memoria).
        """
        self.model.eval()
        s_val = self._to(s_val)
        c_val = self._to(c_val) if c_val is not None else None

        total_nll = 0.0
        n_batches = 0

        for i in range(0, len(s_val), batch_size):
            s_b = s_val[i:i + batch_size]
            c_b = c_val[i:i + batch_size] if c_val is not None else None
            total_nll += self.model.nll_loss(s_b, c_b).item()
            n_batches += 1

        val_nll = total_nll / n_batches
        self.scheduler.step(val_nll)
        return val_nll

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
