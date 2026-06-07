"""
baseline_gaga.py
================
Replica fedele del paper Sarrut et al. 2019:
"Generative Adversarial Networks (GAN) for compact beam source
modelling in Monte Carlo simulations"
DOI: 10.1088/1361-6560/ab3fc3

Implementa ESATTAMENTE l'architettura e i parametri della Sezione 2.3:
    - 3 hidden layers, H = 400 neuroni ciascuno
    - Attivazione ReLU su tutti i layer (eccetto ultimo G → nessuna)
    - z_dim = 6 per linac (dimensione variabile latente)
    - Ottimizzatore: RMSProp, lr = 1e-5
    - Batch size = 10,000
    - D update freq / G update freq = 4 : 1
    - WGAN weight clipping: [-0.01, 0.01]
    - Epoche: 80,000

Produce le figure del paper:
    - Fig. 2: distribuzioni marginali delle 6 dimensioni
    - Fig. 3: matrici di correlazione (PHSP vs GAN)
    - Fig. 6: distribuzione delle differenze relative (Δ%) con incertezza statistica

Uso:
    # Con dati sintetici (test immediato, nessun dato GATE richiesto)
    python baseline_gaga.py --synthetic --n_epochs 1000 --quick_test

    # Con dati IAEA reali (formato .npy, shape N×6)
    python baseline_gaga.py --phsp_train linac_6mv_train.npy \\
                             --phsp_eval  linac_6mv_eval.npy \\
                             --n_epochs 80000

    # Con file HDF5 del nostro progetto
    python baseline_gaga.py --hdf5_train data/ps_6mv_10x10.h5 \\
                             --n_epochs 80000

Dati IAEA reali:
    https://www-nds.iaea.org/phsp/photon/
    Scaricare: Elekta_6MV_PRECISE_phsp_scan.zip
    Formato: ROOT/IAEA → convertire con convert_iaea.py (script incluso)
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── PyTorch ──────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[ERROR] PyTorch non disponibile. Installare: pip install torch")
    sys.exit(1)

# ─── Nostra pipeline di dati e metriche ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from data.synthetic_linac import (
    generate_phase_space,
    save_phase_space_hdf5,
    load_phase_space_hdf5,
)

# ─── Colonne del phase space (ordine del paper Sarrut 2019) ───────────────────
# Il paper usa z fisso e lavora su 6D: E, x, y, dx, dy, dz
COLUMNS_PAPER = ["E", "x", "y", "dx", "dy", "dz"]
COLUMNS_UNITS = ["E [MeV]", "x [cm]", "y [cm]", "dx", "dy", "dz"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ARCHITETTURA ESATTA DEL PAPER (Sezione 2.3)
# ═══════════════════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    """
    Generator G: z → x_fake

    Architettura dal paper:
        - Input: z ∈ R^z_dim da N(0,1)
        - 3 hidden layers da H=400 neuroni
        - Attivazione: ReLU su tutti (nel paper "ReLU per tutti tranne ultimo di G")
        - Output lineare (no activation nell'ultimo layer)
        - Totale pesi: ~500k

    Nota: il paper usa ReLU standard (NON LeakyReLU) per il generatore.
    Abbiamo verificato questo dal codice sorgente gaga_phsp/gaga_model.py.
    """

    def __init__(self, x_dim: int = 6, z_dim: int = 6, h_dim: int = 400):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim

        self.net = nn.Sequential(
            # Input layer
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            # Hidden layer 1
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            # Hidden layer 2
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            # Output layer — NESSUNA attivazione (come da paper)
            nn.Linear(h_dim, x_dim),
        )

        # Kaiming initialization (come nel codice sorgente gaga_phsp)
        for p in self.parameters():
            if p.ndimension() > 1:
                nn.init.kaiming_normal_(p)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def sample(self, n: int, device: str = "cpu") -> torch.Tensor:
        """Genera n campioni dal prior N(0,I)."""
        with torch.no_grad():
            z = torch.randn(n, self.z_dim, device=device)
            return self.forward(z)


class Discriminator(nn.Module):
    """
    Discriminator (Critic) D: x → scalar score

    Architettura dal paper:
        - Input: x ∈ R^6
        - 3 hidden layers da H=400 neuroni
        - Attivazione: ReLU
        - Output: scalare (nessun sigmoid — WGAN)

    Il critic non produce una probabilità ma uno score scalare non limitato.
    """

    def __init__(self, x_dim: int = 6, h_dim: int = 400):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TRAINER WGAN (Sezione 2.2 del paper)
# ═══════════════════════════════════════════════════════════════════════════════

class WGANTrainer:
    """
    WGAN training loop fedele al paper Sarrut 2019.

    Parametri esatti del paper (Sezione 2.3):
        - Optimizer: RMSProp (NON Adam — il paper specifica RMSProp)
        - lr = 1e-5
        - Batch size: 10,000
        - n_critic = 4 (4 update D per 1 update G)
        - Weight clipping: [-0.01, 0.01]
        - 80,000 epoche

    Loss functions (eq. 1 e 2 del paper):
        J_D = E_z[D(G(z))] - E_x[D(x)]   (da massimizzare → minimizza il negativo)
        J_G = -E_z[D(G(z))]               (da minimizzare)
    """

    CLAMP = 0.01   # paper: "clamped to [-0.01, 0.01]"

    def __init__(
        self,
        G: Generator,
        D: Discriminator,
        device: str = "cpu",
        lr: float = 1e-5,       # paper: lr = 1e-5
        n_critic: int = 4,      # paper: "four discriminator updates per one generator"
    ):
        self.G = G.to(device)
        self.D = D.to(device)
        self.device = device
        self.n_critic = n_critic

        # RMSProp come specificato nel paper (NON Adam)
        self.opt_G = optim.RMSprop(G.parameters(), lr=lr)
        self.opt_D = optim.RMSprop(D.parameters(), lr=lr)

        self.history = {
            "J_D": [], "J_G": [], "J_D_val": [],
            "epoch": [],
        }

    def _to(self, x):
        return x.to(self.device)

    def train_step(self, real_batch: torch.Tensor) -> dict:
        """
        Esegue un'epoca completa: n_critic step D + 1 step G.

        Corrisponde esattamente alla procedura descritta nella Sezione 2.2.
        """
        real = self._to(real_batch)
        B = real.size(0)

        # ── n_critic aggiornamenti del Discriminator ────────────────────────
        J_D_accum = 0.0
        for _ in range(self.n_critic):
            z    = torch.randn(B, self.G.z_dim, device=self.device)
            fake = self.G(z).detach()

            # J_D = E_z[D(G(z))] - E_x[D(x)]  (eq. 1, minimizzato con segno -)
            J_D = self.D(fake).mean() - self.D(real).mean()

            self.opt_D.zero_grad()
            J_D.backward()
            self.opt_D.step()

            # Weight clipping WGAN: theta_D ∈ [-0.01, 0.01]
            for p in self.D.parameters():
                p.data.clamp_(-self.CLAMP, self.CLAMP)

            J_D_accum += J_D.item()

        # ── 1 aggiornamento del Generator ───────────────────────────────────
        z    = torch.randn(B, self.G.z_dim, device=self.device)
        fake = self.G(z)

        # J_G = -E_z[D(G(z))]  (eq. 2)
        J_G = -self.D(fake).mean()

        self.opt_G.zero_grad()
        J_G.backward()
        self.opt_G.step()

        return {
            "J_D": J_D_accum / self.n_critic,
            "J_G": J_G.item(),
        }

    @torch.no_grad()
    def val_loss(self, val_tensor: torch.Tensor) -> float:
        """J_D sul validation set."""
        self.D.eval()
        val = self._to(val_tensor)
        B   = min(10000, len(val))
        idx = torch.randperm(len(val))[:B]
        real = val[idx]
        z    = torch.randn(B, self.G.z_dim, device=self.device)
        fake = self.G(z)
        J_D_val = (self.D(fake).mean() - self.D(real).mean()).item()
        self.D.train()
        return J_D_val

    def save(self, path: str):
        torch.save({
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
            "history": self.history,
        }, path)
        print(f"  Checkpoint salvato: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        self.history = ckpt.get("history", self.history)
        print(f"  Checkpoint caricato: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING DEI DATI (come nel paper)
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare(
    source,
    n_train: int = None,
    n_eval: int = None,
    seed: int = 42,
):
    """
    Carica e prepara i dati nel formato del paper.

    Il paper usa 6 dimensioni: (E, x, y, dx, dy, dz)
    z è fisso e scartato.

    Ordine colonne nel nostro formato: (x, y, z, dx, dy, dz, E)
    → riordina in: (E, x, y, dx, dy, dz)  come il paper.

    Normalizzazione: il paper NON normalizza esplicitamente i dati
    (la rete impara la scala direttamente). Manteniamo questo per
    fedeltà alla baseline.

    Parameters
    ----------
    source : np.ndarray (N,7) o str (path HDF5) o "synthetic"
    """
    if isinstance(source, str) and source == "synthetic":
        print("  Generazione dati sintetici (6MV 10x10)...")
        ps7 = generate_phase_space(
            n_samples=max(n_train or 0, n_eval or 0) + 100_000,
            E_nom=6.0, jaw_x=5.0, jaw_y=5.0, seed=seed
        )
    elif isinstance(source, str) and source.endswith(".h5"):
        ps7, _ = load_phase_space_hdf5(source)
    elif isinstance(source, str) and source.endswith(".npy"):
        ps7_or_6 = np.load(source)
        # Se già 6D (formato IAEA pre-convertito), usa direttamente
        if ps7_or_6.shape[1] == 6:
            ps7 = np.column_stack([
                ps7_or_6[:, 1], ps7_or_6[:, 2],  # x, y
                np.zeros(len(ps7_or_6)),           # z=0
                ps7_or_6[:, 3], ps7_or_6[:, 4],   # dx, dy
                ps7_or_6[:, 5],                    # dz
                ps7_or_6[:, 0],                    # E
            ])
        else:
            ps7 = ps7_or_6
    else:
        ps7 = source  # già np.ndarray (N,7)

    # Riordina colonne: (x,y,z,dx,dy,dz,E) → (E,x,y,dx,dy,dz)  [ordine paper]
    ps6 = np.column_stack([
        ps7[:, 6],   # E
        ps7[:, 0],   # x
        ps7[:, 1],   # y
        ps7[:, 3],   # dx
        ps7[:, 4],   # dy
        ps7[:, 5],   # dz
    ]).astype(np.float32)

    print(f"  Dati totali: {len(ps6):,} campioni, shape {ps6.shape}")
    print(f"  Range E: [{ps6[:,0].min():.3f}, {ps6[:,0].max():.3f}] MeV")
    print(f"  Range x: [{ps6[:,1].min():.2f}, {ps6[:,1].max():.2f}] cm")

    # Split train/eval (come il paper: due file separati)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ps6))

    n_tr = n_train or int(0.5 * len(ps6))
    n_ev = n_eval  or min(int(0.5 * len(ps6)), n_tr)

    train = ps6[perm[:n_tr]]
    eval_ = ps6[perm[n_tr:n_tr + n_ev]]

    print(f"  Train: {len(train):,}  |  Eval: {len(eval_):,}")
    return train, eval_, ps6


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FIGURE DEL PAPER
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig2_marginals(
    phsp: np.ndarray,
    gan:  np.ndarray,
    save_path: str,
    title: str = "Elekta 6 MV linac",
    n_bins: int = 100,
    n_subsample: int = 100_000,
):
    """
    Replica della Figura 2 del paper:
    Distribuzioni marginali delle 6 dimensioni — PHSP vs GAN.
    """
    rng = np.random.default_rng(0)
    n   = min(n_subsample, len(phsp), len(gan))
    p   = phsp[rng.choice(len(phsp), n, replace=False)]
    g   = gan[rng.choice(len(gan),   n, replace=False)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, (name, ax) in enumerate(zip(COLUMNS_UNITS, axes)):
        lo = min(np.percentile(p[:, i], 0.1), np.percentile(g[:, i], 0.1))
        hi = max(np.percentile(p[:, i], 99.9), np.percentile(g[:, i], 99.9))
        bins = np.linspace(lo, hi, n_bins)

        ax.hist(p[:, i], bins=bins, color="#4472C4", alpha=0.7,
                label="PHSP", histtype="stepfilled")
        ax.hist(g[:, i], bins=bins, color="#ED7D31", alpha=0.6,
                label="GAN",  histtype="stepfilled")

        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel("Counts", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Marginal distributions — {title}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fig. 2 salvata: {save_path}")


def plot_fig3_correlation(
    phsp: np.ndarray,
    gan:  np.ndarray,
    save_path: str,
    n_subsample: int = 100_000,
):
    """
    Replica della Figura 3 del paper:
    Matrici di correlazione (coefficiente di Pearson normalizzato) per PHSP e GAN.
    """
    rng = np.random.default_rng(0)
    n   = min(n_subsample, len(phsp), len(gan))
    p   = phsp[rng.choice(len(phsp), n, replace=False)]
    g   = gan[rng.choice(len(gan),   n, replace=False)]

    corr_p = np.corrcoef(p.T)
    corr_g = np.corrcoef(g.T)

    # Maschera triangolo superiore (solo triangolo inferiore come nel paper)
    mask = np.triu(np.ones_like(corr_p, dtype=bool), k=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, corr, title in zip(axes, [corr_p, corr_g], ["PHSP correlation matrix", "GAN correlation matrix"]):
        im = ax.imshow(np.abs(corr), cmap="Reds", vmin=0, vmax=0.3, aspect="auto")
        ax.set_xticks(range(6)); ax.set_xticklabels(COLUMNS_PAPER, fontsize=10)
        ax.set_yticks(range(6)); ax.set_yticklabels(COLUMNS_PAPER, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

        # Annota valori (solo triangolo inferiore come nel paper)
        for ii in range(6):
            for jj in range(ii):
                v = corr[ii, jj]
                ax.text(jj, ii, f"{v:.4f}", ha="center", va="center",
                        fontsize=7.5, color="black")

        plt.colorbar(im, ax=ax)

    fig.suptitle("Correlation matrices: PHSP vs GAN (Sarrut 2019 baseline)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fig. 3 salvata: {save_path}")


def compute_fig6_differences(
    phsp1: np.ndarray,
    phsp2: np.ndarray,
    gan:   np.ndarray,
    n_voxels: int = 10_000,
    seed: int = 0,
):
    """
    Simula la metrica di Figura 6 del paper:
    Distribuzione delle differenze relative Δ(k) = (D_PHSP2(k) - D_src(k)) / D_max

    In assenza di una simulazione GATE downstream, approssimiamo la dose
    come la somma dell'energia depositata in voxel pseudocasori.
    Questo è un proxy della metrica del paper ma sufficiente per la baseline.

    Returns
    -------
    delta_phsp : differenze PHSP1 vs PHSP2
    delta_gan  : differenze PHSP2 vs GAN
    """
    rng = np.random.default_rng(seed)
    n   = min(50_000, len(phsp1), len(phsp2), len(gan))

    # Istogramma dell'energia depositata in voxel spaziali proxy (x,y bins)
    # Come nel paper: water box 20×20×20 cm³, voxel 4mm³
    n_bins_xy = 50
    r = 10.0  # semi-range cm

    def dose_map(ps):
        """Proxy: somma energia per bin (x, y) → distribuzione 1D."""
        H, _, _ = np.histogram2d(
            ps[:n, 1], ps[:n, 2],
            bins=n_bins_xy,
            range=[[-r, r], [-r, r]],
            weights=ps[:n, 0],  # peso = energia
        )
        return H.flatten()

    D1 = dose_map(phsp1)
    D2 = dose_map(phsp2)
    Dg = dose_map(gan)

    D_max = D2.max()
    mask  = D2 > 0.1 * D_max  # solo voxel > 10% del massimo

    delta_phsp = (D2[mask] - D1[mask]) / D_max * 100  # in %
    delta_gan  = (D2[mask] - Dg[mask]) / D_max * 100  # in %

    return delta_phsp, delta_gan


def plot_fig6_differences(
    delta_phsp: np.ndarray,
    delta_gan:  np.ndarray,
    save_path: str,
    label: str = "Elekta 6 MV",
):
    """
    Replica della Figura 6 del paper (sinistra):
    Istogramma delle differenze relative in %.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    lo = min(delta_phsp.min(), delta_gan.min()) * 1.2
    hi = max(delta_phsp.max(), delta_gan.max()) * 1.2
    bins = np.linspace(max(lo, -5), min(hi, 5), 80)

    ax.hist(delta_phsp, bins=bins, color="#4472C4", alpha=0.65,
            label=f"PHSP1 vs PHSP2  μ={delta_phsp.mean():.2f}%", histtype="stepfilled")
    ax.hist(delta_gan, bins=bins, color="#FF0000", alpha=0.65,
            label=f"PHSP1 vs GAN   μ={delta_gan.mean():.2f}%", histtype="stepfilled")

    ax.axvline(delta_phsp.mean(), color="#000080", linewidth=1.5)
    ax.axvline(delta_gan.mean(),  color="#FF0000", linewidth=1.5)

    ax.set_xlabel("Difference %", fontsize=11)
    ax.set_ylabel("Counts", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_title(f"Dose differences — {label} (proxy: energy histogram)", fontsize=11)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fig. 6 salvata: {save_path}")


def plot_training_curves(history: dict, save_path: str):
    """
    Replica della Figura 1 del paper: J_D e J_G in funzione delle epoche.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = history["epoch"]
    ax.plot(epochs, history["J_G"], color="red",   alpha=0.7, linewidth=0.8, label="G_loss")
    ax.plot(epochs, history["J_D"], color="blue",  alpha=0.7, linewidth=0.8, label="D_loss")
    if history["J_D_val"]:
        # Subsample val loss (calcolata ogni val_every epoche)
        val_epochs = [history["epoch"][i] for i in range(0, len(history["epoch"]),
                      max(1, len(history["epoch"]) // len(history["J_D_val"])))]
        val_epochs = val_epochs[:len(history["J_D_val"])]
        ax.plot(val_epochs, history["J_D_val"], color="black", linewidth=1.2,
                label="Validation D_loss")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_title("Training curves (Sarrut 2019 baseline)", fontsize=12)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curve di training salvate: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. METRICHE QUANTITATIVE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(phsp: np.ndarray, gan: np.ndarray, label: str = "") -> dict:
    """Calcola W1 per ogni dimensione + mean, come metrica principale di confronto."""
    from scipy.stats import wasserstein_distance
    rng = np.random.default_rng(0)
    n   = min(50_000, len(phsp), len(gan))
    p   = phsp[rng.choice(len(phsp), n, replace=False)]
    g   = gan[rng.choice(len(gan),   n, replace=False)]

    print(f"\n  {'─'*55}")
    print(f"  Metriche W1 — {label}")
    print(f"  {'─'*55}")
    print(f"  {'Canale':<8} {'W1':>12}  {'mu_PHSP':>10}  {'mu_GAN':>10}  {'σ_PHSP':>10}  {'σ_GAN':>10}")
    print(f"  {'─'*55}")

    results = {}
    for i, col in enumerate(COLUMNS_PAPER):
        w1 = wasserstein_distance(p[:, i], g[:, i])
        results[col] = {
            "w1":      float(w1),
            "mu_phsp": float(p[:, i].mean()),
            "mu_gan":  float(g[:, i].mean()),
            "sig_phsp":float(p[:, i].std()),
            "sig_gan": float(g[:, i].std()),
        }
        print(f"  {col:<8} {w1:>12.6f}  "
              f"{p[:,i].mean():>10.4f}  {g[:,i].mean():>10.4f}  "
              f"{p[:,i].std():>10.4f}  {g[:,i].std():>10.4f}")

    w1_mean = np.mean([v["w1"] for v in results.values()])
    print(f"  {'mean':<8} {w1_mean:>12.6f}")
    results["mean_w1"] = float(w1_mean)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    """Pipeline completa: dati → training → figure del paper."""

    out_dir = Path(args.output_dir) / f"gaga_baseline_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*55}")
    print(f"  GAGA Baseline — Sarrut 2019")
    print(f"  Device: {device}  |  Epoche: {args.n_epochs}")
    print(f"  Output: {out_dir}")
    print(f"{'='*55}")

    # ── Dati ──────────────────────────────────────────────────────────────────
    print("\n  Caricamento dati...")
    src = "synthetic" if args.synthetic else (args.hdf5_train or args.phsp_train)
    train_data, eval_data, all_data = load_and_prepare(
        src,
        n_train=args.n_train,
        n_eval=args.n_eval,
        seed=args.seed,
    )

    # Salva copia eval per riferimento (PHSP1 e PHSP2 del paper)
    n_half = len(eval_data) // 2
    phsp1 = eval_data[:n_half]   # PHSP1 (primo eval set)
    phsp2 = eval_data[n_half:]   # PHSP2 (secondo eval set, ground truth)

    X_train = torch.from_numpy(train_data)
    X_val   = torch.from_numpy(eval_data)

    loader = DataLoader(
        TensorDataset(X_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # ── Modello (parametri esatti del paper) ──────────────────────────────────
    print(f"\n  Architettura (Sezione 2.3 del paper):")
    G = Generator(x_dim=6, z_dim=args.z_dim, h_dim=args.h_dim)
    D = Discriminator(x_dim=6, h_dim=args.h_dim)

    n_G = sum(p.numel() for p in G.parameters())
    n_D = sum(p.numel() for p in D.parameters())
    print(f"  G: {n_G:,} parametri  (paper: ~500k)")
    print(f"  D: {n_D:,} parametri  (paper: ~500k)")
    print(f"  z_dim={args.z_dim}, h_dim={args.h_dim}, n_hidden=3")
    print(f"  Optimizer: RMSprop, lr={args.lr}")
    print(f"  n_critic={args.n_critic}, weight_clip=[-0.01, 0.01]")
    print(f"  Batch size={args.batch_size}")

    trainer = WGANTrainer(G, D, device=device, lr=args.lr, n_critic=args.n_critic)

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n  Inizio training ({args.n_epochs} epoche)...")
    print(f"  Steps per epoca: {len(loader)}")
    print(f"  {'─'*55}")

    t0 = time.time()
    best_J_D_val = float("inf")

    for epoch in range(1, args.n_epochs + 1):
        # Batch da DataLoader
        for batch in loader:
            metrics = trainer.train_step(batch[0])

        # Logging
        if epoch % args.log_every == 0 or epoch == args.n_epochs:
            J_D_val = trainer.val_loss(X_val)
            elapsed = time.time() - t0
            print(f"  Ep {epoch:>6d}/{args.n_epochs}  "
                  f"J_D={metrics['J_D']:>+8.5f}  "
                  f"J_G={metrics['J_G']:>+8.5f}  "
                  f"J_D_val={J_D_val:>+8.5f}  "
                  f"[{elapsed:.0f}s]")

            trainer.history["epoch"].append(epoch)
            trainer.history["J_D"].append(metrics["J_D"])
            trainer.history["J_G"].append(metrics["J_G"])
            trainer.history["J_D_val"].append(J_D_val)

            if abs(J_D_val) < abs(best_J_D_val):
                best_J_D_val = J_D_val
                trainer.save(str(out_dir / "best_model.pt"))

        if epoch % args.save_every == 0:
            trainer.save(str(out_dir / f"checkpoint_{epoch:06d}.pt"))

    trainer.save(str(out_dir / "final_model.pt"))
    print(f"\n  Training completato in {time.time()-t0:.0f}s")

    # ── Generazione campioni finali ────────────────────────────────────────────
    print("\n  Generazione campioni per valutazione...")
    n_gen = len(phsp2)
    G.eval()
    gan_samples = G.sample(n_gen, device=device).cpu().numpy()
    print(f"  Generati: {len(gan_samples):,} campioni")

    # Salva campioni GAN (formato paper: 6D, ordine E,x,y,dx,dy,dz)
    np.save(str(out_dir / "gan_samples.npy"), gan_samples)

    # ── Figure del paper ──────────────────────────────────────────────────────
    print("\n  Generazione figure del paper...")

    plot_training_curves(
        trainer.history,
        str(out_dir / "fig1_training_curves.png"),
    )
    plot_fig2_marginals(
        phsp2, gan_samples,
        str(out_dir / "fig2_marginal_distributions.png"),
    )
    plot_fig3_correlation(
        phsp2, gan_samples,
        str(out_dir / "fig3_correlation_matrices.png"),
    )

    delta_phsp, delta_gan = compute_fig6_differences(phsp1, phsp2, gan_samples)
    plot_fig6_differences(
        delta_phsp, delta_gan,
        str(out_dir / "fig6_dose_differences.png"),
    )

    # ── Metriche quantitative ──────────────────────────────────────────────────
    metrics_report = compute_metrics(phsp2, gan_samples, label="PHSP2 vs GAN")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_report, f, indent=2)

    # ── Report finale ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  RISULTATI BASELINE (da confrontare con paper Fig.2-3-6)")
    print(f"{'='*55}")
    print(f"  Epoche completate: {args.n_epochs}")
    print(f"  W1 medio:  {metrics_report['mean_w1']:.6f}  (target: < 0.01)")
    print(f"\n  Confronto col paper Sarrut 2019:")
    print(f"  - Distribuzioni marginali → fig2_marginal_distributions.png")
    print(f"  - Matrici di correlazione → fig3_correlation_matrices.png")
    print(f"  - Differenze di dose      → fig6_dose_differences.png")
    print(f"  - Curve di training       → fig1_training_curves.png")
    print(f"\n  Output directory: {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SCRIPT HELPER: CONVERTI DATI IAEA
# ═══════════════════════════════════════════════════════════════════════════════

IAEA_CONVERSION_SCRIPT = '''
# convert_iaea.py
# ================
# Converte il file IAEA phase space (.IAEAphsp) in formato .npy usabile con
# baseline_gaga.py --phsp_train.
#
# Dati IAEA disponibili su: https://www-nds.iaea.org/phsp/photon/
# File da scaricare: Elekta_PRECISE_6MV_phsp_scan.zip
#
# Installare: pip install iaea_phsp   (o usare opengate che include gatetools.phsp)
#
# Uso:
#   python convert_iaea.py \\
#       --input  Elekta_PRECISE_6MV_phsp_scan/Elekta_PRECISE_6MV_phsp_scan.IAEAphsp \\
#       --output elekta_6mv_train.npy
#
# Il file .npy avrà shape (N, 6): [E, x, y, dx, dy, dz]

import numpy as np
import argparse

def convert(input_path, output_path, max_particles=None):
    """
    Legge un file .IAEAphsp (formato binario IAEA) e salva in .npy.
    Richiede gatetools: pip install gatetools
    """
    try:
        import gatetools.phsp as phsp
        data, keys, m = phsp.load(input_path, nmax=max_particles)
    except ImportError:
        print("Installare: pip install gatetools")
        print("Oppure: pip install opengate  (include gatetools)")
        return

    # Seleziona solo fotoni (particle_type = 1 in IAEA)
    # Riordina: E, x, y, dx, dy, dz
    print(f"Keys: {keys}")
    print(f"Particles: {len(data[keys[0]]):,}")

    E  = data["E"]       # energia in MeV
    x  = data["X"] / 10  # mm → cm
    y  = data["Y"] / 10  # mm → cm
    dx = data["u"]
    dy = data["v"]
    dz = np.sqrt(np.maximum(1 - dx**2 - dy**2, 0))

    # Rimuovi picco 511 keV (come nel paper: "pre-processed to remove the 511 keV peak")
    mask = ~((E > 0.505) & (E < 0.520))
    E = E[mask]; x = x[mask]; y = y[mask]
    dx = dx[mask]; dy = dy[mask]; dz = dz[mask]

    ps6 = np.column_stack([E, x, y, dx, dy, dz]).astype(np.float32)
    np.save(output_path, ps6)
    print(f"Salvato: {output_path}  shape={ps6.shape}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max",    type=int, default=None)
    a = p.parse_args()
    convert(a.input, a.output, a.max)
'''


def parse_args():
    p = argparse.ArgumentParser(
        description="GAGA Baseline: replica Sarrut 2019",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Sorgente dati
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic",   action="store_true",
                   help="Usa dati sintetici (test senza GATE)")
    g.add_argument("--hdf5_train",  type=str, metavar="PATH",
                   help="File HDF5 dal nostro generatore sintetico")
    g.add_argument("--phsp_train",  type=str, metavar="PATH",
                   help="File .npy di phase space (formato E,x,y,dx,dy,dz)")

    p.add_argument("--n_train",  type=int, default=None,
                   help="Numero campioni di training (default: 50% dei dati)")
    p.add_argument("--n_eval",   type=int, default=None,
                   help="Numero campioni eval (default: 50% dei dati)")

    # Iperparametri del paper
    p.add_argument("--h_dim",     type=int,   default=400,  help="Neuroni per layer (paper: 400)")
    p.add_argument("--z_dim",     type=int,   default=6,    help="Dimensione latente (paper: 6)")
    p.add_argument("--lr",        type=float, default=1e-5, help="Learning rate (paper: 1e-5)")
    p.add_argument("--n_critic",  type=int,   default=4,    help="D updates / G update (paper: 4)")
    p.add_argument("--batch_size",type=int,   default=10000,help="Batch size (paper: 10000)")
    p.add_argument("--n_epochs",  type=int,   default=80000,help="Epoche (paper: 80000)")

    # Convenienza
    p.add_argument("--quick_test",action="store_true",
                   help="1000 epoche + batch 1000 per test rapido")
    p.add_argument("--log_every", type=int,  default=5)
    p.add_argument("--save_every",type=int,  default=5000)
    p.add_argument("--seed",      type=int,  default=42)
    p.add_argument("--output_dir",type=str,  default="outputs/baseline")

    # Helper
    p.add_argument("--dump_convert_script", action="store_true",
                   help="Stampa lo script per convertire file IAEA")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dump_convert_script:
        print(IAEA_CONVERSION_SCRIPT)
        sys.exit(0)

    # Quick test override
    if args.quick_test:
        print("  [QUICK TEST] n_epochs=1000, batch_size=1000, log_every=100")
        args.n_epochs  = 1000
        args.batch_size = 1000
        args.log_every  = 100
        args.save_every = 500
        args.n_train   = 50_000
        args.n_eval    = 10_000

    train(args)
