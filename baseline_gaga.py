"""
baseline_gaga.py
================
Replica fedele del paper Sarrut et al. 2019:
"Generative Adversarial Networks (GAN) for compact beam source
modelling in Monte Carlo simulations"
DOI: 10.1088/1361-6560/ab3fc3

VERSIONE REPLICA 100% PURA: DUE FILE REALI SEPARATI (TRAIN VS EVAL INDIPENDENTE)
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
COLUMNS_PAPER = ["E", "x", "y", "dx", "dy", "dz"]
COLUMNS_UNITS = ["E [MeV]", "x [cm]", "y [cm]", "dx", "dy", "dz"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ARCHITETTURA DI SARRUT (Sezione 2.3) — OUTPUT CON SIGMOIDE
# ═══════════════════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    """
    Generator G: z → x_fake
    Architettura esatta dal paper: output 6D confinato in [0, 1] tramite nn.Sigmoid().
    Le variabili dx, dy, dz sono indipendenti durante tutto l'addestramento.
    """

    def __init__(self, x_dim: int = 6, z_dim: int = 6, h_dim: int = 400):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim

        self.net = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),  
            nn.Sigmoid()  # Forza l'output tra 0 e 1 come da paper
        )

        for p in self.parameters():
            if p.ndimension() > 1:
                nn.init.kaiming_normal_(p)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def sample(self, n: int, device: str = "cpu") -> torch.Tensor:
        with torch.no_grad():
            z = torch.randn(n, self.z_dim, device=device)
            return self.forward(z)


class Discriminator(nn.Module):
    """Discriminator (Critic) D: x → scalar score"""

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
    """WGAN training loop fedele al paper Sarrut 2019."""

    CLAMP = 0.01

    def __init__(
        self,
        G: Generator,
        D: Discriminator,
        device: str = "cpu",
        lr: float = 1e-5,
        n_critic: int = 4,
    ):
        self.G = G.to(device)
        self.D = D.to(device)
        self.device = device
        self.n_critic = n_critic

        self.opt_G = optim.RMSprop(G.parameters(), lr=lr)
        self.opt_D = optim.RMSprop(D.parameters(), lr=lr)

        self.history = {
            "J_D": [], "J_G": [], "J_D_val": [],
            "epoch": [],
        }

    def _to(self, x):
        return x.to(self.device)

    def train_step(self, real_batch: torch.Tensor) -> dict:
        real = self._to(real_batch)
        B = real.size(0)

        J_D_accum = 0.0
        for _ in range(self.n_critic):
            z    = torch.randn(B, self.G.z_dim, device=self.device)
            fake = self.G(z).detach()

            J_D = self.D(fake).mean() - self.D(real).mean()

            self.opt_D.zero_grad()
            J_D.backward()
            self.opt_D.step()

            for p in self.D.parameters():
                p.data.clamp_(-self.CLAMP, self.CLAMP)

            J_D_accum += J_D.item()

        z    = torch.randn(B, self.G.z_dim, device=self.device)
        fake = self.G(z)

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
        print(f"   Checkpoint salvato: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        self.history = ckpt.get("history", self.history)
        print(f"   Checkpoint caricato: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING UNIFICATO PER SINGOLO FILE
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare_file(source, seed=42):
    """Carica un singolo file specifico e lo formatta a 6D (E, x, y, dx, dy, dz)"""
    if isinstance(source, str) and source == "synthetic":
        print("   Generazione dati sintetici (6MV 10x10)...")
        ps7 = generate_phase_space(n_samples=2_000_000, E_nom=6.0, jaw_x=5.0, jaw_y=5.0, seed=seed)
    elif isinstance(source, str) and source.endswith(".h5"):
        ps7, _ = load_phase_space_hdf5(source)
    elif isinstance(source, str) and source.endswith(".npy"):
        ps7_or_6 = np.load(source)
        if ps7_or_6.shape[1] == 6:
            ps7 = np.column_stack([
                ps7_or_6[:, 1], ps7_or_6[:, 2],
                np.zeros(len(ps7_or_6)),
                ps7_or_6[:, 3], ps7_or_6[:, 4],
                ps7_or_6[:, 5],
                ps7_or_6[:, 0],
            ])
        else:
            ps7 = ps7_or_6
    else:
        ps7 = source

    ps6 = np.column_stack([
        ps7[:, 6],   # E
        ps7[:, 0],   # x
        ps7[:, 1],   # y
        ps7[:, 3],   # dx
        ps7[:, 4],   # dy
        ps7[:, 5],   # dz
    ]).astype(np.float32)
    return ps6


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
    print(f"   Fig. 2 salvata: {save_path}")


def plot_fig3_correlation(
    phsp: np.ndarray,
    gan:  np.ndarray,
    save_path: str,
    n_subsample: int = 100_000,
):
    rng = np.random.default_rng(0)
    n   = min(n_subsample, len(phsp), len(gan))
    p   = phsp[rng.choice(len(phsp), n, replace=False)]
    g   = gan[rng.choice(len(gan),   n, replace=False)]

    corr_p = np.corrcoef(p.T)
    corr_g = np.corrcoef(g.T)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, corr, title in zip(axes, [corr_p, corr_g], ["PHSP correlation matrix", "GAN correlation matrix"]):
        im = ax.imshow(np.abs(corr), cmap="Reds", vmin=0, vmax=0.3, aspect="auto")
        ax.set_xticks(range(6)); ax.set_xticklabels(COLUMNS_PAPER, fontsize=10)
        ax.set_yticks(range(6)); ax.set_yticklabels(COLUMNS_PAPER, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")

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
    print(f"   Fig. 3 salvata: {save_path}")


def compute_fig6_differences(
    phsp1: np.ndarray,
    phsp2: np.ndarray,
    gan:   np.ndarray,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    n   = min(100_000, len(phsp1), len(phsp2), len(gan))

    n_bins_xy = 50
    r = 10.0

    def dose_map(ps):
        H, _, _ = np.histogram2d(
            ps[:n, 1], ps[:n, 2],
            bins=n_bins_xy,
            range=[[-r, r], [-r, r]],
            weights=ps[:n, 0],
        )
        return H.flatten()

    D1 = dose_map(phsp1)
    D2 = dose_map(phsp2)
    Dg = dose_map(gan)

    D_max = D2.max()
    mask  = D2 > 0.1 * D_max

    delta_phsp = (D2[mask] - D1[mask]) / D_max * 100
    delta_gan  = (D2[mask] - Dg[mask]) / D_max * 100

    return delta_phsp, delta_gan


def plot_fig6_differences(
    delta_phsp: np.ndarray,
    delta_gan:  np.ndarray,
    save_path: str,
    label: str = "Elekta 6 MV",
):
    fig, ax = plt.subplots(figsize=(7, 5))

    lo = min(delta_phsp.min(), delta_gan.min()) * 1.2
    hi = max(delta_phsp.max(), delta_gan.max()) * 1.2
    bins = np.linspace(max(lo, -5), min(hi, 5), 80)

    ax.hist(delta_phsp, bins=bins, color="#4472C4", alpha=0.65,
            label=f"PHSP1 vs PHSP2   μ={delta_phsp.mean():.2f}%", histtype="stepfilled")
    ax.hist(delta_gan, bins=bins, color="#FF0000", alpha=0.65,
            label=f"PHSP2 vs GAN   μ={delta_gan.mean():.2f}%", histtype="stepfilled")

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
    print(f"   Fig. 6 salvata: {save_path}")


def plot_training_curves(history: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = history["epoch"]
    ax.plot(epochs, history["J_G"], color="red",   alpha=0.7, linewidth=0.8, label="G_loss")
    ax.plot(epochs, history["J_D"], color="blue",  alpha=0.7, linewidth=0.8, label="D_loss")
    if history["J_D_val"]:
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
    print(f"   Curve di training salvate: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. METRICHE QUANTITATIVE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(phsp: np.ndarray, gan: np.ndarray, label: str = "") -> dict:
    from scipy.stats import wasserstein_distance
    rng = np.random.default_rng(0)
    n   = min(100_000, len(phsp), len(gan))
    p   = phsp[rng.choice(len(phsp), n, replace=False)]
    g   = gan[rng.choice(len(gan),   n, replace=False)]

    print(f"\n   {'─'*55}")
    print(f"   Metriche W1 — {label}")
    print(f"   {'─'*55}")
    print(f"   {'Canale':<8} {'W1':>12}  {'mu_PHSP':>10}  {'mu_GAN':>10}  {'σ_PHSP':>10}  {'σ_GAN':>10}")
    print(f"   {'─'*55}")

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
        print(f"   {col:<8} {w1:>12.6f}  "
              f"{p[:,i].mean():>10.4f}  {g[:,i].mean():>10.4f}  "
              f"{p[:,i].std():>10.4f}  {g[:,i].std():>10.4f}")

    w1_mean = np.mean([v["w1"] for v in results.values()])
    print(f"   {'mean':<8} {w1_mean:>12.6f}")
    results["mean_w1"] = float(w1_mean)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN TRAINING LOOP (Replica Pura Sarrut 2019)
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    out_dir = Path(args.output_dir) / f"gaga_baseline_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*55}")
    print(f"   GAGA Baseline Replica Assoluta — Sarrut 2019")
    print(f"   Device: {device}  |  Epoche: {args.n_epochs}")
    print(f"   Output: {out_dir}")
    print(f"{'='*55}")

    # 1. Caricamento indipendente dei due file sani (100% Statistica ciascuno)
    print("\n   Caricamento dataset di TRAINING (PHSP1)...")
    src_train = "synthetic" if args.synthetic else args.hdf5_train
    train_data = load_and_prepare_file(src_train, seed=args.seed)

    print("   Caricamento dataset di EVALUATION (PHSP2)...")
    src_eval = "synthetic" if args.synthetic else args.hdf5_eval
    if not src_eval:
        raise ValueError("[ERROR] Devi fornire sia --hdf5_train che --hdf5_eval per la replica pura!")
    eval_data = load_and_prepare_file(src_eval, seed=args.seed + 1)

    # 2. COLLIMAZIONE FISICA GEOMETRICA DI SARRUT (Scatola d'acqua +-10 cm)
    mask_train = (train_data[:, 1] >= -10.0) & (train_data[:, 1] <= 10.0) & \
                 (train_data[:, 2] >= -10.0) & (train_data[:, 2] <= 10.0)
    train_data = train_data[mask_train]

    mask_eval = (eval_data[:, 1] >= -10.0) & (eval_data[:, 1] <= 10.0) & \
                (eval_data[:, 2] >= -10.0) & (eval_data[:, 2] <= 10.0)
    eval_data = eval_data[mask_eval]

    print(f"\n   Statistiche Post-Collimazione:")
    print(f"   Train (PHSP1): {len(train_data):,} campioni")
    print(f"   Eval  (PHSP2): {len(eval_data):,} campioni")

    # 3. NORMALIZZAZIONE MIN-MAX DI SARRUT (Basata sulla massima statistica del Train)
    data_min = train_data.min(axis=0)
    data_max = train_data.max(axis=0)
    denom = data_max - data_min
    denom[denom == 0] = 1.0
    
    # Conserviamo intatte le copie fisiche grezze per i confronti dei plot alla fine
    phsp1_raw = train_data.copy()
    phsp2_raw = eval_data.copy()

    # Scaliamo entrambi i file nell'ipercubo [0, 1] richiesto dalla Sigmoide
    train_data = (train_data - data_min) / denom
    eval_data = (eval_data - data_min) / denom

    X_train = torch.from_numpy(train_data)
    X_val   = torch.from_numpy(eval_data)

    loader = DataLoader(
        TensorDataset(X_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    print(f"\n   Architettura di Rete (Sezione 2.3):")
    G = Generator(x_dim=6, z_dim=args.z_dim, h_dim=args.h_dim)
    D = Discriminator(x_dim=6, h_dim=args.h_dim)

    n_G = sum(p.numel() for p in G.parameters())
    n_D = sum(p.numel() for p in D.parameters())
    print(f"   G: {n_G:,} parametri  |  D: {n_D:,} parametri")
    print(f"   Optimizer: RMSprop, lr={args.lr}  |  Batch size={args.batch_size}")

    trainer = WGANTrainer(G, D, device=device, lr=args.lr, n_critic=args.n_critic)

    print(f"\n   Inizio training ({args.n_epochs} epoche)...")
    print(f"   Steps per epoca: {len(loader)}")
    print(f"   {'─'*55}")

    t0 = time.time()
    best_J_D_val = float("inf")

    for epoch in range(1, args.n_epochs + 1):
        for batch in loader:
            metrics = trainer.train_step(batch[0])

        if epoch % args.log_every == 0 or epoch == args.n_epochs:
            J_D_val = trainer.val_loss(X_val)
            elapsed = time.time() - t0
            print(f"   Ep {epoch:>6d}/{args.n_epochs}  "
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
    print(f"\n   Training completato in {time.time()-t0:.0f}s")

    # ── GENERAZIONE E POST-PROCESSING DI SARRUT ──
    print("\n   Generazione campioni per valutazione...")
    # Generiamo lo stesso esatto numero di particelle contenute nel file PHSP2 sanno
    n_gen = len(phsp2_raw)
    G.eval()
    gan_samples = G.sample(n_gen, device=device).cpu().numpy()
    
    # De-normalizzazione dei campioni GAN riportandoli alla scala fisica reale
    gan_samples = gan_samples * denom + data_min
    
    # Assegniamo le ground truth reali per il confronto
    phsp1 = phsp1_raw
    phsp2 = phsp2_raw

    print(f"   Generati: {len(gan_samples):,} campioni.")
    print("   [POST-PROCESSING] Applicazione normalizzazione vettoriale su direzioni...")
    dx = gan_samples[:, 3]
    dy = gan_samples[:, 4]
    dz = gan_samples[:, 5]
    
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    norm = np.clip(norm, a_min=1e-6, a_max=None) 
    
    gan_samples[:, 3] = dx / norm
    gan_samples[:, 4] = dy / norm
    gan_samples[:, 5] = dz / norm

    np.save(str(out_dir / "gan_samples.npy"), gan_samples)

    # ── GENERAZIONE FIGURE DEL PAPER CON STATISTICA MASSIMA ──
    print("\n   Generazione figure del paper...")
    plot_training_curves(trainer.history, str(out_dir / "fig1_training_curves.png"))
    plot_fig2_marginals(phsp2, gan_samples, str(out_dir / "fig2_marginal_distributions.png"), n_subsample=len(gan_samples))
    plot_fig3_correlation(phsp2, gan_samples, str(out_dir / "fig3_correlation_matrices.png"), n_subsample=len(gan_samples))

    delta_phsp, delta_gan = compute_fig6_differences(phsp1, phsp2, gan_samples)
    plot_fig6_differences(delta_phsp, delta_gan, str(out_dir / "fig6_dose_differences.png"))

    metrics_report = compute_metrics(phsp2, gan_samples, label="PHSP2 (Eval) vs GAN")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_report, f, indent=2)

    print(f"\n{'='*55}")
    print(f"   REPLICA SARRUT 2019 COMPLETA E INTEGRALE")
    print(f"{'='*55}")
    print(f"   Output directory: {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PARSE ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="GAGA Baseline Pure Replica: Sarrut 2019")
    p.add_argument("--synthetic",   action="store_true")
    p.add_argument("--hdf5_train",  type=str, metavar="PATH", help="File HDF5 di Training (PHSP1)")
    p.add_argument("--hdf5_eval",   type=str, metavar="PATH", help="File HDF5 di Evaluation (PHSP2)")

    p.add_argument("--h_dim",     type=int,   default=400)
    p.add_argument("--z_dim",     type=int,   default=6)
    p.add_argument("--lr",        type=float, default=1e-5)
    p.add_argument("--n_critic",  type=int,   default=4)
    p.add_argument("--batch_size",type=int,   default=10000)
    p.add_argument("--n_epochs",  type=int,   default=80000)
    p.add_argument("--quick_test",action="store_true")
    p.add_argument("--log_every", type=int,   default=5)
    p.add_argument("--save_every",type=int,   default=5000)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--output_dir",type=str,   default="outputs/baseline")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.quick_test:
        args.n_epochs  = 1000
        args.batch_size = 1000
        args.log_every  = 100
        args.save_every = 500
    train(args)