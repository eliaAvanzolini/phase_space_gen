"""
recover_sarrut_eval.py
=======================
Recupera la valutazione quantitativa del run sarrut_pure_replica già
completato (training riuscito, crash solo nella valutazione finale per
mismatch di shape 6D vs 7D atteso da evaluate_model).

Carica il checkpoint salvato, rigenera i campioni, reinserisce la
colonna z costante (richiesta da evaluate_model), e calcola le metriche.

Uso:
    python recover_sarrut_eval.py \
        --checkpoint outputs/sarrut_pure_replica_run/sarrut_replica_final.pt \
        --data_path data/elekta_130mv_completo.h5 \
        --eval_data_path data/elekta_130mv_eval_completo.h5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from evaluate import evaluate_model


# ─── Stessa architettura esatta usata in sarrut_pure_replica.py ───────────────
class SarrutGenerator(nn.Module):
    def __init__(self, latent_dim=6, output_dim=6, hidden_dim=400):
        super().__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path a sarrut_replica_final.pt")
    p.add_argument("--data_path", type=str, default="data/elekta_130mv_completo.h5",
                   help="File di training, usato solo per ricostruire lo split "
                        "di test se --eval_data_path non è fornito")
    p.add_argument("--eval_data_path", type=str,
                   default="data/elekta_130mv_eval_completo.h5",
                   help="File di eval indipendente (PHSP2). Se assente, ricade "
                        "sullo split di test interno del file di training.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed usato per lo split (deve combaciare col training "
                        "originale se si usa lo split interno)")
    p.add_argument("--output_dir", type=str,
                   default="outputs/sarrut_pure_replica_run",
                   help="Cartella dove salvare il report di valutazione")
    p.add_argument("--eval_chunk", type=int, default=250_000)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def generate_in_chunks(G, n_total, latent_dim, device, chunk_size=250_000):
    out = []
    n_done = 0
    G.eval()
    with torch.no_grad():
        while n_done < n_total:
            n_chunk = min(chunk_size, n_total - n_done)
            z = torch.randn(n_chunk, latent_dim, device=device)
            g = G(z).cpu().numpy()
            out.append(g)
            n_done += n_chunk
            print(f"\r   Generazione: {n_done:,} / {n_total:,} "
                  f"({100*n_done/n_total:.1f}%)", end="")
    print()
    return np.concatenate(out, axis=0)


def add_z_column(ps_6d: np.ndarray, z_const: float) -> np.ndarray:
    """
    Reinserisce la colonna z (costante) per riportare l'array al formato
    7D atteso da evaluate_model: (x, y, z, dx, dy, dz, E).

    ps_6d ha colonne nell'ordine (x, y, dx, dy, dz, E) — coerente con
    active_indices = [0, 1, 3, 4, 5, 6] usato in sarrut_pure_replica.py.
    """
    n = len(ps_6d)
    ps_7d = np.zeros((n, 7), dtype=np.float32)
    ps_7d[:, 0] = ps_6d[:, 0]              # x
    ps_7d[:, 1] = ps_6d[:, 1]              # y
    ps_7d[:, 2] = z_const                  # z (costante)
    ps_7d[:, 3] = ps_6d[:, 2]              # dx
    ps_7d[:, 4] = ps_6d[:, 3]              # dy
    ps_7d[:, 5] = ps_6d[:, 4]              # dz
    ps_7d[:, 6] = ps_6d[:, 5]              # E
    return ps_7d


def main():
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    print(f"\n{'='*60}")
    print(f"  Recupero valutazione: sarrut_pure_replica")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # ── 1. Carica checkpoint ────────────────────────────────────────────────
    print(f"\n  Caricamento checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    min_vals = np.array(ckpt["min_vals"], dtype=np.float32)
    max_vals = np.array(ckpt["max_vals"], dtype=np.float32)
    z_const  = float(ckpt["z_const"])
    print(f"  z_const = {z_const:.4f} cm")

    G = SarrutGenerator(latent_dim=6, output_dim=6, hidden_dim=400).to(device)
    G.load_state_dict(ckpt["generator"])
    G.eval()
    print(f"  Generator caricato: {sum(p.numel() for p in G.parameters()):,} parametri")

    # ── 2. Costruisci il riferimento reale (6D, stesso ordine del training) ─
    active_indices = [0, 1, 3, 4, 5, 6]  # x, y, dx, dy, dz, E

    if args.eval_data_path and Path(args.eval_data_path).exists():
        print(f"\n  Riferimento reale: {args.eval_data_path} (PHSP2 indipendente)")
        with h5py.File(args.eval_data_path, "r") as f:
            ps_eval_all = f["phase_space"][:]
        real_6d = ps_eval_all[:, active_indices].astype(np.float32)
    else:
        print(f"\n  Riferimento reale: split di test interno di {args.data_path}")
        with h5py.File(args.data_path, "r") as f:
            ps_all = f["phase_space"][:]
        ps_6d_full = ps_all[:, active_indices].astype(np.float32)

        n = len(ps_6d_full)
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(n)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        test_idx = perm[n_train + n_val:]
        real_6d = ps_6d_full[test_idx]

    n_eval = len(real_6d)
    print(f"  Campioni reali di riferimento: {n_eval:,}")

    # ── 3. Genera campioni dal modello (6D, spazio normalizzato) ───────────
    print(f"\n  Generazione di {n_eval:,} campioni (chunk da {args.eval_chunk:,})...")
    gen_norm = generate_in_chunks(G, n_eval, latent_dim=6, device=device,
                                   chunk_size=args.eval_chunk)

    # ── 4. Denormalizza MinMax inversa ──────────────────────────────────────
    gen_6d = gen_norm * (max_vals - min_vals) + min_vals

    # ── 5. Reinserisci z costante: 6D -> 7D per evaluate_model ─────────────
    print(f"\n  Reinserimento colonna z (fix del bug originale)...")
    real_7d = add_z_column(real_6d, z_const)
    gen_7d  = add_z_column(gen_6d,  z_const)
    print(f"  Shape real: {real_7d.shape} | Shape generated: {gen_7d.shape}")

    # ── 6. Valutazione ──────────────────────────────────────────────────────
    print(f"\n  Calcolo metriche (W1, MMD, separability, tail W1, KS)...")
    output_dir = Path(args.output_dir) / "eval_quantitative"
    report = evaluate_model(
        real_7d, gen_7d,
        model_name="sarrut_pure_replica",
        output_dir=str(output_dir),
    )

    print(f"\n{'='*60}")
    print(f"  Replica Sarrut 2019 — Risultati finali")
    print(f"{'='*60}")
    print(f"  W1 mean       = {report.get('w1', {}).get('mean', 'N/A')}")
    print(f"  MMD^2         = {report.get('mmd', 'N/A')}")
    sep = report.get("separability", {})
    print(f"  Separability  = {sep.get('accuracy', 'N/A')} ± {sep.get('std', 'N/A')}")
    print(f"\n  Report completo: {output_dir}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
