# sarrut_pure_replica.py
"""
Script standalone per la replica esatta del paper di Sarrut et al. (2019).
Generative adversarial networks (GAN) for compact beam source modelling in Monte Carlo simulations.
Phys. Med. Biol. 64 215004

Caratteristiche della replica:
  - Preprocessing: Normalizzazione MinMax nell'intervallo [0, 1]
  - Spazio latente (z_dim): 6
  - Architettura G e D: 3 hidden layers da 400 neuroni con attivazione ReLU
  - Layer di output di G: Attivazione Sigmoide
  - Algoritmo: WGAN classico con Weight Clipping a [-0.01, 0.01]
  - Ottimizzatore: RMSProp con Learning Rate = 1e-5
  - Batch Size: 10,000
  - n_critic: 4 aggiornamenti del Critic per ogni aggiornamento di G
  - Iterazioni totali: 80,000 (= aggiornamenti del generator, NON epoche)

Nota epoca vs iterazione:
  Sarrut usa "iterazioni" (chiamate informalmente "epochs" nel paper) per
  indicare il numero di AGGIORNAMENTI DEL GENERATOR, non giri completi sul
  dataset. Questo script usa --iterations direttamente con quel significato,
  quindi --iterations 80000 corrisponde 1:1 alle 80.000 del paper.

Metriche quantitative:
  Alla fine del training, oltre ai plot stile paper, vengono calcolate le
  stesse metriche usate per CFM/NSF (W1 marginali, MMD, separability score,
  tail W1, KS test) tramite evaluate_model() da evaluate.py, per permettere
  un confronto numerico diretto in tesi.
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import h5py
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description="Replica Esatta Sarrut 2019 WGAN (Clipping)")
    p.add_argument("--data_path", type=str, default="data/elekta_130mv_completo.h5",
                   help="Path al Phase Space HDF5 (training)")
    p.add_argument("--eval_data_path", type=str, default="data/elekta_130mv_eval_completo.h5",
                   help="Path al Phase Space HDF5 indipendente per la valutazione finale "
                        "(PHSP2 di Sarrut). Se assente, usa lo split di test interno.")
    p.add_argument("--iterations", type=int, default=80000,
                   help="Numero totale di iterazioni di training (G updates, NON epoche)")
    p.add_argument("--batch_size", type=int, default=10000, help="Batch size (Sarrut: 10000)")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate (Sarrut: 1e-5)")
    p.add_argument("--clip_value", type=float, default=0.01,
                   help="Soglia di clipping dei pesi del Critic (Sarrut: 0.01)")
    p.add_argument("--n_critic", type=int, default=4,
                   help="Step del Critic per ogni step di G (Sarrut: 4)")
    p.add_argument("--seed", type=int, default=42, help="Seed per la riproducibilità dello split")
    p.add_argument("--output_dir", type=str, default="outputs/sarrut_pure_replica",
                   help="Directory di output")
    p.add_argument("--log_every", type=int, default=1000, help="Stampa log ogni N iterazioni")
    p.add_argument("--plot_samples", type=int, default=50000,
                   help="Numero di campioni da usare nei plot di validazione")
    p.add_argument("--eval_chunk", type=int, default=250_000,
                   help="Dimensione chunk per la generazione durante la valutazione finale")
    p.add_argument("--skip_full_eval", action="store_true",
                   help="Se presente, salta la valutazione quantitativa completa "
                        "(usa solo --plot_samples per i grafici, nessuna metrica)")
    return p.parse_args()


# ─── Modelli Originali Sarrut 2019 ─────────────────────────────────────────────

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
            nn.Sigmoid()  # La sigmoide mappa perfettamente nell'intervallo MinMax [0, 1]
        )

    def forward(self, z):
        return self.model(z)

    def sample(self, n_samples, device="cpu"):
        z = torch.randn(n_samples, self.latent_dim, device=device)
        with torch.no_grad():
            return self.forward(z)


class SarrutCritic(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=400):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)  # Nessuna attivazione finale (WGAN Critic score lineare)
        )

    def forward(self, x):
        return self.model(x)


# ─── Helper: generazione a chunk (per valutazione su milioni di campioni) ──────

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


# ─── Main Program ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Setup hardware e directory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  RUNNING SARRUT 2019 EXACT REPLICATION")
    print(f"  Device: {device}")
    print(f"  Output: {out_dir}")
    print(f"  Iterazioni (= G updates): {args.iterations:,}  [NON epoche]")
    print(f"{'='*60}")

    # ── 1. Caricamento dati originali ───────────────────────────────────────
    print(f"-> Caricamento dataset da {args.data_path}...")
    if not os.path.exists(args.data_path):
        print(f"[ERROR] File {args.data_path} non trovato! Controlla il path.")
        sys.exit(1)

    with h5py.File(args.data_path, "r") as f:
        ps_all = f["phase_space"][:]

    print(f"   Caricati {len(ps_all):,} vettori di phase space totali.")

    # Piano z costante come riferimento per la denormalizzazione finale
    z_const = float(np.mean(ps_all[:, 2]))
    print(f"   Rilevato piano isocentrico z costante = {z_const:.4f} cm")

    # Sarrut esclude la coordinata z costante, riducendo lo spazio a 6D
    # Indici estratti: 0=x, 1=y, 3=dx, 4=dy, 5=dz, 6=E
    active_indices = [0, 1, 3, 4, 5, 6]
    ps_6d = ps_all[:, active_indices]

    # ── 2. Split Simmetrico 70/15/15 (Identico a NSF e CFM per rigore) ──────
    n = len(ps_6d)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]

    ps_train = ps_6d[train_idx]
    ps_val   = ps_6d[val_idx]
    ps_test  = ps_6d[test_idx]

    print(f"   Split coerente: train={len(ps_train):,} | val={len(ps_val):,} | "
          f"test={len(ps_test):,}")

    iters_per_epoch_equiv = n_train / args.batch_size
    print(f"   (per riferimento: {args.iterations:,} iterazioni equivalgono a "
          f"~{args.iterations/iters_per_epoch_equiv:.2f} epoche sul training set)")

    # ── 3. Preprocessing Esatto del Paper: Normalizzazione MinMax in [0, 1] ─
    print("-> Applicazione normalizzazione MinMax [0, 1] sulle 6 dimensioni attive...")
    min_vals = ps_train.min(axis=0)
    max_vals = ps_train.max(axis=0)
    max_vals = np.where(max_vals == min_vals, max_vals + 1e-8, max_vals)

    stats = {
        "min_vals": min_vals.tolist(),
        "max_vals": max_vals.tolist(),
        "active_indices": active_indices,
        "z_const": z_const,
    }
    with open(out_dir / "sarrut_minmax_stats.json", "w") as sf:
        json.dump(stats, sf, indent=2)

    ps_train_norm = (ps_train - min_vals) / (max_vals - min_vals)

    train_tensor = torch.from_numpy(ps_train_norm).float()
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=args.batch_size,
                               shuffle=True, drop_last=True)

    def infinite_get_batch(loader):
        while True:
            for batch in loader:
                yield batch[0].to(device)
    batch_generator = infinite_get_batch(train_loader)

    # ── 4. Inizializzazione Modelli e Ottimizzatori del Paper ───────────────
    print("-> Costruzione reti neurali ed ottimizzatore RMSProp...")
    G = SarrutGenerator(latent_dim=6, output_dim=6, hidden_dim=400).to(device)
    D = SarrutCritic(input_dim=6, hidden_dim=400).to(device)

    opt_G = optim.RMSprop(G.parameters(), lr=args.lr)
    opt_D = optim.RMSprop(D.parameters(), lr=args.lr)

    print(f"   G parametri: {sum(p.numel() for p in G.parameters()):,}")
    print(f"   D parametri: {sum(p.numel() for p in D.parameters()):,}")

    # ── 5. Training Loop Basato su Iterazioni Totali ────────────────────────
    print(f"\n-> Inizio loop di training avversario per {args.iterations:,} iterazioni...")
    print(f"   Batch size: {args.batch_size} | Weight Clipping: {args.clip_value} | "
          f"n_critic: {args.n_critic}")
    print("-"*60)

    t_start = time.time()
    history = {"iter": [], "loss_G": [], "loss_D": [], "w_dist": []}

    for iteration in range(1, args.iterations + 1):

        # ─── UPDATE CRITIC (D) ───
        G.eval()
        D.train()
        loss_D_accum = 0.0

        for _ in range(args.n_critic):
            real_batch = next(batch_generator)
            B = real_batch.size(0)

            z = torch.randn(B, 6, device=device)
            fake_batch = G(z).detach()

            score_real = D(real_batch)
            score_fake = D(fake_batch)
            loss_D = -(score_real.mean() - score_fake.mean())

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Weight clipping (WGAN originale)
            for p in D.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)

            loss_D_accum += loss_D.item()

        # ─── UPDATE GENERATOR (G) ───
        G.train()
        D.eval()

        z = torch.randn(args.batch_size, 6, device=device)
        fake_batch = G(z)
        loss_G = -D(fake_batch).mean()

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        if iteration % args.log_every == 0 or iteration == 1:
            elapsed = time.time() - t_start
            w_distance_est = -(loss_D_accum / args.n_critic)
            history["iter"].append(iteration)
            history["loss_G"].append(loss_G.item())
            history["loss_D"].append(loss_D_accum / args.n_critic)
            history["w_dist"].append(w_distance_est)
            print(f" Iter {iteration:>6d}/{args.iterations} | "
                  f"loss_G: {loss_G.item():>8.5f} | "
                  f"loss_D: {loss_D_accum/args.n_critic:>8.5f} | "
                  f"Est_W_Dist: {w_distance_est:>8.5f} | {elapsed:.0f}s")

    train_elapsed = time.time() - t_start
    print(f"\n-> Training completato in {train_elapsed:.0f}s "
          f"({train_elapsed/3600:.2f}h). Salvataggio pesi...")

    torch.save({
        "generator": G.state_dict(),
        "critic": D.state_dict(),
        "min_vals": min_vals,
        "max_vals": max_vals,
        "z_const": z_const,
        "history": history,
    }, out_dir / "sarrut_replica_final.pt")

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── 6. Plot stile paper (campione ridotto, veloce) ──────────────────────
    print("\n-> Generazione plot marginali stile paper...")
    G.eval()
    n_plot = min(args.plot_samples, len(ps_test))

    z_samples = torch.randn(n_plot, 6, device=device)
    with torch.no_grad():
        gen_norm_plot = G(z_samples).cpu().numpy()

    gen_phys_plot = gen_norm_plot * (max_vals - min_vals) + min_vals
    real_phys_plot = ps_test[:n_plot]

    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    axes = axes.flatten()

    # Ordine nel sottoarray 6D: x=0, y=1, dx=2, dy=3, dz=4, E=5
    plot_info = [
        {"idx": 5, "title": "E [MeV]", "name": "E"},
        {"idx": 0, "title": "x [cm]",  "name": "x"},
        {"idx": 1, "title": "y [cm]",  "name": "y"},
        {"idx": 2, "title": "dx",       "name": "dx"},
        {"idx": 3, "title": "dy",       "name": "dy"},
        {"idx": 4, "title": "dz",       "name": "dz"},
    ]

    for ax_idx, item in enumerate(plot_info):
        ax = axes[ax_idx]
        i = item["idx"]

        lo, hi = np.percentile(real_phys_plot[:, i], [0.1, 99.9])
        bins = np.linspace(lo, hi, 120)

        ax.hist(real_phys_plot[:, i], bins=bins, color="#5b84c4", alpha=0.6,
                label="PHSP", density=False)
        ax.hist(gen_phys_plot[:, i], bins=bins, color="#e69a6a", alpha=0.6,
                label="GAN", density=False)

        ax.set_title(f"Distribution: {item['title']}", fontsize=10, fontweight="bold")
        ax.set_xlabel(item["name"], fontsize=9)
        ax.set_ylabel("Counts", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plot_path = out_dir / "sarrut_replica_marginals.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Grafici marginali salvati in: {plot_path}")

    # ── 7. Valutazione quantitativa completa (W1, MMD, separability, ecc.) ──
    if args.skip_full_eval:
        print("\n-> --skip_full_eval attivo: nessuna metrica quantitativa calcolata.")
        print("="*60 + "\n")
        return

    print("\n" + "="*60)
    print("  VALUTAZIONE QUANTITATIVA COMPLETA")
    print("="*60)

    from evaluate import evaluate_model

    # Sceglie il riferimento reale per la valutazione finale:
    # preferibilmente il file di eval indipendente (PHSP2), altrimenti
    # ricade sullo split di test interno (PHSP1, 15%)
    if args.eval_data_path and os.path.exists(args.eval_data_path):
        print(f"  Riferimento reale: {args.eval_data_path} (PHSP2 indipendente)")
        with h5py.File(args.eval_data_path, "r") as f:
            ps_eval_all = f["phase_space"][:]
        real_eval_6d = ps_eval_all[:, active_indices].astype(np.float32)
    else:
        print(f"  Riferimento reale: split di test interno (15% di PHSP1, "
              f"{len(ps_test):,} campioni)")
        real_eval_6d = ps_test.astype(np.float32)

    n_eval = len(real_eval_6d)
    print(f"  Campioni reali di riferimento: {n_eval:,}")

    print(f"\n  Generazione di {n_eval:,} campioni dal generatore "
          f"(chunk da {args.eval_chunk:,})...")
    gen_norm_full = generate_in_chunks(G, n_eval, latent_dim=6, device=device,
                                        chunk_size=args.eval_chunk)

    # Denormalizzazione MinMax inversa
    gen_phys_full = gen_norm_full * (max_vals - min_vals) + min_vals

    # evaluate_model si aspetta colonne nell'ordine standard del progetto.
    # Qui lavoriamo con l'ordine ridotto a 6D (x,y,dx,dy,dz,E) coerente per
    # entrambi real e generated, quindi il confronto interno è valido.
    print("\n  Calcolo metriche (W1, MMD, separability, tail W1, KS)...")
    report = evaluate_model(
        real_eval_6d, gen_phys_full,
        model_name="sarrut_pure_replica",
        output_dir=str(out_dir / "eval_quantitative"),
    )

    print(f"\n{'='*60}")
    print(f"  Replica Sarrut 2019 — Risultati finali")
    print(f"{'='*60}")
    print(f"  W1 mean       = {report.get('w1', {}).get('mean', 'N/A')}")
    print(f"  MMD^2         = {report.get('mmd', 'N/A')}")
    sep = report.get("separability", {})
    print(f"  Separability  = {sep.get('accuracy', 'N/A')} ± {sep.get('std', 'N/A')}")
    print(f"\n  Report completo: {out_dir}/eval_quantitative/")
    print(f"  Checkpoint:      {out_dir / 'sarrut_replica_final.pt'}")
    print(f"  Plot marginali:  {plot_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
