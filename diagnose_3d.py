"""
diagnose_3d.py
==============
Diagnosi leggera (CPU, pochi secondi) sul modello 3D cfm_condE e su stats_at(3.5):
  1. Confronta stats_at(3.5) interpolata con le statistiche reali TOPAS a 3.5 cm e mostra l'evoluzione con d.
  2. Isola quanto dell'errore W1/std su (y, dx, dy) dipende da:
       - offset di media (dmean ~ -0.025)
       - errore di scala sigma(3.5) da stats_at
       - errore puro di forma (code di scattering della rete 3D)
  3. Valuta l'errore angolare per fasce di energia (E < 10, 10-18, > 18 MeV).
"""
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.stats import wasserstein_distance

import read_topas_phsp as rtp
from phsp_transforms import load_stats, stats_at, e_fwd

REAL_35 = "data/raw_phsp/SurfaceFile_3.5_peak_1_2_5.phsp"
GEN_RESQ = "gen/gen_3.5_neural_resq_condE.h5"
GEN_ORAC = "gen/gen_3.5_condE_oracle.h5"
STATS_JSON = "runs/cfm_condE_s42/stats.json"


def load_real_4d(phsp_path):
    df = rtp.read_phsp(str(phsp_path))
    if "pdg" in df.columns:
        df = df[df["pdg"] == 2212]
    ps7 = rtp.to_7d(df)
    # (y, dx, dy, E) -> colonne 1, 3, 4, 6
    return ps7[:, [1, 3, 4, 6]].astype(np.float64)


def load_gen_4d(h5_path):
    with h5py.File(h5_path, "r") as h:
        ps7 = h["phase_space"][:]
    return ps7[:, [1, 3, 4, 6]].astype(np.float64)


def main():
    print("=== 1. EVOLUZIONE DI MEDIA E STD CON LA PROFONDITÀ (stats.json vs Reale 3.5 cm) ===")
    stats = load_stats(STATS_JSON)
    raw_stats = json.load(open(STATS_JSON))
    print("Contenuto chiavi stats.json:", list(raw_stats.keys()))

    for d in [0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0]:
        mu, sd = stats_at(stats, d)
        tag = " <-- INTERPOLATO (hold-out)" if d == 3.5 else " (training)"
        print(f"  d={d:4.2f} cm | mu=[y:{mu[0]:+.4f}, dx:{mu[1]:+.4f}, dy:{mu[2]:+.4f}, fE:{mu[3]:6.2f}] "
              f"| sd=[y:{sd[0]:.4f}, dx:{sd[1]:.4f}, dy:{sd[2]:.4f}, fE:{sd[3]:5.2f}]{tag}")

    print("\nCaricamento dati reali a 3.5 cm...")
    real = load_real_4d(REAL_35)
    mu_r = real.mean(axis=0)
    sd_r = real.std(axis=0)
    fE_r = e_fwd(real[:, 3], stats.get("etrans", "lin"), stats.get("p", 1.77))
    mu_interp, sd_interp = stats_at(stats, 3.5)

    names = ["y", "dx", "dy", "f(E)"]
    mu_r_all = [mu_r[0], mu_r[1], mu_r[2], fE_r.mean()]
    sd_r_all = [sd_r[0], sd_r[1], sd_r[2], fE_r.std()]

    print("\n--- CONFRONTO DIRETTO A 3.5 CM: stats_at(3.5) vs REALE TOPAS ---")
    print(f"  {'Var':6s} | {'mu_real':>10s} {'mu_interp':>10s} {'diff/sd':>9s} | "
          f"{'sd_real':>10s} {'sd_interp':>10s} {'err_rel_sd':>10s}")
    for i, nm in enumerate(names):
        dm = (mu_interp[i] - mu_r_all[i]) / sd_r_all[i]
        ds = (sd_interp[i] - sd_r_all[i]) / sd_r_all[i]
        print(f"  {nm:6s} | {mu_r_all[i]:10.5f} {mu_interp[i]:10.5f} {dm:+9.4f} | "
              f"{sd_r_all[i]:10.5f} {sd_interp[i]:10.5f} {100*ds:+9.2f}%")

    for label, path in [("ResidualQ + CFM 3D", GEN_RESQ), ("Oracle + CFM 3D", GEN_ORAC)]:
        if not Path(path).exists():
            continue
        gen = load_gen_4d(path)
        print(f"\n=== 2. SCOMPOSIZIONE ERRORE 3D PER: {label} ({path}) ===")
        print(f"  {'Var':4s} | {'W1/sd (raw)':>12s} | {'W1/sd (zero-mean)':>18s} | "
              f"{'W1/sd (oracle mu,sd)':>20s} | {'dmean/sd':>9s} {'dstd_rel':>9s}")
        for i, nm in enumerate(["y", "dx", "dy"]):
            r_col = real[:, i]
            g_col = gen[:, i]
            sr = r_col.std()

            w1_raw = wasserstein_distance(r_col, g_col) / sr
            # Se forziamo solo media zero (o media uguale al reale)
            g_zm = g_col - g_col.mean() + r_col.mean()
            w1_zm = wasserstein_distance(r_col, g_zm) / sr
            # Se standardizziamo anche con la std reale (isola il puro errore di FORMA della rete 3D)
            g_shape = (g_col - g_col.mean()) / g_col.std() * sr + r_col.mean()
            w1_shape = wasserstein_distance(r_col, g_shape) / sr

            dm = (g_col.mean() - r_col.mean()) / sr
            ds = (g_col.std() - sr) / sr
            print(f"  {nm:4s} | {w1_raw:12.4f} | {w1_zm:18.4f} | {w1_shape:20.4f} | {dm:+9.4f} {100*ds:+8.2f}%")

        # Verifica cosa ha prodotto la rete nello spazio latente Z prima di moltiplicare per sd_interp + mu_interp
        print("  -> Verifica uscita latente Z della rete 3D (atteso: mean=0.000, std=1.000):")
        for i, nm in enumerate(["y", "dx", "dy"]):
            z_gen = (gen[:, i] - mu_interp[i]) / sd_interp[i]
            z_real = (real[:, i] - mu_interp[i]) / sd_interp[i]
            print(f"     Z({nm}): gen_mean={z_gen.mean():+.4f} (real={z_real.mean():+.4f}) | "
                  f"gen_std={z_gen.std():.4f} (real={z_real.std():.4f})")

        print(f"\n=== 3. ERRORE ANGOLARE PER FASCE DI ENERGIA ({label}) ===")
        bands = [(0.0, 10.0, "Coda bassa (<10 MeV)"),
                 (10.0, 18.0, "Valle (10-18 MeV)"),
                 (18.0, 35.0, "Picco (>=18 MeV)")]
        for e_lo, e_hi, bname in bands:
            mr = (real[:, 3] >= e_lo) & (real[:, 3] < e_hi)
            mg = (gen[:, 3] >= e_lo) & (gen[:, 3] < e_hi)
            if mr.sum() < 100 or mg.sum() < 100:
                continue
            w_dx = wasserstein_distance(real[mr, 1], gen[mg, 1]) / real[mr, 1].std()
            w_dy = wasserstein_distance(real[mr, 2], gen[mg, 2]) / real[mr, 2].std()
            ds_dx = (gen[mg, 1].std() - real[mr, 1].std()) / real[mr, 1].std()
            ds_dy = (gen[mg, 2].std() - real[mr, 2].std()) / real[mr, 2].std()
            print(f"  {bname:22s} | N_real={mr.sum():6d} | dx: W1/sd={w_dx:.4f} (dstd={100*ds_dx:+5.1f}%) "
                  f"| dy: W1/sd={w_dy:.4f} (dstd={100*ds_dy:+5.1f}%)")

    print("\n=== 4. FILE .H5 PRESENTI IN gen/ (per pulizia tabella NSF/GAN) ===")
    for p in sorted(Path("gen").glob("*.h5")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
