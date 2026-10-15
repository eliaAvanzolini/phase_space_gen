"""
verify_3d_flip.py
=================
Verifica rigorosa del flip di simmetria fisica:
  1. Stesso seed e stesso campione (raw vs flip).
  2. Matrice di correlazione completa (tutte le 6 coppie cinematiche).
  3. Errore per fasce di energia (lenti vs veloci) prima e dopo il flip.
"""
from pathlib import Path
import h5py
import numpy as np
from scipy.stats import wasserstein_distance

import read_topas_phsp as rtp

REAL_35 = "data/raw_phsp/SurfaceFile_3.5_peak_1_2_5.phsp"
GEN_FILE = "gen/gen_3.5_resq_cfm_condE_s43.h5"


def load_real():
    df = rtp.read_phsp(REAL_35)
    if "pdg" in df.columns:
        df = df[df["pdg"] == 2212]
    ps7 = rtp.to_7d(df).astype(np.float64)
    # Colonne: 1:y, 3:dx, 4:dy, 6:E
    return ps7[:, [1, 3, 4, 6]]


def apply_flip_exact(data, seed=42):
    rng = np.random.default_rng(seed)
    out = data.copy()
    n = len(out)
    flip_x = rng.choice([-1.0, 1.0], size=n)
    flip_y = rng.choice([-1.0, 1.0], size=n)
    out[:, 1] *= flip_x          # dx -> -dx
    out[:, 0] *= flip_y          # y  -> -y
    out[:, 2] *= flip_y          # dy -> -dy
    return out


def main():
    print("=== CARICAMENTO DATI PER VERIFICA FLIP (SEED FISSO S43) ===")
    real = load_real()
    with h5py.File(GEN_FILE, "r") as h:
        raw_gen = h["phase_space"][:, [1, 3, 4, 6]].astype(np.float64)

    flip_gen = apply_flip_exact(raw_gen, seed=42)

    names = ["y", "dx", "dy", "E"]
    sr = real.std(axis=0)

    print("\n--- 1. METRICHE MARGINALI (RAW vs FLIP sullo stesso campione) ---")
    print(f"{'Var':4s} | {'W1/sd (RAW)':>12s} {'dmean (RAW)':>12s} {'dstd (RAW)':>11s} | "
          f"{'W1/sd (FLIP)':>13s} {'dmean (FLIP)':>13s} {'dstd (FLIP)':>12s}")
    for i, nm in enumerate(names):
        w1_r = wasserstein_distance(real[:, i], raw_gen[:, i]) / sr[i]
        dm_r = (raw_gen[:, i].mean() - real[:, i].mean()) / sr[i]
        ds_r = (raw_gen[:, i].std() - sr[i]) / sr[i]

        w1_f = wasserstein_distance(real[:, i], flip_gen[:, i]) / sr[i]
        dm_f = (flip_gen[:, i].mean() - real[:, i].mean()) / sr[i]
        ds_f = (flip_gen[:, i].std() - sr[i]) / sr[i]

        print(f"{nm:4s} | {w1_r:12.4f} {dm_r:+12.4f} {100*ds_r:+10.2f}% | "
              f"{w1_f:13.4f} {dm_f:+13.4f} {100*ds_f:+11.2f}%")

    print("\n--- 2. MATRICE DELLE CORRELAZIONI COMPLETE (Tutte le 6 coppie) ---")
    pairs = [(0, 1, "y-dx"), (0, 2, "y-dy"), (0, 3, "y-E"),
             (1, 2, "dx-dy"), (1, 3, "dx-E"), (2, 3, "dy-E")]

    corr_r = np.corrcoef(real.T)
    corr_raw = np.corrcoef(raw_gen.T)
    corr_flp = np.corrcoef(flip_gen.T)

    print(f"{'Coppia':8s} | {'Reale':>9s} | {'RAW':>9s} {'Delta RAW':>10s} | {'FLIP':>9s} {'Delta FLIP':>11s}")
    for i, j, label in pairs:
        cr = corr_r[i, j]
        cg_raw = corr_raw[i, j]
        cg_flp = corr_flp[i, j]
        print(f"{label:8s} | {cr:+9.4f} | {cg_raw:+9.4f} {cg_raw - cr:+10.4f} | {cg_flp:+9.4f} {cg_flp - cr:+11.4f}")

    print("\n--- 3. ERRORE ANGOLARE PER FASCE DI ENERGIA (RAW vs FLIP) ---")
    bands = [(0.0, 10.0, "Coda (<10 MeV)"),
             (10.0, 18.0, "Valle (10-18 MeV)"),
             (18.0, 80.0, "Picco (>=18 MeV)")]

    print(f"{'Fascia':18s} | {'dx W1 RAW':>10s} {'dx W1 FLIP':>11s} {'dstd dx FLIP':>13s} | "
          f"{'dy W1 RAW':>10s} {'dy W1 FLIP':>11s} {'dstd dy FLIP':>13s}")
    for e_lo, e_hi, bname in bands:
        mr = (real[:, 3] >= e_lo) & (real[:, 3] < e_hi)
        mg_r = (raw_gen[:, 3] >= e_lo) & (raw_gen[:, 3] < e_hi)
        mg_f = (flip_gen[:, 3] >= e_lo) & (flip_gen[:, 3] < e_hi)

        w_dx_raw = wasserstein_distance(real[mr, 1], raw_gen[mg_r, 1]) / real[mr, 1].std()
        w_dx_flp = wasserstein_distance(real[mr, 1], flip_gen[mg_f, 1]) / real[mr, 1].std()
        ds_dx_flp = (flip_gen[mg_f, 1].std() - real[mr, 1].std()) / real[mr, 1].std()

        w_dy_raw = wasserstein_distance(real[mr, 2], raw_gen[mg_r, 2]) / real[mr, 2].std()
        w_dy_flp = wasserstein_distance(real[mr, 2], flip_gen[mg_f, 2]) / real[mr, 2].std()
        ds_dy_flp = (flip_gen[mg_f, 2].std() - real[mr, 2].std()) / real[mr, 2].std()

        print(f"{bname:18s} | {w_dx_raw:10.4f} {w_dx_flp:11.4f} {100*ds_dx_flp:+12.2f}% | "
              f"{w_dy_raw:10.4f} {w_dy_flp:11.4f} {100*ds_dy_flp:+12.2f}%")


if __name__ == "__main__":
    main()
