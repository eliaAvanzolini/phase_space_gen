"""
analyze_beta_ydy.py
===================
Analisi del coefficiente di regressione beta(d) tra y e dy:
  dy = beta(d) * y + residuo
  beta(d) = Cov(y, dy) / Var(y) = corr(y, dy) * (std(dy) / std(y))

Verifica se il disallineamento della correlazione y-dy (0.217 vs 0.240)
e' correggibile linearmente su CPU senza riaddestrare.
"""
from pathlib import Path
import h5py
import numpy as np
from scipy.stats import wasserstein_distance
import read_topas_phsp as rtp

TRAIN_DEPTHS = [0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0]
DATA_DIR = "data/raw_phsp"


def compute_beta(y, dy):
    var_y = np.var(y)
    cov_y_dy = np.cov(y, dy)[0, 1]
    corr = np.corrcoef(y, dy)[0, 1]
    beta = cov_y_dy / (var_y + 1e-12)
    res = dy - beta * y
    std_res = np.std(res)
    return beta, corr, std_res


def main():
    print("=== 1. EVOLUZIONE FISICA REALE DI beta(d) NEL FANTOCCIO TOPAS ===")
    print(f"{'d [cm]':>7s} | {'corr(y, dy)':>12s} | {'std(y) [cm]':>12s} | {'std(dy)':>10s} | {'beta [rad/cm]':>14s} | {'std(residuo)':>13s}")

    beta_dict = {}
    for p in sorted(Path(DATA_DIR).glob("*.phsp")):
        if "peak" not in p.name.lower():
            continue
        d = rtp.parse_depth_from_name(p.name)
        if d is None:
            continue
        d_val = round(float(d), 3)
        if d_val not in TRAIN_DEPTHS:
            continue
        df = rtp.read_phsp(str(p))
        if "pdg" in df.columns:
            df = df[df["pdg"] == 2212]
        ps7 = rtp.to_7d(df).astype(np.float64)
        ps7 = ps7[ps7[:, 6] > 0.0]

        y = ps7[:, 1]
        dy = ps7[:, 4]
        beta, corr, std_res = compute_beta(y, dy)
        beta_dict[d_val] = (beta, corr, y.std(), dy.std(), std_res)
        tag = " <-- TARGET 3.5 CM" if d_val == 3.5 else ""
        print(f"{d_val:7.2f} | {corr:12.4f} | {y.std():12.5f} | {dy.std():10.5f} | {beta:14.5f} | {std_res:13.5f}{tag}")

    # Confronto sui file generati a 3.5 cm
    print("\n=== 2. DIAGNOSI beta(3.5 cm): TOPAS REALE vs MODELLI GENERATI ===")
    real_beta, real_corr, real_sy, real_sdy, real_sres = beta_dict[3.5]

    gen_files = [
        ("cfm_condE_s42 (base)",       "gen/gen_3.5_cfm_condE_s42.h5"),
        ("resq_s42 (base)",           "gen/gen_3.5_resq_cfm_condE_s42.h5"),
        ("resq_s43 (base)",           "gen/gen_3.5_resq_cfm_condE_s43.h5"),
        ("resq_flip_s42",             "gen/gen_3.5_resq_cfm_condE_flip_s42.h5"),
        ("resq_flip_s43",             "gen/gen_3.5_resq_cfm_condE_flip_s43.h5"),
    ]

    print(f"{'Modello':28s} | {'corr(y,dy)':>10s} {'Delta corr':>11s} | {'beta':>10s} {'Delta beta':>11s} | {'W1(dy)':>8s}")
    for label, path in gen_files:
        p = Path(path)
        if not p.exists():
            continue
        with h5py.File(p, "r") as h:
            ps7 = h["phase_space"][:, [1, 4]].astype(np.float64)
        y_g, dy_g = ps7[:, 0], ps7[:, 1]
        b_g, c_g, sres_g = compute_beta(y_g, dy_g)

        # Calcola W1(dy) contro reale
        real_df = rtp.read_phsp("data/raw_phsp/SurfaceFile_3.5_peak_1_2_5.phsp")
        if "pdg" in real_df.columns:
            real_df = real_df[real_df["pdg"] == 2212]
        real_dy = rtp.to_7d(real_df)[:, 4].astype(np.float64)
        w1_dy = wasserstein_distance(real_dy, dy_g) / real_dy.std()

        print(f"{label:28s} | {c_g:10.4f} {c_g - real_corr:+11.4f} | {b_g:10.5f} {b_g - real_beta:+11.5f} | {w1_dy:8.4f}")

    # 3. Test di correzione lineare a posteriori
    print("\n=== 3. TEST CORREZIONE LINEARE A POSTERIORI SU resq_flip_s43 ===")
    target_path = Path("gen/gen_3.5_resq_cfm_condE_flip_s43.h5")
    if target_path.exists():
        with h5py.File(target_path, "r") as h:
            ps7 = h["phase_space"][:]
        y_g = ps7[:, 1].astype(np.float64)
        dy_g = ps7[:, 4].astype(np.float64)
        b_g, c_g, _ = compute_beta(y_g, dy_g)

        # Correzione dell'inclinazione: aggiunge (beta_reale - beta_generato) * y
        delta_beta = real_beta - b_g
        dy_corr = dy_g + delta_beta * y_g

        # Verifica della nuova correlazione e W1
        new_corr = np.corrcoef(y_g, dy_corr)[0, 1]
        new_w1 = wasserstein_distance(real_dy, dy_corr) / real_dy.std()
        new_sdy = dy_corr.std()
        dstd = (new_sdy - real_dy.std()) / real_dy.std()

        print(f"  Prima della correzione : corr(y, dy) = {c_g:.4f} | W1(dy) = {w1_dy:.4f} | dstd = {100*(dy_g.std()-real_dy.std())/real_dy.std():+.2f}%")
        print(f"  Delta beta applicato   : {delta_beta:+.5f} rad/cm")
        print(f"  Dopo la correzione     : corr(y, dy) = {new_corr:.4f} | W1(dy) = {new_w1:.4f} | dstd = {100*dstd:+.2f}%")
        if abs(new_corr - real_corr) < 0.005:
            print("  --> ESITO: La discrepanza su corr(y, dy) è al 100% una debole sotto-stima di beta!")


if __name__ == "__main__":
    main()
