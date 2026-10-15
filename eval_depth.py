"""
eval_depth.py  (numpy/scipy/matplotlib)
=======================================
Confronta un phase space generato con quello reale alla stessa profondità (tipicamente una profondità
di HOLD-OUT mai vista in training).

    python eval_depth.py --real data/raw_phsp/SurfaceFile_3.5_peak_1_2_5.phsp --gen gen_3.5.h5 --out eval/3.5

Metriche (sui protoni, variabili y, dx, dy, E):
  * Wasserstein-1 per variabile, normalizzata sulla std del reale
  * differenza di media e std (in unità della std reale)
  * differenza della matrice di correlazione (max |Δρ|)
  * KS sulle code di E (E < 5° percentile reale): la zona critica vicino a fine range
Baseline di rumore: stessa metrica tra due metà casuali del reale (il generato non può fare meglio di così).
"""
import argparse, json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp

from read_topas_phsp import load_filtered

VARS = {"y": 1, "dx": 3, "dy": 4, "E": 6}


def load_any(path):
    if str(path).endswith(".h5"):
        import h5py
        with h5py.File(path, "r") as h:
            return h["phase_space"][:]
    return load_filtered(path, protons_only=True)[0]


def metrics(real, gen, rng, n_max=300_000):
    if len(real) > n_max: real = real[rng.choice(len(real), n_max, replace=False)]
    if len(gen) > n_max:  gen = gen[rng.choice(len(gen), n_max, replace=False)]
    out = {}
    for name, j in VARS.items():
        r, g = real[:, j].astype(np.float64), gen[:, j].astype(np.float64)
        s = r.std()
        out[name] = dict(W1_over_std=wasserstein_distance(r, g) / s,
                         dmean_over_std=(g.mean() - r.mean()) / s, dstd_rel=g.std() / s - 1)
    cols = list(VARS.values()); names = list(VARS)
    Cr, Cg = np.corrcoef(real[:, cols].T), np.corrcoef(gen[:, cols].T)
    D = np.abs(Cr - Cg); np.fill_diagonal(D, 0); i, j = np.unravel_index(D.argmax(), D.shape)
    out["max_abs_dcorr"] = float(D.max())
    out["worst_corr_pair"] = dict(pair=f"{names[i]}-{names[j]}", real=float(Cr[i, j]), gen=float(Cg[i, j]))
    # massa generata sotto i quantili REALI di E (attesi: 1,5,25,50,75,95 %): misura la massa, non solo la forma
    qs = [1, 5, 25, 50, 75, 95]; thr = np.percentile(real[:, 6], qs)
    out["E_mass_below_real_quantiles"] = {f"q{q}": float((gen[:, 6] < t).mean()) for q, t in zip(qs, thr)}
    thr = np.percentile(real[:, 6], 5)
    out["KS_lowE_tail"] = float(ks_2samp(real[real[:, 6] < thr, 6], gen[gen[:, 6] < thr, 6]).statistic)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True); ap.add_argument("--gen", nargs="+", required=True,
                    help="uno o più file generati (h5/phsp): vengono confrontati in un'unica tabella")
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    real = load_any(a.real)
    half = rng.permutation(len(real)); h1, h2 = real[half[: len(real) // 2]], real[half[len(real) // 2:]]
    floor = metrics(h1, h2, rng)
    res = {"noise_floor_real_vs_real": floor}
    gens = {Path(g).stem: load_any(g) for g in a.gen}
    for name, g in gens.items():
        res[name] = metrics(real, g, rng)

    for name in gens:
        m = res[name]
        print(f"\n=== {name} ===")
        print(f"{'var':>4} | {'W1/std gen':>10} {'floor':>8} | {'dmean/std':>9} | {'dstd_rel':>8}")
        for v in VARS:
            print(f"{v:>4} | {m[v]['W1_over_std']:10.4f} {floor[v]['W1_over_std']:8.4f} | {m[v]['dmean_over_std']:+9.4f} | {m[v]['dstd_rel']:+8.4f}")
        w = m["worst_corr_pair"]
        print(f"max|dcorr|: gen {m['max_abs_dcorr']:.4f} floor {floor['max_abs_dcorr']:.4f}   coppia peggiore: {w['pair']} (reale {w['real']:+.3f}, gen {w['gen']:+.3f})")
        print(f"KS coda E<5% (forma): gen {m['KS_lowE_tail']:.4f} floor {floor['KS_lowE_tail']:.4f}")

    print("\n=== CONFRONTO su E: massa generata sotto i quantili reali (attesi 1/5/25/50/75/95 %) ===")
    print(f"{'run':<28} {'W1/std':>7} {'dmean':>7} {'dstd':>7} | " + " ".join(f"{q:>6}" for q in ("q1", "q5", "q25", "q50", "q75", "q95")))
    for name in gens:
        m = res[name]; mq = m["E_mass_below_real_quantiles"]
        print(f"{name:<28} {m['E']['W1_over_std']:7.3f} {m['E']['dmean_over_std']:+7.3f} {m['E']['dstd_rel']:+7.3f} | " +
              " ".join(f"{100 * mq[k]:5.1f}%" for k in ("q1", "q5", "q25", "q50", "q75", "q95")))
    json.dump(res, open(out / "metrics.json", "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for a_, (v, j) in zip(ax.ravel(), VARS.items()):
        lo, hi = np.percentile(real[:, j], [0.05, 99.95]); b = np.linspace(lo, hi, 120)
        a_.hist(real[:, j], b, density=True, histtype="step", color="k", lw=1.6, label="reale")
        for name, g in gens.items():
            a_.hist(g[:, j], b, density=True, histtype="step", label=name)
        a_.set_xlabel(v); a_.set_yscale("log" if v == "E" else "linear")
    ax[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "marginals.png", dpi=110)
    print(f"\nsalvato in {out}")


if __name__ == "__main__":
    main()
