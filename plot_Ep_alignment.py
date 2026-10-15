"""
plot_Ep_alignment.py  (numpy/scipy/matplotlib)
==============================================
Verifica l'ipotesi: nello spazio v = E^p (p ~ 1.77, Bragg-Kleeman) lo spettro di energia a profondità
diverse è lo STESSO spettro traslato di  k*d  (ogni protone perde la stessa quantità di v per cm).
Se è vero, u = E^p + k*d  ("energia di ingresso retro-proiettata") ha un picco-struttura quasi uguale a tutte
le profondità, e i picchi multipli che il flow sbaglia a 3.5 diventano un problema facile.

    python plot_Ep_alignment.py --data data/proton_depth_train.h5 data/proton_depth_train_holdout.h5 --out eval/Ep_alignment

Stampa, per diversi p, la distanza media (W1/std) tra gli spettri u di profondità adiacenti (più piccola = meglio
allineati) e salva i grafici per il p migliore.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import wasserstein_distance


def load(files, n_max):
    rng = np.random.default_rng(0); E, D = [], []
    for f in files:
        with h5py.File(f, "r") as h:
            e = h["phase_space"][:, 6]; d = h["conditions"][:, 0]
        for dd in np.unique(d):
            m = np.where(np.isclose(d, dd))[0]
            if len(m) > n_max: m = rng.choice(m, n_max, replace=False)
            E.append(e[m]); D.append(np.full(len(m), dd))
    return np.concatenate(E).astype(np.float64), np.concatenate(D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True); ap.add_argument("--out", default="Ep_alignment")
    ap.add_argument("--ps", nargs="+", type=float, default=[1.5, 1.6, 1.7, 1.77, 1.85, 1.95, 2.1])
    ap.add_argument("--fit_dmax", type=float, default=2.5, help="profondità max per fittare k (prima dell'effetto fine range)")
    ap.add_argument("--n_max", type=int, default=100000)
    a = ap.parse_args()
    E, D = load(a.data, a.n_max); ds = np.unique(D)
    print("profondità:", ds.tolist())

    best = None
    print(f"\n{'p':>5} {'k [per cm]':>11} {'W1 adiacenti (u)':>18}")
    for p in a.ps:
        v = E ** p
        sel = ds[ds <= a.fit_dmax]
        mv = np.array([v[np.isclose(D, d)].mean() for d in sel])
        k = -np.polyfit(sel, mv, 1)[0]
        u = v + k * D
        sc = u[D <= a.fit_dmax].std()
        pairs = [(ds[i], ds[i + 1]) for i in range(len(ds) - 1) if ds[i + 1] <= 3.0]     # confronto fino a 3.0 (zona pulita)
        w = np.mean([wasserstein_distance(u[np.isclose(D, x)], u[np.isclose(D, y)]) / sc for x, y in pairs])
        print(f"{p:5.2f} {k:11.1f} {w:18.4f}")
        if best is None or w < best[0]: best = (w, p, k)
    _, p, k = best
    print(f"\nmigliore: p={p}, k={k:.1f} per cm")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    v = E ** p; u = v + k * D
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for d in ds:
        m = np.isclose(D, d)
        for a_, x, lab in ((ax[0], v[m], "v = E^p"), (ax[1], u[m], "u = E^p + k d")):
            h, b = np.histogram(x, bins=200, range=(np.percentile(u, 0.01) if lab != "v = E^p" else 0, np.percentile(u, 99.99) if lab != "v = E^p" else np.percentile(v, 99.99)), density=True)
            a_.step(0.5 * (b[1:] + b[:-1]), h, where="mid", label=f"d={d:g}")
            a_.set_xlabel(lab)
    for a_ in ax: a_.set_yscale("log"); a_.set_ylim(1e-6, None)
    ax[0].legend(fontsize=7); ax[0].set_title(f"spettro in v (p={p})"); ax[1].set_title(f"spettro in u, k={k:.0f}: se allineati, struttura = traslazione")
    fig.tight_layout(); fig.savefig(str(out) + ".png", dpi=110); print(f"salvato {out}.png")


if __name__ == "__main__":
    main()
