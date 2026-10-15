"""
shift_baseline.py  (numpy)
==========================
BASELINE FISICA SENZA TRAINING: genera il phase space a una profondità d' prendendo i protoni di una
profondità SORGENTE più superficiale (d_src < d') e "degradandoli" di k*(d' - d_src) in v = E^p:

    v' = v - k*(d' - d_src);   i protoni con v' <= 0 si sono fermati e vengono scartati;   E' = v'^(1/p)

y, dx, dy restano quelli della sorgente (nessuna correzione di scattering aggiuntivo; opzionale --extra_sigma_v
aggiunge un allargamento gaussiano in v per mimare lo straggling in più).
Serve a (1) dare un punto di confronto onesto al flow, (2) verificare l'ipotesi "profondità = traslazione in E^p".

    python shift_baseline.py --data data/proton_depth_train.h5 --depth 3.5 --out gen_3.5_shift.h5
    python eval_depth.py --real data/raw_phsp/SurfaceFile_3.5_peak_1_2_5.phsp --gen gen_3.5_shift.h5 gen_3.5_hybrid.h5 --out eval/compare_shift

k: --k_mode local (default) = pendenza della retta fittata sulle MEDIANE di v alle ultime --k_n profondità di training <= d_src
   (la pendenza locale cresce un po' verso fine range); --k_mode global = retta su tutte le profondità <= --fit_dmax.
"""
import argparse
import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="HDF5 di TRAINING (phase_space, conditions)")
    ap.add_argument("--depth", type=float, required=True, help="profondità da generare [cm]")
    ap.add_argument("--src", type=float, default=None, help="profondità sorgente (default: la più vicina <= depth tra quelle di training)")
    ap.add_argument("--p", type=float, default=1.77)
    ap.add_argument("--k_mode", choices=["local", "global"], default="local")
    ap.add_argument("--k_n", type=int, default=3, help="n. di profondità per la pendenza locale")
    ap.add_argument("--fit_dmax", type=float, default=3.0)
    ap.add_argument("--k_quantile", type=float, default=50.0,
                    help="quantile (%%) di v usato per misurare la pendenza k: 50 = mediana (default); valori più alti (es. 75) "
                         "sono meno sensibili al taglio v>0 che cresce con la profondità")
    ap.add_argument("--k_scale", type=float, default=1.0, help="fattore moltiplicativo su k (calibrazione fine dello shift, es. 1.02)")
    ap.add_argument("--extra_sigma_v", type=float, default=0.0, help="allargamento gaussiano aggiuntivo in v (unità di v)")
    ap.add_argument("--angle_mode", choices=["none", "scale", "kick"], default="none",
                    help="correzione degli angoli: la larghezza (IQR/1.349) di dx,dy a depth è interpolata (log-lineare) dalle profondità di training; "
                         "scale = moltiplica dx,dy per w_target/w_src; kick = aggiunge rumore gaussiano con varianza w_target^2 - w_src^2")
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    with h5py.File(a.data, "r") as h:
        ps, dep = h["phase_space"][:], h["conditions"][:, 0]
    ds = np.unique(np.round(dep, 6))
    below = ds[ds <= a.depth + 1e-9]
    if len(below) == 0:
        raise SystemExit("nessuna profondità di training <= depth")
    src = a.src if a.src is not None else float(below.max())
    if src >= a.depth:
        raise SystemExit(f"la sorgente ({src}) deve essere più superficiale di depth ({a.depth})")

    v_all = ps[:, 6].astype(np.float64) ** a.p
    med = {d: np.percentile(v_all[np.isclose(dep, d)], a.k_quantile) for d in ds}
    if a.k_mode == "local":
        use = ds[ds <= src + 1e-9][-a.k_n:]
    else:
        use = ds[ds <= a.fit_dmax + 1e-9]
    k = -np.polyfit(use, [med[d] for d in use], 1)[0] * a.k_scale
    print(f"sorgente d={src}, target d={a.depth}, k={k:.1f} per cm (profondità usate: {use.tolist()}), shift in v = {k * (a.depth - src):.1f}")

    m = np.isclose(dep, src)
    q = ps[m].copy()
    v = q[:, 6].astype(np.float64) ** a.p - k * (a.depth - src)
    if a.extra_sigma_v > 0:
        v = v + rng.normal(0, a.extra_sigma_v, len(v))
    keep = v > 0
    q = q[keep]; q[:, 6] = v[keep] ** (1.0 / a.p); q[:, 2] = -(158.0 + a.depth)
    print(f"protoni sorgente: {m.sum()}, sopravvissuti (v>0): {keep.sum()} ({100 * keep.mean():.1f}%)")

    if a.angle_mode != "none":
        def width(x):
            q25, q75 = np.percentile(x, [25, 75]); return (q75 - q25) / 1.349
        W = np.array([[width(ps[np.isclose(dep, d), 3]), width(ps[np.isclose(dep, d), 4])] for d in ds])    # (n_depth, 2)
        w_t = np.exp([np.interp(a.depth, ds, np.log(W[:, c])) for c in range(2)])
        w_s = W[list(np.round(ds, 6)).index(round(src, 6))]
        print(f"larghezza core angoli: sorgente {w_s.round(4).tolist()} -> target (interp) {w_t.round(4).tolist()}  [modo: {a.angle_mode}]")
        if a.angle_mode == "scale":
            q[:, 3] *= w_t[0] / w_s[0]; q[:, 4] *= w_t[1] / w_s[1]
        else:
            sd = np.sqrt(np.maximum(w_t ** 2 - w_s ** 2, 0.0))
            q[:, 3] += rng.normal(0, sd[0], len(q)); q[:, 4] += rng.normal(0, sd[1], len(q))
        r2 = q[:, 3] ** 2 + q[:, 4] ** 2; bad = r2 >= 0.999999
        if bad.any():
            f = np.sqrt(0.999999 / r2[bad]); q[bad, 3] *= f; q[bad, 4] *= f; r2[bad] = 0.999999
        q[:, 5] = -np.sqrt(1.0 - r2)
    with h5py.File(a.out, "w") as h:
        h.create_dataset("phase_space", data=q.astype(np.float32), compression="gzip")
        h.create_dataset("conditions", data=np.full((len(q), 1), a.depth, np.float32))
    print(f"scritto {a.out}: {q.shape}")


if __name__ == "__main__":
    main()
