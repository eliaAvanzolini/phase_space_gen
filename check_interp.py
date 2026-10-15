"""
check_interp.py  (numpy/scipy, niente torch)
============================================
Leave-one-out sulle statistiche per profondità: per ogni profondità interna, predice media e std di
(y, dx, dy, logE) usando SOLO le altre e confronta con il valore vero. Dice, PRIMA di addestrare,
quanto è buona l'interpolazione di mu(d), sigma(d) e quale metodo scegliere.

    python check_interp.py --data data/proton_depth_train.h5 data/proton_depth_train_holdout.h5

Errore media in unità della std vera; errore std relativo. Sotto ~0.05 va bene; a 3.5 ci aspettiamo il peggio.
"""
import argparse
import h5py
import numpy as np
from phsp_transforms import CHANNELS, fit_depth_stats, interp_from_table, to_features


def check_translate(PS, Dc, fit_dmax, p=1.77):
    """LOO del modello 'traslazione' per il canale E^p: posizione = a - k d, scala costante.
    Errori di posizione in unità della scala (IQR) vera di quella profondità, errore relativo della scala."""
    v = to_features(PS, "Ep", p)[:, -1]
    ds = np.unique(np.round(Dc, 6))
    med = np.array([np.median(v[np.isclose(Dc, d)]) for d in ds])
    sc = np.array([np.subtract(*np.percentile(v[np.isclose(Dc, d)], [75, 25])) / 1.349 for d in ds])
    print(f"\n########  TRASLAZIONE in E^p (p={p}), retta fittata su d <= {fit_dmax}")
    print("  d      loc_vera  loc_pred  err_loc/scala   scala_vera  scala_pred  err_scala")
    for i in range(1, len(ds) - 1):
        keep = (np.arange(len(ds)) != i) & (ds <= fit_dmax + 1e-9)
        slope, a0 = np.polyfit(ds[keep], med[keep], 1); s = sc[keep].mean()
        pred = a0 + slope * ds[i]
        print(f"  {ds[i]:4.2f}  {med[i]:9.1f} {pred:9.1f}   {(pred - med[i]) / sc[i]:+10.3f}   {sc[i]:10.1f} {s:10.1f}   {s / sc[i] - 1:+9.3f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["pchip_Ep", "hybrid", "linear"])
    ap.add_argument("--stats", nargs="+", default=["moments"])
    ap.add_argument("--fit_dmax", type=float, default=3.0)
    ap.add_argument("--etrans", nargs="+", default=["log", "lin", "sqrt", "Ep"], help="trasformazioni di E da confrontare")
    a = ap.parse_args()
    F, D = [], []
    for f in a.data:
        with h5py.File(f, "r") as h:
            F.append(h["phase_space"][:]); D.append(h["conditions"][:, 0])
    PS, Dc = np.concatenate(F), np.concatenate(D)
    for et in a.etrans:
        Fc = to_features(PS, et)
        if et == "Ep":
            check_translate(PS, Dc, a.fit_dmax)
        for stat in a.stats:
            st = fit_depth_stats(Fc, Dc, stat=stat, etrans=et)
            ds, M, S = np.array(st["depths"]), np.array(st["mean_k"]), np.array(st["std_k"])
            print(f"\n########  etrans = {et}   stat = {stat}   profondità: {ds.tolist()}")
            for meth in a.methods:
                print(f"== {meth} ==   errore loc/scala_vera | errore scala relativo      (y, dx, dy, f(E))")
                for k in range(1, len(ds) - 1):
                    keep = np.arange(len(ds)) != k
                    mu, sd = interp_from_table(ds[keep], M[keep], S[keep], ds[k], interp=meth, etrans=et)
                    em = (mu - M[k]) / S[k]; es = sd / S[k] - 1
                    print(f"  d={ds[k]:4.2f}  loc " + " ".join(f"{v:+7.3f}" for v in em) + "  | scala " + " ".join(f"{v:+7.3f}" for v in es))


if __name__ == "__main__":
    main()
