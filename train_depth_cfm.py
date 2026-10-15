"""
train_depth_cfm.py
==================
Addestra un Conditional Flow Matching p(y, dx, dy, logE | profondità) sui protoni.

    python train_depth_cfm.py --data data/proton_depth_train.h5 --out runs/cfm_depth_v1 \
        --epochs 200 --batch 8192 --hidden 512 --layers 6 --lr 5e-5

Input: HDF5 prodotto da build_depth_dataset.py (phase_space (N,7), conditions (N,1)).
Il modello è `PhaseSpaceCFM` del progetto fotoni (models/cfm.py), con dim=4 e cond_dim=1.
"""
import argparse, json, time
from pathlib import Path

import h5py
import numpy as np
import torch

from models.cfm import PhaseSpaceCFM, CFMTrainer
from phsp_transforms import (to_features, fit_depth_stats, standardize_for_training, norm_depth, save_stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--val_every", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=None, help="per smoke test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--norm", choices=["depth", "global"], default="depth",
                    help="standardizzazione per profondità (default) o globale (vecchio comportamento)")
    ap.add_argument("--stat", choices=["moments", "robust"], default="moments",
                    help="posizione/scala per la standardizzazione: media/std oppure mediana/IQR")
    ap.add_argument("--etrans", choices=["log", "lin", "sqrt", "Ep"], default="log",
                    help="trasformazione di E per il 4° canale")
    ap.add_argument("--p", type=float, default=1.77, help="esponente Bragg-Kleeman per --etrans Ep")
    ap.add_argument("--translate", action="store_true",
                    help="(con --etrans Ep) E: posizione = a - k*d, scala costante; niente interpolazione di tabella")
    ap.add_argument("--asinh", action="store_true", help="(con --translate) arcsinh sul canale E: comprime la coda fino a v=0")
    ap.add_argument("--fit_dmax", type=float, default=3.0, help="profondità max per fittare la retta di --translate")
    ap.add_argument("--interp", choices=["hybrid", "pchip_Ep", "pchip", "linear"], default="pchip_Ep",
                    help="interpolazione di mu(d), sigma(d) in generazione")
    a = ap.parse_args()

    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    with h5py.File(a.data, "r") as h:
        ps7 = h["phase_space"][:]; depth = h["conditions"][:, 0]
    if a.max_samples and len(ps7) > a.max_samples:
        idx = rng.choice(len(ps7), a.max_samples, replace=False); ps7, depth = ps7[idx], depth[idx]
    print(f"dati: {ps7.shape}, profondità: {np.unique(depth)}, device={dev}")

    feat = to_features(ps7, a.etrans, a.p)
    stats = fit_depth_stats(feat, depth, mode=a.norm, interp=a.interp, stat=a.stat, etrans=a.etrans,
                           p=a.p, translate=a.translate, fit_dmax=a.fit_dmax, asinh=a.asinh)
    save_stats(stats, out / "stats.json")
    X = standardize_for_training(feat, depth, stats)
    if stats.get("translate"): print("translate:", stats["translate"])
    print(f"standardizzazione: {a.norm}, stat: {a.stat}, etrans: {a.etrans}, interp: {a.interp}")
    C = norm_depth(depth, stats)[:, None]

    # split train/val: NOTA - se il phsp di training deriva da input riciclato le particelle non sono
    # indipendenti; la validazione vera è sulla profondità di hold-out (eval_depth.py).
    perm = rng.permutation(len(X)); n_val = int(len(X) * a.val_frac)
    va, tr = perm[:n_val], perm[n_val:]
    Xtr, Ctr = torch.from_numpy(X[tr]).to(dev), torch.from_numpy(C[tr]).to(dev)
    Xva, Cva = torch.from_numpy(X[va]).to(dev), torch.from_numpy(C[va]).to(dev)

    model = PhaseSpaceCFM(dim=X.shape[1], cond_dim=1, hidden_dim=a.hidden, n_layers=a.layers)
    tr_ = CFMTrainer(model, device=dev, lr=a.lr, epochs=a.epochs)
    print(f"parametri: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    best = 1e9; t0 = time.time()
    for ep in range(1, a.epochs + 1):
        p = torch.randperm(len(Xtr), device=dev); losses = []
        for i in range(0, len(p), a.batch):
            b = p[i:i + a.batch]
            losses.append(tr_.train_step(Xtr[b], Ctr[b]))
        tr_.scheduler.step()
        tr_.history["train_loss"].append(float(np.mean(losses)))
        msg = f"ep {ep:4d}  train {np.mean(losses):.4f}"
        if ep % a.val_every == 0:
            v = tr_.val_step(Xva, Cva); tr_.history["val_loss"].append(v); msg += f"  val {v:.4f}"
            if v < best:
                best = v; tr_.save(str(out / "best.pt"))
        print(msg + f"  [{time.time()-t0:.0f}s]", flush=True)
    tr_.save(str(out / "last.pt"))
    json.dump(dict(vars(a), best_val=best), open(out / "run.json", "w"), indent=2)


if __name__ == "__main__":
    main()
