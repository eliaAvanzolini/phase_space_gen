"""
train_depth_cfm_condE.py
========================
CFM p(y, dx, dy | d, E). Target dim=3, condizione cond_dim=2 (d, E).
Opzione --flip: augmentation con simmetrie del fascio
  dx -> -dx (50%), (y, dy) -> (-y, -dy) (50%, indipendente).
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
    ap.add_argument("--norm", choices=["depth", "global"], default="depth")
    ap.add_argument("--stat", choices=["moments", "robust"], default="moments")
    ap.add_argument("--etrans", choices=["log", "lin", "sqrt", "Ep"], default="lin")
    ap.add_argument("--p", type=float, default=1.77)
    ap.add_argument("--translate", action="store_true")
    ap.add_argument("--asinh", action="store_true")
    ap.add_argument("--fit_dmax", type=float, default=3.0)
    ap.add_argument("--interp", choices=["hybrid", "pchip_Ep", "pchip", "linear"], default="hybrid")
    ap.add_argument("--flip", action="store_true", help="augmentation per simmetria dx e (y,dy)")
    a = ap.parse_args()

    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    with h5py.File(a.data, "r") as h:
        ps7 = h["phase_space"][:]; depth = h["conditions"][:, 0]
    if a.max_samples and len(ps7) > a.max_samples:
        idx = rng.choice(len(ps7), a.max_samples, replace=False); ps7, depth = ps7[idx], depth[idx]
    assert not np.any(np.isclose(depth, 3.5)), "3.5 cm presente nel training!"
    print(f"Dati: {ps7.shape}, profondità: {np.unique(depth)}, device={dev}")

    # Controllo delle correlazioni che giustificano il flip
    for d in np.unique(depth):
        m = depth == d
        sub = ps7[m][:200_000]
        c_ydx = np.corrcoef(sub[:, 1], sub[:, 3])[0, 1]
        c_dxdy = np.corrcoef(sub[:, 3], sub[:, 4])[0, 1]
        c_ydy = np.corrcoef(sub[:, 1], sub[:, 4])[0, 1]
        warn = "  <-- ATTENZIONE: flip non lecito" if (abs(c_ydx) > 0.01 or abs(c_dxdy) > 0.01) else ""
        print(f"  d={d:.2f}: corr(y,dx)={c_ydx:+.4f} corr(dx,dy)={c_dxdy:+.4f} corr(y,dy)={c_ydy:+.4f}{warn}")

    if a.flip:
        sx = rng.choice([-1.0, 1.0], size=len(ps7)).astype(ps7.dtype)
        sy = rng.choice([-1.0, 1.0], size=len(ps7)).astype(ps7.dtype)
        ps7 = ps7.copy()
        ps7[:, 3] *= sx            # dx
        ps7[:, 1] *= sy            # y
        ps7[:, 4] *= sy            # dy
        print("Flip di simmetria attivo (stats calcolate su dati simmetrizzati).")

    feat = to_features(ps7, a.etrans, a.p)
    stats = fit_depth_stats(feat, depth, mode=a.norm, interp=a.interp, stat=a.stat, etrans=a.etrans,
                            p=a.p, translate=a.translate, fit_dmax=a.fit_dmax, asinh=a.asinh)
    save_stats(stats, out / "stats.json")

    X_all = standardize_for_training(feat, depth, stats)
    X_spatial = X_all[:, :3]
    C_energy = X_all[:, 3:4]
    C_depth = norm_depth(depth, stats)[:, None]
    C_joint = np.concatenate([C_depth, C_energy], axis=1)
    print(f"Formato training: target X={X_spatial.shape} (y, dx, dy), condizione C={C_joint.shape} (d, E)")

    perm = rng.permutation(len(X_spatial)); n_val = int(len(X_spatial) * a.val_frac)
    va, tr = perm[:n_val], perm[n_val:]
    Xtr, Ctr = torch.from_numpy(X_spatial[tr]).to(dev), torch.from_numpy(C_joint[tr]).to(dev)
    Xva, Cva = torch.from_numpy(X_spatial[va]).to(dev), torch.from_numpy(C_joint[va]).to(dev)

    model = PhaseSpaceCFM(dim=3, cond_dim=2, hidden_dim=a.hidden, n_layers=a.layers)
    tr_ = CFMTrainer(model, device=dev, lr=a.lr, epochs=a.epochs)
    print(f"Parametri CFM cond-E: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    best = 1e9; t0 = time.time()
    for ep in range(1, a.epochs + 1):
        p = torch.randperm(len(Xtr), device=dev); losses = []
        for i in range(0, len(p), a.batch):
            b = p[i:i + a.batch]
            xb = Xtr[b]
            if a.flip:
                xb = xb.clone()
                sx = (torch.randint(0, 2, (len(b), 1), device=dev) * 2 - 1).to(xb.dtype)
                sy = (torch.randint(0, 2, (len(b), 1), device=dev) * 2 - 1).to(xb.dtype)
                xb[:, 1:2] *= sx                 # dx
                xb[:, 0:1] *= sy                 # y
                xb[:, 2:3] *= sy                 # dy
            losses.append(tr_.train_step(xb, Ctr[b]))
        tr_.scheduler.step()
        tr_.history["train_loss"].append(float(np.mean(losses)))
        msg = f"ep {ep:4d}  train {np.mean(losses):.4f}"
        if ep % a.val_every == 0:
            v = tr_.val_step(Xva, Cva); tr_.history["val_loss"].append(v); msg += f"  val {v:.4f}"
            if v < best:
                best = v; tr_.save(str(out / "best.pt"))
        print(msg + f"  [{time.time()-t0:.0f}s]", flush=True)

    tr_.save(str(out / "last.pt"))
    json.dump(dict(vars(a), best_val=best, cond_dim=2, dim=3), open(out / "run.json", "w"), indent=2)
    print("Addestramento cond-E terminato con successo.")


if __name__ == "__main__":
    main()
