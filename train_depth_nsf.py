"""
train_depth_nsf.py - Neural Spline Flow condizionato sulla profondità (modello models/nsf.py del progetto fotoni).

    python -u train_depth_nsf.py --data data/proton_depth_train.h5 --out runs/nsf_R2 --epochs 100

Stessa preparazione dati del run R2 del CFM (E lineare, standardizzazione per profondità, interpolazione hybrid).
Iperparametri di default = config nsf_conditional_multi_energy.yaml dei fotoni (12 coupling, 16 bin, hidden 256),
tranne lr (1e-4 invece di 3e-5) ed epoche (100). Richiede nflows.
Nota: con tails="linear" i valori standardizzati oltre ±tail_bound (code angolari molto estreme) sono trasformati con l'identità.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from depth_data import add_data_args, load_standardized
from depth_models import build
from phsp_transforms import save_stats


def main():
    ap = argparse.ArgumentParser(); add_data_args(ap)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--val_every", type=int, default=1)
    ap.add_argument("--n_transforms", type=int, default=12); ap.add_argument("--n_bins", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--tail_bound", type=float, default=8.0)
    ap.add_argument("--context_dim", type=int, default=32)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    X, C, stats = load_standardized(a, rng); save_stats(stats, out / "stats.json")
    perm = rng.permutation(len(X)); n_val = int(len(X) * a.val_frac); va, tr = perm[:n_val], perm[n_val:]
    Xtr, Ctr = torch.from_numpy(X[tr]).to(dev), torch.from_numpy(C[tr]).to(dev)
    Xva, Cva = torch.from_numpy(X[va]).to(dev), torch.from_numpy(C[va]).to(dev)

    cfg = dict(n_transforms=a.n_transforms, n_bins=a.n_bins, hidden=a.hidden, tail_bound=a.tail_bound, context_dim=a.context_dim)
    model = build("nsf", cfg, X.shape[1], dev)
    from models.nsf import NSFTrainer
    tr_ = NSFTrainer(model, device=dev, lr=a.lr)
    print(f"parametri: {sum(p.numel() for p in model.parameters())/1e6:.2f} M, device={dev}")

    best = 1e9; t0 = time.time()
    for ep in range(1, a.epochs + 1):
        p = torch.randperm(len(Xtr), device=dev); losses = []
        for i in range(0, len(p), a.batch):
            b = p[i:i + a.batch]; losses.append(tr_.train_step(Xtr[b], Ctr[b]))
        msg = f"ep {ep:4d}  train NLL {np.mean(losses):.4f}"
        if ep % a.val_every == 0:
            v = tr_.val_step(Xva, Cva); msg += f"  val NLL {v:.4f}  lr {tr_.opt.param_groups[0]['lr']:.1e}"
            if v < best: best = v; tr_.save(str(out / "best.pt"))
        print(msg + f"  [{time.time()-t0:.0f}s]", flush=True)
    tr_.save(str(out / "last.pt"))
    json.dump(dict(vars(a), model="nsf", best_val=best, **{k: cfg[k] for k in cfg}), open(out / "run.json", "w"), indent=2)


if __name__ == "__main__":
    main()
