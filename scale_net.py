"""
scale_net.py - rete piccola per log sigma_dx, log sigma_dy in funzione di (E, d).
Target: sigma robusta (IQR/1.349) per bin quantilici di E, per ogni profondita'.
Uso standalone (verifica LOO):  python scale_net.py --data data/proton_depth_train.h5
"""
import argparse
import numpy as np
import torch
import torch.nn as nn


def rsd(a):
    q1, q3 = np.percentile(a, [25, 75])
    return (q3 - q1) / 1.349


class ScaleNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2))
        nn.init.constant_(self.net[-1].bias, -2.8)

    def forward(self, E, d):
        E = E.clamp(min=0.1)
        f = torch.stack([torch.log(E) / 4.0, d / 4.0, E / 70.0], dim=-1)
        return self.net(f)  # log sigma (dx, dy)


def collect_bins(ps7, d, min_n=3000):
    E, dx, dy = ps7[:, 6], ps7[:, 3], ps7[:, 4]
    qs = np.unique(np.concatenate([np.linspace(0, 0.1, 11), np.linspace(0.1, 1.0, 50)]))
    edges = np.quantile(E, qs)
    rows = []
    for j in range(len(edges) - 1):
        last = j == len(edges) - 2
        m = (E >= edges[j]) & ((E <= edges[j + 1]) if last else (E < edges[j + 1]))
        if m.sum() < min_n:
            continue
        rows.append((float(np.median(E[m])), float(d),
                     float(np.log(rsd(dx[m]))), float(np.log(rsd(dy[m]))), float(np.sqrt(m.sum()))))
    return rows


def fit_scale_net(rows, steps=3000, seed=0, device="cpu"):
    torch.manual_seed(seed)
    a = torch.tensor(rows, dtype=torch.float32, device=device)
    E, d, tgt, w = a[:, 0], a[:, 1], a[:, 2:4], a[:, 4]
    w = w / w.mean()
    net = ScaleNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.mean(w[:, None] * (net(E, d) - tgt) ** 2)
        loss.backward(); opt.step(); sch.step()
    return net.eval()


@torch.no_grad()
def sigma(net, E_np, d, device="cpu", chunk=1_000_000):
    """sigma (N,2) per energie E_np; d scalare o array."""
    net = net.to(device)
    out = []
    d_arr = np.broadcast_to(np.asarray(d, dtype=np.float32), E_np.shape)
    for i in range(0, len(E_np), chunk):
        E = torch.from_numpy(E_np[i:i + chunk].astype(np.float32)).to(device)
        dd = torch.from_numpy(np.ascontiguousarray(d_arr[i:i + chunk])).to(device)
        out.append(torch.exp(net(E, dd)).cpu().numpy())
    return np.concatenate(out, axis=0)


def loo_report(by_depth, holds, device="cpu"):
    print("LOO della rete sigma(E,d): errore medio |dlog sigma| (=errore relativo di scala)")
    for h in holds:
        rows = [r for d, p in by_depth.items() if d != h for r in collect_bins(p, d)]
        net = fit_scale_net(rows, device=device)
        te = np.array(collect_bins(by_depth[h], h))
        pred = np.log(sigma(net, te[:, 0], h, device))
        err = np.abs(pred - te[:, 2:4])
        print(f"  hold {h}: dx {100 * err[:, 0].mean():.1f}%  dy {100 * err[:, 1].mean():.1f}%  ({len(te)} bin)")


if __name__ == "__main__":
    import h5py
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    with h5py.File(a.data, "r") as h:
        ps7 = h["phase_space"][:]; depth = h["conditions"][:, 0]
    by = {float(d): ps7[depth == d] for d in np.unique(depth)}
    print("profondita':", sorted(by))
    loo_report(by, [h for h in (1.0, 2.0, 2.5, 3.0) if h in by])
