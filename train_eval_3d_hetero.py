"""
train_eval_3d_hetero.py
=======================
Punto 2: Standardizzazione eteroschedastica e disaccoppiamento y-dy nel CFM 3D.
  1. Allena una rete leggera sigma_scale per predire log sigma_dx(E, d) e log sigma_dy(E, d) e beta(E, d).
  2. Allena il CFM 3D sulle feature standardizzate ad energia costante:
       dx_norm = dx / sigma_dx(E, d)
       dy_res  = (dy - beta(E, d) * y) / sigma_dy_res(E, d)
  3. Valuta le fasce di energia su test 3.5 cm (Oracle).
"""
import argparse, time
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.cfm import PhaseSpaceCFM
from phsp_transforms import load_stats, norm_depth, stats_at, e_fwd
import read_topas_phsp as rtp

TRAIN_DEPTHS = [0.5, 1.0, 2.0, 2.5, 3.0, 4.0]
REAL_35 = "data/raw_phsp/SurfaceFile_3.5_peak_1_2_5.phsp"


class AngleScaleNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3)
        )

    def forward(self, E, d):
        log_e = torch.log(E.clamp(min=0.3)) / 4.0
        d_n = d / 4.0
        inv_e = (10.0 / E.clamp(min=0.3)).clamp(max=20.0) / 10.0
        inp = torch.stack([d_n, log_e, inv_e], dim=-1)
        out = self.net(inp)
        sig_dx = torch.exp(out[..., 0])
        sig_dy_res = torch.exp(out[..., 1])
        beta = out[..., 2]
        return sig_dx, sig_dy_res, beta


def fit_angle_scale_net(data_dir, device="cuda"):
    rows_d, rows_E, rows_y, rows_dx, rows_dy = [], [], [], [], []
    for p in sorted(Path(data_dir).glob("*.phsp")):
        if "peak" not in p.name.lower():
            continue
        d = rtp.parse_depth_from_name(p.name)
        if d is None or round(float(d), 3) not in TRAIN_DEPTHS:
            continue
        df = rtp.read_phsp(str(p))
        if "pdg" in df.columns:
            df = df[df["pdg"] == 2212]
        ps7 = rtp.to_7d(df).astype(np.float32)
        ps7 = ps7[ps7[:, 6] > 0.0]
        rows_d.append(np.full(len(ps7), float(d), dtype=np.float32))
        rows_E.append(ps7[:, 6])
        rows_y.append(ps7[:, 1])
        rows_dx.append(ps7[:, 3])
        rows_dy.append(ps7[:, 4])

    D = torch.from_numpy(np.concatenate(rows_d)).float().to(device)
    E = torch.from_numpy(np.concatenate(rows_E)).float().to(device)
    Y = torch.from_numpy(np.concatenate(rows_y)).float().to(device)
    DX = torch.from_numpy(np.concatenate(rows_dx)).float().to(device)
    DY = torch.from_numpy(np.concatenate(rows_dy)).float().to(device)

    model = AngleScaleNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(D, E, Y, DX, DY), batch_size=8192, shuffle=True)

    for ep in range(1, 41):
        for d_b, e_b, y_b, dx_b, dy_b in loader:
            sig_dx, sig_dy_res, beta = model(e_b, d_b)
            dy_res = dy_b - beta * y_b
            loss_dx = torch.log(sig_dx) + 0.5 * (dx_b / sig_dx) ** 2
            loss_dy = torch.log(sig_dy_res) + 0.5 * (dy_res / sig_dy_res) ** 2
            loss = torch.mean(loss_dx + loss_dy)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw_phsp")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = load_stats("runs/cfm_condE_s42/stats.json")

    print("=== 1. FIT RETE DI SCALA ETEROSCHEDASTICA ANGOLARE sigma(E, d) ===")
    scale_net = fit_angle_scale_net(a.data_dir, device=dev)
    scale_net.eval()

    print("=== 2. PREPARAZIONE DATASET STANDARDIZZATO ===")
    rng = np.random.default_rng(a.seed)
    all_sp, all_cond = [], []
    for p in sorted(Path(a.data_dir).glob("*.phsp")):
        if "peak" not in p.name.lower():
            continue
        d = rtp.parse_depth_from_name(p.name)
        if d is None or round(float(d), 3) not in TRAIN_DEPTHS:
            continue
        d_val = round(float(d), 3)
        df = rtp.read_phsp(str(p))
        if "pdg" in df.columns:
            df = df[df["pdg"] == 2212]
        ps7 = rtp.to_7d(df).astype(np.float32)
        ps7 = ps7[ps7[:, 6] > 0.0]

        fx = rng.choice([-1.0, 1.0], size=len(ps7)).astype(np.float32)
        fy = rng.choice([-1.0, 1.0], size=len(ps7)).astype(np.float32)
        ps7[:, 3] *= fx
        ps7[:, 1] *= fy
        ps7[:, 4] *= fy

        with torch.no_grad():
            e_t = torch.from_numpy(ps7[:, 6]).float().to(dev)
            d_t = torch.full_like(e_t, float(d_val))
            sig_dx, sig_dy_res, beta = scale_net(e_t, d_t)
            sig_dx = sig_dx.cpu().numpy()
            sig_dy_res = sig_dy_res.cpu().numpy()
            beta = beta.cpu().numpy()

        mu, sd = stats_at(stats, d_val)
        y_norm = (ps7[:, 1] - mu[0]) / sd[0]
        dx_norm = ps7[:, 3] / (sig_dx + 1e-7)
        dy_res_norm = (ps7[:, 4] - beta * ps7[:, 1]) / (sig_dy_res + 1e-7)
        sp = np.column_stack([y_norm, dx_norm, dy_res_norm]).astype(np.float32)

        fE = e_fwd(ps7[:, 6].astype(np.float64), stats.get("etrans", "lin"), stats.get("p", 1.77))
        E_norm = ((fE - mu[3]) / sd[3]).astype(np.float32)
        d_norm = float(norm_depth(d_val, stats))
        cond = np.column_stack([np.full(len(ps7), d_norm, dtype=np.float32), E_norm]).astype(np.float32)

        all_sp.append(sp)
        all_cond.append(cond)

    X = torch.from_numpy(np.concatenate(all_sp, axis=0)).float()
    C = torch.from_numpy(np.concatenate(all_cond, axis=0)).float()
    loader = DataLoader(TensorDataset(X, C), batch_size=4096, shuffle=True, drop_last=True)

    print("=== 3. TRAINING CFM 3D ETEROSCHEDASTICO ===")
    torch.manual_seed(a.seed)
    model = PhaseSpaceCFM(dim=3, cond_dim=2, hidden_dim=128, n_layers=4).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)

    for ep in range(1, a.epochs + 1):
        model.train()
        for x1, c in loader:
            x1, c = x1.to(dev).float(), c.to(dev).float()
            loss = model.cfm_loss(x1, c)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    print("\n=== 4. VALUTAZIONE SU TEST HOLD-OUT 3.5 CM (ORACLE) ===")
    model.eval()
    real_df = rtp.read_phsp(REAL_35)
    if "pdg" in real_df.columns:
        real_df = real_df[real_df["pdg"] == 2212]
    real_ps7 = rtp.to_7d(real_df).astype(np.float64)
    real_ps7 = real_ps7[real_ps7[:, 6] > 0.0]

    E_real = real_ps7[:, 6].astype(np.float64)
    n = len(E_real)
    mu, sd = stats_at(stats, 3.5)
    fE = e_fwd(E_real, stats.get("etrans", "lin"), stats.get("p", 1.77))
    E_norm = ((fE - mu[3]) / sd[3]).astype(np.float32)
    d_norm = float(norm_depth(3.5, stats))
    cond = torch.from_numpy(np.column_stack([np.full(n, d_norm, dtype=np.float32), E_norm])).float().to(dev)

    parts = []
    chunk = 200_000
    for i in range(0, n, chunk):
        c_b = cond[i:i + chunk]
        z_b = model.sample_fast(len(c_b), c_b, n_steps=200)
        parts.append(z_b.cpu().numpy().astype(np.float32))
    z_all = np.concatenate(parts, axis=0)

    with torch.no_grad():
        e_t = torch.from_numpy(E_real).float().to(dev)
        d_t = torch.full_like(e_t, 3.5)
        sig_dx, sig_dy_res, beta = scale_net(e_t, d_t)
        sig_dx = sig_dx.cpu().numpy()
        sig_dy_res = sig_dy_res.cpu().numpy()
        beta = beta.cpu().numpy()

    y_gen = z_all[:, 0] * sd[0] + mu[0]
    dx_gen = z_all[:, 1] * sig_dx
    dy_gen = z_all[:, 2] * sig_dy_res + beta * y_gen

    fx = rng.choice([-1.0, 1.0], size=n)
    fy = rng.choice([-1.0, 1.0], size=n)
    dx_gen *= fx
    y_gen *= fy
    dy_gen *= fy

    names = ["y", "dx", "dy"]
    g_arr = [y_gen, dx_gen, dy_gen]
    r_arr = [real_ps7[:, 1], real_ps7[:, 3], real_ps7[:, 4]]

    print("\n--- METRICHE GLOBALI (3.5 CM) ---")
    for nm, g, r in zip(names, g_arr, r_arr):
        sr = r.std()
        w1 = wasserstein_distance(r, g) / sr
        dm = (g.mean() - r.mean()) / sr
        ds = (g.std() - sr) / sr
        print(f"  {nm:2s} | W1/sd={w1:.4f} | dmean/sd={dm:+.4f} | dstd_rel={100*ds:+6.2f}%")

    corr_r = np.corrcoef(np.column_stack(r_arr).T)
    corr_g = np.corrcoef(np.column_stack(g_arr).T)
    print(f"  corr(y, dy): reale={corr_r[0, 2]:.4f} | gen={corr_g[0, 2]:.4f}")
    print(f"  corr(dx, dy): reale={corr_r[1, 2]:.4f} | gen={corr_g[1, 2]:.4f}")

    print("\n--- ERRORE ANGOLARE PER FASCE DI ENERGIA (Verifica appiattimento) ---")
    bands = [(0.0, 10.0, "Coda (<10 MeV)"),
             (10.0, 18.0, "Valle (10-18 MeV)"),
             (18.0, 80.0, "Picco (>=18 MeV)")]
    for e_lo, e_hi, bname in bands:
        m = (E_real >= e_lo) & (E_real < e_hi)
        w_dx = wasserstein_distance(r_arr[1][m], g_arr[1][m]) / r_arr[1][m].std()
        w_dy = wasserstein_distance(r_arr[2][m], g_arr[2][m]) / r_arr[2][m].std()
        ds_dx = (g_arr[1][m].std() - r_arr[1][m].std()) / r_arr[1][m].std()
        ds_dy = (g_arr[2][m].std() - r_arr[2][m].std()) / r_arr[2][m].std()
        print(f"  {bname:18s} | dx: W1/sd={w_dx:.4f} (dstd={100*ds_dx:+6.2f}%) | "
              f"dy: W1/sd={w_dy:.4f} (dstd={100*ds_dy:+6.2f}%)")


if __name__ == "__main__":
    main()
