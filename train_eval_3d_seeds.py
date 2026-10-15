"""
train_eval_3d_seeds.py
======================
Punto 1: Controllo rigoroso della variabilita' dei seed nel CFM 3D condizionato.
Addestra e valuta a 3.5 cm (Oracle):
  - no_flip  seed 43
  - flip     seed 43
  - no_flip  seed 44
  - flip     seed 44
"""
import argparse, json, time
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import wasserstein_distance
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.cfm import PhaseSpaceCFM
from phsp_transforms import load_stats, norm_depth, stats_at, e_fwd
import read_topas_phsp as rtp

REAL_35 = "data/raw_phsp/SurfaceFile_3.5_peak_1_2_5.phsp"
TRAIN_DEPTHS = [0.5, 1.0, 2.0, 2.5, 3.0, 4.0]


def load_raw_train_cache(data_dir):
    """Carica in memoria i protoni di training una sola volta per tutti i seed."""
    t0 = time.time()
    print("--> Caricamento dati grezzi di training in RAM...")
    data_by_depth = {}
    for p in sorted(Path(data_dir).glob("*.phsp")):
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
        data_by_depth[d_val] = ps7
        print(f"    d={d_val:4.2f} cm: {len(ps7):8d} protoni")
    print(f"--> Caricamento completato in {time.time() - t0:.1f}s")
    return data_by_depth


def build_tensors(data_by_depth, stats, use_flip=False, seed=42):
    rng = np.random.default_rng(seed)
    all_spatial, all_cond = [], []
    for d_val, ps7_orig in data_by_depth.items():
        ps7 = ps7_orig.copy()
        if use_flip:
            fx = rng.choice([-1.0, 1.0], size=len(ps7)).astype(np.float32)
            fy = rng.choice([-1.0, 1.0], size=len(ps7)).astype(np.float32)
            ps7[:, 3] *= fx
            ps7[:, 1] *= fy
            ps7[:, 4] *= fy

        mu, sd = stats_at(stats, d_val)
        y_norm = (ps7[:, 1] - mu[0]) / sd[0]
        dx_norm = (ps7[:, 3] - mu[1]) / sd[1]
        dy_norm = (ps7[:, 4] - mu[2]) / sd[2]
        sp = np.column_stack([y_norm, dx_norm, dy_norm]).astype(np.float32)

        fE = e_fwd(ps7[:, 6].astype(np.float64), stats.get("etrans", "lin"), stats.get("p", 1.77))
        E_norm = ((fE - mu[3]) / sd[3]).astype(np.float32)
        d_norm = float(norm_depth(d_val, stats))
        cond = np.column_stack([np.full(len(ps7), d_norm, dtype=np.float32), E_norm]).astype(np.float32)

        all_spatial.append(sp)
        all_cond.append(cond)

    X = torch.from_numpy(np.concatenate(all_spatial, axis=0)).float()
    C = torch.from_numpy(np.concatenate(all_cond, axis=0)).float()
    return TensorDataset(X, C)


def train_cfm(dataset, epochs, seed, device="cuda"):
    torch.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True, drop_last=True)
    model = PhaseSpaceCFM(dim=3, cond_dim=2, hidden_dim=128, n_layers=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    n_batches = len(loader)
    print(f"      Inizio training CFM (batch={4096}, step/epoca={n_batches})...")
    t0_train = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for x1, c in loader:
            x1, c = x1.to(device).float(), c.to(device).float()
            loss = model.cfm_loss(x1, c)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        sched.step()

        if ep % 10 == 0 or ep == 1:
            mean_l = ep_loss / n_batches
            elapsed = time.time() - t0_train
            print(f"      [ep {ep:3d}/{epochs}] loss = {mean_l:.4f} | tempo: {elapsed:5.1f}s")
    return model


@torch.no_grad()
def evaluate_on_35(model, stats, real_35, device="cuda", seed=42):
    model.eval()
    print("      Campionamento 3.5 cm (200 step ODE, Oracle E)...")
    t0_sample = time.time()
    E_phys = real_35[:, 6].astype(np.float64)
    n = len(E_phys)

    mu, sd = stats_at(stats, 3.5)
    fE = e_fwd(E_phys, stats.get("etrans", "lin"), stats.get("p", 1.77))
    E_norm = ((fE - mu[3]) / sd[3]).astype(np.float32)
    d_norm = float(norm_depth(3.5, stats))
    cond = torch.from_numpy(np.column_stack([np.full(n, d_norm, dtype=np.float32), E_norm])).float().to(device)

    chunk = 200_000
    parts = []
    for i in range(0, n, chunk):
        c_b = cond[i:i + chunk]
        z_b = model.sample_fast(len(c_b), c_b, n_steps=200)
        sp_b = z_b.cpu().numpy() * sd[:3] + mu[:3]
        parts.append(sp_b.astype(np.float32))
    gen_sp = np.concatenate(parts, axis=0)
    print(f"      Campionamento terminato in {time.time() - t0_sample:.1f}s")

    real_sp = real_35[:, [1, 3, 4]]
    res = {}
    for idx, nm in enumerate(["y", "dx", "dy"]):
        sr = real_sp[:, idx].std()
        w1 = wasserstein_distance(real_sp[:, idx], gen_sp[:, idx]) / sr
        dm = (gen_sp[:, idx].mean() - real_sp[:, idx].mean()) / sr
        ds = (gen_sp[:, idx].std() - sr) / sr
        res[f"w1_{nm}"] = w1
        res[f"dmean_{nm}"] = dm
        res[f"dstd_{nm}"] = ds

    corr_r = np.corrcoef(real_sp.T)
    corr_g = np.corrcoef(gen_sp.T)
    res["max_dcorr"] = float(np.max(np.abs(corr_g - corr_r)))
    res["corr_y_dy"] = float(corr_g[0, 2])
    res["corr_dx_dy"] = float(corr_g[1, 2])
    return res


def load_real_ps7(path):
    df = rtp.read_phsp(str(path))
    if "pdg" in df.columns:
        df = df[df["pdg"] == 2212]
    ps7 = rtp.to_7d(df).astype(np.float64)
    return ps7[ps7[:, 6] > 0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw_phsp")
    ap.add_argument("--epochs", type=int, default=120)
    a = ap.parse_args()

    assert torch.cuda.is_available(), "ERRORE CRITICO: GPU CUDA non disponibile!"
    dev = "cuda"
    print(f"=== DEVICE DISPONIBILE: {torch.cuda.get_device_name(0)} ===")

    stats = load_stats("runs/cfm_condE_s42/stats.json")
    real_35 = load_real_ps7(REAL_35)
    train_cache = load_raw_train_cache(a.data_dir)

    experiments = [
        ("no_flip_s43", False, 43),
        ("flip_s43",    True,  43),
        ("no_flip_s44", False, 44),
        ("flip_s44",    True,  44),
    ]

    print("\n=== VERIFICA RIGOROSA DEI SEED SUL CFM 3D (3.5 CM ORACLE) ===")
    results = {}
    for name, use_flip, s in experiments:
        t0 = time.time()
        print(f"\n--> Avvio: {name} (flip={use_flip}, seed={s}, epochs={a.epochs})")
        ds = build_tensors(train_cache, stats, use_flip=use_flip, seed=s)
        model = train_cfm(ds, a.epochs, seed=s, device=dev)
        m = evaluate_on_35(model, stats, real_35, device=dev, seed=s)
        results[name] = m
        print(f"    ==> RISULTATO [{name}] (tempo tot: {time.time()-t0:.1f}s):")
        print(f"        W1/sd:  dx={m['w1_dx']:.4f}  dy={m['w1_dy']:.4f}  y={m['w1_y']:.4f}")
        print(f"        dmean:  dx={m['dmean_dx']:+.4f}  dy={m['dmean_dy']:+.4f}")
        print(f"        corr:   y-dy={m['corr_y_dy']:.4f} (reale ~0.240)  dx-dy={m['corr_dx_dy']:.4f}  max|dcorr|={m['max_dcorr']:.4f}")

    print("\n" + "="*70)
    print("TABELLA RIASSUNTIVA FINALE: CONFRONTO SEED E FLIP")
    print("="*70)
    print(f"{'Configurazione':15s} | {'W1(dx)':>8s} {'W1(dy)':>8s} {'W1(y)':>8s} | "
          f"{'dmean(dx)':>10s} {'dmean(dy)':>10s} | {'corr(y,dy)':>10s} {'max_dcorr':>10s}")
    for k, v in results.items():
        print(f"{k:15s} | {v['w1_dx']:8.4f} {v['w1_dy']:8.4f} {v['w1_y']:8.4f} | "
              f"{v['dmean_dx']:+10.4f} {v['dmean_dy']:+10.4f} | {v['corr_y_dy']:10.4f} {v['max_dcorr']:10.4f}")

    Path("eval/seed_check").mkdir(parents=True, exist_ok=True)
    with open("eval/seed_check/results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
