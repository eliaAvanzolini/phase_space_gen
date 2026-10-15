"""
generate_depth_condE.py
=======================
Genera il phase space campionando (y, dx, dy) condizionato su (d, E).
Supporta sorgenti energetiche sia .h5 che .phsp (con filtro PDG 2212).
"""
import argparse, json
from pathlib import Path

import h5py
import numpy as np
import torch

from models.cfm import PhaseSpaceCFM
from phsp_transforms import load_stats, norm_depth, stats_at, e_fwd, Z0_CM
import read_topas_phsp as rtp


def load_condE_model(run_dir, ckpt=None, device=None):
    run = Path(run_dir)
    stats = load_stats(run / "stats.json")
    cfg = json.load(open(run / "run.json"))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PhaseSpaceCFM(
        dim=3,
        cond_dim=2,
        hidden_dim=cfg["hidden"],
        n_layers=cfg["layers"]
    ).to(dev)
    
    weights_path = run / (ckpt or "best.pt")
    if not weights_path.exists():
        weights_path = run / "last.pt"
    ckpt_data = torch.load(weights_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt_data["model"])
    model.eval()
    return model, stats, dev


def load_energies(energy_src):
    p = Path(energy_src)
    if p.suffix == ".h5":
        with h5py.File(p, "r") as h:
            key = "phase_space" if "phase_space" in h else list(h.keys())[0]
            E = h[key][:, 6].astype(np.float64)
    else:
        # File TOPAS .phsp grezzo: prendiamo solo i protoni primari (PDG 2212)
        df = rtp.read_phsp(str(p))
        if "pdg" in df.columns:
            df = df[df["pdg"] == 2212]
        ps7 = rtp.to_7d(df)
        E = ps7[:, 6].astype(np.float64)

    print(f"File {p.name}: caricate {len(E)} particelle (SOLO PROTONI PDG 2212) | Media E = {E.mean():.2f} MeV, std = {E.std():.2f} MeV")
    return E


@torch.no_grad()
def generate(run_dir, depth, energy_src, n=None, steps=200, chunk=500_000, ckpt=None, seed=0):
    model, stats, dev = load_condE_model(run_dir, ckpt)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    E_phys = load_energies(energy_src)
    if n is not None and n < len(E_phys):
        idx = rng.choice(len(E_phys), n, replace=False)
        E_phys = E_phys[idx]
    tot_n = len(E_phys)
    print(f"Campionamento di {tot_n} particelle condizionate su E...")

    mu, sd = stats_at(stats, float(depth))
    mu_spatial, sd_spatial = mu[:3], sd[:3]
    mu_E, sd_E = mu[3], sd[3]

    fE = e_fwd(E_phys, stats.get("etrans", "lin"), stats.get("p", 1.77))
    E_norm = ((fE - mu_E) / sd_E).astype(np.float32)
    d_norm = float(norm_depth(depth, stats))

    c_joint_all = np.column_stack([np.full(tot_n, d_norm, dtype=np.float32), E_norm])

    spatial_list = []
    for i in range(0, tot_n, chunk):
        end = min(i + chunk, tot_n)
        c_batch = torch.from_numpy(c_joint_all[i:end]).to(dev)
        z_batch = model.sample_fast(len(c_batch), c_batch, n_steps=steps)
        spatial_batch = z_batch.cpu().numpy() * sd_spatial + mu_spatial
        spatial_list.append(spatial_batch.astype(np.float32))

    spatial_all = np.concatenate(spatial_list, axis=0)
    y = spatial_all[:, 0]
    dx = spatial_all[:, 1]
    dy = spatial_all[:, 2]

    r2 = dx**2 + dy**2
    bad = r2 >= 0.999999
    if bad.any():
        s = np.sqrt(0.999999 / r2[bad])
        dx = dx.copy()
        dy = dy.copy()
        dx[bad] *= s
        dy[bad] *= s

    dz = -np.sqrt(1.0 - dx**2 - dy**2)
    x = rng.uniform(-stats["x_halfwidth"], stats["x_halfwidth"], tot_n).astype(np.float32)
    z = np.full(tot_n, -(stats.get("z0", Z0_CM) + depth), dtype=np.float32)

    ps7 = np.column_stack([x, y, z, dx, dy, dz, E_phys.astype(np.float32)])
    return ps7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--depth", type=float, required=True)
    ap.add_argument("--energy_src", required=True)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ps = generate(a.run, a.depth, a.energy_src, n=a.n, steps=a.steps, ckpt=a.ckpt, seed=a.seed)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.out, "w") as h:
        h.create_dataset("phase_space", data=ps, compression="gzip")
        h.create_dataset("conditions", data=np.full((len(ps), 1), a.depth, dtype=np.float32))
    print(f"Scritto con successo {a.out}: {ps.shape}")


if __name__ == "__main__":
    main()
