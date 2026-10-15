"""
diagnose_and_improve_3d.py
==========================
Esegue i punti di Claude sulla parte 3D:
  1. Diagnosi a 3.0 cm e 4.0 cm (training) vs 3.5 cm (hold-out) per bande di energia e Z latente.
  2. Test a 500 step ODE + correzione dell'offset tramite simmetria fisica esatta:
     dx -> -dx e (y, dy) -> (-y, -dy).
  3. Rete neurale di scala condizionale locale sigma_psi(E, d) addestrata sulle 6 profondita'
     di training (escluso 3.5 cm) per eliminare l'appiattimento di sigma(theta | E).
"""
from pathlib import Path
import h5py
import numpy as np
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn
import torch.nn.functional as F

import read_topas_phsp as rtp
from phsp_transforms import stats_at, e_fwd, Z0_CM
from generate_depth_condE import generate, load_condE_model

TRAIN_DEPTHS = [0.5, 1.0, 2.0, 2.5, 3.0, 4.0]


def load_real_ps7(phsp_path):
    df = rtp.read_phsp(str(phsp_path))
    if "pdg" in df.columns:
        df = df[df["pdg"] == 2212]
    ps7 = rtp.to_7d(df).astype(np.float64)
    return ps7[ps7[:, 6] > 0.0]


def print_depth_diagnostics(real_ps7, gen_ps7, stats, depth, label):
    real = real_ps7[:, [1, 3, 4, 6]]  # y, dx, dy, E
    gen = gen_ps7[:, [1, 3, 4, 6]]
    mu_interp, sd_interp = stats_at(stats, float(depth))

    corr_dx_dy_real = np.corrcoef(real[:, 1], real[:, 2])[0, 1]
    corr_y_dy_real = np.corrcoef(real[:, 0], real[:, 2])[0, 1]
    print(f"\n==================== {label} (d = {depth:.2f} cm) ====================")
    print(f"  Correlazioni reali: corr(dx, dy) = {corr_dx_dy_real:+.4f} | corr(y, dy) = {corr_y_dy_real:+.4f}")

    for i, nm in enumerate(["y", "dx", "dy"]):
        z_gen = (gen[:, i] - mu_interp[i]) / sd_interp[i]
        z_real = (real[:, i] - mu_interp[i]) / sd_interp[i]
        w1 = wasserstein_distance(real[:, i], gen[:, i]) / real[:, i].std()
        dm = (gen[:, i].mean() - real[:, i].mean()) / real[:, i].std()
        ds = (gen[:, i].std() - real[:, i].std()) / real[:, i].std()
        print(f"  {nm:2s} | W1/sd={w1:.4f} | dmean/sd={dm:+.4f} | dstd_rel={100*ds:+6.2f}% || "
              f"Z_mean: gen={z_gen.mean():+.4f} (real={z_real.mean():+.4f}) | "
              f"Z_std: gen={z_gen.std():.4f} (real={z_real.std():.4f})")

    # Fasce di energia (adattate per 4.0 cm dove E_mean = 4.27 MeV)
    if depth >= 3.9:
        bands = [(0.0, 2.5, "Bassa (<2.5 MeV)"),
                 (2.5, 5.0, "Media (2.5-5 MeV)"),
                 (5.0, 20.0, "Alta (>=5 MeV)")]
    else:
        bands = [(0.0, 10.0, "Coda bassa (<10 MeV)"),
                 (10.0, 18.0, "Valle (10-18 MeV)"),
                 (18.0, 80.0, "Picco (>=18 MeV)")]

    print("  --- Errore angolare per fasce di energia ---")
    for e_lo, e_hi, bname in bands:
        mr = (real[:, 3] >= e_lo) & (real[:, 3] < e_hi)
        mg = (gen[:, 3] >= e_lo) & (gen[:, 3] < e_hi)
        if mr.sum() < 200 or mg.sum() < 200:
            continue
        w_dx = wasserstein_distance(real[mr, 1], gen[mg, 1]) / real[mr, 1].std()
        w_dy = wasserstein_distance(real[mr, 2], gen[mg, 2]) / real[mr, 2].std()
        ds_dx = (gen[mg, 1].std() - real[mr, 1].std()) / real[mr, 1].std()
        ds_dy = (gen[mg, 2].std() - real[mr, 2].std()) / real[mr, 2].std()
        print(f"  {bname:22s} (quota {100*mr.mean():4.1f}%) | "
              f"dx: W1/sd={w_dx:.4f} (dstd={100*ds_dx:+6.2f}%) | "
              f"dy: W1/sd={w_dy:.4f} (dstd={100*ds_dy:+6.2f}%)")


def apply_symmetry_flip(ps7, seed=42):
    """
    Punto 2 di Claude: sfrutta la simmetria cilindrica/rettangolare del fascio.
    Inverte casualmente al 50% dx -> -dx e indipendentemente al 50% (y, dy) -> (-y, -dy).
    Garantisce media esattamente zero preservando corr(y, dy) e tutte le correlazioni con E.
    """
    rng = np.random.default_rng(seed)
    out = ps7.copy()
    n = len(out)
    flip_x = rng.choice([-1.0, 1.0], size=n).astype(out.dtype)
    flip_y = rng.choice([-1.0, 1.0], size=n).astype(out.dtype)
    out[:, 3] *= flip_x          # dx
    out[:, 1] *= flip_y          # y
    out[:, 4] *= flip_y          # dy
    return out


class LocalScaleNet(nn.Module):
    """
    Punto 3 di Claude: piccola rete che impara la log-deviazione standard locale
    log sigma(E, d) per (y, dx, dy) sulle sole profondita' di training.
    Input: [d/4, log(E)/4, (E/70)^1.76] + feature di Fourier.
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2)  # log_sigma per dx e dy
        )

    def _features(self, E, d):
        d_n = d / 4.0
        log_e = torch.log(E.clamp(min=0.2)) / 4.0
        v_n = (E.clamp(min=0.2) / 70.0) ** 1.76
        return torch.stack([
            d_n, log_e, v_n,
            torch.sin(np.pi * d_n), torch.cos(np.pi * d_n),
            torch.sin(np.pi * log_e), torch.cos(np.pi * log_e)
        ], dim=-1)

    def forward(self, E, d):
        return torch.exp(self.net(self._features(E, d)))


def train_local_scale_net(data_dir, device="cpu"):
    """
    Calcola le deviazioni standard empiriche in bin di energia alle 6 profondita' di training
    (escludendo 3.5 cm) e addestra LocalScaleNet per predire sigma_dx(E, d) e sigma_dy(E, d).
    """
    rows_E, rows_d, rows_sd, rows_w = [], [], [], []
    for p in sorted(Path(data_dir).glob("*.phsp")):
        if "peak" not in p.name.lower():
            continue
        d = rtp.parse_depth_from_name(p.name)
        if d is None or round(float(d), 3) not in TRAIN_DEPTHS:
            continue
        ps7 = load_real_ps7(p)
        E = ps7[:, 6]
        dx, dy = ps7[:, 3], ps7[:, 4]
        # 64 bin quantilici per profondita' + bin dedicati alla coda bassa
        qs = np.unique(np.concatenate([np.linspace(0.0, 0.10, 16), np.linspace(0.10, 1.0, 50)]))
        edges = np.quantile(E, qs)
        for j in range(len(edges) - 1):
            m = (E >= edges[j]) & (E <= edges[j + 1])
            if m.sum() < 200:
                continue
            rows_E.append(float(np.median(E[m])))
            rows_d.append(float(d))
            rows_sd.append([float(dx[m].std()), float(dy[m].std())])
            rows_w.append(float(np.sqrt(m.sum())))

    E_t = torch.tensor(rows_E, dtype=torch.float32, device=device)
    d_t = torch.tensor(rows_d, dtype=torch.float32, device=device)
    sd_t = torch.tensor(rows_sd, dtype=torch.float32, device=device)
    w_t = torch.tensor(rows_w, dtype=torch.float32, device=device)
    w_t = w_t / w_t.mean()

    model = LocalScaleNet(hidden=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1500, eta_min=1e-4)

    for ep in range(1, 1501):
        opt.zero_grad()
        pred = model(E_t, d_t)
        loss = torch.mean(w_t[:, None] * (torch.log(pred) - torch.log(sd_t)) ** 2)
        loss.backward()
        opt.step()
        sched.step()

    return model


@torch.no_grad()
def apply_local_scale_correction(ps7_gen, scale_net, depth=3.5, device="cpu"):
    """
    Ricalibra la scala locale sigma(E, d) degli angoli generati dal CFM 3D:
    Per ogni bin di energia, sostituisce la dispersione piatta del CFM con quella appresa da LocalScaleNet.
    """
    out = ps7_gen.copy()
    E = out[:, 6].astype(np.float32)
    dx = out[:, 3].astype(np.float64)
    dy = out[:, 4].astype(np.float64)

    E_t = torch.from_numpy(E).to(device)
    d_t = torch.full_like(E_t, float(depth))
    target_sd = scale_net(E_t, d_t).cpu().numpy()  # (N, 2)

    # Stima la std locale generata dal CFM in finestre quantiliche lisce di E
    order = np.argsort(E)
    n = len(E)
    win = max(2000, n // 50)
    gen_sd_dx = np.empty(n, dtype=np.float64)
    gen_sd_dy = np.empty(n, dtype=np.float64)

    dx_ord = dx[order]
    dy_ord = dy[order]
    # Calcolo rapido a blocchi sovrapposti con interpolazione
    centers = np.linspace(0, n - 1, 60, dtype=int)
    c_sd_dx, c_sd_dy = [], []
    for c in centers:
        lo = max(0, c - win // 2)
        hi = min(n, c + win // 2)
        c_sd_dx.append(dx_ord[lo:hi].std())
        c_sd_dy.append(dy_ord[lo:hi].std())

    gen_sd_dx[order] = np.interp(np.arange(n), centers, c_sd_dx)
    gen_sd_dy[order] = np.interp(np.arange(n), centers, c_sd_dy)

    dx_new = dx * (target_sd[:, 0] / (gen_sd_dx + 1e-8))
    dy_new = dy * (target_sd[:, 1] / (gen_sd_dy + 1e-8))

    r2 = dx_new ** 2 + dy_new ** 2
    bad = r2 >= 0.999999
    if bad.any():
        s = np.sqrt(0.999999 / r2[bad])
        dx_new[bad] *= s
        dy_new[bad] *= s
    dz_new = -np.sqrt(1.0 - dx_new ** 2 - dy_new ** 2)

    out[:, 3] = dx_new.astype(np.float32)
    out[:, 4] = dy_new.astype(np.float32)
    out[:, 5] = dz_new.astype(np.float32)
    return out


def save_h5(path, ps7, depth):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.create_dataset("phase_space", data=ps7.astype(np.float32), compression="gzip")
        h.create_dataset("conditions", data=np.full((len(ps7), 1), depth, dtype=np.float32))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = "runs/cfm_condE_s42"
    _, stats, _ = load_condE_model(run_dir)

    print("=== PUNTO 1 DI CLAUDE: VERIFICA CFM 3D A 3.0 CM E 4.0 CM (TRAINING) VS 3.5 CM ===")
    for d, fname in [(3.0, "SurfaceFile_3_peak_1_2_5.phsp"),
                     (3.5, "SurfaceFile_3.5_peak_1_2_5.phsp"),
                     (4.0, "SurfaceFile_4_peak_1_2_5.phsp")]:
        phsp_path = f"data/raw_phsp/{fname}"
        real_ps7 = load_real_ps7(phsp_path)
        gen_ps7 = generate(run_dir, d, phsp_path, n=500_000, steps=200, seed=42)
        print_depth_diagnostics(real_ps7, gen_ps7, stats, d, f"CFM 3D (200 step) con E reale @ {d} cm")

    print("\n=== PUNTO 2 DI CLAUDE: TEST 500 STEP ODE E SIMMETRIA A 3.5 CM ===")
    real_35 = load_real_ps7("data/raw_phsp/SurfaceFile_3.5_peak_1_2_5.phsp")
    gen_35_500 = generate(run_dir, 3.5, "data/raw_phsp/SurfaceFile_3.5_peak_1_2_5.phsp", n=500_000, steps=500, seed=42)
    print_depth_diagnostics(real_35, gen_35_500, stats, 3.5, "CFM 3D (500 step ODE) @ 3.5 cm")

    with h5py.File("gen/gen_3.5_neural_resq_condE.h5", "r") as h:
        resq_ps7 = h["phase_space"][:]

    resq_sym = apply_symmetry_flip(resq_ps7, seed=42)
    print_depth_diagnostics(real_35, resq_sym, stats, 3.5, "ResidualQ + CFM 3D + Flip Simmetria @ 3.5 cm")
    save_h5("gen/gen_3.5_resq_condE_sym.h5", resq_sym, 3.5)

    print("\n=== PUNTO 3 DI CLAUDE: RETE DI SCALA CONDIZIONALE LOCALE sigma(E, d) ===")
    scale_net = train_local_scale_net("data/raw_phsp", device=dev)
    resq_sym_local = apply_local_scale_correction(resq_sym, scale_net, depth=3.5, device=dev)
    print_depth_diagnostics(real_35, resq_sym_local, stats, 3.5,
                            "ResidualQ + CFM 3D + Simmetria + Scala Locale sigma(E,d) @ 3.5 cm")
    save_h5("gen/gen_3.5_resq_condE_sym_local.h5", resq_sym_local, 3.5)


if __name__ == "__main__":
    main()
