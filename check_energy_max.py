import json
from pathlib import Path
import h5py
import numpy as np
import torch

# Path dei dataset e checkpoint
TRAIN_H5_PATH = "data/energy_only_train_dataset.h5"
REFERENCE_10MV_PATH = "data/energy_only_10mv_reference.h5"
CFM_CKPT = "outputs/cfm_energy_only/best_model.pt"
CFM_STATS = "outputs/cfm_energy_only/normalization_stats.json"
CFM_COND_STATS = "outputs/cfm_energy_only/condition_stats.json"

SAFETY_MARGIN = 10.2  # MeV


def load_reference(n_samples=500_000):
    with h5py.File(REFERENCE_10MV_PATH, "r") as f:
        ps = f["phase_space"][:]
        if len(ps) > n_samples:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(len(ps), size=n_samples, replace=False))
            ps = ps[idx]
        return ps


def build_naive_mixture(energy=10.0, n_samples=500_000, seed=42):
    e_low, e_high = 6.0, 25.0
    t = (energy - e_low) / (e_high - e_low)
    n_high = int(round(n_samples * t))
    n_low = n_samples - n_high

    rng = np.random.default_rng(seed)
    with h5py.File(TRAIN_H5_PATH, "r") as f:
        cond = f["conditions"][:, 0]
        ps = f["phase_space"]

        idx_low_all = np.where(np.abs(cond - e_low) < 0.1)[0]
        idx_high_all = np.where(np.abs(cond - e_high) < 0.1)[0]

        idx_low = np.sort(
            rng.choice(idx_low_all, size=min(n_low, len(idx_low_all)), replace=False)
        )
        idx_high = np.sort(
            rng.choice(
                idx_high_all, size=min(n_high, len(idx_high_all)), replace=False
            )
        )

        ps_low = ps[idx_low]
        ps_high = ps[idx_high]

    mixture = np.concatenate([ps_low, ps_high], axis=0).astype(np.float32)
    rng.shuffle(mixture)
    return mixture


def generate_cfm_samples(energy=10.0, n_samples=500_000, device="cuda"):
    from models.cfm import CFMTrainer, PhaseSpaceCFM
    from data.synthetic_linac import denormalize_phase_space

    device = (
        "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    with open(CFM_STATS) as f:
        stats = json.load(f)
    dim = len(stats.get("col_names", ["x", "y", "dx", "dy", "dz", "E"]))

    with open(CFM_COND_STATS) as f:
        cond_stats = json.load(f)
    mu_c = np.array(cond_stats["mu"], dtype=np.float32)
    sig_c = np.array(cond_stats["sigma"], dtype=np.float32)
    cond_norm = (
        (np.array([energy], dtype=np.float32) - mu_c) / sig_c
    ).tolist()

    ckpt = torch.load(CFM_CKPT, map_location=device, weights_only=False)
    sd = ckpt.get("model") or ckpt
    hidden_dim = (
        sd["velocity_net.input_proj.weight"].shape[0]
        if "velocity_net.input_proj.weight" in sd
        else 256
    )

    max_idx = -1
    for k in sd:
        if "velocity_net.res_layers." in k:
            max_idx = max(
                max_idx, int(k.split("res_layers.")[1].split(".")[0])
            )
    n_layers = max_idx + 1 if max_idx >= 0 else 4

    model = PhaseSpaceCFM(
        dim=dim, cond_dim=1, hidden_dim=hidden_dim, n_layers=n_layers
    )
    trainer = CFMTrainer(model, device=device, lr=1e-4)
    trainer.load(CFM_CKPT)
    model = model.to(device).eval()

    chunk_size = 250_000
    chunks = []
    n_done = 0

    with torch.no_grad():
        while n_done < n_samples:
            n_chunk = min(chunk_size, n_samples - n_done)
            cond_tensor = (
                torch.tensor(cond_norm, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .repeat(n_chunk, 1)
            )
            chunk_out = model.sample(
                n_chunk, cond_tensor, method="dopri5", atol=1e-4, rtol=1e-4
            ).cpu().numpy()
            chunks.append(chunk_out)
            n_done += n_chunk

    ps_norm = np.concatenate(chunks, axis=0)
    ps_out = denormalize_phase_space(ps_norm, stats).astype(np.float32)

    if ps_out.shape[1] == 6:
        ps_7d = np.zeros((len(ps_out), 7), dtype=np.float32)
        ps_7d[:, [0, 1, 3, 4, 5, 6]] = ps_out
        ps_7d[:, 2] = stats.get("z_const", 0.0)
        ps_out = ps_7d

    return ps_out


def main():
    n_samples = 500_000
    print("Caricamento ed estrazione dati in corso...")

    datasets = {}

    # 1. Reference
    if Path(REFERENCE_10MV_PATH).exists():
        ps_ref = load_reference(n_samples)
        datasets["Reference Reale (10 MV)"] = (
            ps_ref[:, 6] if ps_ref.shape[1] == 7 else ps_ref[:, 5]
        )

    # 2. Baseline
    if Path(TRAIN_H5_PATH).exists():
        ps_base = build_naive_mixture(10.0, n_samples)
        datasets["Baseline Ingenua (6+25 MV)"] = (
            ps_base[:, 6] if ps_base.shape[1] == 7 else ps_base[:, 5]
        )

    # 3. CFM
    if Path(CFM_CKPT).exists():
        ps_cfm = generate_cfm_samples(10.0, n_samples)
        datasets["CFM Interpolato (10 MV)"] = (
            ps_cfm[:, 6] if ps_cfm.shape[1] == 7 else ps_cfm[:, 5]
        )

    print("\n" + "=" * 75)
    print(
        f"{'DATASET':<28} | {'E_max (MeV)':<11} | {'E_99.9%':<9} | {'E_mean':<8} | {'> 10.2 MeV':<12}"
    )
    print("=" * 75)

    for label, E in datasets.items():
        n_total = len(E)
        e_max = np.max(E)
        e_mean = np.mean(E)
        e_p999 = np.percentile(E, 99.9)
        count_illegal = np.sum(E > SAFETY_MARGIN)
        pct_illegal = (count_illegal / n_total) * 100

        print(
            f"{label:<28} | "
            f"{e_max:>11.4f} | "
            f"{e_p999:>9.4f} | "
            f"{e_mean:>8.4f} | "
            f"{count_illegal:>6,} ({pct_illegal:.2f}%)"
        )

    print("=" * 75)
    print("Cutoff fisico Bremsstrahlung per 10 MV: E <= 10.0 MeV.\n")


if __name__ == "__main__":
    main()
