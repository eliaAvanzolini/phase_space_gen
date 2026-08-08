import json
from pathlib import Path
import h5py
import numpy as np
import torch

TRAIN_H5_PATH = "data/energy_only_train_dataset.h5"
REFERENCE_10MV_PATH = "data/energy_only_10mv_reference.h5"
CFM_CKPT = "outputs/cfm_energy_only/best_model.pt"
CFM_STATS = "outputs/cfm_energy_only/normalization_stats.json"
CFM_COND_STATS = "outputs/cfm_energy_only/condition_stats.json"

# Valori di riferimento noti in letteratura per fasci clinici linac (vedi anche
# print_stats() in read_iaea_phsp.py, gia' usato altrove nel progetto)
REF_CORR_E_DZ = 0.18
REF_CORR_X_DX = 0.89


def load_reference(n_samples=500_000, seed=42):
    with h5py.File(REFERENCE_10MV_PATH, "r") as f:
        ps = f["phase_space"][:]
        if len(ps) > n_samples:
            rng = np.random.default_rng(seed)
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

        idx_low = np.sort(rng.choice(idx_low_all, size=min(n_low, len(idx_low_all)), replace=False))
        idx_high = np.sort(rng.choice(idx_high_all, size=min(n_high, len(idx_high_all)), replace=False))

        ps_low = ps[idx_low]
        ps_high = ps[idx_high]

    mixture = np.concatenate([ps_low, ps_high], axis=0).astype(np.float32)
    rng.shuffle(mixture)
    return mixture


def generate_cfm_samples(energy=10.0, n_samples=500_000, device="cuda", seed=42):
    from models.cfm import CFMTrainer, PhaseSpaceCFM
    from data.synthetic_linac import denormalize_phase_space

    torch.manual_seed(seed)  # fissato per riproducibilita' (mancava prima)
    device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"

    with open(CFM_STATS) as f:
        stats = json.load(f)
    dim = len(stats.get("col_names", ["x", "y", "dx", "dy", "dz", "E"]))

    with open(CFM_COND_STATS) as f:
        cond_stats = json.load(f)
    mu_c = np.array(cond_stats["mu"], dtype=np.float32)
    sig_c = np.array(cond_stats["sigma"], dtype=np.float32)
    cond_norm = ((np.array([energy], dtype=np.float32) - mu_c) / sig_c).tolist()

    ckpt = torch.load(CFM_CKPT, map_location=device, weights_only=False)
    sd = ckpt.get("model") or ckpt
    hidden_dim = sd["velocity_net.input_proj.weight"].shape[0] if "velocity_net.input_proj.weight" in sd else 256

    max_idx = -1
    for k in sd:
        if "velocity_net.res_layers." in k:
            max_idx = max(max_idx, int(k.split("res_layers.")[1].split(".")[0]))
    n_layers = max_idx + 1 if max_idx >= 0 else 4

    model = PhaseSpaceCFM(dim=dim, cond_dim=1, hidden_dim=hidden_dim, n_layers=n_layers)
    trainer = CFMTrainer(model, device=device, lr=1e-4)
    trainer.load(CFM_CKPT)
    model = model.to(device).eval()

    chunk_size = 250_000
    chunks = []
    n_done = 0
    with torch.no_grad():
        while n_done < n_samples:
            n_chunk = min(chunk_size, n_samples - n_done)
            cond_tensor = torch.tensor(cond_norm, dtype=torch.float32, device=device).unsqueeze(0).repeat(n_chunk, 1)
            chunk_out = model.sample(n_chunk, cond_tensor, method="dopri5", atol=1e-4, rtol=1e-4).cpu().numpy()
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


def compute_correlations(ps):
    """ps: array (N,7) colonne [X,Y,Z,dX,dY,dZ,E]"""
    corr_e_dz = np.corrcoef(ps[:, 6], ps[:, 5])[0, 1]
    corr_x_dx = np.corrcoef(ps[:, 0], ps[:, 3])[0, 1]
    return corr_e_dz, corr_x_dx


def main():
    n_samples = 500_000
    print("Caricamento ed estrazione dati in corso...")

    datasets = {}

    if Path(REFERENCE_10MV_PATH).exists():
        datasets["Reference Reale (10 MV)"] = load_reference(n_samples)

    if Path(TRAIN_H5_PATH).exists():
        datasets["Baseline Ingenua (6+25 MV)"] = build_naive_mixture(10.0, n_samples)

    if Path(CFM_CKPT).exists():
        datasets["CFM Interpolato (10 MV)"] = generate_cfm_samples(10.0, n_samples)

    print("\n" + "=" * 90)
    print(f"{'DATASET':<28} | {'Corr(E,dz)':<12} | {'Δ da rif.(0.18)':<16} | {'Corr(x,dx)':<12} | {'Δ da rif.(0.89)':<16}")
    print("=" * 90)

    ref_corr_e_dz = ref_corr_x_dx = None
    for label, ps in datasets.items():
        c_e_dz, c_x_dx = compute_correlations(ps)
        if label.startswith("Reference"):
            ref_corr_e_dz, ref_corr_x_dx = c_e_dz, c_x_dx
        print(f"{label:<28} | {c_e_dz:>12.4f} | {abs(c_e_dz - REF_CORR_E_DZ):>16.4f} | "
              f"{c_x_dx:>12.4f} | {abs(c_x_dx - REF_CORR_X_DX):>16.4f}")

    print("=" * 90)
    print("Valori 'Δ da rif.' calcolati contro i valori di letteratura (~0.18 e ~0.89) "
          "usati altrove nel progetto come riferimento.")

    if ref_corr_e_dz is not None:
        print(f"\n{'-'*90}")
        print(" Δ RISPETTO AL REFERENCE REALE (il confronto che conta di piu' per l'interpolazione)")
        print(f"{'-'*90}")
        for label, ps in datasets.items():
            if label.startswith("Reference"):
                continue
            c_e_dz, c_x_dx = compute_correlations(ps)
            print(f"{label:<28} | Δcorr(E,dz) vs reale = {abs(c_e_dz - ref_corr_e_dz):.4f} | "
                  f"Δcorr(x,dx) vs reale = {abs(c_x_dx - ref_corr_x_dx):.4f}")


if __name__ == "__main__":
    main()
