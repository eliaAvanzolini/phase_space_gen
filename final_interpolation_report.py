import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_interpolation_10mv import MODELS, generate_at_energy  # riuso i loader gia' scritti

TRAIN_H5_PATH = "data/energy_only_train_dataset.h5"
REFERENCE_10MV_PATH = "data/energy_only_10mv_reference.h5"
SAFETY_MARGIN = 10.2  # MeV, soglia oltre il cutoff fisico del 10MV

REF_CORR_E_DZ = 0.18
REF_CORR_X_DX = 0.89

TARGET_ENERGY = 10.0
N_SAMPLES = 500_000
SEED = 42


def load_reference(n_samples=N_SAMPLES, seed=SEED):
    with h5py.File(REFERENCE_10MV_PATH, "r") as f:
        n_available = f["phase_space"].shape[0]
        n = min(n_available, n_samples)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_available, size=n, replace=False))
        return f["phase_space"][idx]


def build_naive_mixture(energy=TARGET_ENERGY, n_samples=N_SAMPLES, seed=SEED):
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


def energy_stats(ps):
    E = ps[:, 6]
    n_total = len(E)
    n_illegal = int(np.sum(E > SAFETY_MARGIN))
    return {
        "e_max": float(E.max()),
        "e_p999": float(np.percentile(E, 99.9)),
        "e_mean": float(E.mean()),
        "n_illegal": n_illegal,
        "pct_illegal": 100.0 * n_illegal / n_total,
    }


def correlation_stats(ps):
    corr_e_dz = float(np.corrcoef(ps[:, 6], ps[:, 5])[0, 1])
    corr_x_dx = float(np.corrcoef(ps[:, 0], ps[:, 3])[0, 1])
    return {"corr_e_dz": corr_e_dz, "corr_x_dx": corr_x_dx}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f" REPORT FINALE INTERPOLAZIONE A {TARGET_ENERGY} MeV")
    print("=" * 60)

    datasets = {}

    print("\nCaricamento reference reale...")
    datasets["Reference Reale"] = load_reference()

    print("Costruzione baseline ingenua (6+25MV mescolati)...")
    datasets["Baseline Ingenua"] = build_naive_mixture()

    for model_name in MODELS:
        ckpt_path = Path(MODELS[model_name]["checkpoint"])
        if not ckpt_path.exists():
            print(f"[SKIP] {model_name.upper()}: checkpoint non trovato ({ckpt_path})")
            continue
        print(f"Generazione {model_name.upper()}...")
        torch.manual_seed(SEED)
        ps_gen = generate_at_energy(model_name, TARGET_ENERGY, N_SAMPLES, device)
        datasets[model_name.upper()] = ps_gen

    # ── Calcolo statistiche per ciascun dataset ─────────────────────────────
    e_results = {}
    c_results = {}
    for label, ps in datasets.items():
        e_results[label] = energy_stats(ps)
        c_results[label] = correlation_stats(ps)

    ref_e = e_results["Reference Reale"]
    ref_c = c_results["Reference Reale"]

    # ── Tabella 1: endpoint energetico ──────────────────────────────────────
    print(f"\n{'='*95}")
    print(" TABELLA 1 — ENDPOINT ENERGETICO (vincolo fisico bremsstrahlung)")
    print(f"{'='*95}")
    print(f"{'DATASET':<20} | {'E_max':<8} | {'E_P99.9':<8} | {'E_mean':<8} | {'>'+str(SAFETY_MARGIN)+'MeV':<14} | {'Δ E_max vs reale':<16}")
    print("-" * 95)
    for label, s in e_results.items():
        delta = "" if label == "Reference Reale" else f"{abs(s['e_max'] - ref_e['e_max']):.4f}"
        print(f"{label:<20} | {s['e_max']:>8.4f} | {s['e_p999']:>8.4f} | {s['e_mean']:>8.4f} | "
              f"{s['n_illegal']:>6,} ({s['pct_illegal']:.3f}%) | {delta:>16}")

    # ── Tabella 2: correlazioni fisiche ──────────────────────────────────────
    print(f"\n{'='*95}")
    print(" TABELLA 2 — CORRELAZIONI FISICHE (struttura congiunta)")
    print(f"{'='*95}")
    print(f"{'DATASET':<20} | {'Corr(E,dz)':<11} | {'Δ vs reale':<11} | {'Corr(x,dx)':<11} | {'Δ vs reale':<11}")
    print("-" * 95)
    for label, c in c_results.items():
        d_e_dz = "" if label == "Reference Reale" else f"{abs(c['corr_e_dz'] - ref_c['corr_e_dz']):.4f}"
        d_x_dx = "" if label == "Reference Reale" else f"{abs(c['corr_x_dx'] - ref_c['corr_x_dx']):.4f}"
        print(f"{label:<20} | {c['corr_e_dz']:>11.4f} | {d_e_dz:>11} | {c['corr_x_dx']:>11.4f} | {d_x_dx:>11}")

    # ── Riepilogo: chi si avvicina di piu' al reale, per ciascun test ───────
    print(f"\n{'='*95}")
    print(" RIEPILOGO — CHI SI AVVICINA DI PIU' AL REFERENCE (esclusa la riga Reference stessa)")
    print(f"{'='*95}")
    candidates = {k: v for k, v in e_results.items() if k != "Reference Reale"}
    best_e = min(candidates, key=lambda k: abs(candidates[k]["e_max"] - ref_e["e_max"]))
    print(f"  Endpoint E_max piu' vicino al reale: {best_e}")

    candidates_c = {k: v for k, v in c_results.items() if k != "Reference Reale"}
    best_edz = min(candidates_c, key=lambda k: abs(candidates_c[k]["corr_e_dz"] - ref_c["corr_e_dz"]))
    best_xdx = min(candidates_c, key=lambda k: abs(candidates_c[k]["corr_x_dx"] - ref_c["corr_x_dx"]))
    print(f"  Corr(E,dz) piu' vicina al reale:     {best_edz}")
    print(f"  Corr(x,dx) piu' vicina al reale:     {best_xdx}")


if __name__ == "__main__":
    main()
