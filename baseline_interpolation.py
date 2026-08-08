import argparse
import json
from pathlib import Path

import h5py
import numpy as np

TRAIN_H5_PATH = "data/energy_only_train_dataset.h5"
REFERENCE_10MV_PATH = "data/energy_only_10mv_reference.h5"


def build_naive_mixture(energy, n_samples, seed=42):
    """Mescola vettori REALI di 6MV e 25MV nelle proporzioni lineari corrette
    per l'energia target, senza alcun modello - la baseline piu' semplice
    possibile per l'interpolazione."""
    e_low, e_high = 6.0, 25.0
    t = (energy - e_low) / (e_high - e_low)  # frazione verso l'alto
    n_high = int(round(n_samples * t))
    n_low = n_samples - n_high

    print(f"  Interpolazione lineare: t={t:.4f} -> {n_low:,} da 6MV + {n_high:,} da 25MV")

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", type=float, default=10.0)
    ap.add_argument("--n_samples", type=int, default=500_000)
    ap.add_argument("--output_dir", default="outputs/interpolation_10mv_eval")
    args = ap.parse_args()

    out = Path(args.output_dir) / "naive_baseline"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f" BASELINE INGENUA: miscela lineare 6MV+25MV -> {args.energy}MeV (nessun modello)")
    print("=" * 70)

    print(f"\nCaricamento reference reale ({REFERENCE_10MV_PATH})...")
    with h5py.File(REFERENCE_10MV_PATH, "r") as f:
        n_available = f["phase_space"].shape[0]
        n_ref = min(n_available, args.n_samples)
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n_available, size=n_ref, replace=False))
        ps_real = f["phase_space"][idx]
    print(f"  {len(ps_real):,} particelle reali campionate")

    print(f"\nCostruzione miscela ingenua...")
    ps_naive = build_naive_mixture(args.energy, len(ps_real))
    print(f"  {len(ps_naive):,} particelle nella miscela")

    from evaluate import evaluate_model
    report = evaluate_model(
        ps_real, ps_naive,
        model_name="naive_mixture_baseline",
        output_dir=str(out),
    )

    with open(out / "naive_baseline_summary.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(" CONFRONTO: se la separability della baseline ingenua e' vicina")
    print(" a quella di CFM (0.5495), il compito era piu' facile del previsto.")
    print(" Se invece e' molto piu' alta (vicina a 1.0), CFM ha davvero imparato")
    print(" qualcosa che una semplice miscela lineare non cattura.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
