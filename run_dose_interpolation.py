import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import uproot

sys.path.insert(0, str(Path(__file__).parent))
from dose_validation_conditional_PATCHED import run_gate_dose
from evaluate_interpolation_10mv import generate_at_energy

REFERENCE_10MV_PATH = "data/energy_only_10mv_reference.h5"


def load_reference_sample(n_samples, seed=42):
    with h5py.File(REFERENCE_10MV_PATH, "r") as f:
        n_available = f["phase_space"].shape[0]
        n = min(n_available, n_samples)
        rng = np.random.default_rng(seed)
        start_idx = int(rng.integers(0, n_available - n + 1)) if n_available > n else 0
        print(f"  Lettura HDF5: {n:,} particelle dall'indice {start_idx:,}...")
        ps = f["phase_space"][start_idx : start_idx + n]
    print(f"  Reference: {n:,} particelle campionate (su {n_available:,} disponibili)")
    return ps


def write_phsp_root(ps, out_path):
    """Scrive il phase space nel formato ROOT atteso da run_gate_dose():
    X,Y in mm (da cm*10), Z sempre 0 (piano sorgente per la sim di dose),
    dX,dY,dZ direzione (dZ forzata positiva, verso il fantoccio), E in MeV."""
    with uproot.recreate(out_path) as f:
        f["PhaseSpace"] = {
            "X": ps[:, 0] * 10.0,
            "Y": ps[:, 1] * 10.0,
            "Z": np.zeros(len(ps), dtype=np.float32),
            "dX": ps[:, 3],
            "dY": ps[:, 4],
            "dZ": np.abs(ps[:, 5]).astype(np.float32),
            "E": ps[:, 6],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["reference", "cfm"],
        required=True,
        help="Fase da eseguire in questo processo (OpenGATE consente un solo sim.run() per processo Python)",
    )
    ap.add_argument("--n_particles", type=int, default=10_000_000)
    ap.add_argument("--n_threads", type=int, default=8)
    ap.add_argument("--voxel_mm", type=float, default=2.0)
    ap.add_argument("--energy", type=float, default=10.0)
    ap.add_argument(
        "--solver",
        choices=["euler", "dopri5"],
        default="euler",
        help="euler (sample_fast) per generare milioni di campioni rapidamente",
    )
    ap.add_argument("--cfm_steps", type=int, default=30)
    ap.add_argument("--output_dir", default="outputs/dose_interpolation_10mv")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print(f" DOSE INTERPOLAZIONE 10MV [FASE: {args.phase.upper()}] — {args.n_particles:,} particelle")
    print("=" * 70)

    if args.phase == "reference":
        print("\nCostruzione reference...")
        ps_ref = load_reference_sample(args.n_particles)
        ref_root = out / "gen_reference.root"
        write_phsp_root(ps_ref, ref_root)
        del ps_ref
        print(f"  Scritto: {ref_root}")

        print("\nSimulazione dose reference...")
        run_gate_dose(ref_root, out, "reference", args.n_threads, args.voxel_mm)

    elif args.phase == "cfm":
        cfm_root = out / "gen_cfm.root"
        # Se gen_cfm.root era già stato generato prima del crash, possiamo riusarlo
        if not cfm_root.exists():
            print(f"\nGenerazione CFM a E={args.energy}...")
            ps_cfm = generate_at_energy("cfm", args.energy, args.n_particles, device)
            write_phsp_root(ps_cfm, cfm_root)
            del ps_cfm
            print(f"  Scritto: {cfm_root}")
        else:
            print(f"\nFile CFM già presente su disco ({cfm_root}), avvio diretto simulazione GATE...")

        print("\nSimulazione dose CFM...")
        run_gate_dose(cfm_root, out, "cfm", args.n_threads, args.voxel_mm)

    print(f"\n{'=' * 70}")
    print(f" Fase {args.phase.upper()} completata con successo in: {out}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
