import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dose_validation_conditional_PATCHED import run_gate_dose

SRC_DIR = Path("outputs/dose_interpolation_10mv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["reference", "cfm"], required=True)
    ap.add_argument("--voxel_mm", type=float, default=4.0)
    ap.add_argument("--n_threads", type=int, default=8)
    ap.add_argument("--output_dir", default="outputs/dose_interpolation_10mv_coarse")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    root_path = SRC_DIR / f"gen_{args.phase}.root"
    if not root_path.exists():
        raise FileNotFoundError(
            f"{root_path} non trovato in {SRC_DIR} — "
            "i file ROOT originali devono essere presenti per riutilizzare gli eventi generati."
        )

    print(f"[{args.phase}] Riuso {root_path} (in sola lettura)")
    print(f"[{args.phase}] Dimensione voxel: {args.voxel_mm} mm")
    print(f"[{args.phase}] Destinazione output: {out}")
    
    run_gate_dose(root_path, out, args.phase, args.n_threads, args.voxel_mm)
    print(f"[{args.phase}] Completato con successo.")


if __name__ == "__main__":
    main()

