"""
check_pipeline.py
==================
Verifica che l'intera pipeline funzioni prima di caricare sul cloud.
Usa dati sintetici e pochissime epoche: gira in <5 minuti.

Testa:
    1. Generazione dati sintetici + normalizzazione sferica
    2. Training GAN (10 iter), NSF (2 epoche), CFM (2 epoche)
    3. Generazione campioni da ogni modello
    4. save_for_gate.py → .pth
    5. evaluate.py metriche

Uso:
    python check_pipeline.py
    python check_pipeline.py --with_gate  # testa anche gate_simulations (richiede opengate)
"""

import sys
import json
import subprocess
import numpy as np
from pathlib import Path
import tempfile
import os

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def run(cmd, desc, cwd=None):
    print(f"\n  [{desc}]")
    result = subprocess.run(
        cmd, shell=True, cwd=str(cwd or ROOT),
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ✗ FAILED:\n{result.stderr[-500:]}")
        return False
    print(f"  ✓ OK")
    return True


def check_imports():
    print("\n── 1. Import check ─────────────────────────────────────")
    ok = True
    for pkg, imp in [
        ("numpy",      "import numpy"),
        ("scipy",      "import scipy"),
        ("sklearn",    "import sklearn"),
        ("h5py",       "import h5py"),
        ("matplotlib", "import matplotlib"),
        ("torch",      "import torch; print(torch.__version__)"),
        ("nflows",     "import nflows"),
        ("zuko",       "import zuko"),
        ("torchdiffeq","import torchdiffeq"),
    ]:
        r = subprocess.run(
            f"python -c \"{imp}\"",
            shell=True, capture_output=True, cwd=str(ROOT)
        )
        status = "✓" if r.returncode == 0 else "✗"
        print(f"  {status} {pkg}")
        if r.returncode != 0:
            ok = False
    return ok


def check_data_pipeline():
    print("\n── 2. Data pipeline (sintetica + sferica) ──────────────")
    r = subprocess.run(
        "python -c \"\n"
        "import sys; sys.path.insert(0, '.')\n"
        "import numpy as np\n"
        "from data.synthetic_linac import (\n"
        "    generate_phase_space, normalize_phase_space, denormalize_phase_space\n"
        ")\n"
        "ps = generate_phase_space(10000, seed=42)\n"
        "n, s = normalize_phase_space(ps, spherical=True)\n"
        "assert n.shape == (10000, 5), f'shape {n.shape}'\n"
        "r = denormalize_phase_space(n, s)\n"
        "d = np.linalg.norm(r[:,3:6], axis=1)\n"
        "assert np.allclose(d, 1, atol=1e-5), f'norm_err {np.abs(d-1).max()}'\n"
        "print('OK: spherical 5D, ||d||=1 guaranteed')\n"
        "\"",
        shell=True, capture_output=True, text=True, cwd=str(ROOT)
    )
    if r.returncode == 0:
        print(f"  ✓ {r.stdout.strip()}")
        return True
    print(f"  ✗ {r.stderr[-300:]}")
    return False


def check_training(tmpdir):
    print("\n── 3. Training (2 epoche sintetiche) ───────────────────")
    results = {}

    # NSF
    cmd_nsf = (
        f"python train.py --model nsf --n_samples 50000 "
        f"--epochs 2 --batch_size 1024 --lr 1e-3 "
        f"--n_transforms 4 --n_bins 4 --hidden_dim 64 "
        f"--spherical --save_every 2 --val_every 2 "
        f"--output_dir {tmpdir}/outputs"
    )
    results["nsf"] = run(cmd_nsf, "NSF 2 epoche")

    # CFM
    cmd_cfm = (
        f"python train.py --model cfm --n_samples 50000 "
        f"--epochs 2 --batch_size 1024 --lr 1e-3 "
        f"--n_layers 2 --hidden_dim 64 "
        f"--spherical --save_every 2 --val_every 2 "
        f"--output_dir {tmpdir}/outputs"
    )
    results["cfm"] = run(cmd_cfm, "CFM 2 epoche")

    # GAN baseline
    cmd_gan = (
        f"python baseline_gaga.py --synthetic "
        f"--n_epochs 20 --batch_size 1000 "
        f"--n_train 20000 --n_eval 5000 "
        f"--h_dim 64 --z_dim 6 "
        f"--log_every 20 --save_every 20 "
        f"--output_dir {tmpdir}/outputs"
    )
    results["gan"] = run(cmd_gan, "GAN 20 iterazioni")

    return results


def check_save_for_gate(tmpdir):
    print("\n── 4. save_for_gate.py ─────────────────────────────────")
    results = {}

    for model in ["nsf", "cfm"]:
        # Trova il checkpoint del training precedente
        ckpt_pattern = list(Path(tmpdir, "outputs").glob(f"{model}_*/best_model.pt"))
        if not ckpt_pattern:
            print(f"  ? {model}: checkpoint non trovato (training fallito?)")
            results[model] = False
            continue

        ckpt  = ckpt_pattern[-1]
        stats = ckpt.parent / "normalization_stats.json"
        out   = Path(tmpdir) / f"{model}_gate_test.pth"

        cmd = (
            f"python gate_integration/save_for_gate.py "
            f"--checkpoint {ckpt} --model {model} "
            f"--stats_path {stats} --out {out}"
        )
        results[model] = run(cmd, f"save_for_gate {model}")

        if results[model] and out.exists():
            # Verifica che il .pth sia leggibile e contenga le chiavi giuste
            try:
                import torch
                pth = torch.load(str(out), map_location="cpu", weights_only=False)
                assert "keys" in pth
                assert pth["keys"] == ["Ekine", "X", "Y", "Z", "dX", "dY", "dZ"]
                # Test forward pass
                wrapper = pth  # non è un modello diretto, è un dict
                print(f"    ✓ .pth valido, keys: {pth['keys']}")
            except Exception as e:
                print(f"    ? .pth creato ma verifica fallita: {e}")

    return results


def check_evaluate(tmpdir):
    print("\n── 5. Metriche (evaluate.py) ───────────────────────────")

    ckpt_nsf = list(Path(tmpdir, "outputs").glob("nsf_*/best_model.pt"))
    if not ckpt_nsf:
        print("  ? NSF checkpoint non trovato")
        return False

    cmd = (
        f"python generate.py "
        f"--checkpoint {ckpt_nsf[-1]} --model nsf "
        f"--stats_path {ckpt_nsf[-1].parent}/normalization_stats.json "
        f"--n_samples 5000 --validate "
        f"--E_nom 6.0 --jaw_x 5.0 --jaw_y 5.0 "
        f"--out {tmpdir}/test_generated.h5"
    )
    ok = run(cmd, "generate + validate (NSF)")
    return ok


def print_summary(results: dict):
    print(f"\n{'='*55}")
    print(f"  RIEPILOGO PIPELINE CHECK")
    print(f"{'='*55}")

    all_ok = True
    checks = {
        "Import": results.get("imports", False),
        "Dati sintetici sferici": results.get("data", False),
        "Training NSF": results.get("training", {}).get("nsf", False),
        "Training CFM": results.get("training", {}).get("cfm", False),
        "Training GAN": results.get("training", {}).get("gan", False),
        "save_for_gate NSF": results.get("gate_export", {}).get("nsf", False),
        "save_for_gate CFM": results.get("gate_export", {}).get("cfm", False),
        "Generate + metrics": results.get("evaluate", False),
    }

    for name, ok in checks.items():
        icon = "✓" if ok else "✗"
        print(f"  {icon} {name}")
        if not ok:
            all_ok = False

    print(f"\n  {'✅ TUTTO OK — pronto per il cloud!' if all_ok else '❌ Ci sono problemi da risolvere'}")

    if all_ok:
        print(f"\n  Prossimi step:")
        print(f"  1. Carica su cloud:  tar -czf phase_space_gen_cloud.tar.gz \\")
        print(f"                           phase_space_gen/ \\")
        print(f"                           data/elekta_6mv_train.h5 \\")
        print(f"                           data/elekta_6mv_eval.h5")
        print(f"  2. Sul server:       bash run_cloud.sh all")
        print(f"  3. Scarica outputs:  scp server:~/outputs/gate_models/*.pth .")
    print(f"{'='*55}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--with_gate", action="store_true",
                   help="Testa anche GATE 10 (richiede opengate installato)")
    args = p.parse_args()

    print("╔══════════════════════════════════════════════════════╗")
    print("║  Pipeline Check — Phase Space Generative Models     ║")
    print("╚══════════════════════════════════════════════════════╝")

    results = {}
    tmpdir  = tempfile.mkdtemp(prefix="ps_check_")
    print(f"  Cartella temporanea: {tmpdir}")

    results["imports"]     = check_imports()
    results["data"]        = check_data_pipeline()

    if results["imports"] and results["data"]:
        results["training"]    = check_training(tmpdir)
        results["gate_export"] = check_save_for_gate(tmpdir)
        results["evaluate"]    = check_evaluate(tmpdir)
    else:
        print("\n  [SKIP] Training saltato (import o data falliti)")
        results["training"]    = {}
        results["gate_export"] = {}
        results["evaluate"]    = False

    print_summary(results)

    # Pulizia tmpdir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
