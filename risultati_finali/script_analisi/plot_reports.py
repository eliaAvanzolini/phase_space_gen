"""
plot_reports.py
===============
Rigenera i plot stile GAN (stepfilled, PHSP vs modello) e la tabella
riassuntiva delle metriche per OGNI run nella cartella outputs/.

Uso:
    python plot_reports.py --outputs_dir outputs --save_dir ./report_plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Tenta di caricare h5py (necessario per caricare i file del phase space) ───
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("[WARN] h5py non trovato — i plot marginals saranno saltati.")

# ─── Costanti del Phase Space (Ordine esatto e stile Sarrut 2019) ─────────────
COLUMNS_PAPER = ["E", "x", "y", "dx", "dy", "dz"]
COLUMNS_UNITS = ["E [MeV]", "x [cm]", "y [cm]", "dx", "dy", "dz"]

# Colori ufficiali del paper Sarrut 2019 Fig.2
COLOR_REAL  = "#4472C4"  # Blu PHSP
COLOR_MODEL = "#ED7D31"  # Arancio Modello

def _safe_get(d: dict, *keys, default="N/A"):
    """Naviga nei dizionari delle metriche in modo sicuro."""
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d if d is not None else default

def find_valid_runs(outputs_dir: Path) -> List[Path]:
    """Trova tutte le cartelle di run valide dentro la directory di output."""
    runs = []
    if not outputs_dir.exists():
        return runs
    for item in sorted(outputs_dir.iterdir()):
        if item.is_dir():
            if list(item.glob("*metrics.json")) or list(item.glob("eval/*_report.json")) or list(item.glob("*_report.json")):
                runs.append(item)
    return runs

def load_metrics_for_run(run_dir: Path) -> Optional[Dict]:
    """Carica il file JSON delle metriche adattandosi ai vari formati del progetto."""
    search_paths = [run_dir / "eval", run_dir, run_dir / "baseline_GAN_iaea"]
    for path in search_paths:
        if not path.exists():
            continue
        json_files = list(path.glob("*metrics.json")) + list(path.glob("*_report.json"))
        if json_files:
            try:
                with open(json_files[0], "r") as f:
                    return json.load(f)
            except Exception:
                pass
    return None

def load_config_for_run(run_dir: Path) -> Dict:
    """Carica config.json se presente nella cartella della run."""
    for path in [run_dir, run_dir / "eval"]:
        cfg_f = path / "config.json"
        if cfg_f.exists():
            with open(cfg_f, "r") as f:
                return json.load(f)
    return {}

def load_data_vectors(run_dir: Path, config: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Carica i vettori fisici reali (PHSP) e generati dal modello.
    Converte tutto nell'ordine 6D del paper Sarrut: [E, x, y, dx, dy, dz]
    """
    if not HAS_H5PY:
        return None, None

    real_7d = None
    # Cerca prima nei percorsi centrali dei dati
    for ref_path in [Path("data/elekta_6mv_eval.h5"), Path("data/elekta_6mv_train.h5")]:
        if ref_path.exists():
            try:
                with h5py.File(ref_path, "r") as f:
                    real_7d = f["phase_space"][:]
                    break
            except Exception:
                pass

    # Se non trovato, cerca dentro la cartella del run (data_raw.h5)
    if real_7d is None:
        for p in [run_dir, run_dir / "eval"]:
            h5_raw = p / "data_raw.h5"
            if h5_raw.exists():
                try:
                    with h5py.File(h5_raw, "r") as f:
                        real_7d = f["phase_space"][:]
                        break
                except Exception:
                    pass

    if real_7d is None:
        return None, None

    # Riordina il reale 7D [x, y, z, dx, dy, dz, E] nel formato 6D del paper [E, x, y, dx, dy, dz]
    real_6d = np.column_stack([
        real_7d[:, 6],   # E
        real_7d[:, 0],   # x
        real_7d[:, 1],   # y
        real_7d[:, 3],   # dx
        real_7d[:, 4],   # dy
        real_7d[:, 5],   # dz
    ]).astype(np.float32)

    # --- CASO 1: Baseline GAN (Sarrut 2019) ---
    if "gan" in run_dir.name.lower() or "gaga" in run_dir.name.lower():
        gan_samples_path = run_dir / "gan_samples.npy"
        if gan_samples_path.exists():
            gan_6d = np.load(gan_samples_path)
            n = min(len(real_6d), len(gan_6d))
            return real_6d[:n], gan_6d[:n]

    # --- CASO 2: Modelli Generativi NSF e CFM ---
    h5_gen = run_dir / "generated_ps.h5"
    if not h5_gen.exists():
        h5_gen = run_dir / "eval" / "generated_ps.h5"

    if h5_gen.exists():
        try:
            with h5py.File(h5_gen, "r") as f_gen:
                gen_7d = f_gen["phase_space"][:]
                gen_6d = np.column_stack([
                    gen_7d[:, 6],   # E
                    gen_7d[:, 0],   # x
                    gen_7d[:, 1],   # y
                    gen_7d[:, 3],   # dx
                    gen_7d[:, 4],   # dy
                    gen_7d[:, 5],   # dz
                ]).astype(np.float32)
                
                if len(real_6d) > len(gen_6d):
                    seed = config.get("seed", 42)
                    rng = np.random.default_rng(seed)
                    perm = rng.permutation(len(real_6d))
                    n_train = int(len(real_6d) * 0.70)
                    n_val   = int(len(real_6d) * 0.15)
                    test_idx = perm[n_train + n_val:]
                    if len(test_idx) == len(gen_6d):
                        return real_6d[test_idx], gen_6d
                
                n = min(len(real_6d), len(gen_6d))
                return real_6d[:n], gen_6d[:n]
        except Exception:
            pass

    return None, None

def print_and_save_local_table(run_name: str, metrics: Dict, run_save_dir: Path):
    """Genera e stampa una tabella di metriche esclusiva per la cartella corrente."""
    w1_mean = _safe_get(metrics, "w1", "mean")
    w1_E    = _safe_get(metrics, "w1", "E")
    if w1_mean == "N/A" and "mean_w1" in metrics:
        w1_mean = metrics["mean_w1"]
        w1_E    = _safe_get(metrics, "E", "w1")
    
    mmd2    = metrics.get("mmd", "N/A")
    sep_acc = _safe_get(metrics, "separability", "accuracy")
    sep_std = _safe_get(metrics, "separability", "std")
    tail_E  = _safe_get(metrics, "tail_w1", "E")

    def fmt(v):
        return f"{v:.5f}" if isinstance(v, (int, float)) else str(v)

    sep_str = f"{fmt(sep_acc)} ± {fmt(sep_std)}" if sep_std != "N/A" else fmt(sep_acc)

    output_lines = [
        f"=====================================================================================",
        f"  TABELLA COMPARATIVA METRICHE DI VALIDAZIONE LOCALE - RUN: {run_name}",
        f"=====================================================================================",
        f"| Metrica / Parametro Fisico     | Valore Ottenuto nel Run                           |",
        f"|--------------------------------|---------------------------------------------------|",
        f"| Wasserstein-1 Medio (6D)       | {fmt(w1_mean):<49} |",
        f"| Wasserstein-1 Spettro Energia  | {fmt(w1_E):<49} |",
        f"| Maximum Mean Discrepancy (MMD²)| {fmt(mmd2):<49} |",
        f"| Separability Score (RF Acc)    | {sep_str:<49} |",
        f"| Tail W1 (>2σ Spettro Energia)  | {fmt(tail_E):<49} |",
        f"=====================================================================================",
        f"  * Nota: Separability -> 0.50 ottimo (indistinguibile), 1.00 fallimento totale.",
        f"  * Report JSON letto correttamente per questa sessione.",
    ]

    for line in output_lines:
        print(line)

    with open(run_save_dir / "tabella_metriche_locale.txt", "w") as f:
        f.write("".join(output_lines))

def plot_stepfilled_marginals(real: np.ndarray, gen: np.ndarray, run_name: str, save_path: Path, n_subsample: int, n_bins: int):
    """Genera il plot 2x3 con istogrammi sovrapposti di tipo stepfilled in stile GAN (Sarrut 2019 Fig.2)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    rng = np.random.default_rng(42)
    sub_r = real[rng.choice(len(real), min(n_subsample, len(real)), replace=False)]
    sub_g = gen[rng.choice(len(gen), min(n_subsample, len(gen)), replace=False)]

    if "gan" in run_name.lower() or "gaga" in run_name.lower():
        model_label = "GAN"
    elif "nsf" in run_name.lower():
        model_label = "NSF"
    elif "cfm" in run_name.lower():
        model_label = "CFM"
    else:
        model_label = "Modello"

    for i in range(6):
        ax = axes[i]
        name = COLUMNS_UNITS[i]

        lo = min(np.percentile(sub_r[:, i], 0.1), np.percentile(sub_g[:, i], 0.1))
        hi = max(np.percentile(sub_r[:, i], 99.9), np.percentile(sub_g[:, i], 99.9))
        bins = np.linspace(lo, hi, n_bins)

        # 1. LAYER REALE (PHSP) -> Stepfilled Blu (alpha=0.7) come nel paper
        ax.hist(sub_r[:, i], bins=bins, density=True,
                histtype="stepfilled", color=COLOR_REAL, alpha=0.7, label="PHSP")

        # 2. LAYER GENERATO -> Stepfilled Arancio (alpha=0.6) identico allo stile della GAN
        ax.hist(sub_g[:, i], bins=bins, density=True,
                histtype="stepfilled", color=COLOR_MODEL, alpha=0.6, label=model_label)

        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel("Counts", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(lo, hi)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Grafico stepfilled convertito in stile Sarrut 2019 Fig.2 salvato.")

def main():
    parser = argparse.ArgumentParser(description="Generatore tabelle locali e plot stepfilled per cartella outputs")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Cartella dei modelli")
    parser.add_argument("--save_dir", type=str, default="report_plots", help="Cartella dei report finali")
    parser.add_argument("--n_bins", type=int, default=100, help="Bin per istogramma (Sarrut standard: 100)")
    parser.add_argument("--n_subsample", type=int, default=100000, help="Particelle da analizzare")
    args = parser.parse_args()

    outputs_path = Path(args.outputs_dir)
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    runs = find_valid_runs(outputs_path)
    if not runs:
        print(f"[ERROR] Nessun modello trovato in {outputs_path} contenente report JSON.")
        sys.exit(1)

    print(f"=====================================================================================")
    print(f"  ELABORAZIONE REPORT INDIPENDENTI PER OGNI MODELLO (TABELLE + PLOTS STEPFILLED)")
    print(f"=====================================================================================")

    for run_dir in runs:
        run_name = run_dir.name
        metrics = load_metrics_for_run(run_dir)
        
        if metrics is None:
            continue

        run_save_dir = save_path / run_name
        run_save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Genera e stampa la tabella locale esclusiva per questa cartella
        print_and_save_local_table(run_name, metrics, run_save_dir)

        # 2. Carica i dati e genera il plot forzando lo stile della GAN (reorder 6D + stepfilled)
        config = load_config_for_run(run_dir)
        real_data, gen_data = load_data_vectors(run_dir, config)

        if real_data is not None and gen_data is not None:
            print(f"  [Plot] Caricamento vettori di spazio delle fasi (subsampling a {args.n_subsample:,} particelle)...")
            plot_file = run_save_dir / "marginals_stepfilled_local.png"
            plot_stepfilled_marginals(real_data, gen_data, run_name, plot_file, args.n_subsample, args.n_bins)
        else:
            print(f"  [INFO] File di dati raw o generati non trovati per {run_name}. Plot stepfilled saltato.")
        
        print(f"  [OK] Tutti i report della cartella salvati in: {run_save_dir}")

if __name__ == "__main__":
    main()
plot_reports.py