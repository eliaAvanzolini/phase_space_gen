"""
compare_conditional_models.py
==============================
Confronto finale tra CFM, NSF e GAN condizionali.

Carica i report JSON da eval_conditional_real.py per ciascun modello
e genera tabelle comparative, heatmap e barplot.

Uso:
    python compare_conditional_models.py \\
        --cfm_dir outputs/cfm_conditional_6mv_10mv_139k \\
        --nsf_dir outputs/nsf_conditional_6mv_10mv_139k \\
        --gan_dir outputs/gan_conditional_6mv_10mv_139k \\
        --output_dir outputs/comparison_conditional
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description="Confronto finale CFM vs NSF vs GAN condizionali",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cfm_dir", type=str, default=None,
                   help="Cartella run CFM condizionale")
    p.add_argument("--nsf_dir", type=str, default=None,
                   help="Cartella run NSF condizionale")
    p.add_argument("--gan_dir", type=str, default=None,
                   help="Cartella run GAN condizionale")
    p.add_argument("--output_dir", type=str, default="outputs/comparison_conditional")
    return p.parse_args()


def load_report(run_dir: str, model_name: str) -> dict:
    """Carica il report JSON di valutazione condizionale."""
    report_path = Path(run_dir) / "eval_conditional" / f"{model_name}_conditional_eval.json"
    if not report_path.exists():
        print(f"  [WARNING] Report non trovato: {report_path}")
        return None
    with open(report_path) as f:
        return json.load(f)


def plot_comparison_barplot(reports: dict, save_path: str):
    """
    Barplot delle metriche chiave per modello × energia.
    """
    energies = ["6MV", "10MV"]
    models = list(reports.keys())
    n_models = len(models)

    metrics = [
        ("W1 (mean)", lambda r, e: r[e]["w1"]["mean"]),
        ("W1 (E)", lambda r, e: r[e]["w1"]["E"]),
        ("MMD²", lambda r, e: r[e]["mmd"]),
        ("Separability", lambda r, e: r[e]["separability"]["accuracy"]),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    colors = {"CFM": "#4c72b0", "NSF": "#55a868", "GAN": "#c44e52"}
    bar_width = 0.25

    for ax_idx, (metric_name, metric_fn) in enumerate(metrics):
        ax = axes[ax_idx]
        x = np.arange(len(energies))

        for m_idx, model_name in enumerate(models):
            values = []
            for e in energies:
                try:
                    values.append(metric_fn(reports[model_name], e))
                except (KeyError, TypeError):
                    values.append(0)

            offset = (m_idx - n_models / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, values, bar_width,
                         label=model_name, color=colors.get(model_name, "#888888"),
                         alpha=0.85, edgecolor="white", linewidth=0.5)

            # Valori sulle barre
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       f"{val:.4f}", ha="center", va="bottom", fontsize=7)

        ax.set_xlabel("Energia", fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(metric_name, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(energies)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    plt.suptitle("Confronto Condizionale — CFM vs NSF vs GAN",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Barplot salvato: {save_path}")


def plot_comparison_heatmap(reports: dict, save_path: str):
    """
    Heatmap delle metriche per modello × energia con colori verde/rosso.
    """
    models = list(reports.keys())
    energies = ["6MV", "10MV"]

    metric_defs = [
        ("W1 (mean)", lambda r, e: r[e]["w1"]["mean"]),
        ("W1 (E)", lambda r, e: r[e]["w1"]["E"]),
        ("MMD²", lambda r, e: r[e]["mmd"]),
        ("Separability", lambda r, e: r[e]["separability"]["accuracy"]),
        ("Leakage", lambda r, e: r[e].get("leakage_frac", 0)),
    ]

    # Dati: righe = metriche × energie, colonne = modelli
    row_labels = []
    data = []
    for metric_name, fn in metric_defs:
        for e in energies:
            row_labels.append(f"{metric_name} ({e})")
            row_data = []
            for m in models:
                try:
                    row_data.append(fn(reports[m], e))
                except (KeyError, TypeError):
                    row_data.append(np.nan)
            data.append(row_data)

    data = np.array(data)

    # Normalizzazione per colori relativi
    data_norm = np.zeros_like(data)
    for i in range(len(data)):
        row = data[i]
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            vmin, vmax = valid.min(), valid.max()
            if vmax > vmin:
                data_norm[i] = (row - vmin) / (vmax - vmin)
            else:
                data_norm[i] = 0.5

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(models)),
                                     max(4, 0.6 * len(row_labels))))
    im = ax.imshow(data_norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(row_labels)):
        for j in range(len(models)):
            v = data[i, j]
            text = f"{v:.4f}" if not np.isnan(v) else "N/A"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=8, color="black", fontweight="bold")

    ax.set_title("Confronto Condizionale — Metriche per Energia\n"
                 "(verde = migliore, rosso = peggiore)", fontsize=11)
    plt.colorbar(im, ax=ax, label="Score relativo")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap salvata: {save_path}")


def print_summary_table(reports: dict):
    """Stampa tabella riassuntiva."""
    energies = ["6MV", "10MV"]
    models = list(reports.keys())

    print(f"\n{'='*80}")
    print(f"  {'Modello':<8} {'Energia':<8} {'W1_mean':>10} {'W1_E':>10} "
          f"{'MMD²':>12} {'Sep':>8} {'Leakage':>10}")
    print(f"  {'-'*72}")

    for m in models:
        r = reports[m]
        for e in energies:
            try:
                w1_mean = r[e]["w1"]["mean"]
                w1_E = r[e]["w1"]["E"]
                mmd = r[e]["mmd"]
                sep = r[e]["separability"]["accuracy"]
                leakage = r[e].get("leakage_frac", 0)
                print(f"  {m:<8} {e:<8} {w1_mean:>10.6f} {w1_E:>10.6f} "
                      f"{mmd:>12.6f} {sep:>8.4f} {leakage:>10.4%}")
            except (KeyError, TypeError):
                print(f"  {m:<8} {e:<8} {'N/A':>10} {'N/A':>10} "
                      f"{'N/A':>12} {'N/A':>8} {'N/A':>10}")
        print(f"  {'-'*72}")

    print(f"{'='*80}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Confronto finale condizionale CFM vs NSF vs GAN")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # Carica report disponibili
    reports = {}
    model_dirs = {
        "CFM": (args.cfm_dir, "cfm"),
        "NSF": (args.nsf_dir, "nsf"),
        "GAN": (args.gan_dir, "gan"),
    }

    for display_name, (run_dir, model_name) in model_dirs.items():
        if run_dir is None:
            continue
        r = load_report(run_dir, model_name)
        if r is not None:
            reports[display_name] = r
            print(f"  ✓ {display_name}: report caricato da {run_dir}")

    if not reports:
        print("\n  [ERROR] Nessun report trovato. Eseguire prima eval_conditional_real.py.")
        sys.exit(1)

    # Stampa tabella
    print_summary_table(reports)

    # Plot
    if len(reports) >= 2:
        plot_comparison_barplot(reports, str(output_dir / "comparison_barplot.png"))
        plot_comparison_heatmap(reports, str(output_dir / "comparison_heatmap.png"))
    else:
        print("  [INFO] Serve almeno 2 modelli per il confronto visuale.")

    # Salva report unificato
    unified_path = output_dir / "comparison_report.json"
    with open(unified_path, "w") as f:
        json.dump(reports, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\n  ✓ Report unificato: {unified_path}")


if __name__ == "__main__":
    main()
