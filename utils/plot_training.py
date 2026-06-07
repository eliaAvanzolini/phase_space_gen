"""
utils/plot_training.py
======================
Visualizzazione delle curve di training e comparazione finale dei modelli.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_training_history(
    history: dict,
    model_name: str,
    save_path: str,
) -> None:
    """
    Plotta le curve di training e validation loss nel tempo.

    Funziona per tutti e tre i modelli:
        - NSF:  train_nll / val_nll
        - CFM:  train_loss / val_loss
        - GAN:  loss_G / loss_D / w_dist / gp
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training History — {model_name.upper()}", fontsize=13, fontweight="bold")

    # Loss principale
    ax = axes[0]
    if "train_nll" in history:
        ax.plot(history["train_nll"], label="Train NLL", color="steelblue", alpha=0.8)
        if "val_nll" in history and history["val_nll"]:
            ax.plot(history["val_nll"], label="Val NLL", color="darkorange",
                    linewidth=2, marker="o", markersize=3)
        ax.set_ylabel("Negative Log-Likelihood")
    elif "train_loss" in history:
        ax.plot(history["train_loss"], label="Train Loss", color="steelblue", alpha=0.8)
        if "val_loss" in history and history["val_loss"]:
            ax.plot(history["val_loss"], label="Val Loss", color="darkorange",
                    linewidth=2, marker="o", markersize=3)
        ax.set_ylabel("CFM Loss (MSE)")
    elif "w_dist" in history:
        ax.plot(history["w_dist"], label="Wasserstein Estimate", color="steelblue", alpha=0.8)
        ax.set_ylabel("Wasserstein Distance")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate / GAN losses
    ax = axes[1]
    if "lr" in history and history["lr"]:
        ax.semilogy(history["lr"], color="green", label="Learning Rate")
        ax.set_ylabel("Learning Rate")
        ax.set_xlabel("Epoch")
        ax.legend()
    elif "loss_G" in history:
        ax.plot(history["loss_G"], label="G Loss", color="steelblue", alpha=0.7)
        ax.plot(history["loss_D"], label="D Loss", color="darkorange", alpha=0.7)
        if "gp" in history:
            ax.plot(history["gp"], label="GP", color="green", alpha=0.7, linestyle="--")
        ax.set_ylabel("GAN Losses")
        ax.set_xlabel("Step")
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training history salvata: {save_path}")


def plot_comparison_table(
    reports: Dict[str, dict],
    save_path: str,
) -> None:
    """
    Tabella visuale di comparazione finale tra modelli.
    Ogni colore indica la qualità relativa (verde = meglio, rosso = peggio).
    """
    models  = list(reports.keys())
    metrics = [
        ("W1 (mean)",     lambda r: r["w1"]["mean"],             True),
        ("W1 (E)",        lambda r: r["w1"]["E"],                True),
        ("W1 (x)",        lambda r: r["w1"]["x"],                True),
        ("MMD^2",         lambda r: r["mmd"],                    True),
        ("Separability",  lambda r: r["separability"]["accuracy"],True),
        ("Tail W1 (E)",   lambda r: r["tail_w1"].get("E", np.nan), True),
    ]

    data   = np.zeros((len(metrics), len(models)))
    labels = [m[0] for m in metrics]

    for j, model in enumerate(models):
        r = reports[model]
        for i, (_, fn, _) in enumerate(metrics):
            try:
                data[i, j] = fn(r)
            except Exception:
                data[i, j] = np.nan

    # Normalizza ogni riga per colori relativi
    data_norm = np.zeros_like(data)
    for i in range(len(metrics)):
        row = data[i]
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            vmin, vmax = valid.min(), valid.max()
            if vmax > vmin:
                data_norm[i] = (row - vmin) / (vmax - vmin)
            else:
                data_norm[i] = 0.5

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(models)), max(4, 0.6 * len(metrics))))
    im = ax.imshow(data_norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(metrics)))
    ax.set_xticklabels([m.upper() for m in models], fontsize=11, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=10)

    # Valori nelle celle
    for i in range(len(metrics)):
        for j in range(len(models)):
            v = data[i, j]
            text = f"{v:.4f}" if not np.isnan(v) else "N/A"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=9, color="black", fontweight="bold")

    ax.set_title("Confronto Modelli — Metriche di Validazione\n"
                 "(verde = migliore, rosso = peggiore)", fontsize=12)
    plt.colorbar(im, ax=ax, label="Score relativo (0=best, 1=worst)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Tabella comparazione salvata: {save_path}")


def plot_energy_spectrum(
    real: np.ndarray,
    generated_dict: Dict[str, np.ndarray],
    save_path: str,
    n_subsample: int = 50_000,
) -> None:
    """
    Plot dettagliato dello spettro energetico — la distribuzione più critica
    dal punto di vista fisico (le GAN tipicamente falliscono sulle code).
    """
    rng = np.random.default_rng(0)
    n_r = min(n_subsample, len(real))
    real_E = real[rng.choice(len(real), n_r, replace=False), 6]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Spettro Energetico (E [MeV]) — Scala lineare e logaritmica",
                 fontsize=12, fontweight="bold")

    bins = np.linspace(real_E.min() * 0.95, real_E.max() * 1.02, 100)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(generated_dict) + 1))

    for ax_idx, (ax, yscale) in enumerate(zip(axes, ["linear", "log"])):
        ax.hist(real_E, bins=bins, density=True, alpha=0.6,
                color=colors[0], label="MC (reale)", histtype="stepfilled")

        for j, (name, gen) in enumerate(generated_dict.items()):
            n_g = min(n_subsample, len(gen))
            gen_E = gen[rng.choice(len(gen), n_g, replace=False), 6]
            ax.hist(gen_E, bins=bins, density=True, alpha=0.8,
                    color=colors[j+1], label=name, histtype="step", linewidth=2)

        ax.set_xlabel("E [MeV]", fontsize=11)
        ax.set_ylabel("Densità", fontsize=11)
        ax.set_yscale(yscale)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Scala {yscale}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Spettro energetico salvato: {save_path}")
