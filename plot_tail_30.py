"""
plot_tail_30.py
===============
Isola e visualizza la coda energetica (E < 5% percentile) sul fold hold-out a 3.0 cm:
  - Confronta TOPAS reale, shift fisico e DeepRangeResQ.
  - Stampa la tabella dei micro-quantili sotto il 5%.
  - Salva la figura 'eval/diagnostica_coda_3.0cm.png' (istogramma zoomato + CDF empirica).
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from deep_ai_1d import DeepRangeResQ, train_deep_resq
from full_ai_1d import quantile_tables, TRAIN6
from train_gen_energy_1d import load_proton_energies, make_quantile_grid, physics_transport

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== CARICAMENTO DATI PER DIAGNOSTICA CODA A 3.0 CM ===")
    E_by = load_proton_energies("data/raw_phsp", set(TRAIN6))
    tau = make_quantile_grid(2048, dev)
    tau_np = tau.cpu().numpy()

    # Fold 3.0 cm (hold-out, sorgente 2.5 cm)
    hold, src = 3.0, 2.5
    train_d = [d for d in TRAIN6 if d != hold]
    q_np, q_t = quantile_tables(E_by, train_d, tau_np, dev)

    E_real = E_by[hold]
    E_src = E_by[src]

    print("\n1. Calcolo campioni Shift Fisico...")
    E_shift = physics_transport(E_src, src, hold, 1.77, 450.0 * 1.01, 0.0)

    print("2. Addestramento DeepRangeResQ sul fold 3.0 cm...")
    torch.manual_seed(42)
    drq = DeepRangeResQ().to(dev)
    train_deep_resq(drq, q_t, tau, warmup=300, epochs=600)
    E_drq = drq.sample_from_source(E_src, src, hold, tau)

    # Soglia del 5% reale
    q5_val = float(np.quantile(E_real, 0.05))
    print(f"\nSoglia 5° percentile reale a 3.0 cm: {q5_val:.2f} MeV")

    # Micro-quantili della coda
    tau_tail = np.array([0.001, 0.005, 0.010, 0.020, 0.030, 0.050])
    q_real = np.quantile(E_real, tau_tail)
    q_shift = np.quantile(E_shift, tau_tail)
    q_drq = np.quantile(E_drq, tau_tail)

    print("\n--- CONFRONTO MICRO-QUANTILI NELLA CODA BASSA [MeV] ---")
    print(f"{'Livello tau':12s} | {'TOPAS Reale':>12s} | {'Shift Fisico':>12s} | {'DeepRangeResQ':>14s} | {'Diff DRQ-Reale':>14s}")
    for i, t in enumerate(tau_tail):
        diff = q_drq[i] - q_real[i]
        print(f"tau = {100*t:4.1f}%   | {q_real[i]:12.3f} | {q_shift[i]:12.3f} | {q_drq[i]:14.3f} | {diff:+14.3f} MeV")

    # Isolamento particelle della coda
    tail_real = np.sort(E_real[E_real <= q5_val])
    tail_shift = np.sort(E_shift[E_shift <= np.quantile(E_shift, 0.05)])
    tail_drq = np.sort(E_drq[E_drq <= np.quantile(E_drq, 0.05)])

    # Generazione Grafico Diagnostico
    Path("eval").mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # 1. Istogramma della coda
    bins = np.linspace(min(tail_real.min(), tail_drq.min()), q5_val, 40)
    ax1.hist(tail_real, bins=bins, density=True, histtype="step", color="black", linewidth=2, label="TOPAS Reale")
    ax1.hist(tail_shift, bins=bins, density=True, histtype="step", color="tab:blue", linewidth=1.5, linestyle="--", label="Shift Fisico (KS=0.063)")
    ax1.hist(tail_drq, bins=bins, density=True, histtype="step", color="tab:orange", linewidth=1.5, label="DeepRangeResQ (KS=0.155)")
    ax1.set_title("Densità nella coda (E <= 5° perc.)", fontsize=12)
    ax1.set_xlabel("Energia E [MeV]", fontsize=11)
    ax1.set_ylabel("Densità di probabilità", fontsize=11)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Curve CDF empiriche
    cdf_y = np.linspace(0, 1, len(tail_real))
    ax2.plot(tail_real, cdf_y, color="black", linewidth=2, label="TOPAS Reale")
    ax2.plot(tail_shift, np.linspace(0, 1, len(tail_shift)), color="tab:blue", linewidth=1.5, linestyle="--", label="Shift Fisico")
    ax2.plot(tail_drq, np.linspace(0, 1, len(tail_drq)), color="tab:orange", linewidth=1.5, label="DeepRangeResQ")
    ax2.set_title("CDF empirica standardizzata (E <= 5° perc.)", fontsize=12)
    ax2.set_xlabel("Energia E [MeV]", fontsize=11)
    ax2.set_ylabel("F(E) cumulativa", fontsize=11)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_png = "eval/diagnostica_coda_3.0cm.png"
    plt.savefig(out_png)
    print(f"\nGrafico diagnostico salvato con successo in: {out_png}")

if __name__ == "__main__":
    main()
