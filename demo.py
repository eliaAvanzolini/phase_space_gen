"""
demo.py
=======
Demo eseguibile SENZA PyTorch — valida la pipeline di dati e le metriche.

Cosa fa:
    1. Genera un phase space sintetico (5 configurazioni di linac)
    2. Calcola tutte le metriche di valutazione (W1, MMD, separability)
    3. Produce i plot delle distribuzioni marginali
    4. Simula un "modello imperfetto" (con mode collapse artificiale) per
       verificare che le metriche lo rilevino correttamente

Questo script è pensato per:
    - Verificare che la pipeline funzioni sulla macchina locale
    - Capire visualmente la forma delle distribuzioni del phase space
    - Testare la sensibilità delle metriche prima di avere i modelli addestrati

Esecuzione:
    python demo.py

Output in: outputs/demo/
"""

import sys
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.synthetic_linac import (
    generate_phase_space,
    generate_multi_condition_dataset,
    save_phase_space_hdf5,
    normalize_phase_space,
    DEFAULT_CONFIGS,
)
from evaluate import (
    wasserstein1_marginals,
    mmd_rbf,
    separability_score,
    tail_wasserstein,
    plot_marginals,
    plot_2d_projection,
    compare_models,
)


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def demo_data_generation():
    """Sezione 1: generazione e visualizzazione dei dati."""
    print_section("1. Generazione Phase Space Sintetico")

    out_dir = Path("outputs/demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Singola configurazione ────────────────────────────────────────────
    print("\n  Generazione 6MV 10x10 (200k campioni)...")
    ps_6mv = generate_phase_space(
        n_samples=200_000,
        E_nom=6.0, jaw_x=5.0, jaw_y=5.0,
        seed=42,
    )
    print(f"  Shape: {ps_6mv.shape}")
    print(f"\n  Statistiche per canale:")
    col_names = ["x [cm]", "y [cm]", "z [cm]", "dx", "dy", "dz", "E [MeV]"]
    for i, name in enumerate(col_names):
        col = ps_6mv[:, i]
        print(f"    {name:>10s}: "
              f"mu={col.mean():+7.4f}  "
              f"σ={col.std():7.4f}  "
              f"min={col.min():+8.4f}  "
              f"max={col.max():+8.4f}")

    # Verifica vincolo fisico ||d||=1
    d_norms = np.linalg.norm(ps_6mv[:, 3:6], axis=1)
    print(f"\n  Verifica ||d||=1:")
    print(f"    mean={d_norms.mean():.8f}  "
          f"std={d_norms.std():.2e}  "
          f"max_dev={np.abs(d_norms - 1).max():.2e}")
    assert np.allclose(d_norms, 1.0, atol=1e-5), "ERRORE: vincolo ||d||=1 violato!"
    print(f"    ✓ Vincolo ||d||=1 rispettato")

    # Salva HDF5
    save_phase_space_hdf5(
        ps_6mv, None, str(out_dir / "ps_6mv_10x10.h5"),
        metadata={"E_nom": "6.0", "jaw_x": "5.0", "jaw_y": "5.0", "n": "200000"}
    )
    print(f"\n  Salvato: {out_dir}/ps_6mv_10x10.h5")

    return ps_6mv, out_dir


def demo_multi_condition(out_dir: Path):
    """Sezione 2: dataset multi-condizione."""
    print_section("2. Dataset Multi-Condizione (5 configurazioni)")

    ps_all, c_all, label_map = generate_multi_condition_dataset(
        n_per_config=50_000,
        seed=42,
        save_path=str(out_dir / "ps_multicond.h5"),
    )

    print(f"\n  Dataset totale: {len(ps_all):,} campioni")
    print(f"  Condizioni:     {ps_all.shape}")
    print(f"\n  Configurazioni:")
    for name, idx in label_map.items():
        cfg = DEFAULT_CONFIGS[name]
        mask = np.all(c_all == [cfg["E_nom"], cfg["jaw_x"], cfg["jaw_y"]], axis=1)
        print(f"    {name:>12s} (E={cfg['E_nom']:4.1f}MV, "
              f"jaw={cfg['jaw_x']:4.1f}×{cfg['jaw_y']:4.1f}cm): "
              f"{mask.sum():>8,} campioni")

    return ps_all, c_all


def demo_metrics(ps_real: np.ndarray, out_dir: Path):
    """
    Sezione 3: validazione delle metriche.

    Confronta:
        - Modello "perfetto": campioni dalla stessa distribuzione
        - Modello "imperfetto": simula mode collapse (manca le code di E)
        - Modello "parziale": perturbazione leggera (piccolo bias)
    """
    print_section("3. Validazione delle Metriche")

    rng = np.random.default_rng(99)
    N = min(50_000, len(ps_real))
    real_sub = ps_real[rng.choice(len(ps_real), N, replace=False)]

    # ── Modello "perfetto": altra metà degli stessi dati ─────────────────
    print("\n  Generazione campioni di confronto...")
    perfect = ps_real[rng.choice(len(ps_real), N, replace=False)]

    # ── Modello "mode_collapse": tronca le code di E (> 1σ) ─────────────
    mode_collapse = ps_real.copy()
    E_col = mode_collapse[:, 6]
    E_mu, E_sig = E_col.mean(), E_col.std()
    # Schiaccia le code: tutti i valori a > 1σ vengono portati a 1σ
    E_col_clipped = np.clip(E_col, E_mu - E_sig, E_mu + E_sig)
    mode_collapse[:, 6] = E_col_clipped + rng.normal(0, E_sig * 0.05, len(E_col_clipped))
    mode_collapse = mode_collapse[rng.choice(len(mode_collapse), N, replace=False)]

    # ── Modello "biased": piccolo shift sistematico ────────────────────────
    biased = ps_real.copy()
    biased[:, 0] += 0.3    # shift in x di 3mm
    biased[:, 6] *= 1.05   # energia sovrastimata del 5%
    biased = biased[rng.choice(len(biased), N, replace=False)]

    models = {
        "Perfetto (altro split)": perfect,
        "Mode Collapse (E clip)": mode_collapse,
        "Biased (shift+scale)":  biased,
    }

    print(f"\n  {'Metrica':<30} {'Perfetto':>12} {'Mode Collapse':>15} {'Biased':>12}")
    print(f"  {'-'*72}")

    results = {}
    for name, gen in models.items():
        w1   = wasserstein1_marginals(real_sub, gen)
        mmd  = mmd_rbf(real_sub, gen, n_subsample=5000)
        sep  = separability_score(real_sub, gen, n_subsample=5000)
        tail = tail_wasserstein(real_sub, gen)
        results[name] = {"w1": w1, "mmd": mmd, "sep": sep, "tail": tail}

    # Stampa confronto
    metrics_rows = [
        ("W1 (mean)", lambda r: f"{r['w1']['mean']:.6f}"),
        ("W1 (E)",    lambda r: f"{r['w1']['E']:.6f}"),
        ("MMD^2",     lambda r: f"{r['mmd']:.6f}"),
        ("Sep. acc.", lambda r: f"{r['sep']['accuracy']:.4f}"),
        ("Tail W1 E", lambda r: f"{r['tail'].get('E', float('nan')):.6f}"),
    ]

    vals = list(results.values())
    for label, fn in metrics_rows:
        row = [fn(v) for v in vals]
        print(f"  {label:<30} {row[0]:>12} {row[1]:>15} {row[2]:>12}")

    print(f"\n  Interpretazione:")
    print(f"    W1 perfetto ≈ 0 → modello ottimo")
    print(f"    W1 mode_collapse >> 0 su E → metriche rilevano il problema")
    print(f"    Sep. perfetto ≈ 0.50 → indistinguibile da reale")
    print(f"    Sep. mode_collapse >> 0.50 → facilmente identificabile")

    # Salva risultati
    with open(out_dir / "demo_metrics.json", "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, '__float__') else str(x))

    return real_sub, models


def demo_plots(real: np.ndarray, models: dict, out_dir: Path):
    """Sezione 4: visualizzazioni."""
    print_section("4. Generazione Plot")

    print("\n  Plot distribuzioni marginali (tutti i modelli)...")
    plot_marginals(
        real, models,
        str(out_dir / "comparison_marginals.png"),
        title="Confronto: Reale vs Modelli Generativi (DEMO)",
    )

    print("  Plot proiezione 2D (x, y)...")
    plot_2d_projection(
        real, models,
        str(out_dir / "comparison_2d_xy.png"), dims=(0, 1),
    )

    print("  Plot proiezione 2D (E, theta)...")
    plot_2d_projection(
        real, models,
        str(out_dir / "comparison_2d_E_theta.png"), dims=(6, 3),
    )

    print(f"\n  Plot salvati in: {out_dir}/")


def demo_normalization(ps: np.ndarray, out_dir: Path):
    """Sezione 5: verifica della pipeline di normalizzazione."""
    print_section("5. Pipeline di Normalizzazione")

    ps_norm, stats = normalize_phase_space(ps)

    print(f"\n  {'Canale':<8} {'mu_raw':>10} {'sig_raw':>10} "
          f"{'mu_norm':>10} {'sig_norm':>10}")
    print(f"  {'-'*52}")

    from data.synthetic_linac import denormalize_phase_space
    col_names = stats.get("col_names", ["x", "y", "dx", "dy", "dz", "E"])
    for i, col in enumerate(col_names):
        mu_raw  = stats[f"{col}_mu"]
        sig_raw = stats[f"{col}_sigma"]
        mu_n    = float(ps_norm[:, i].mean())
        sig_n   = float(ps_norm[:, i].std())
        print(f"  {col:<8} {mu_raw:>10.4f} {sig_raw:>10.4f} "
              f"{mu_n:>10.4f} {sig_n:>10.4f}")

    # Verifica invertibilità
    ps_denorm = denormalize_phase_space(ps_norm, stats)
    # Compare only the non-z columns (z is reconstructed as 0, so skip col 2)
    ps_cmp = np.concatenate([ps[:, :2], ps[:, 3:]], axis=1)  # x,y,dx,dy,dz,E
    ps_denorm_cmp = np.concatenate([ps_denorm[:, :2], ps_denorm[:, 3:]], axis=1)
    max_err = np.abs(ps_cmp - ps_denorm_cmp).max()
    print(f"\n  Errore di ricostruzione (normalizza → denormalizza):")
    print(f"    max_err = {max_err:.2e}  (deve essere ~0)")
    assert max_err < 1e-5, f"ERRORE: denormalizzazione imprecisa! max_err={max_err}"
    print(f"    ✓ Pipeline invertibile")

    with open(out_dir / "normalization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def main():
    print("\n" + "╔" + "═"*58 + "╗")
    print("║   Phase Space Generative Models — Demo Pipeline       ║")
    print("║   Fisica Medica MC — Università                       ║")
    print("╚" + "═"*58 + "╝")

    print("\n  Dipendenze: numpy, scipy, sklearn, matplotlib, h5py")
    print("  PyTorch NON richiesto per questo demo\n")

    # Verifica imports
    try:
        import scipy, sklearn, matplotlib, h5py
        print("  ✓ Tutte le dipendenze disponibili")
    except ImportError as e:
        print(f"  ✗ Dipendenza mancante: {e}")
        print("  Installare con: pip install scipy scikit-learn matplotlib h5py")
        sys.exit(1)

    # Esegui le sezioni
    ps_real, out_dir = demo_data_generation()
    demo_normalization(ps_real, out_dir)
    ps_all, c_all = demo_multi_condition(out_dir)
    real_sub, models = demo_metrics(ps_real, out_dir)
    demo_plots(real_sub, models, out_dir)

    print("\n" + "╔" + "═"*58 + "╗")
    print("║   Demo completato con successo!                       ║")
    print("╚" + "═"*58 + "╝")
    print(f"\n  Output in: outputs/demo/")
    print(f"    - ps_6mv_10x10.h5         (phase space HDF5)")
    print(f"    - ps_multicond.h5          (multi-config)")
    print(f"    - demo_metrics.json        (W1, MMD, separability)")
    print(f"    - comparison_marginals.png (distribuzioni 1D)")
    print(f"    - comparison_2d_*.png      (proiezioni 2D)")
    print(f"    - normalization_stats.json (statistiche normaliz.)")
    print(f"\n  Prossimo step: eseguire con PyTorch disponibile:")
    print(f"    python train.py --model cfm --conditional")
    print(f"    python train.py --model nsf --n_samples 1000000")
    print(f"    python train.py --model gan  # baseline")


if __name__ == "__main__":
    main()
