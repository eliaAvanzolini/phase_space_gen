"""
evaluate.py
===========
Pipeline di valutazione per modelli generativi di phase space.

Metriche implementate:
    1. Wasserstein-1 (W1) per ogni dimension marginale
    2. MMD (Maximum Mean Discrepancy) con kernel RBF sulla distribuzione 7D
    3. Separability score con classificatore Random Forest
    4. Statistiche sulle code (> 2 sigma) per diagnosticare mode collapse
    5. Plot delle distribuzioni marginali (pdf + CDFs) con stile 2x3 pastello

Dipendenze: numpy, scipy, sklearn, matplotlib — nessun torch richiesto.
"""

import numpy as np
import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")  # non-interactive per server
import matplotlib.pyplot as plt


# ─── Nomi dei canali ──────────────────────────────────────────────────────────
CHANNEL_NAMES  = ["x [cm]", "y [cm]", "z [cm]", "dx", "dy", "dz", "E [MeV]"]
CHANNEL_LABELS = ["x", "y", "z", "dx", "dy", "dz", "E"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Wasserstein-1 distance (per dimensione marginale)
# ═══════════════════════════════════════════════════════════════════════════════

def wasserstein1_marginals(
    real: np.ndarray,
    generated: np.ndarray,
) -> Dict[str, float]:
    """
    Calcola W1 per ogni dimensione marginale.

    W1(p, q) = integral |CDF_p(x) - CDF_q(x)| dx

    Returns
    -------
    dict {channel_label: W1_value}
    """
    assert real.shape[1] == 7 and generated.shape[1] == 7
    results = {}
    for i, label in enumerate(CHANNEL_LABELS):
        w1 = wasserstein_distance(real[:, i], generated[:, i])
        results[label] = float(w1)
    results["mean"] = float(np.mean(list(results.values())))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MMD con kernel RBF (distribuzione congiunta 7D)
# ═══════════════════════════════════════════════════════════════════════════════

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Kernel RBF: k(x, y) = exp(-||x-y||^2 / (2*sigma^2))"""
    diff = X[:, None, :] - Y[None, :, :]          # (n, m, d)
    sq_dist = np.sum(diff**2, axis=-1)             # (n, m)
    return np.exp(-sq_dist / (2 * sigma**2))


def mmd_rbf(
    real: np.ndarray,
    generated: np.ndarray,
    n_subsample: int = 2000,
    sigma: Optional[float] = None,
    seed: int = 0,
) -> float:
    """
    Maximum Mean Discrepancy con kernel RBF.

    MMD^2(p, q) = E_p[k(x,x')] - 2*E_{p,q}[k(x,y)] + E_q[k(y,y')]

    Il sigma viene scelto automaticamente con la median heuristic se non
    specificato: sigma = median(||xi - xj||) / sqrt(2).

    Per N grande usa un subsample per efficienza computazionale.
    """
    rng = np.random.default_rng(seed)
    n = min(n_subsample, len(real), len(generated))

    idx_r = rng.choice(len(real),      size=n, replace=False)
    idx_g = rng.choice(len(generated), size=n, replace=False)
    X = real[idx_r]
    Y = generated[idx_g]

    # Standardizzazione per rendere le scale confrontabili
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    Y = scaler.transform(Y)

    # Median heuristic per sigma
    if sigma is None:
        all_pts = np.concatenate([X, Y])
        idx1 = rng.choice(len(all_pts), size=min(500, len(all_pts)), replace=False)
        idx2 = rng.choice(len(all_pts), size=min(500, len(all_pts)), replace=False)
        diffs = all_pts[idx1] - all_pts[idx2]
        median_dist = np.median(np.sqrt(np.sum(diffs**2, axis=1)))
        sigma = max(median_dist / np.sqrt(2), 1e-3)

    Kxx = _rbf_kernel(X, X, sigma)
    Kyy = _rbf_kernel(Y, Y, sigma)
    Kxy = _rbf_kernel(X, Y, sigma)

    # Rimuovi termini diagonali (bias correction)
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    mmd2 = (Kxx.sum() / (n * (n-1))
            + Kyy.sum() / (n * (n-1))
            - 2 * Kxy.mean())
    return float(max(mmd2, 0.0))  # garantisce non-negatività numerica


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Separability score
# ═══════════════════════════════════════════════════════════════════════════════

def separability_score(
    real: np.ndarray,
    generated: np.ndarray,
    n_subsample: int = 5000,
    seed: int = 42,
    drop_cols: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Addestra un Random Forest a distinguere campioni reali da generati.

    Score = accuracy di classificazione binaria (5-fold CV).
    Interpretazione:
        0.50 → distribuzioni indistinguibili (modello perfetto)
        1.00 → facilmente distinguibili (modello fallisce)

    drop_cols : indici di colonna da escludere prima del fit (es.
        colonne costanti come z, vedi _detect_zero_variance_columns).
        Necessario perché lo StandardScaler dividerebbe per uno std
        quasi-zero su una colonna costante, amplificando un residuo
        numerico float32 insignificante in una feature perfettamente
        discriminante — un falso leakage che non riflette la qualità
        reale del modello sulle variabili fisiche.

    Returns
    -------
    {accuracy, std, n_real, n_gen}
    """
    rng = np.random.default_rng(seed)
    n = min(n_subsample, len(real), len(generated))

    idx_r = rng.choice(len(real),      size=n, replace=False)
    idx_g = rng.choice(len(generated), size=n, replace=False)

    X = np.concatenate([real[idx_r], generated[idx_g]])
    y = np.concatenate([np.ones(n), np.zeros(n)])

    if drop_cols:
        keep_cols = [i for i in range(X.shape[1]) if i not in drop_cols]
        X = X[:, keep_cols]

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                 random_state=seed, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy")

    return {
        "accuracy": float(scores.mean()),
        "std":      float(scores.std()),
        "n_real":   n,
        "n_gen":    n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Analisi delle code
# ═══════════════════════════════════════════════════════════════════════════════

def tail_wasserstein(
    real: np.ndarray,
    generated: np.ndarray,
    sigma_threshold: float = 2.0,
) -> Dict[str, float]:
    """
    W1 calcolata solo sulle code della distribuzione (> sigma_threshold sigma).

    Questa è la metrica più importante per diagnosticare il mode collapse
    delle GAN: le code fisiche (fotoni ad alta energia, angoli estremi)
    sono spesso le più mal riprodotte.
    """
    results = {}
    for i, label in enumerate(CHANNEL_LABELS):
        mu   = real[:, i].mean()
        sig  = real[:, i].std()
        # Seleziona solo i campioni nelle code
        mask_real = np.abs(real[:, i] - mu) > sigma_threshold * sig
        mask_gen  = np.abs(generated[:, i] - mu) > sigma_threshold * sig

        if mask_real.sum() < 10 or mask_gen.sum() < 10:
            results[label] = np.nan
            continue

        w1_tail = wasserstein_distance(
            real[mask_real, i],
            generated[mask_gen, i]
        )
        results[label] = float(w1_tail)
        results[f"{label}_frac_real"] = float(mask_real.mean())
        results[f"{label}_frac_gen"]  = float(mask_gen.mean())

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Kolmogorov-Smirnov test
# ═══════════════════════════════════════════════════════════════════════════════

def ks_test_marginals(
    real: np.ndarray,
    generated: np.ndarray,
) -> Dict[str, dict]:
    """
    Test KS per ogni dimensione marginale.
    p-value > 0.05 significa che l'ipotesi nulla (stessa distribuzione)
    non è rifiutata al 5% — il modello è statisticamente compatibile.
    """
    results = {}
    for i, label in enumerate(CHANNEL_LABELS):
        stat, pval = ks_2samp(real[:, i], generated[:, i])
        results[label] = {"statistic": float(stat), "p_value": float(pval)}
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5b. Auto-fix data leakage su colonne a varianza zero (es. z costante)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_zero_variance_columns(
    real: np.ndarray,
    generated: np.ndarray,
    tol: float = 1e-3,
    verbose: bool = True,
) -> List[int]:
    """
    Rileva le colonne costanti (varianza ~0) in ENTRAMBI gli array reale
    e generato — tipicamente z, il piano isocentrico fisso del fascio.

    Motivazione: una costante geometrica come z può differire tra real
    e generated solo per rumore numerico residuo di precisione float32
    (es. 27.210000 vs 27.210005). Una differenza così piccola è
    fisicamente nulla, ma se la colonna entra nel separability_score,
    lo StandardScaler la normalizza dividendo per uno std quasi-zero
    (es. 1e-6) — questo AMPLIFICA il residuo numerico fino a renderlo
    una feature perfettamente discriminante per il Random Forest,
    anche dopo aver "allineato" i due valori (l'allineamento stesso
    introduce un nuovo arrotondamento float32 leggermente diverso).

    L'unica soluzione robusta è ESCLUDERE la colonna dal calcolo del
    separability score, non provare ad allinearla numericamente.
    Per le altre metriche (W1, MMD, tail, KS) la colonna non va
    esclusa: lì il contributo di una costante è già ~0 per
    costruzione (nessuno scaler divisivo nel mezzo), quindi restano
    informative e comparabili allo stato originale.

    Returns
    -------
    Lista degli indici di colonna rilevati come costanti in entrambi.
    """
    const_cols = []
    n_cols = real.shape[1]
    for i in range(n_cols):
        real_std = float(real[:, i].std())
        gen_std  = float(generated[:, i].std())
        if real_std < tol and gen_std < tol:
            const_cols.append(i)
            if verbose:
                label = CHANNEL_LABELS[i] if i < len(CHANNEL_LABELS) else f"col{i}"
                print(f"  [Auto-Detect] Colonna '{label}' costante in entrambi "
                      f"(std_real={real_std:.2e}, std_gen={gen_std:.2e}) — "
                      f"verrà esclusa dal separability score per evitare "
                      f"data leakage numerico (lo StandardScaler "
                      f"amplificherebbe il residuo float32 a una feature "
                      f"perfettamente discriminante).")
    return const_cols


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Plot distribuzioni marginali (Stile Elegante 2x3 Pastello)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_marginals(
    real: np.ndarray,
    generated_dict: Dict[str, np.ndarray],
    save_path: str,
    n_subsample: int = 50_000,
    n_bins: int = 100,
    title: Optional[str] = None,
) -> None:
    """
    Confronto visivo delle distribuzioni 1D marginali in un layout compatto 2x3.
    Esclude la variabile 'z' (costante) per replicare perfettamente lo stile richiesto.
    """
    rng = np.random.default_rng(0)
    n_r = min(n_subsample, len(real))
    idx_r = rng.choice(len(real), n_r, replace=False)
    real_sub = real[idx_r]

    # Creazione griglia 2x3 identica al file di riferimento
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))

    color_phsp = "#5b84c4"  # Blu pastello (PHSP)
    color_cfm  = "#e69a6a"  # Arancione pastello (CFM)
    alpha_val  = 0.65       # Trasparenza ottimale per la sovrapposizione

    # Mapping esatto: [Indice canale, Etichetta asse X, Asse di riferimento]
    # Ordine riga 1: E, x, y  | Ordine riga 2: dx, dy, dz
    # Indici originali: x=0, y=1, z=2, dx=3, dy=4, dz=5, E=6
    plot_mapping = [
        {"idx": 6, "label": "E [MeV]", "ax": axes[0, 0]},
        {"idx": 0, "label": "x [cm]",  "ax": axes[0, 1]},
        {"idx": 1, "label": "y [cm]",  "ax": axes[0, 2]},
        {"idx": 3, "label": "dx",       "ax": axes[1, 0]},
        {"idx": 4, "label": "dy",       "ax": axes[1, 1]},
        {"idx": 5, "label": "dz",       "ax": axes[1, 2]}
    ]

    for item in plot_mapping:
        i = item["idx"]
        ax = item["ax"]

        # Intervallo basato sui percentili del reale per tagliare code vuote estreme
        lo, hi = np.percentile(real_sub[:, i], [0.5, 99.5])
        bins = np.linspace(lo, hi, n_bins)

        # Plot Ground Truth (PHSP)
        ax.hist(real_sub[:, i], bins=bins, density=True, color=color_phsp,
                alpha=alpha_val, histtype="stepfilled", edgecolor="none", label="PHSP")

        # Plot Modelli Generati (CFM)
        for j, (model_name, gen) in enumerate(generated_dict.items()):
            n_g = min(n_subsample, len(gen))
            idx_g = rng.choice(len(gen), n_g, replace=False)
            
            # Se c'è solo un modello usa l'arancione fisso, altrimenti scala con la colormap
            color = color_cfm if len(generated_dict) == 1 else plt.cm.tab10(j / max(1, len(generated_dict)))
            label = model_name
            
            ax.hist(gen[idx_g, i], bins=bins, density=True, color=color,
                    alpha=alpha_val, histtype="stepfilled", edgecolor="none", label=label)

        # Formattazione estetica assi e griglia
        ax.set_xlabel(item["label"], fontsize=10)
        ax.set_ylabel("Counts", fontsize=10)
        ax.set_xlim([lo, hi])
        ax.grid(True, linestyle='-', alpha=0.2, color='gray')
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
        
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot salvato: {save_path}")


def plot_2d_projection(
    real: np.ndarray,
    generated_dict: Dict[str, np.ndarray],
    save_path: str,
    dims: Tuple[int, int] = (0, 1),
    n_subsample: int = 20_000,
) -> None:
    """
    Confronto density plot 2D tra modello e ground truth per due dimensioni.
    Default: proiezione (x, y) del piano del fascio.
    """
    rng  = np.random.default_rng(0)
    dim1, dim2 = dims
    n_models   = len(generated_dict)

    fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 4.5))
    if n_models == 0:
        axes = [axes]

    def _plot_one(ax, data, title, cmap="viridis"):
        n = min(n_subsample, len(data))
        idx = rng.choice(len(data), n, replace=False)
        ax.hexbin(data[idx, dim1], data[idx, dim2],
                  gridsize=60, cmap=cmap, mincnt=1)
        ax.set_xlabel(CHANNEL_NAMES[dim1], fontsize=10)
        ax.set_ylabel(CHANNEL_NAMES[dim2], fontsize=10)
        ax.set_title(title, fontsize=11)

    _plot_one(axes[0], real, "MC (reale)", cmap="Blues")
    for ax, (name, gen) in zip(axes[1:], generated_dict.items()):
        _plot_one(ax, gen, name, cmap="Reds")

    plt.suptitle(
        f"Proiezione 2D: {CHANNEL_NAMES[dim1]} vs {CHANNEL_NAMES[dim2]}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot 2D salvato: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Funzione di valutazione completa
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    real: np.ndarray,
    generated: np.ndarray,
    model_name: str = "model",
    output_dir: str = "outputs/eval",
    verbose: bool = True,
) -> Dict:
    """
    Esegue tutta la pipeline di valutazione e restituisce un report completo.

    Returns
    -------
    report : dizionario con tutte le metriche
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*55}")
        print(f"  Valutazione: {model_name}")
        print(f"  Campioni reali:    {len(real):>10,}")
        print(f"  Campioni generati: {len(generated):>10,}")
        print(f"{'='*55}")

    report = {"model": model_name, "n_real": len(real), "n_gen": len(generated)}

    # Rileva colonne costanti in entrambi gli array (es. z) — verranno
    # escluse SOLO dal separability score (dove lo StandardScaler
    # amplificherebbe un residuo numerico float32 in un falso leakage).
    # Le altre metriche (W1, MMD, tail, KS) restano sui dati originali:
    # lì una colonna costante contribuisce ~0 senza bisogno di esclusione.
    if verbose: print("\n[0/4] Controllo data leakage su colonne costanti...")
    const_cols = _detect_zero_variance_columns(real, generated, verbose=verbose)

    # 1. W1 marginali
    if verbose: print("\n[1/4] Wasserstein-1 marginali...")
    w1 = wasserstein1_marginals(real, generated)
    report["w1"] = w1
    if verbose:
        for k, v in w1.items():
            if k != "mean": print(f"  W1({k:>3s}) = {v:.6f}")
        print(f"  W1(mean) = {w1['mean']:.6f}")

    # 2. MMD
    if verbose: print("\n[2/4] MMD (RBF kernel, 7D)...")
    mmd = mmd_rbf(real, generated)
    report["mmd"] = mmd
    if verbose: print(f"  MMD^2 = {mmd:.6f}  (sqrt = {np.sqrt(mmd):.6f})")

    # 3. Separability (colonne costanti escluse per evitare leakage)
    if verbose: print("\n[3/4] Separability score (Random Forest)...")
    sep = separability_score(real, generated, drop_cols=const_cols)
    report["separability"] = sep
    if verbose:
        print(f"  Accuracy = {sep['accuracy']:.4f} ± {sep['std']:.4f}")
        print(f"  (0.50 = perfetto, 1.00 = fallimento totale)")

    # 4. Analisi code
    if verbose: print("\n[4/4] Analisi code (> 2σ)...")
    tails = tail_wasserstein(real, generated)
    report["tail_w1"] = tails
    if verbose:
        for label in CHANNEL_LABELS:
            v = tails.get(label, np.nan)
            frac = tails.get(f"{label}_frac_gen", np.nan)
            if not np.isnan(v):
                print(f"  W1_tail({label:>3s}) = {v:.6f}  | frac_gen = {frac:.3f}")

    # KS test
    ks = ks_test_marginals(real, generated)
    report["ks"] = ks
    if verbose:
        print("\n  KS test p-values (>0.05 = OK):")
        for label, res in ks.items():
            ok = "✓" if res["p_value"] > 0.05 else "✗"
            print(f"  {ok} KS({label:>3s}): stat={res['statistic']:.4f}  p={res['p_value']:.4e}")

    # Salva report JSON
    report_path = output_dir / f"{model_name}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    if verbose: print(f"\n  Report salvato: {report_path}")

    # Generazione dei Plot aggiornati
    plot_marginals(
        real, {model_name: generated},
        str(output_dir / f"{model_name}_marginals.png")
    )
    plot_2d_projection(
        real, {model_name: generated},
        str(output_dir / f"{model_name}_2d_xy.png"), dims=(0, 1)
    )
    plot_2d_projection(
        real, {model_name: generated},
        str(output_dir / f"{model_name}_2d_eangle.png"), dims=(6, 3)
    )

    return report


def compare_models(
    real: np.ndarray,
    models: Dict[str, np.ndarray],
    output_dir: str = "outputs/eval",
) -> Dict[str, Dict]:
    """
    Confronta più modelli e produce un plot di comparazione riassuntivo.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    for name, gen in models.items():
        reports[name] = evaluate_model(real, gen, model_name=name,
                                       output_dir=str(output_dir))

    # Plot marginals comparativo (tutti i modelli insieme nel formato 2x3)
    plot_marginals(real, models,
                   str(output_dir / "comparison_marginals.png"))

    # Tabella riassuntiva
    print(f"\n{'='*65}")
    print(f"{'Modello':<20} {'W1_mean':>10} {'MMD^2':>12} {'Sep.':>10} {'Sep.std':>10}")
    print(f"{'-'*65}")
    for name, r in reports.items():
        print(f"{name:<20} {r['w1']['mean']:>10.6f} {r['mmd']:>12.6f} "
              f"{r['separability']['accuracy']:>10.4f} "
              f"{r['separability']['std']:>10.4f}")
    print(f"{'='*65}")

    return reports


# ═══════════════════════════════════════════════════════════════════════════════
# Valutazione condizionale per-configurazione
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_conditional(
    model,
    stats: dict,
    cond_stats: dict,
    configs: dict,
    n_samples_per_config: int = 50_000,
    output_dir: str = "outputs/eval_conditional",
    device: str = "cpu",
) -> dict:
    """
    Valuta un CFM condizionato config per config.
    """
    import torch
    from pathlib import Path
    from data.synthetic_linac import (
        generate_phase_space, DEFAULT_CONFIGS, denormalize_phase_space
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    col_names = stats.get("col_names", ["x", "y", "dx", "dy", "dz", "E"])
    mu_c    = np.array(cond_stats["mu"],    dtype=np.float32)
    sig_c   = np.array(cond_stats["sigma"], dtype=np.float32)

    all_results = {}

    print(f"\n{'='*60}")
    print(f"  Valutazione condizionale per configurazione")
    print(f"  {len(configs)} configurazioni × {n_samples_per_config:,} campioni")
    print(f"{'='*60}")

    summary_rows = []

    for cfg_name, cfg_params in configs.items():
        E_nom = cfg_params["E_nom"]
        jaw_x = cfg_params["jaw_x"]
        jaw_y = cfg_params["jaw_y"]

        print(f"\n  [{cfg_name}] E={E_nom}MV, jaw={jaw_x}×{jaw_y}cm")

        # ── Ground truth sintetico per questa config ──────────────────────
        ps7_real = generate_phase_space(
            n_samples_per_config, E_nom=E_nom,
            jaw_x=jaw_x, jaw_y=jaw_y, seed=999
        )

        # ── Genera campioni condizionati ───────────────────────────────────
        c_raw  = np.array([[E_nom, jaw_x, jaw_y]], dtype=np.float32)
        c_norm = (c_raw - mu_c) / sig_c
        c_t    = torch.from_numpy(
            np.tile(c_norm, (n_samples_per_config, 1))
        ).float().to(device)

        with torch.no_grad():
            model.eval()
            try:
                gen_norm = model.sample(n_samples_per_config, c=c_t).cpu().numpy()
            except Exception:
                gen_norm = model.sample_fast(n_samples_per_config, c=c_t,
                                             n_steps=50).cpu().numpy()

        gen_phys = denormalize_phase_space(gen_norm, stats)

        # ── Metriche ──────────────────────────────────────────────────────
        w1  = wasserstein1_marginals(ps7_real, gen_phys)
        mmd = mmd_rbf(ps7_real, gen_phys, n_subsample=10_000)
        sep = separability_score(ps7_real, gen_phys, n_subsample=10_000)

        print(f"    W1_mean={w1['mean']:.6f}  W1_E={w1['E']:.6f}  "
              f"MMD²={mmd:.6f}  Sep={sep['accuracy']:.4f}")

        all_results[cfg_name] = {"w1": w1, "mmd": mmd, "separability": sep,
                                 "config": cfg_params}
        summary_rows.append((cfg_name, w1["mean"], w1["E"], mmd,
                              sep["accuracy"]))

        # Plot marginali per questa config nel formato 2x3 pulito
        plot_marginals(
            ps7_real, {cfg_name: gen_phys},
            str(output_dir / f"marginals_{cfg_name}.png")
        )

    # Tabella riassuntiva
    print(f"\n{'='*65}")
    print(f"  {'Config':<14} {'W1_mean':>10} {'W1_E':>10} {'MMD²':>12} {'Sep':>8}")
    print(f"  {'-'*57}")
    for row in summary_rows:
        print(f"  {row[0]:<14} {row[1]:>10.6f} {row[2]:>10.6f} "
              f"{row[3]:>12.6f} {row[4]:>8.4f}")

    mean_w1  = np.mean([r[1] for r in summary_rows])
    mean_sep = np.mean([r[4] for r in summary_rows])
    print(f"  {'MEDIA':<14} {mean_w1:>10.6f} {'':>10} {'':>12} {mean_sep:>8.4f}")
    print(f"{'='*65}")

    with open(output_dir / "conditional_eval.json", "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))

    return all_results
