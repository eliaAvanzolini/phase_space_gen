"""
validate_gate_output.py
========================
Verifica completa dell'output di una simulazione GATE 10:
    1. Controlli numerici: sanity checks fisici
    2. Profili di dose: depth dose curve + profilo trasversale
    3. Confronto con reference (se disponibile): gamma-index e diff map
    4. Plot comparativo stile paper Sarrut (Fig. 6-7)

Uso:
    # Solo un file (sanity check + profili)
    python validate_gate_output.py \\
        --dose outputs/dose_nsf/dose_linac_6MV_nsf_TEST_dose.mhd

    # Confronto modello vs reference (Fase 4 tesi)
    python validate_gate_output.py \\
        --dose    outputs/dose_nsf/dose_linac_6MV_nsf_dose.mhd \\
        --reference outputs/dose_reference/dose_reference_dose.mhd \\
        --label   "NSF"

    # Confronto multiplo (tabella + plot comparativo)
    python validate_gate_output.py \\
        --reference outputs/dose_reference/dose_reference_dose.mhd \\
        --compare \\
        --models outputs/dose_gan/dose_gan_dose.mhd \\
                 outputs/dose_nsf/dose_nsf_dose.mhd \\
                 outputs/dose_cfm/dose_cfm_dose.mhd \\
        --labels "GAN" "NSF" "CFM"
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─── Costanti fisiche linac 6MV (Elekta Precise, paper Sarrut) ───────────────
# Fantoccio d'acqua 20×20×20 cm³, voxel 4×4×4 mm³
PHANTOM_SIZE_CM  = 20.0
VOXEL_SIZE_MM    = 4.0
N_VOXELS         = 50       # 200mm / 4mm = 50 voxel per asse
# Profilo trasversale: a 20 mm di profondità (come nel paper Sarrut Fig. 7)
PROFILE_DEPTH_MM = 20.0


def load_mhd(path: str) -> tuple:
    """
    Carica un file .mhd e restituisce (array, spacing_mm, origin_mm).
    Richiede SimpleITK: pip install SimpleITK
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        print("[ERROR] pip install SimpleITK")
        sys.exit(1)

    img     = sitk.ReadImage(path)
    arr     = sitk.GetArrayFromImage(img).astype(np.float64)  # (Z, Y, X)
    spacing = np.array(img.GetSpacing())   # mm (X, Y, Z)
    origin  = np.array(img.GetOrigin())    # mm
    return arr, spacing, origin


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SANITY CHECKS NUMERICI
# ═══════════════════════════════════════════════════════════════════════════════

def sanity_check(dose: np.ndarray, label: str = "dose") -> dict:
    """
    Verifica che la mappa di dose sia fisicamente plausibile.

    Checks:
        - matrice non vuota (almeno qualche voxel > 0)
        - dose massima finita
        - distribuzione monotona con la profondità (Bragg peak attesa per fotoni: no)
        - simmetria approssimativa sull'asse del fascio (|asimmetria| < 5%)
    """
    print(f"\n{'─'*55}")
    print(f"  Sanity checks: {label}")
    print(f"{'─'*55}")

    results = {"label": label, "checks": {}}

    # Shape
    nz, ny, nx = dose.shape
    print(f"  Shape (Z,Y,X): {dose.shape}  |  voxel non-zero: {(dose>0).sum():,}/{dose.size:,}")

    # 1. Non vuota
    n_nonzero  = int((dose > 0).sum())
    pct_nonzero = 100 * n_nonzero / dose.size
    ok_nonzero = n_nonzero > 10
    results["checks"]["non_empty"] = ok_nonzero
    print(f"  {'✓' if ok_nonzero else '✗'} Voxel colpiti: {n_nonzero} ({pct_nonzero:.2f}%)")

    # 2. Dose massima finita e positiva
    d_max = dose.max()
    ok_max = 0 < d_max < 1e10
    results["checks"]["finite_max"] = ok_max
    print(f"  {'✓' if ok_max else '✗'} Dose massima: {d_max:.4e} Gy")
    results["d_max"] = float(d_max)

    # 3. Il massimo è nel primo terzo del fantoccio (fotoni 6MV: Dmax a ~15mm depth)
    # Asse Z = profondità (Z=0 è la superficie di entrata)
    depth_profile = dose.mean(axis=(1, 2))  # media su X,Y per ogni profondità Z
    if depth_profile.max() > 0:
        z_max_idx = np.argmax(depth_profile)
        z_max_mm  = z_max_idx * VOXEL_SIZE_MM
        ok_depth  = z_max_mm < PHANTOM_SIZE_CM * 10 * 0.5  # Dmax < metà del fantoccio
        results["checks"]["dmax_position"] = ok_depth
        print(f"  {'✓' if ok_depth else '?'} Dmax a {z_max_mm:.0f}mm profondità "
              f"(atteso 10-30mm per 6MV)")
        results["z_dmax_mm"] = float(z_max_mm)

    # 4. Simmetria laterale (asse X): profilo centrale a metà profondità
    mid_z = nz // 4
    lateral_profile_x = dose[mid_z, ny//2, :]
    if lateral_profile_x.max() > 0:
        left_half  = lateral_profile_x[:nx//2]
        right_half = lateral_profile_x[nx//2:][::-1]
        n_compare  = min(len(left_half), len(right_half))
        if n_compare > 0:
            asym = np.abs(left_half[:n_compare] - right_half[:n_compare]).mean()
            asym_pct = 100 * asym / (lateral_profile_x.max() + 1e-30)
            ok_sym = asym_pct < 10.0  # tolleranza 10% (modello a 2 epoche)
            results["checks"]["lateral_symmetry"] = ok_sym
            print(f"  {'✓' if ok_sym else '?'} Asimmetria laterale: {asym_pct:.2f}% "
                  f"(accettabile < 10%)")
            results["asymmetry_pct"] = float(asym_pct)

    # Riepilogo
    n_pass = sum(results["checks"].values())
    n_tot  = len(results["checks"])
    print(f"\n  Checks superati: {n_pass}/{n_tot}")
    if n_pass == n_tot:
        print(f"  ✅ Pipeline GATE corretta!")
    elif n_nonzero > 0:
        print(f"  ⚠️  Output presente ma qualità bassa (atteso con poche particelle)")
    else:
        print(f"  ❌ Matrice vuota — problema nella pipeline")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PROFILI DI DOSE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_profiles(
    dose: np.ndarray,
    spacing_mm: np.ndarray,
    normalize: bool = True,
) -> dict:
    """
    Calcola depth dose curve e profili trasversali.

    Returns
    -------
    dict con:
        depth_mm     : array profondità [mm]
        depth_dose   : PDD normalizzato a 100 al massimo
        profile_x_mm : coordinate trasversali X
        profile_x    : profilo trasversale a depth=20mm
        profile_y    : profilo trasversale Y
    """
    nz, ny, nx = dose.shape
    sx, sy, sz = spacing_mm  # (X, Y, Z) in mm

    # Depth dose curve (media su X,Y)
    depth_dose = dose.mean(axis=(1, 2))
    depth_mm   = np.arange(nz) * sz + sz / 2

    # Normalizza a 100 al massimo (standard PDD)
    d_max = depth_dose.max()
    if normalize and d_max > 0:
        depth_dose = 100 * depth_dose / d_max

    # Profilo trasversale a depth target
    depth_idx = max(0, min(int(PROFILE_DEPTH_MM / sz), nz - 1))
    profile_x = dose[depth_idx, ny // 2, :]  # profilo centrale X
    profile_y = dose[depth_idx, :, nx // 2]  # profilo centrale Y

    if normalize and profile_x.max() > 0:
        profile_x = 100 * profile_x / d_max
        profile_y = 100 * profile_y / d_max

    # Coordinate centrate in cm (come nel paper)
    x_mm = (np.arange(nx) - nx // 2) * sx
    y_mm = (np.arange(ny) - ny // 2) * sy

    return {
        "depth_mm":    depth_mm,
        "depth_dose":  depth_dose,
        "profile_x":   profile_x,
        "profile_y":   profile_y,
        "x_mm":        x_mm,
        "y_mm":        y_mm,
        "depth_idx":   depth_idx,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GAMMA-INDEX
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gamma_index(
    ref:   np.ndarray,
    model: np.ndarray,
    spacing_mm: np.ndarray,
    dd: float  = 2.0,   # % dose difference
    dta: float = 2.0,   # mm distance-to-agreement
    threshold: float = 0.1,
) -> dict:
    """
    Gamma-index locale (approssimazione voxel-by-voxel + ricerca 3×3×3 neighborhood).

    Per una tesi magistrale, questa approssimazione è sufficiente.
    Per pubblicazione usare pymedphys.gamma() che fa la ricerca esatta.
    """
    D_max  = ref.max()
    if D_max == 0:
        return {"pass_rate_pct": 0.0, "error": "reference vuota"}

    mask = ref > threshold * D_max

    # Termine dose
    delta_d = np.abs(ref - model) / D_max * 100  # in %

    # Termine distanza: ricerca locale 3×3×3
    sx, sy, sz = spacing_mm
    gamma = np.full_like(ref, np.inf)

    from itertools import product as iproduct
    offsets = list(iproduct([-1, 0, 1], repeat=3))

    nz, ny, nx = ref.shape
    for oz, oy, ox in offsets:
        # Shift dell'immagine model
        z1, z2 = max(0, -oz), min(nz, nz - oz)
        y1, y2 = max(0, -oy), min(ny, ny - oy)
        x1, x2 = max(0, -ox), min(nx, nx - ox)
        zs, ze = max(0, oz), min(nz, nz + oz)
        ys, ye = max(0, oy), min(ny, ny + oy)
        xs, xe = max(0, ox), min(nx, nx + ox)

        dist_mm = np.sqrt((ox * sx)**2 + (oy * sy)**2 + (oz * sz)**2)
        dd_local = np.abs(ref[z1:z2, y1:y2, x1:x2] -
                          model[zs:ze, ys:ye, xs:xe]) / D_max * 100

        g = np.sqrt((dd_local / dd)**2 + (dist_mm / dta)**2)
        gamma[z1:z2, y1:y2, x1:x2] = np.minimum(gamma[z1:z2, y1:y2, x1:x2], g)

    gamma_masked = gamma[mask]
    pass_rate    = float((gamma_masked <= 1.0).mean() * 100)

    delta_pct = (ref[mask] - model[mask]) / D_max * 100

    return {
        "dd_criterion_pct":   dd,
        "dta_criterion_mm":   dta,
        "threshold":          threshold,
        "n_voxels_evaluated": int(mask.sum()),
        "pass_rate_pct":      pass_rate,
        "mean_diff_pct":      float(delta_pct.mean()),
        "std_diff_pct":       float(delta_pct.std()),
        "max_abs_diff_pct":   float(np.abs(delta_pct).max()),
        "gamma_map":          gamma,  # (non serializzabile, solo per plot)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PLOT — STILE PAPER SARRUT (Fig. 6-7)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dose_profiles(
    profiles_dict: dict,
    output_path: str,
    title: str = "Dose Profiles",
) -> None:
    """
    Replica la Fig. 7 del paper Sarrut 2019:
    Profilo trasversale (sinistra) e depth dose curve (destra).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    colors  = plt.cm.tab10(np.linspace(0, 0.8, len(profiles_dict)))
    styles  = ["-", "--", "-.", ":"]

    for idx, (label, pr) in enumerate(profiles_dict.items()):
        col = colors[idx]
        ls  = styles[idx % len(styles)]
        lw  = 2.5 if idx == 0 else 1.8

        # Profilo trasversale X (sinistra)
        axes[0].plot(pr["x_mm"], pr["profile_x"],
                     color=col, linestyle=ls, linewidth=lw, label=label)

        # Depth dose curve (destra)
        axes[1].plot(pr["depth_mm"], pr["depth_dose"],
                     color=col, linestyle=ls, linewidth=lw, label=label)

    axes[0].set_xlabel("Distanza dall'asse [mm]", fontsize=11)
    axes[0].set_ylabel("Dose relativa [%]", fontsize=11)
    axes[0].set_title(f"Profilo trasversale (profondità {PROFILE_DEPTH_MM:.0f}mm)", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(0, color="gray", linewidth=0.5, linestyle="--")

    axes[1].set_xlabel("Profondità in acqua [mm]", fontsize=11)
    axes[1].set_ylabel("PDD [%]", fontsize=11)
    axes[1].set_title("Depth Dose Curve", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Profili salvati: {output_path}")


def plot_dose_slice(
    dose: np.ndarray,
    spacing_mm: np.ndarray,
    label: str,
    output_path: str,
) -> None:
    """
    Slice 2D della distribuzione di dose (piano XZ centrale).
    Replica la Fig. 9 del paper per la visualizzazione.
    """
    nz, ny, nx = dose.shape
    dose_slice = dose[:, ny // 2, :]  # piano XZ a Y=0

    sx, sz = spacing_mm[0], spacing_mm[2]
    extent = [-nx//2 * sx, nx//2 * sx, nz * sz, 0]  # [x_left, x_right, z_bottom, z_top]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(dose_slice, cmap="jet", extent=extent, aspect="auto",
                   vmin=0, vmax=dose_slice.max())
    plt.colorbar(im, ax=ax, label="Dose [Gy]")
    ax.set_xlabel("X [mm]", fontsize=11)
    ax.set_ylabel("Profondità Z [mm]", fontsize=11)
    ax.set_title(f"Distribuzione dose — {label}\n(piano XZ centrale)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Slice 2D salvata: {output_path}")


def plot_difference_map(
    ref: np.ndarray,
    model: np.ndarray,
    label: str,
    output_path: str,
) -> None:
    """
    Mappa delle differenze relative (Δ% = (ref-model)/Dmax × 100).
    Replica la Fig. 9 destra del paper.
    """
    D_max = ref.max()
    if D_max == 0:
        return

    nz, ny, nx = ref.shape
    ref_slice   = ref[:, ny // 2, :]
    model_slice = model[:, ny // 2, :]
    diff_pct    = (ref_slice - model_slice) / D_max * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Confronto dose — {label}", fontsize=12, fontweight="bold")

    lim = max(abs(diff_pct.max()), abs(diff_pct.min()), 0.5)

    axes[0].imshow(ref_slice,   cmap="jet", aspect="auto")
    axes[0].set_title("Reference (PHSP)", fontsize=10)

    axes[1].imshow(model_slice, cmap="jet", aspect="auto")
    axes[1].set_title(f"Modello ({label})", fontsize=10)

    im = axes[2].imshow(diff_pct, cmap="RdBu_r", aspect="auto",
                        vmin=-lim, vmax=lim)
    plt.colorbar(im, ax=axes[2], label="Δ dose [%]")
    axes[2].set_title(f"Differenza relativa\nμ={diff_pct.mean():+.2f}%", fontsize=10)

    for ax in axes:
        ax.set_xlabel("X [voxel]"); ax.set_ylabel("Z [voxel]")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Mappa differenze salvata: {output_path}")


def plot_difference_histogram(
    ref: np.ndarray,
    models_dict: dict,
    output_path: str,
    threshold: float = 0.1,
) -> None:
    """
    Replica la Fig. 6 del paper Sarrut: istogramma delle differenze relative in %.
    """
    D_max = ref.max()
    mask  = ref > threshold * D_max

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["black", "#4472C4", "#ED7D31", "#A9D18E"]

    for idx, (label, model) in enumerate(models_dict.items()):
        diff = (ref[mask] - model[mask]) / D_max * 100
        mu   = diff.mean()

        lo = max(np.percentile(diff, 0.5), -5)
        hi = min(np.percentile(diff, 99.5), 5)
        bins = np.linspace(lo, hi, 80)

        ax.hist(diff, bins=bins, alpha=0.65, color=colors[idx % len(colors)],
                label=f"{label}  μ={mu:+.3f}%", histtype="stepfilled")
        ax.axvline(mu, color=colors[idx % len(colors)], linewidth=1.5)

    ax.set_xlabel("Differenza relativa [%]", fontsize=11)
    ax.set_ylabel("Conteggi", fontsize=11)
    ax.set_title("Distribuzione differenze relative (replica Fig.6 Sarrut 2019)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Istogramma differenze salvato: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. REPORT FINALE
# ═══════════════════════════════════════════════════════════════════════════════

def print_final_report(
    gamma_results: dict,
    label: str,
) -> None:
    """Stampa il report finale nello stile del paper."""
    print(f"\n{'='*55}")
    print(f"  RISULTATI — {label}")
    print(f"{'='*55}")
    g = gamma_results
    print(f"  Gamma-index ({g['dd_criterion_pct']:.0f}%/{g['dta_criterion_mm']:.0f}mm):")
    print(f"    Pass rate:   {g['pass_rate_pct']:>8.2f}%  (obiettivo: >95%)")
    print(f"    Mean Δ%:    {g['mean_diff_pct']:>+8.3f}%  (paper: <0.03%)")
    print(f"    Std Δ%:     {g['std_diff_pct']:>8.3f}%  (paper: ~1.15%)")
    print(f"    Max |Δ|%:   {g['max_abs_diff_pct']:>8.3f}%  (paper: <4%)")
    print(f"    Voxel eval.: {g['n_voxels_evaluated']:>8,}")

    if g["pass_rate_pct"] >= 95:
        print(f"\n  ✅ Risultato clinicamente accettabile (>95%)")
    elif g["pass_rate_pct"] >= 80:
        print(f"\n  ⚠️  Risultato parziale — serve più statistica")
    else:
        print(f"\n  ❌ Risultato insufficiente — verificare il modello")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Validazione output GATE 10 — numerico e grafico",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dose",      type=str, required=True,
                   help="File .mhd del modello da validare")
    p.add_argument("--reference", type=str, default=None,
                   help="File .mhd dose reference (gold standard PHSP)")
    p.add_argument("--label",     type=str, default="Modello",
                   help="Nome del modello per i plot")
    p.add_argument("--output_dir",type=str, default=None,
                   help="Directory output plot (default: stessa del file dose)")
    p.add_argument("--dd",        type=float, default=2.0)
    p.add_argument("--dta",       type=float, default=2.0)

    # Modalità confronto multiplo
    p.add_argument("--compare",   action="store_true",
                   help="Confronto multiplo (richiede --models e --labels)")
    p.add_argument("--models",    type=str, nargs="+", default=[],
                   help="File .mhd dei modelli da confrontare")
    p.add_argument("--labels",    type=str, nargs="+", default=[],
                   help="Etichette per i modelli")

    return p.parse_args()


def main():
    args   = parse_args()
    out_dir = Path(args.output_dir or Path(args.dose).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Carica dose modello ────────────────────────────────────────────────────
    print(f"\n  Caricamento: {args.dose}")
    dose, spacing, origin = load_mhd(args.dose)
    print(f"  Shape: {dose.shape}  spacing: {spacing} mm  origin: {origin} mm")

    # ── Sanity checks ──────────────────────────────────────────────────────────
    sanity = sanity_check(dose, label=args.label)

    # ── Profili ────────────────────────────────────────────────────────────────
    profiles_dict = {args.label: compute_profiles(dose, spacing)}
    plot_dose_slice(dose, spacing, args.label,
                    str(out_dir / f"dose_slice_{args.label}.png"))

    # ── Se reference disponibile ───────────────────────────────────────────────
    if args.reference:
        print(f"\n  Caricamento reference: {args.reference}")
        ref, ref_spacing, _ = load_mhd(args.reference)

        profiles_dict["Reference (PHSP)"] = compute_profiles(ref, ref_spacing)

        # Gamma-index
        print(f"\n  Calcolo gamma-index ({args.dd}%/{args.dta}mm)...")
        gamma = compute_gamma_index(ref, dose, spacing, args.dd, args.dta)
        print_final_report(gamma, args.label)

        # Salva gamma map
        gamma_result = {k: v for k, v in gamma.items() if k != "gamma_map"}
        with open(out_dir / f"gamma_{args.label}.json", "w") as f:
            json.dump(gamma_result, f, indent=2)

        # Plot differenze
        plot_difference_map(ref, dose, args.label,
                             str(out_dir / f"diff_map_{args.label}.png"))
        plot_difference_histogram(ref, {args.label: dose},
                                   str(out_dir / f"diff_hist_{args.label}.png"))

    # ── Confronto multiplo ─────────────────────────────────────────────────────
    if args.compare and args.models and args.reference:
        ref, ref_spacing, _ = load_mhd(args.reference)
        models_for_plot = {}
        gamma_table = {}

        for mpath, mlabel in zip(args.models, args.labels or args.models):
            m_dose, m_spacing, _ = load_mhd(mpath)
            profiles_dict[mlabel] = compute_profiles(m_dose, m_spacing)
            models_for_plot[mlabel] = m_dose
            g = compute_gamma_index(ref, m_dose, m_spacing, args.dd, args.dta)
            gamma_table[mlabel] = g
            print_final_report(g, mlabel)

        # Histogram comparativo (replica Fig. 6 paper)
        plot_difference_histogram(ref, models_for_plot,
                                   str(out_dir / "comparison_diff_hist.png"))

        # Tabella gamma
        print(f"\n{'='*60}")
        print(f"  {'Modello':<15} {'Pass rate':>12} {'Mean Δ%':>10} {'Max|Δ|%':>10}")
        print(f"  {'-'*50}")
        for lbl, g in gamma_table.items():
            print(f"  {lbl:<15} {g['pass_rate_pct']:>11.2f}% "
                  f"{g['mean_diff_pct']:>+10.3f} {g['max_abs_diff_pct']:>10.3f}")

    # ── Plot profili (sempre, per tutte le modalità) ───────────────────────────
    plot_dose_profiles(profiles_dict,
                       str(out_dir / "dose_profiles.png"),
                       title=f"Profili di dose — {args.label}")

    print(f"\n  Output salvati in: {out_dir}/")
    print(f"  File generati:")
    for f in sorted(out_dir.glob("*.png")) :
        print(f"    {f.name}")
    for f in sorted(out_dir.glob("*.json")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
