import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymedphys
import SimpleITK as sitk

BASE_DIR = "outputs/dose_interpolation_10mv_coarse"
LABEL = "10mv_open_field"
MODEL_NAME = "cfm"

CRITERIA = [
    {"name": "3pct_3mm", "dose_pct": 3.0, "dist_mm": 3.0},
    {"name": "2pct_2mm_clinico", "dose_pct": 2.0, "dist_mm": 2.0},
]
LOWER_DOSE_CUTOFF = 10.0
HALF_WIDTH_VOX = 3

ANALYZE_UNCERTAINTY = True
UNCERTAINTY_MODE = "auto"
Z_ABS_MAX_PLOT = 5.0


def load_dose(path):
  img = sitk.ReadImage(str(path))
  arr = sitk.GetArrayFromImage(img).astype(np.float64)
  spacing_zyx = img.GetSpacing()[::-1]
  return arr, spacing_zyx


def robust_max(arr, cy=None, cx=None, hw=3):
  """Normalizza alla media di una ROI centrata sull'asse centrale (PDD),

  evitando di agganciarsi a voxel rumorosi isolati.
  """
  if cy is None or cx is None:
    nz, ny, nx = arr.shape
    cy, cx = ny // 2, nx // 2
  y0, y1 = max(0, cy - hw), min(arr.shape[1], cy + hw + 1)
  x0, x1 = max(0, cx - hw), min(arr.shape[2], cx + hw + 1)
  pdd_roi = arr[:, y0:y1, x0:x1].mean(axis=(1, 2))
  return float(pdd_roi.max())


def extract_pdd(arr, cy, cx, hw):
  y0, y1 = max(0, cy - hw), min(arr.shape[1], cy + hw + 1)
  x0, x1 = max(0, cx - hw), min(arr.shape[2], cx + hw + 1)
  return arr[:, y0:y1, x0:x1].mean(axis=(1, 2))


def extract_transverse(arr, z_idx, cy, hw):
  y0, y1 = max(0, cy - hw), min(arr.shape[1], cy + hw + 1)
  return arr[z_idx, y0:y1, :].mean(axis=0)


def find_uncertainty_path(dose_path: Path):
  stem = dose_path.name
  candidates = [
      dose_path.with_name(stem.replace("_dose.mhd", "_dose_uncertainty.mhd")),
      dose_path.with_name(stem.replace(".mhd", "_uncertainty.mhd")),
      dose_path.with_name(stem.replace("_dose.mhd", "-dose_uncertainty.mhd")),
  ]
  for c in candidates:
    if c.exists():
      return c
  run_tag = stem.split("_dose")[0]
  for p in dose_path.parent.glob(f"*{run_tag}*uncertain*.mhd"):
    return p
  return None


def _to_absolute_uncertainty(unc_arr, dose_arr, mode):
  if mode == "relative":
    return unc_arr * dose_arr
  if mode == "absolute":
    return unc_arr.copy()
  nz = unc_arr[unc_arr > 0]
  if len(nz) == 0:
    return unc_arr.copy()
  p90 = np.percentile(nz, 90)
  inferred = "relative" if p90 < 2.0 else "absolute"
  print(
      f"    [auto] incertezza rilevata come '{inferred}' (p90 valori grezzi ="
      f" {p90:.3g})"
  )
  return unc_arr * dose_arr if inferred == "relative" else unc_arr.copy()


def load_uncertainty_absolute(dose_path: Path, dose_arr, mode):
  unc_path = find_uncertainty_path(dose_path)
  if unc_path is None:
    return None, None
  unc_img, _ = load_dose(unc_path)
  return _to_absolute_uncertainty(unc_img, dose_arr, mode), unc_path


def main():
  base = Path(BASE_DIR)
  ref_path = base / "dose_reference_dose.mhd"
  model_path = base / f"dose_{MODEL_NAME}_dose.mhd"

  if not ref_path.exists():
    raise FileNotFoundError(f"Reference non trovato: {ref_path}")
  if not model_path.exists():
    raise FileNotFoundError(f"{MODEL_NAME.upper()} non trovato: {model_path}")

  print(f"Caricamento reference: {ref_path}")
  ref_arr, spacing_zyx = load_dose(ref_path)
  print(f"Caricamento {MODEL_NAME}: {model_path}")
  model_arr, model_spacing = load_dose(model_path)

  if not np.allclose(model_spacing, spacing_zyx, rtol=1e-3):
    print(
        f"⚠️ spacing diverso: reference={spacing_zyx} vs"
        f" {MODEL_NAME}={model_spacing}"
    )
  assert ref_arr.shape == model_arr.shape, (
      f"Shape mismatch: ref={ref_arr.shape} vs {MODEL_NAME}={model_arr.shape} "
      "(controlla che voxel_mm sia stato lo stesso in entrambe le run)"
  )

  ref_gnorm = robust_max(ref_arr)
  ref_norm = (ref_arr / ref_gnorm) * 100.0
  model_gnorm = robust_max(model_arr)
  model_norm = (model_arr / model_gnorm) * 100.0

  sp_z, sp_y, sp_x = spacing_zyx
  nz, ny, nx = ref_arr.shape
  cz, cy, cx = nz // 2, ny // 2, nx // 2

  out_dir = base
  depth_cm_slice = 10.0
  z_idx_slice = max(0, min(int(round(depth_cm_slice * 10.0 / sp_z)), nz - 1))
  extent_xy = [
      -nx / 2 * sp_x / 10,
      nx / 2 * sp_x / 10,
      -ny / 2 * sp_y / 10,
      ny / 2 * sp_y / 10,
  ]
  extent_xz = [-nx / 2 * sp_x / 10, nx / 2 * sp_x / 10, 0, nz * sp_z / 10]
  vmax = max(ref_norm.max(), model_norm.max())

  # ── 1. Mappe 2D ───────────────────────────────────────────────────────
  fig, axes = plt.subplots(2, 3, figsize=(18, 11))
  im0 = axes[0, 0].imshow(
      ref_norm[z_idx_slice],
      extent=extent_xy,
      origin="lower",
      cmap="jet",
      vmin=0,
      vmax=vmax,
  )
  axes[0, 0].set_title(f"REFERENCE — XY @ Z={depth_cm_slice}cm ({LABEL})")
  axes[0, 0].set_xlabel("X (cm)")
  axes[0, 0].set_ylabel("Y (cm)")
  plt.colorbar(im0, ax=axes[0, 0], label="Dose (% PDD max)")

  im1 = axes[0, 1].imshow(
      model_norm[z_idx_slice],
      extent=extent_xy,
      origin="lower",
      cmap="jet",
      vmin=0,
      vmax=vmax,
  )
  axes[0, 1].set_title(f"{MODEL_NAME.upper()} — XY @ Z={depth_cm_slice}cm")
  axes[0, 1].set_xlabel("X (cm)")
  axes[0, 1].set_ylabel("Y (cm)")
  plt.colorbar(im1, ax=axes[0, 1], label="Dose (% PDD max)")

  diff_xy = model_norm[z_idx_slice] - ref_norm[z_idx_slice]
  dmax_abs = np.percentile(np.abs(diff_xy), 99) or 1.0
  im2 = axes[0, 2].imshow(
      diff_xy,
      extent=extent_xy,
      origin="lower",
      cmap="RdBu_r",
      vmin=-dmax_abs,
      vmax=dmax_abs,
  )
  axes[0, 2].set_title(f"Differenza {MODEL_NAME.upper()} - Reference (XY)")
  axes[0, 2].set_xlabel("X (cm)")
  axes[0, 2].set_ylabel("Y (cm)")
  plt.colorbar(im2, ax=axes[0, 2], label="Δ Dose (% PDD max)")

  im3 = axes[1, 0].imshow(
      ref_norm[:, cy, :],
      extent=extent_xz,
      origin="lower",
      cmap="jet",
      vmin=0,
      vmax=vmax,
      aspect="auto",
  )
  axes[1, 0].set_title("REFERENCE — XZ (assiale)")
  axes[1, 0].set_xlabel("X (cm)")
  axes[1, 0].set_ylabel("Profondita' Z (cm)")
  plt.colorbar(im3, ax=axes[1, 0], label="Dose (% PDD max)")

  im4 = axes[1, 1].imshow(
      model_norm[:, cy, :],
      extent=extent_xz,
      origin="lower",
      cmap="jet",
      vmin=0,
      vmax=vmax,
      aspect="auto",
  )
  axes[1, 1].set_title(f"{MODEL_NAME.upper()} — XZ (assiale)")
  axes[1, 1].set_xlabel("X (cm)")
  axes[1, 1].set_ylabel("Profondita' Z (cm)")
  plt.colorbar(im4, ax=axes[1, 1], label="Dose (% PDD max)")

  diff_xz = model_norm[:, cy, :] - ref_norm[:, cy, :]
  dmax_abs_xz = np.percentile(np.abs(diff_xz), 99) or 1.0
  im5 = axes[1, 2].imshow(
      diff_xz,
      extent=extent_xz,
      origin="lower",
      cmap="RdBu_r",
      vmin=-dmax_abs_xz,
      vmax=dmax_abs_xz,
      aspect="auto",
  )
  axes[1, 2].set_title(f"Differenza {MODEL_NAME.upper()} - Reference (XZ)")
  axes[1, 2].set_xlabel("X (cm)")
  axes[1, 2].set_ylabel("Profondita' Z (cm)")
  plt.colorbar(im5, ax=axes[1, 2], label="Δ Dose (% PDD max)")

  plt.tight_layout()
  maps_path = out_dir / f"dose_maps_2d_{MODEL_NAME}.png"
  plt.savefig(maps_path, dpi=130)
  plt.close(fig)
  print(f"  Salvato: {maps_path}")

  # ── 2. Profili 1D ─────────────────────────────────────────────────────
  fig, axes = plt.subplots(1, 3, figsize=(18, 5))
  z_axis_cm = np.arange(nz) * sp_z / 10.0
  x_axis_cm = (np.arange(nx) - cx) * sp_x / 10.0

  ref_pdd = extract_pdd(ref_norm, cy, cx, HALF_WIDTH_VOX)
  model_pdd = extract_pdd(model_norm, cy, cx, HALF_WIDTH_VOX)
  axes[0].plot(z_axis_cm, ref_pdd, label="REFERENCE", color="black")
  axes[0].plot(z_axis_cm, model_pdd, label=MODEL_NAME.upper(), color="tab:blue")
  axes[0].set_xlabel("Profondita' Z (cm)")
  axes[0].set_ylabel("Dose (% PDD max)")
  axes[0].set_title(f"PDD ({LABEL}, mediato 7x7 voxel)")
  axes[0].legend()
  axes[0].grid(alpha=0.3)

  for i, depth_cm in enumerate([3.0, 10.0]):
    z_idx = max(0, min(int(round(depth_cm * 10.0 / sp_z)), nz - 1))
    ref_t = extract_transverse(ref_norm, z_idx, cy, HALF_WIDTH_VOX)
    model_t = extract_transverse(model_norm, z_idx, cy, HALF_WIDTH_VOX)
    axes[i + 1].plot(x_axis_cm, ref_t, label="REFERENCE", color="black")
    axes[i + 1].plot(
        x_axis_cm, model_t, label=MODEL_NAME.upper(), color="tab:blue"
    )
    axes[i + 1].set_xlabel("Posizione X (cm)")
    axes[i + 1].set_ylabel("Dose (% PDD max)")
    axes[i + 1].set_title(f"Trasversale @ Z={depth_cm}cm")
    axes[i + 1].legend()
    axes[i + 1].grid(alpha=0.3)

  plt.tight_layout()
  profiles_path = out_dir / f"dose_profiles_1d_{MODEL_NAME}.png"
  plt.savefig(profiles_path, dpi=130)
  plt.close(fig)
  print(f"  Salvato: {profiles_path}")

  # ── 3. Gamma index ────────────────────────────────────────────────────
  axes_3d = tuple(np.arange(s) * sp for s, sp in zip(ref_arr.shape, spacing_zyx))
  results = {}
  for crit in CRITERIA:
    gamma_map = pymedphys.gamma(
        axes_3d,
        ref_arr,
        axes_3d,
        model_arr,
        dose_percent_threshold=crit["dose_pct"],
        distance_mm_threshold=crit["dist_mm"],
        lower_percent_dose_cutoff=LOWER_DOSE_CUTOFF,
        global_normalisation=ref_gnorm,
        max_gamma=2,
        skip_once_passed=True,
        quiet=True,
    )
    valid = gamma_map[~np.isnan(gamma_map)]
    pass_rate = (
        float((valid <= 1.0).mean() * 100) if len(valid) > 0 else float("nan")
    )
    results[crit["name"]] = {"pass_rate_pct": pass_rate, "n_valid": len(valid)}

    gamma_slice = gamma_map[z_idx_slice]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        gamma_slice,
        extent=extent_xy,
        origin="lower",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=2,
    )
    ax.set_title(
        f"Gamma 2D {LABEL} — {MODEL_NAME.upper()} — {crit['name']}\nPass rate 3D:"
        f" {pass_rate:.1f}%"
    )
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    plt.colorbar(im, ax=ax, label="Gamma (<=1 = pass)")
    plt.tight_layout()
    gpath = out_dir / f"gamma_map_2d_{MODEL_NAME}_{crit['name']}.png"
    plt.savefig(gpath, dpi=130)
    plt.close(fig)
    print(f"  Salvato: {gpath}  (pass rate 3D = {pass_rate:.1f}%)")

  # ── 4. Significatività statistica (z-score), volume 3D ────────────────
  zscore_summary = None
  dmax_pdd = extract_pdd(ref_arr, cy, cx, HALF_WIDTH_VOX)
  z_idx_dmax = int(np.argmax(dmax_pdd))
  depth_cm_dmax = z_idx_dmax * sp_z / 10.0
  print(f"  Profondità dmax (asse centrale): {depth_cm_dmax:.1f} cm")

  if ANALYZE_UNCERTAINTY:
    sigma_ref_abs, ref_unc_path = load_uncertainty_absolute(
        ref_path, ref_arr, UNCERTAINTY_MODE
    )
    sigma_model_abs, model_unc_path = load_uncertainty_absolute(
        model_path, model_arr, UNCERTAINTY_MODE
    )

    if sigma_ref_abs is None or sigma_model_abs is None:
      print("⚠️ Mappa di incertezza non trovata. Salto analisi z-score.")
    else:
      print(f"  Incertezza reference: {ref_unc_path.name}")
      print(f"  Incertezza {MODEL_NAME}: {model_unc_path.name}")

      mask_3d = ref_norm >= LOWER_DOSE_CUTOFF
      denom_3d = np.sqrt(sigma_ref_abs**2 + sigma_model_abs**2)
      with np.errstate(divide="ignore", invalid="ignore"):
        z_map_3d = np.where(
            denom_3d > 0, (model_arr - ref_arr) / denom_3d, np.nan
        )
      z_map_3d = np.where(mask_3d, z_map_3d, np.nan)

      z_valid = z_map_3d[~np.isnan(z_map_3d)]
      frac_within_2sigma = (
          float((np.abs(z_valid) <= 2).mean() * 100)
          if len(z_valid)
          else float("nan")
      )
      frac_within_3sigma = (
          float((np.abs(z_valid) <= 3).mean() * 100)
          if len(z_valid)
          else float("nan")
      )
      zscore_summary = {
          "mean_abs_z": (
              float(np.nanmean(np.abs(z_valid)))
              if len(z_valid)
              else float("nan")
          ),
          "frac_within_2sigma_pct": frac_within_2sigma,
          "frac_within_3sigma_pct": frac_within_3sigma,
          "n_valid": int(len(z_valid)),
      }

      z_map_slice = z_map_3d[z_idx_dmax]
      fig, axes = plt.subplots(1, 2, figsize=(14, 6))
      im = axes[0].imshow(
          z_map_slice,
          extent=extent_xy,
          origin="lower",
          cmap="RdBu_r",
          vmin=-Z_ABS_MAX_PLOT,
          vmax=Z_ABS_MAX_PLOT,
      )
      axes[0].set_title(
          "Significatività statistica z = ΔD/σ (3D, slice @"
          f" dmax={depth_cm_dmax:.1f}cm)\n{LABEL} — {MODEL_NAME.upper()}"
      )
      axes[0].set_xlabel("X (cm)")
      axes[0].set_ylabel("Y (cm)")
      plt.colorbar(im, ax=axes[0], label="z (σ)")

      axes[1].hist(
          z_valid, bins=80, range=(-8, 8), color="tab:blue", alpha=0.8
      )
      axes[1].axvline(2, color="orange", ls="--", label="|z|=2")
      axes[1].axvline(-2, color="orange", ls="--")
      axes[1].axvline(3, color="red", ls="--", label="|z|=3")
      axes[1].axvline(-3, color="red", ls="--")
      axes[1].set_xlabel("z")
      axes[1].set_ylabel("Numero di voxel (volume 3D)")
      axes[1].set_title(
          f"Distribuzione z (3D) — {frac_within_2sigma:.1f}% entro 2σ,"
          f" {frac_within_3sigma:.1f}% entro 3σ, n={len(z_valid):,}"
      )
      axes[1].legend()

      plt.tight_layout()
      zpath = out_dir / f"zscore_map_{MODEL_NAME}.png"
      plt.savefig(zpath, dpi=130)
      plt.close(fig)
      print(
          f"  Salvato: {zpath}  ({frac_within_2sigma:.1f}% entro 2σ,"
          f" {frac_within_3sigma:.1f}% entro 3σ, n={len(z_valid):,})"
      )

  results["zscore_uncertainty"] = zscore_summary

  with open(out_dir / "dose_comparison_summary.json", "w") as f:
    json.dump({LABEL: {MODEL_NAME: results}}, f, indent=2)

  print(f"\n{'=' * 70}\n Completato. Risultati in {out_dir}\n{'=' * 70}")


if __name__ == "__main__":
  main()
