import os
import json
import numpy as np
import SimpleITK as sitk
import pymedphys

BASE_DIR = "outputs/dose_validation"
FIELDS = ["6mv_5x5", "6mv_10x10", "6mv_20x20", "10mv_5x5", "10mv_10x10", "10mv_20x20"]
MODELS = ["cfm", "nsf"]

CRITERIA = [
    {"name": "3pct_3mm", "dose_pct": 3.0, "dist_mm": 3.0},
    {"name": "2pct_2mm_clinico", "dose_pct": 2.0, "dist_mm": 2.0},
]
LOWER_DOSE_CUTOFF = 10.0
HALF_WIDTH_VOX = 3  # media su finestra 7x7 voxel, stessa logica di plot_dose_profiles.py
DEPTHS_CM_TO_PLOT = [3.0, 10.0]


def load_dose(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float64)  # (Z, Y, X)
    spacing_zyx = img.GetSpacing()[::-1]
    return arr, spacing_zyx


def robust_max(arr):
    nz = arr[arr > 0]
    return float(np.percentile(nz, 99.9)) if len(nz) > 0 else float(arr.max())


def extract_pdd(arr, cy, cx, hw):
    y0, y1 = max(0, cy - hw), min(arr.shape[1], cy + hw + 1)
    x0, x1 = max(0, cx - hw), min(arr.shape[2], cx + hw + 1)
    return arr[:, y0:y1, x0:x1].mean(axis=(1, 2))


def extract_transverse(arr, z_idx, cy, hw):
    y0, y1 = max(0, cy - hw), min(arr.shape[1], cy + hw + 1)
    return arr[z_idx, y0:y1, :].mean(axis=0)


def gamma_1d(ref_profile, mod_profile, axis_mm, global_norm, crit):
    axes = (axis_mm,)
    gamma_map = pymedphys.gamma(
        axes, ref_profile,
        axes, mod_profile,
        dose_percent_threshold=crit["dose_pct"],
        distance_mm_threshold=crit["dist_mm"],
        lower_percent_dose_cutoff=LOWER_DOSE_CUTOFF,
        global_normalisation=global_norm,
        max_gamma=2,
        skip_once_passed=True,
        quiet=True,
    )
    valid = gamma_map[~np.isnan(gamma_map)]
    if len(valid) == 0:
        return float("nan"), 0
    return float((valid <= 1.0).mean() * 100), len(valid)


def main():
    summary = {}

    for field in FIELDS:
        field_dir = os.path.join(BASE_DIR, field)
        ref_path = os.path.join(field_dir, "dose_reference_dose.mhd")
        if not os.path.exists(ref_path):
            print(f"❌ [{field}] reference non trovato, salto.")
            continue

        ref_arr, spacing_zyx = load_dose(ref_path)
        sp_z, sp_y, sp_x = spacing_zyx
        nz, ny, nx = ref_arr.shape
        cy, cx = ny // 2, nx // 2

        ref_gnorm = robust_max(ref_arr)
        ref_pdd = extract_pdd(ref_arr, cy, cx, HALF_WIDTH_VOX)
        z_axis_mm = np.arange(nz) * sp_z
        x_axis_mm = (np.arange(nx) - cx) * sp_x

        ref_transverse = {}
        for depth_cm in DEPTHS_CM_TO_PLOT:
            z_idx = max(0, min(int(round(depth_cm * 10.0 / sp_z)), nz - 1))
            ref_transverse[depth_cm] = extract_transverse(ref_arr, z_idx, cy, HALF_WIDTH_VOX)

        print(f"\n{'='*80}\n CLASSE: {field}\n{'='*80}")
        summary[field] = {}

        for model in MODELS:
            model_path = os.path.join(field_dir, f"dose_{model}_dose.mhd")
            if not os.path.exists(model_path):
                print(f"  ⚠️ [{model.upper()}] mappa non trovata, salto.")
                continue

            m_arr, m_spacing = load_dose(model_path)
            if m_arr.shape != ref_arr.shape:
                print(f"  ⚠️ [{model.upper()}] shape diversa, salto.")
                continue

            m_pdd = extract_pdd(m_arr, cy, cx, HALF_WIDTH_VOX)

            print(f"\n  --- Modello: {model.upper()} ---")
            summary[field][model] = {"pdd": {}, "transverse": {}}

            for crit in CRITERIA:
                pass_rate, n_valid = gamma_1d(ref_pdd, m_pdd, z_axis_mm, ref_gnorm, crit)
                print(f"    [PDD        | {crit['name']:<18}] n={n_valid:>4} | PASS RATE={pass_rate:.2f}%")
                summary[field][model]["pdd"][crit["name"]] = {"pass_rate_pct": pass_rate, "n_valid": n_valid}

            for depth_cm in DEPTHS_CM_TO_PLOT:
                m_transverse = extract_transverse(m_arr, max(0, min(int(round(depth_cm * 10.0 / sp_z)), nz - 1)), cy, HALF_WIDTH_VOX)
                for crit in CRITERIA:
                    pass_rate, n_valid = gamma_1d(ref_transverse[depth_cm], m_transverse, x_axis_mm, ref_gnorm, crit)
                    key = f"z{depth_cm}cm"
                    print(f"    [TRASV {key:<6} | {crit['name']:<18}] n={n_valid:>4} | PASS RATE={pass_rate:.2f}%")
                    summary[field][model]["transverse"].setdefault(key, {})[crit["name"]] = {
                        "pass_rate_pct": pass_rate, "n_valid": n_valid,
                    }

        out_json = os.path.join(BASE_DIR, "gamma_summary_1d_profiles.json")
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)

    # ── Riepilogo finale ─────────────────────────────────────────────────
    print(f"\n\n{'='*100}\n RIEPILOGO FINALE — GAMMA 1D PASS RATE (%)\n{'='*100}")
    header = f"{'Classe':<12}{'Modello':<8}{'PDD 3/3':>10}{'PDD 2/2':>10}{'Trasv3cm 3/3':>14}{'Trasv3cm 2/2':>14}{'Trasv10cm 3/3':>15}{'Trasv10cm 2/2':>15}"
    print(header)
    print("-" * len(header))
    for field in FIELDS:
        if field not in summary:
            continue
        for model in MODELS:
            if model not in summary[field]:
                continue
            d = summary[field][model]
            pdd33 = d["pdd"].get("3pct_3mm", {}).get("pass_rate_pct", float("nan"))
            pdd22 = d["pdd"].get("2pct_2mm_clinico", {}).get("pass_rate_pct", float("nan"))
            t3_33 = d["transverse"].get("z3.0cm", {}).get("3pct_3mm", {}).get("pass_rate_pct", float("nan"))
            t3_22 = d["transverse"].get("z3.0cm", {}).get("2pct_2mm_clinico", {}).get("pass_rate_pct", float("nan"))
            t10_33 = d["transverse"].get("z10.0cm", {}).get("3pct_3mm", {}).get("pass_rate_pct", float("nan"))
            t10_22 = d["transverse"].get("z10.0cm", {}).get("2pct_2mm_clinico", {}).get("pass_rate_pct", float("nan"))
            print(f"{field:<12}{model.upper():<8}{pdd33:>9.2f}%{pdd22:>9.2f}%{t3_33:>13.2f}%{t3_22:>13.2f}%{t10_33:>14.2f}%{t10_22:>14.2f}%")

    print(f"\nReport completo salvato in: {os.path.join(BASE_DIR, 'gamma_summary_1d_profiles.json')}")


if __name__ == "__main__":
    main()
