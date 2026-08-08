import os
import json
import numpy as np
import SimpleITK as sitk
import pymedphys

BASE_DIR = "outputs/dose_validation"
FIELDS = ["6mv_5x5", "6mv_10x10", "6mv_20x20", "10mv_5x5", "10mv_10x10", "10mv_20x20"]
MODELS = ["cfm", "nsf"]  # gan escluso deliberatamente (baseline non ottimizzata)

CRITERIA = [
    {"name": "3pct_3mm", "dose_pct": 3.0, "dist_mm": 3.0},
    {"name": "2pct_2mm_clinico", "dose_pct": 2.0, "dist_mm": 2.0},
]
LOWER_DOSE_CUTOFF = 10.0  # % di Dmax sotto cui i voxel non vengono valutati


def load_dose(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float64)  # ordine (Z, Y, X)
    spacing = img.GetSpacing()  # (X, Y, Z) - attenzione all'ordine invertito vs array!
    # np array e' in ordine (Z,Y,X), spacing di sitk e' (X,Y,Z): invertiamo
    spacing_zyx = spacing[::-1]
    axes = tuple(np.arange(s) * sp for s, sp in zip(arr.shape, spacing_zyx))
    return arr, axes, spacing_zyx


def main():
    summary = {}

    for field in FIELDS:
        field_dir = os.path.join(BASE_DIR, field)
        ref_path = os.path.join(field_dir, "dose_reference_dose.mhd")
        if not os.path.exists(ref_path):
            print(f"❌ [{field}] Reference non trovato in {ref_path}, salto la classe.")
            continue

        ref_dose, axes, spacing_zyx = load_dose(ref_path)
        # Normalizzazione robusta: P99.9 invece del max grezzo, che con statistica MC
        # limitata puo' essere un singolo voxel/piccolo cluster di rumore (deposito
        # localizzato di un elettrone secondario energetico), non vero segnale.
        ref_nonzero = ref_dose[ref_dose > 0]
        ref_robust_max = np.percentile(ref_nonzero, 99.9) if len(ref_nonzero) > 0 else ref_dose.max()
        ref_norm = (ref_dose / ref_robust_max) * 100.0
        mask = ref_norm > LOWER_DOSE_CUTOFF

        print(f"\n{'='*70}")
        print(f" CLASSE: {field}  (voxel spacing rilevato: {spacing_zyx} mm)")
        print(f"{'='*70}")

        summary[field] = {}

        for model in MODELS:
            model_path = os.path.join(field_dir, f"dose_{model}_dose.mhd")
            if not os.path.exists(model_path):
                print(f"  ⚠️ [{model.upper()}] mappa non trovata in {model_path}, salto.")
                continue

            model_dose, model_axes, model_spacing = load_dose(model_path)
            if model_spacing != spacing_zyx:
                print(f"  ⚠️ ATTENZIONE: spacing di {model.upper()} ({model_spacing}) "
                      f"diverso dal reference ({spacing_zyx})! Il confronto potrebbe non avere senso.")

            # STESSA normalizzazione robusta usata per il reference, non il max
            # grezzo: il massimo di una distribuzione a code pesanti (rumore da
            # elettroni secondari energetici) non si stabilizza con piu' statistica
            # quanto un percentile, quindi usare max grezzo solo qui introduce un
            # offset sistematico persistente indipendente dalla statistica disponibile.
            model_nonzero = model_dose[model_dose > 0]
            model_robust_max = np.percentile(model_nonzero, 99.9) if len(model_nonzero) > 0 else model_dose.max()
            model_norm = (model_dose / model_robust_max) * 100.0
            diff_media = float(np.mean(model_norm[mask] - ref_norm[mask]))

            print(f"\n  --- Modello: {model.upper()} ---")
            print(f"  Δ medio relativo (>{LOWER_DOSE_CUTOFF}% Dmax): {diff_media:+.4f}%")

            summary[field][model] = {"delta_medio_pct": diff_media, "criteri": {}}

            for crit in CRITERIA:
                gamma_map = pymedphys.gamma(
                    axes, ref_dose,
                    axes, model_dose,
                    dose_percent_threshold=crit["dose_pct"],
                    distance_mm_threshold=crit["dist_mm"],
                    lower_percent_dose_cutoff=LOWER_DOSE_CUTOFF,
                    global_normalisation=ref_robust_max,  # stessa normalizzazione robusta, non il max grezzo di default
                    max_gamma=2,             # taglia la ricerca oltre gamma=2, non altera il pass rate (<=1)
                    skip_once_passed=True,   # smette di cercare un punto migliore appena gamma<=1
                    quiet=True,
                )
                valid = gamma_map[~np.isnan(gamma_map)]
                n_valid = len(valid)
                pass_rate = float((valid <= 1.0).mean() * 100) if n_valid > 0 else float("nan")
                gamma_mean = float(np.mean(valid)) if n_valid > 0 else float("nan")

                print(f"    [{crit['name']}] voxel validi={n_valid:,} | "
                      f"gamma medio={gamma_mean:.4f} | PASS RATE={pass_rate:.2f}%")

                summary[field][model]["criteri"][crit["name"]] = {
                    "dose_pct": crit["dose_pct"],
                    "dist_mm": crit["dist_mm"],
                    "n_valid_voxels": n_valid,
                    "gamma_mean": gamma_mean,
                    "pass_rate_pct": pass_rate,
                }

        # Salvataggio incrementale: se il job viene ucciso a meta', non si perde
        # tutto il lavoro gia' fatto per le classi precedenti
        out_json = os.path.join(BASE_DIR, "gamma_summary_all_classes.json")
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)

    # ── Riepilogo finale a tabella ──────────────────────────────────────
    print(f"\n\n{'='*90}")
    print(" RIEPILOGO FINALE — GAMMA PASS RATE (%)")
    print(f"{'='*90}")
    header = f"{'Classe':<14}{'Modello':<8}{'3%/3mm':>12}{'2%/2mm clinico':>18}{'Δ medio %':>14}"
    print(header)
    print("-" * len(header))
    for field in FIELDS:
        if field not in summary:
            continue
        for model in MODELS:
            if model not in summary[field]:
                continue
            d = summary[field][model]
            c3 = d["criteri"].get("3pct_3mm", {}).get("pass_rate_pct", float("nan"))
            c2 = d["criteri"].get("2pct_2mm_clinico", {}).get("pass_rate_pct", float("nan"))
            print(f"{field:<14}{model.upper():<8}{c3:>11.2f}%{c2:>17.2f}%{d['delta_medio_pct']:>13.3f}%")

    out_json = os.path.join(BASE_DIR, "gamma_summary_all_classes.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nReport completo salvato in: {out_json}")


if __name__ == "__main__":
    main()
