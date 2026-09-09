import json
from pathlib import Path
from plot_reports_dose_extended import process_pair, load_dose, robust_max

field_dir = Path("outputs/dose_interpolation_25m")
field = "10mv_open_field"
model = "cfm"

print("=" * 70)
print(f" VALUTAZIONE DOSIMETRICA 25M: {field.upper()} — {model.upper()}")
print("=" * 70)

ref_path = field_dir / "dose_reference_dose.mhd"
if not ref_path.exists():
    raise FileNotFoundError(f"Reference non trovato in {ref_path}")

print("Caricamento matrici di dose 3D...")
ref_arr, spacing_zyx = load_dose(ref_path)
ref_gnorm = robust_max(ref_arr)
ref_norm = (ref_arr / ref_gnorm) * 100.0

out_dir = field_dir

print("\nAvvio calcolo Gamma 3D, profili 1D e mappa di significativita' z...")
res = process_pair(
    field_dir=field_dir,
    field=field,
    model=model,
    ref_arr=ref_arr,
    ref_norm=ref_norm,
    ref_gnorm=ref_gnorm,
    spacing_zyx=spacing_zyx,
    out_dir=out_dir,
    ref_dose_path=ref_path,
)

if res is not None:
    summary_path = out_dir / "gamma_summary_25m.json"
    with open(summary_path, "w") as f:
        json.dump({field: {model: res}}, f, indent=2)
    print(f"\n[+] Risultati quantitativi salvati in: {summary_path}")

print("=" * 70)
print(" VALUTAZIONE COMPLETATA CON SUCCESSO")
print("=" * 70)
