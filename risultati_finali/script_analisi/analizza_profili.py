import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

base = "outputs/outputs_condizionali_6mv_5x5/run_scale_test"

def load_dose_matrix(pattern):
    files = sorted(glob.glob(f"{base}/{pattern}"))
    if not files:
        print(f"❌ Errore: impossibile trovare file per il pattern '{pattern}' in {base}")
        return None
    print(f"📂 Caricato: {files[0]}")
    return sitk.GetArrayFromImage(sitk.ReadImage(files[0])).astype(np.float64)

print("=========================================================")
print("🔬 ESTRAZIONE PROFILI E CALCOLO DELLA PENOMBRA (50%)")
print("=========================================================")

# Caricamento robusto con globbing
ref = load_dose_matrix("dose_reference_nativo*.mhd")
cfm_118k = load_dose_matrix("dose_cfm_118k*.mhd")
cfm_1m = load_dose_matrix("dose_cfm_1M*.mhd")

if ref is None or cfm_118k is None or cfm_1m is None:
    print("❌ Errore critico: Mancano alcuni file mhd nella cartella. Verificare i nomi.")
    sys.exit(1)

# Normalizzazione locale al rispettivo picco di dose (confronto in %)
ref_norm = (ref / ref.max()) * 100.0
cfm_118k_norm = (cfm_118k / cfm_118k.max()) * 100.0
cfm_1m_norm = (cfm_1m / cfm_1m.max()) * 100.0

# Individuazione del voxel Z di massima dose (picco di build-up)
z_peak = np.unravel_index(np.argmax(ref), ref.shape)[0]
y_center = ref.shape[1] // 2

# Estrazione dei profili monodimensionali lungo l'asse X
profile_ref = ref_norm[z_peak, y_center, :]
profile_118k = cfm_118k_norm[z_peak, y_center, :]
profile_1m = cfm_1m_norm[z_peak, y_center, :]

# Costruzione dell'asse spaziale millimetrico (voxel_size = 4mm)
voxel_mm = 4.0
x_axis = (np.arange(len(profile_ref)) - len(profile_ref) / 2) * voxel_mm

# --- COSTRUZIONE DEL GRAFICO PER LA TESI ---
plt.figure(figsize=(10, 6))
plt.plot(x_axis, profile_ref, label="Reference Nativo (118k storie)", linewidth=2.0, color="black", alpha=0.8)
plt.plot(x_axis, profile_118k, label="CFM Baseline 1:1 (118k storie)", linewidth=1.5, linestyle=":", color="crimson")
plt.plot(x_axis, profile_1m, label="CFM Alta Statistica (1M storie)", linewidth=2.5, linestyle="--", color="forestgreen")

plt.xlabel("Posizione X [mm]", fontsize=12)
plt.ylabel("Dose Normalizzata [%]", fontsize=12)
plt.title(f"Confronto Profili di Dose Trasversali al Picco di Build-up (Z Voxel = {z_peak})", fontsize=13, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=11, loc="lower center")
plt.xlim(-60, 60) # Zoom sulla zona centrale del campo 5x5

plot_path = f"{base}/profile_comparison_total.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\n🎨 ✅ Grafico comparativo salvato con successo in: {plot_path}")

# --- STIMA QUANTITATIVA DELLA LARGHEZZA DEL CAMPO AL 50% ---
def field_width_at_level(profile, x_axis, level=50.0):
    above = profile >= level
    if not above.any(): return None
    idx = np.where(above)[0]
    return x_axis[idx[-1]] - x_axis[idx[0]]

w_ref = field_width_at_level(profile_ref, x_axis)
w_118k = field_width_at_level(profile_118k, x_axis)
w_1m = field_width_at_level(profile_1m, x_axis)

print("\n📊 VERDETTO DELLE LARGHEZZE DI CAMPO (FWHM al 50%):")
print("-" * 57)
print(f" 🟢 Width REFERENCE: {w_ref:.1f} mm")
print(f" 🔴 Width CFM 118k:  {w_118k:.1f} mm  |  Δ vs Ref = {w_118k - w_ref:+.1f} mm")
print(f" 🍏 Width CFM 1M:    {w_1m:.1f} mm  |  Δ vs Ref = {w_1m - w_ref:+.1f} mm")
print("-" * 57)
print("Se Δ è vicino a 0, il problema è il rumore del Reference.")
print("Se Δ è negativo (es -4mm), c'è un bias geometrico di restringimento.")
print("=========================================================\n")
