import os
import sys
import glob
import uproot
import numpy as np
import argparse
import subprocess
from pathlib import Path
import scipy.ndimage as ndimage
import SimpleITK as sitk
import pymedphys

# Parametri Geometrici del Fantoccio
WATER_BOX_CM = 20.0
VOXEL_MM = 2.0

# Mappatura corretta dei 6 campi reali sul filesystem
FIELDS = [
    "6mv_5x5",
    "6mv_10x10",
    "6mv_20x20",
    "10mv_5x5",
    "10mv_10x10",
    "10mv_20x20"
]

BASE_OUT_DIR = Path("outputs/outputs_noise_floor_analysis")

def load_raw_gate_pool(field_name):
    """
    Carica tutti i file ROOT originari generati da GATE per uno specifico campo.
    """
    path_pattern = f"outputs/gate_jaw_ref/{field_name}/{field_name}_phsp_part*.root"
    files = sorted(glob.glob(path_pattern))
    
    if not files:
        print(f"❌ Errore critico: File ROOT originari non trovati per {field_name} al pattern: {path_pattern}")
        sys.exit(1)

    gate_branches = [
        "PrePosition_X", "PrePosition_Y", "PrePosition_Z",
        "PreDirection_X", "PreDirection_Y", "PreDirection_Z",
        "KineticEnergy"
    ]

    all_chunks = []
    for fpath in files:
        with uproot.open(fpath) as f:
            keys = f.keys()
            if not keys: 
                continue
            tree = f[keys[0]]
            arrays = tree.arrays(gate_branches, library="np")
            n_events = len(arrays["KineticEnergy"])

            chunk = np.zeros((n_events, 7), dtype=np.float32)
            chunk[:, 0] = arrays["PrePosition_X"] / 10.0  # mm -> cm
            chunk[:, 1] = arrays["PrePosition_Y"] / 10.0  # mm -> cm
            chunk[:, 2] = arrays["PrePosition_Z"] / 10.0  # mm -> cm
            chunk[:, 3] = arrays["PreDirection_X"]
            chunk[:, 4] = arrays["PreDirection_Y"]
            chunk[:, 5] = arrays["PreDirection_Z"]
            chunk[:, 6] = arrays["KineticEnergy"]
            all_chunks.append(chunk)

    pool = np.concatenate(all_chunks, axis=0)
    print(f"📦 [{field_name}] Pool totale RAW estratto: {len(pool)} fotoni.")
    return pool

def run_gate_dose(source_path, field_dir, run_name, n_particles):
    """
    Esegue la simulazione OpenGATE sul fantoccio d'acqua (singola istanza isolata).
    """
    import opengate as gate
    from opengate import g4_units
    mm, cm, MeV = g4_units.mm, g4_units.cm, g4_units.MeV

    sim = gate.Simulation()
    sim.g4_verbose, sim.visu, sim.number_of_threads, sim.random_seed = False, False, 4, 42
    sim.world.size, sim.world.material = [100 * cm, 100 * cm, 120 * cm], "G4_AIR"
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.set_production_cut("world", "all", 2 * mm)

    water = sim.add_volume("Box", "water_phantom")
    water.size, water.material, water.translation = [WATER_BOX_CM * cm, WATER_BOX_CM * cm, WATER_BOX_CM * cm], "G4_WATER", [0, 0, (WATER_BOX_CM / 2) * cm]

    n_vox = int(WATER_BOX_CM * 10 / VOXEL_MM)
    dose = sim.add_actor("DoseActor", "dose")
    dose.attached_to, dose.size, dose.spacing, dose.output_filename = water.name, [n_vox, n_vox, n_vox], [VOXEL_MM * mm, VOXEL_MM * mm, VOXEL_MM * mm], str(field_dir / f"dose_{run_name}.mhd")
    dose.hit_type, dose.dose.active, dose.dose_uncertainty.active = "random", True, True

    src = sim.add_source("PhaseSpaceSource", "phsp_src")
    src.phsp_file, src.particle, src.n = str(source_path), "gamma", n_particles
    src.position_key_x, src.position_key_y, src.position_key_z = "X", "Y", "Z"
    src.direction_key_x, src.direction_key_y, src.direction_key_z = "dX", "dY", "dZ"
    src.energy_key, src.primary_PDGCode, src.primary_lower_energy_threshold = "E", 22, 0.01 * MeV
    sim.run()

def task_setup(field_name):
    """
    Estrae e suddivide la Phase Space in due metà disgiunte A e B (50% e 50%).
    """
    field_dir = BASE_OUT_DIR / field_name
    field_dir.mkdir(parents=True, exist_ok=True)
    
    raw_pool = load_raw_gate_pool(field_name)
    total_events = len(raw_pool)
    half_count = total_events // 2

    np.random.seed(42)
    indices = np.arange(total_events)
    np.random.shuffle(indices)

    idx_A = indices[:half_count]
    idx_B = indices[half_count:2 * half_count]

    # Salvataggio ROOT Metà A
    ps_A = raw_pool[idx_A]
    with uproot.recreate(field_dir / "gen_true_ref_A.root") as f:
        f["PhaseSpace"] = {
            "X": ps_A[:, 0] * 10.0, "Y": ps_A[:, 1] * 10.0, "Z": np.zeros(len(ps_A), dtype=np.float32),
            "dX": ps_A[:, 3], "dY": ps_A[:, 4], "dZ": np.abs(ps_A[:, 5]).astype(np.float32),
            "E": ps_A[:, 6]
        }

    # Salvataggio ROOT Metà B
    ps_B = raw_pool[idx_B]
    with uproot.recreate(field_dir / "gen_true_ref_B.root") as f:
        f["PhaseSpace"] = {
            "X": ps_B[:, 0] * 10.0, "Y": ps_B[:, 1] * 10.0, "Z": np.zeros(len(ps_B), dtype=np.float32),
            "dX": ps_B[:, 3], "dY": ps_B[:, 4], "dZ": np.abs(ps_B[:, 5]).astype(np.float32),
            "E": ps_B[:, 6]
        }

    print(f"✓ [{field_name}] Creati due blocchi disgiunti da {half_count} storie ciascuno.")

def task_run_single_sim(field_name, target_half):
    """
    Esegue la simulazione OpenGATE per una singola metà in un processo isolato.
    """
    field_dir = BASE_OUT_DIR / field_name
    phsp_file = field_dir / f"gen_true_ref_{target_half}.root"
    
    if not phsp_file.exists():
        print(f"❌ File {phsp_file} non trovato. Esegui prima lo --task setup!")
        sys.exit(1)
        
    with uproot.open(phsp_file) as f:
        n_particles = len(f["PhaseSpace"]["E"].array())

    print(f"\n🚀 Esecuzione GATE su {field_name} - Metà {target_half} ({n_particles} storie)...")
    run_gate_dose(phsp_file, field_dir, f"true_ref_{target_half}", n_particles)

def task_analyze(sigma_voxels):
    """
    Calcola la Gamma Analysis tra Metà A e Metà B per tutti i campi, sia RAW che con Smoothing.
    """
    print("\n=========================================================================")
    print(f"📊 CALCOLO NOISE FLOOR REALE (A vs B) - SIGMA SMOOTHING = {sigma_voxels} voxel ({sigma_voxels*VOXEL_MM:.1f} mm)")
    print("=========================================================================\n")

    header = f"{'CAMPO':<15} | {'STAT (1/2)':<10} | {'3%/3mm RAW':<12} | {'2%/2mm RAW':<12} | {f'3%/3mm (σ={sigma_voxels})':<16} | {f'2%/2mm (σ={sigma_voxels})':<16}"
    print(header)
    print("-" * len(header))

    for field in FIELDS:
        field_dir = BASE_OUT_DIR / field
        file_A = field_dir / "dose_true_ref_A_dose.mhd"
        file_B = field_dir / "dose_true_ref_B_dose.mhd"

        if not file_A.exists() or not file_B.exists():
            print(f"{field:<15} | ⚠️ File di dose mancanti. Saltato.")
            continue

        img_A = sitk.GetArrayFromImage(sitk.ReadImage(str(file_A))).astype(np.float64)
        img_B = sitk.GetArrayFromImage(sitk.ReadImage(str(file_B))).astype(np.float64)

        # Normalizzazione RAW (max locale)
        norm_A_raw = (img_A / img_A.max()) * 100.0
        norm_B_raw = (img_B / img_B.max()) * 100.0
        axes = tuple(np.arange(s) * VOXEL_MM for s in img_A.shape)

        # Gamma RAW
        g33_raw = pymedphys.gamma(axes, norm_A_raw, axes, norm_B_raw, dose_percent_threshold=3.0, distance_mm_threshold=3.0, lower_percent_dose_cutoff=20.0, max_gamma=1.1, quiet=True)
        g22_raw = pymedphys.gamma(axes, norm_A_raw, axes, norm_B_raw, dose_percent_threshold=2.0, distance_mm_threshold=2.0, lower_percent_dose_cutoff=20.0, max_gamma=1.1, quiet=True)
        
        p33_raw = (g33_raw[~np.isnan(g33_raw)] <= 1.0).mean() * 100.0
        p22_raw = (g22_raw[~np.isnan(g22_raw)] <= 1.0).mean() * 100.0

        # Normalizzazione con Smoothing Gaussiano
        if sigma_voxels > 0:
            sm_A = ndimage.gaussian_filter(img_A, sigma=sigma_voxels)
            sm_B = ndimage.gaussian_filter(img_B, sigma=sigma_voxels)
            norm_A_sm = (sm_A / sm_A.max()) * 100.0
            norm_B_sm = (sm_B / sm_B.max()) * 100.0

            g33_sm = pymedphys.gamma(axes, norm_A_sm, axes, norm_B_sm, dose_percent_threshold=3.0, distance_mm_threshold=3.0, lower_percent_dose_cutoff=20.0, max_gamma=1.1, quiet=True)
            g22_sm = pymedphys.gamma(axes, norm_A_sm, axes, norm_B_sm, dose_percent_threshold=2.0, distance_mm_threshold=2.0, lower_percent_dose_cutoff=20.0, max_gamma=1.1, quiet=True)

            p33_sm = (g33_sm[~np.isnan(g33_sm)] <= 1.0).mean() * 100.0
            p22_sm = (g22_sm[~np.isnan(g22_sm)] <= 1.0).mean() * 100.0
            str_33_sm = f"{p33_sm:.2f}%"
            str_22_sm = f"{p22_sm:.2f}%"
        else:
            str_33_sm = "N/A"
            str_22_sm = "N/A"

        # Conteggio storie per metà
        phsp_file = field_dir / "gen_true_ref_A.root"
        with uproot.open(phsp_file) as f:
            n_events = len(f["PhaseSpace"]["E"].array())

        print(f"{field:<15} | {n_events:<10} | {p33_raw:>10.2f}% | {p22_raw:>10.2f}% | {str_33_sm:>14} | {str_22_sm:>14}")

    print("=========================================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Split-Half Noise Floor Analysis per tutti i campi")
    parser.add_argument("--task", choices=["setup", "run_sims", "analyze"], required=True)
    parser.add_argument("--field", choices=FIELDS + ["all"], default="all")
    parser.add_argument("--target_half", choices=["A", "B", "both"], default="both")
    parser.add_argument("--sigma", type=float, default=0.5)

    args = parser.parse_args()

    if args.task == "setup":
        target_fields = FIELDS if args.field == "all" else [args.field]
        print("=========================================================")
        print("🪓 [FASE 1] ESTRAZIONE DISGIUNTA E CREAZIONE METÀ A/B")
        print("=========================================================")
        for f in target_fields:
            task_setup(f)

    elif args.task == "run_sims":
        target_fields = FIELDS if args.field == "all" else [args.field]
        halves = ["A", "B"] if args.target_half == "both" else [args.target_half]

        # Se ci sono più simulazioni da fare, le isoliamo lanciando sotto-processi Python
        if len(target_fields) > 1 or len(halves) > 1:
            print("=========================================================")
            print("🚀 [FASE 2/3] ISOLAMENTO PROCESSI PER SIMULAZIONI GATE")
            print("=========================================================")
            for f in target_fields:
                for h in halves:
                    cmd = [
                        sys.executable, __file__,
                        "--task", "run_sims",
                        "--field", f,
                        "--target_half", h
                    ]
                    # Esecuzione in sotto-processo isolato
                    subprocess.run(cmd, check=True)
        else:
            # Singola simulazione all'interno del processo pulito
            task_run_single_sim(target_fields[0], halves[0])

    elif args.task == "analyze":
        task_analyze(sigma_voxels=args.sigma)

if __name__ == "__main__":
    main()
