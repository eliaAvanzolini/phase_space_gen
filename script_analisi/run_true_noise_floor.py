import os
import sys
import glob
import uproot
import numpy as np
import argparse
from pathlib import Path

WATER_BOX_CM = 20.0
VOXEL_MM = 2.0
OUT_DIR = Path("outputs/outputs_condizionali_6mv_5x5/run_true_noise_floor")

def load_raw_gate_pool():
    path_pattern = "outputs/gate_jaw/6mv_5x5/6mv_5x5_phsp_part*.root"
    files = sorted(glob.glob(path_pattern))
    if not files:
        print("❌ Errore critico: File ROOT originari di GATE non trovati!")
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
            if not keys: continue
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
            
    return np.concatenate(all_chunks, axis=0)

def run_gate_dose(source_path, run_name, n_particles):
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
    dose.attached_to, dose.size, dose.spacing, dose.output_filename = water.name, [n_vox, n_vox, n_vox], [VOXEL_MM * mm, VOXEL_MM * mm, VOXEL_MM * mm], str(OUT_DIR / f"dose_{run_name}.mhd")
    dose.hit_type, dose.dose.active, dose.dose_uncertainty.active = "random", True, True
    
    src = sim.add_source("PhaseSpaceSource", "phsp_src")
    src.phsp_file, src.particle, src.n = str(source_path), "gamma", n_particles
    src.position_key_x, src.position_key_y, src.position_key_z = "X", "Y", "Z"
    src.direction_key_x, src.direction_key_y, src.direction_key_z = "dX", "dY", "dZ"
    src.energy_key, src.primary_PDGCode, src.primary_lower_energy_threshold = "E", 22, 0.01 * MeV
    sim.run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["setup", "run_A", "run_B", "analyze"], required=True)
    args = parser.parse_args()
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target_count = 118724
    
    if args.task == "setup":
        print("=========================================================")
        print("🪓 [FASE 1] ESTRAZIONE DISGIUNTA A PIENA STATISTICA (118k)")
        print("=========================================================")
        raw_pool = load_raw_gate_pool()
        print(f"📦 Pool totale RAW estratto dai file ROOT: {len(raw_pool)} fotoni.")
        
        # Mescolamento stocastico per eliminare l'ordine di scrittura dei file
        np.random.seed(42)
        indices = np.arange(len(raw_pool))
        np.random.shuffle(indices)
        
        idx_A = indices[:target_count]
        idx_B = indices[target_count:2*target_count]
        
        # Salvataggio ROOT Metà A (118k)
        ps_A = raw_pool[idx_A]
        with uproot.recreate(OUT_DIR / "gen_true_ref_A.root") as f:
            f["PhaseSpace"] = {"X": ps_A[:, 0]*10.0, "Y": ps_A[:, 1]*10.0, "Z": np.zeros(len(ps_A), dtype=np.float32), "dX": ps_A[:, 3], "dY": ps_A[:, 4], "dZ": np.abs(ps_A[:, 5]).astype(np.float32), "E": ps_A[:, 6]}
            
        # Salvataggio ROOT Metà B (118k)
        ps_B = raw_pool[idx_B]
        with uproot.recreate(OUT_DIR / "gen_true_ref_B.root") as f:
            f["PhaseSpace"] = {"X": ps_B[:, 0]*10.0, "Y": ps_B[:, 1]*10.0, "Z": np.zeros(len(ps_B), dtype=np.float32), "dX": ps_B[:, 3], "dY": ps_B[:, 4], "dZ": np.abs(ps_B[:, 5]).astype(np.float32), "E": ps_B[:, 6]}
            
        print(f"✓ Creati due blocchi identici e non sovrapposti da esattamente {target_count} storie ciascuno.")

    elif args.task == "run_A":
        print(f"🚀 [FASE 2] Esecuzione GATE su True Reference A ({target_count} storie)...")
        run_gate_dose(OUT_DIR / "gen_true_ref_A.root", "true_ref_A", target_count)

    elif args.task == "run_B":
        print(f"🚀 [FASE 3] Esecuzione GATE su True Reference B ({target_count} storie)...")
        run_gate_dose(OUT_DIR / "gen_true_ref_B.root", "true_ref_B", target_count)

    elif args.task == "analyze":
        print("=========================================================")
        print("📊 [FASE 4] CALCOLO NOISE FLOOR REALE 'MELE CON MELE'")
        print("=========================================================")
        import SimpleITK as sitk
        import pymedphys
        
        img_A = sitk.GetArrayFromImage(sitk.ReadImage(str(OUT_DIR / "dose_true_ref_A_dose.mhd"))).astype(np.float64)
        img_B = sitk.GetArrayFromImage(sitk.ReadImage(str(OUT_DIR / "dose_true_ref_B_dose.mhd"))).astype(np.float64)
        
        norm_A = (img_A / img_A.max()) * 100.0
        norm_B = (img_B / img_B.max()) * 100.0
        axes = tuple(np.arange(s) * VOXEL_MM for s in img_A.shape)
        
        g33 = pymedphys.gamma(axes, norm_A, axes, norm_B, dose_percent_threshold=3.0, distance_mm_threshold=3.0, lower_percent_dose_cutoff=20.0, max_gamma=1.1, quiet=True)
        v33 = g33[~np.isnan(g33)]
        print(f"🎯 TRUE NOISE FLOOR EQUO (Gamma 3%/3mm a 118k): {(v33 <= 1.0).mean() * 100:.2f}%")
        
        g22 = pymedphys.gamma(axes, norm_A, axes, norm_B, dose_percent_threshold=2.0, distance_mm_threshold=2.0, lower_percent_dose_cutoff=20.0, max_gamma=1.1, quiet=True)
        v22 = g22[~np.isnan(g22)]
        print(f"🎯 TRUE NOISE FLOOR EQUO (Gamma 2%/2mm a 118k): {(v22 <= 1.0).mean() * 100:.2f}%")
        print("=========================================================")

if __name__ == "__main__":
    main()
