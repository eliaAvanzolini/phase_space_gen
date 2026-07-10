import os
import sys
import h5py
import numpy as np
import uproot
import argparse
from pathlib import Path

PHSP2_PATH = "data/conditional_jaws_dataset.h5"
WATER_BOX_CM = 20.0
VOXEL_MM = 2.0
OUT_DIR = Path("outputs/outputs_condizionali_6mv_5x5/run_split_half")

def get_split_indices():
    with h5py.File(PHSP2_PATH, "r") as f:
        cond_all = f["conditions"][:]
    
    cond_vector = [6.0, 2.5, 2.5]
    mask = (np.abs(cond_all[:, 0] - cond_vector[0]) < 0.1) & \
           (np.abs(cond_all[:, 1] - cond_vector[1]) < 0.1) & \
           (np.abs(cond_all[:, 2] - cond_vector[2]) < 0.1)
    
    available_indices = np.where(mask)[0]
    
    # Mescolamento con seed bloccato per garantire stabilità cross-processo
    np.random.seed(42)
    np.random.shuffle(available_indices)
    
    half = len(available_indices) // 2
    return available_indices[:half], available_indices[half:2*half]

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
    idx_A, idx_B = get_split_indices()
    
    if args.task == "setup":
        print("=========================================================")
        print("🪓 [FASE 1] CREAZIONE MATRICI DISGIUNTE ROOT")
        print("=========================================================")
        with h5py.File(PHSP2_PATH, "r") as f:
            ps_all = f["phase_space"][:]
            
        # Scrittura file ROOT per Metà A
        ps_A = ps_all[idx_A].astype(np.float32)
        with uproot.recreate(OUT_DIR / "gen_ref_A.root") as f:
            f["PhaseSpace"] = {"X": ps_A[:, 0]*10.0, "Y": ps_A[:, 1]*10.0, "Z": np.zeros(len(ps_A), dtype=np.float32), "dX": ps_A[:, 3], "dY": ps_A[:, 4], "dZ": np.abs(ps_A[:, 5]).astype(np.float32), "E": ps_A[:, 6]}
            
        # Scrittura file ROOT per Metà B
        ps_B = ps_all[idx_B].astype(np.float32)
        with uproot.recreate(OUT_DIR / "gen_ref_B.root") as f:
            f["PhaseSpace"] = {"X": ps_B[:, 0]*10.0, "Y": ps_B[:, 1]*10.0, "Z": np.zeros(len(ps_B), dtype=np.float32), "dX": ps_B[:, 3], "dY": ps_B[:, 4], "dZ": np.abs(ps_B[:, 5]).astype(np.float32), "E": ps_B[:, 6]}
            
        print(f"✓ File ROOT generati con successo in {OUT_DIR}. Taglio disgiunto: {len(idx_A)} fotoni a testa.")

    elif args.task == "run_A":
        print(f"🚀 [FASE 2] Lancio OpenGATE su Metà A ({len(idx_A)} particelle)...")
        run_gate_dose(OUT_DIR / "gen_ref_A.root", "ref_A", len(idx_A))

    elif args.task == "run_B":
        print(f"🚀 [FASE 3] Lancio OpenGATE su Metà B ({len(idx_B)} particelle)...")
        run_gate_dose(OUT_DIR / "gen_ref_B.root", "ref_B", len(idx_B))

    elif args.task == "analyze":
        print("=========================================================")
        print("📊 [FASE 4] CALCOLO GAMMA INDEX TRA LE METÀ (NOISE FLOOR)")
        print("=========================================================")
        import SimpleITK as sitk
        import pymedphys
        
        img_A = sitk.GetArrayFromImage(sitk.ReadImage(str(OUT_DIR / "dose_ref_A_dose.mhd"))).astype(np.float64)
        img_B = sitk.GetArrayFromImage(sitk.ReadImage(str(OUT_DIR / "dose_ref_B_dose.mhd"))).astype(np.float64)
        
        norm_A = (img_A / img_A.max()) * 100.0
        norm_B = (img_B / img_B.max()) * 100.0
        axes = tuple(np.arange(s) * VOXEL_MM for s in img_A.shape)
        
        g33 = pymedphys.gamma(axes, norm_A, axes, norm_B, dose_percent_threshold=3.0, distance_mm_threshold=3.0, lower_percent_dose_cutoff=20.0, max_gamma=1.1, quiet=True)
        v33 = g33[~np.isnan(g33)]
        print(f"🔥 NOISE FLOOR ORIZZONTE (Gamma 3%/3mm al 50k): {(v33 <= 1.0).mean() * 100:.2f}%")
        
        g22 = pymedphys.gamma(axes, norm_A, axes, norm_B, dose_percent_threshold=2.0, distance_mm_threshold=2.0, lower_percent_dose_cutoff=20.0, max_gamma=1.1, quiet=True)
        v22 = g22[~np.isnan(g22)]
        print(f"🔥 NOISE FLOOR ORIZZONTE (Gamma 2%/2mm al 50k): {(v22 <= 1.0).mean() * 100:.2f}%")
        print("=========================================================")

if __name__ == "__main__":
    main()
