#!/usr/bin/env python3
"""
dose_validation_conditional_PATCHED.py
=======================================
Simulazione di Dose 3D con OpenGATE.
Usa automaticamente la statistica ESATTA (1:1) presente nell'HDF5 di riferimento.
"""

import argparse
import json
import sys
import time
from pathlib import Path
import h5py
import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

MODELS = {
    "cfm": {
        "checkpoint": "outputs/cfm_conditional_6mv_10mv/best_model.pt",
        "stats_json": "outputs/cfm_conditional_6mv_10mv/normalization_stats.json",
        "model_type": "cfm",
        "label": "CFM",
    },
    "nsf": {
        "checkpoint": "outputs/nsf_conditional_6mv_10mv/best_model.pt",
        "stats_json": "outputs/nsf_conditional_6mv_10mv/normalization_stats.json",
        "model_type": "nsf",
        "label": "NSF",
    },
    "gan": {
        "checkpoint": "outputs/gan_conditional_6mv_10mv/best_model.pt",
        "stats_json": "outputs/gan_conditional_6mv_10mv/normalization_stats.json",
        "model_type": "gan_sarrut",
        "label": "GAN",
    },
}

PHSP2_PATH = "data/conditional_jaws_ref_dataset.h5"
CHUNK_SIZE = 500_000
WATER_BOX_CM = 20.0


def _load_cfm(checkpoint_path: str, dim: int, device: str):
    from models.cfm import CFMTrainer, PhaseSpaceCFM

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model") or ckpt
    hidden_dim = 256
    if "velocity_net.input_proj.weight" in sd:
        hidden_dim = sd["velocity_net.input_proj.weight"].shape[0]
    n_layers = 4
    max_idx = -1
    for k in sd:
        if "velocity_net.res_layers." in k:
            idx = int(k.split("res_layers.")[1].split(".")[0])
            max_idx = max(max_idx, idx)
    if max_idx >= 0:
        n_layers = max_idx + 1
    model = PhaseSpaceCFM(dim=dim, cond_dim=3, hidden_dim=hidden_dim, n_layers=n_layers)
    trainer = CFMTrainer(model, device=device, lr=1e-4)
    trainer.load(checkpoint_path)
    return model.to(device).eval()


def _load_nsf(checkpoint_path: str, dim: int, device: str):
    from models.nsf import NSFTrainer, PhaseSpaceNSF

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("model") or ckpt
    hidden_dim = 256
    for k, v in sd.items():
        if "transform_net.initial_layer.weight" in k:
            hidden_dim = v.shape[0]
            break
    n_transforms = 6
    max_idx = -1
    for k in sd:
        if "_transforms." in k:
            s = k.split("_transforms.")[1].split(".")[0]
            if s.isdigit():
                max_idx = max(max_idx, int(s))
    if max_idx >= 0:
        n_transforms = (max_idx + 1) // 2
    model = PhaseSpaceNSF(
        dim=dim, cond_dim=3, n_transforms=n_transforms, hidden_dim=hidden_dim,
        n_bins=16, tail_bound=7.0,
    )
    trainer = NSFTrainer(model, device=device, lr=1e-4)
    trainer.load(checkpoint_path)
    return model.to(device).eval()


def generate_cfm_nsf(model_cfg: dict, n_samples: int, cond_vector: list, device: str, cfm_steps: int, solver: str) -> np.ndarray:
    from data.synthetic_linac import denormalize_phase_space

    stats_path = model_cfg["stats_json"]
    with open(stats_path) as f:
        stats = json.load(f)
    dim = len(stats.get("col_names", ["x", "y", "theta", "phi", "E"]))
    is_cfm = model_cfg["model_type"] == "cfm"

    cond_stats_path = Path(model_cfg["stats_json"]).parent / "condition_stats.json"
    with open(cond_stats_path) as f:
        cond_stats = json.load(f)
    mu_c = np.array(cond_stats["mu"], dtype=np.float32)
    sig_c = np.array(cond_stats["sigma"], dtype=np.float32)
    cond_norm = ((np.array(cond_vector, dtype=np.float32) - mu_c) / sig_c).tolist()

    model = _load_cfm(model_cfg["checkpoint"], dim, device) if is_cfm else _load_nsf(model_cfg["checkpoint"], dim, device)
    chunks_phys = []
    n_done = 0

    with torch.no_grad():
        while n_done < n_samples:
            n_chunk = min(CHUNK_SIZE, n_samples - n_done)

            if is_cfm:
                cond_tensor = torch.tensor(cond_norm, dtype=torch.float32, device=device).unsqueeze(0).repeat(n_chunk, 1)
                if solver == "euler":
                    chunk_t = model.sample_fast(n_chunk, cond_tensor, n_steps=cfm_steps)
                else:
                    chunk_t = model.sample(n_chunk, cond_tensor, method="dopri5", atol=1e-4, rtol=1e-4)
                chunk = chunk_t.cpu().numpy()
                del cond_tensor
            else:
                nsf_subchunk = 50000
                nsf_chunks = []
                for start_idx in range(0, n_chunk, nsf_subchunk):
                    curr_sub = min(nsf_subchunk, n_chunk - start_idx)
                    cond_tensor_nsf = torch.tensor(cond_norm, dtype=torch.float32, device=device).unsqueeze(0)
                    chunk_t = model.sample(curr_sub, cond_tensor_nsf)
                    nsf_chunks.append(chunk_t.cpu().numpy())
                    del cond_tensor_nsf
                chunk = np.concatenate(nsf_chunks, axis=0)

            chunks_phys.append(chunk)
            n_done += n_chunk
            del chunk_t
            if device == "cuda":
                torch.cuda.empty_cache()

    return denormalize_phase_space(np.concatenate(chunks_phys, axis=0), stats).astype(np.float32)


def _apply_physical_filters(ps7: np.ndarray, model_name: str) -> np.ndarray:
    E, dx, dy, dz = ps7[:, 6], ps7[:, 3], ps7[:, 4], ps7[:, 5]
    norm_d = np.sqrt(dx**2 + dy**2 + dz**2)
    mask = (E > 0.01) & (norm_d > 0.4) & (np.abs(ps7[:, 0]) < 20.0) & (np.abs(ps7[:, 1]) < 20.0)
    if mask.sum() == 0:
        return ps7[:10].copy()
    ps7_clean = ps7[mask].copy()
    norms = np.sqrt(ps7_clean[:, 3] ** 2 + ps7_clean[:, 4] ** 2 + ps7_clean[:, 5] ** 2)
    ps7_clean[:, 3] /= norms
    ps7_clean[:, 4] /= norms
    ps7_clean[:, 5] /= norms
    return ps7_clean


def run_gate_dose(source_path: Path, output_dir: Path, run_name: str, n_threads: int, voxel_mm: float):
    import opengate as gate
    from opengate import g4_units
    import uproot

    with uproot.open(source_path) as root_f:
        key = [k for k in root_f.keys() if "PhaseSpace" in k][0]
        actual_n = int(root_f[key].num_entries)

    mm, cm, MeV = g4_units.mm, g4_units.cm, g4_units.MeV
    sim = gate.Simulation()
    sim.g4_verbose, sim.visu, sim.number_of_threads, sim.random_seed = False, False, n_threads, 42
    sim.world.size, sim.world.material = [100 * cm, 100 * cm, 120 * cm], "G4_AIR"
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.set_production_cut("world", "all", 2 * mm)

    water = sim.add_volume("Box", "water_phantom")
    water.size, water.material, water.translation = (
        [WATER_BOX_CM * cm, WATER_BOX_CM * cm, WATER_BOX_CM * cm], "G4_WATER", [0, 0, (WATER_BOX_CM / 2) * cm],
    )

    n_vox = int(WATER_BOX_CM * 10 / voxel_mm)
    dose = sim.add_actor("DoseActor", "dose")
    dose.attached_to, dose.size, dose.spacing, dose.output_filename = (
        water.name, [n_vox, n_vox, n_vox], [voxel_mm * mm, voxel_mm * mm, voxel_mm * mm],
        str(output_dir / f"dose_{run_name}.mhd"),
    )
    dose.hit_type, dose.dose.active, dose.dose_uncertainty.active = "random", True, True

    src = sim.add_source("PhaseSpaceSource", "phsp_src")
    src.phsp_file, src.particle = str(source_path), "gamma"

    # --- FIX MULTI-THREADING: entry_start esplicito per thread ---
    # Senza questo, con n_threads>1 ogni thread rilegge in modo indipendente
    # (e sovrapposto) il file sorgente: misurato empiricamente un fattore di
    # inflazione della dose vicino al numero di thread (8.09x con 8 thread,
    # confermato dal test 1thread vs 8thread a n_particles fisso).
    if n_threads > 1:
        n_per_thread = actual_n // n_threads
        src.n = n_per_thread
        src.entry_start = [i * n_per_thread for i in range(n_threads)]
    else:
        src.n = actual_n
        src.entry_start = 0

    src.position_key_x, src.position_key_y, src.position_key_z = "X", "Y", "Z"
    src.direction_key_x, src.direction_key_y, src.direction_key_z = ("dX", "dY", "dZ")
    src.energy_key, src.primary_PDGCode, src.primary_lower_energy_threshold = ("E", 22, 0.01 * MeV)
    sim.run()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--field", choices=["6mv_5x5", "6mv_10x10", "6mv_20x20", "10mv_5x5", "10mv_10x10", "10mv_20x20"], required=True)
    p.add_argument("--n_particles", type=int, default=0, help="0 o non specificato per fare il match automatico con il Reference")
    p.add_argument("--n_threads", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--subtask", choices=["reference", "cfm", "nsf", "gan"], required=True)
    p.add_argument("--suffix", default="")
    p.add_argument("--voxel_mm", type=float, default=2.0)
    p.add_argument("--cfm_steps", type=int, default=30)
    p.add_argument("--solver", choices=["euler", "dopri5"], default="dopri5", help="Tipo di solutore ODE")
    args = p.parse_args()

    fields_map = {
        "6mv_5x5": [6.0, 2.5, 2.5], "6mv_10x10": [6.0, 5.0, 5.0], "6mv_20x20": [6.0, 10.0, 10.0],
        "10mv_5x5": [10.0, 2.5, 2.5], "10mv_10x10": [10.0, 5.0, 5.0], "10mv_20x20": [10.0, 10.0, 10.0],
    }
    cond_vector = fields_map[args.field]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    run_name = f"{args.subtask}_{args.suffix}" if args.suffix else args.subtask

    import uproot

    with h5py.File(PHSP2_PATH, "r") as f:
        if args.field not in f:
            raise KeyError(f"Gruppo '{args.field}' non trovato in {PHSP2_PATH}.")
        grp = f[args.field]
        ps_ref_all = grp["phase_space"][:]
        cond_saved = grp["condition"][:]
        if not np.allclose(cond_saved, cond_vector, atol=0.1):
            raise ValueError(f"Mismatch condition per gruppo '{args.field}'")

    n_ref_available = len(ps_ref_all)

    if args.n_particles <= 0 or args.n_particles > n_ref_available:
        target_n_particles = n_ref_available
    else:
        target_n_particles = args.n_particles

    print(f"🎯 [{args.field} | {args.subtask.upper()}] Particelle richieste: {target_n_particles:,} (su {n_ref_available:,} disponibili nel ref)")

    if args.subtask == "reference":
        ref_root = out / f"gen_{run_name}.root"
        ps_ref = ps_ref_all[:target_n_particles].astype(np.float32)

        with uproot.recreate(ref_root) as f:
            f["PhaseSpace"] = {
                "X": ps_ref[:, 0] * 10.0, "Y": ps_ref[:, 1] * 10.0,
                "Z": np.zeros(len(ps_ref), dtype=np.float32),
                "dX": ps_ref[:, 3], "dY": ps_ref[:, 4],
                "dZ": np.abs(ps_ref[:, 5]).astype(np.float32), "E": ps_ref[:, 6],
            }
        run_gate_dose(ref_root, out, run_name, args.n_threads, args.voxel_mm)

    else:
        name = args.subtask
        rpath = out / f"gen_{run_name}.root"
        ps7 = generate_cfm_nsf(MODELS[name], target_n_particles, cond_vector, dev, args.cfm_steps, args.solver)
        ps7_clean = _apply_physical_filters(ps7, name)

        with uproot.recreate(rpath) as f:
            f["PhaseSpace"] = {
                "X": ps7_clean[:, 0] * 10.0, "Y": ps7_clean[:, 1] * 10.0,
                "Z": np.zeros(len(ps7_clean), dtype=np.float32),
                "dX": ps7_clean[:, 3], "dY": ps7_clean[:, 4],
                "dZ": np.abs(ps7_clean[:, 5]).astype(np.float32), "E": ps7_clean[:, 6],
            }
        run_gate_dose(rpath, out, run_name, args.n_threads, args.voxel_mm)


if __name__ == "__main__":
    main()
