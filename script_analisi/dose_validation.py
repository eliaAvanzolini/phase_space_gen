"""
dose_validation.py (UNIFICATO)
==============================
Validazione downstream della dose per i tre modelli generativi
(CFM, NSF, GAN Sarrut) tramite GATE 10 / Geant4.

Analisi Dosimetrica: Gamma Index 3D Clinico Reale tramite PyMedPhys
Normalizzazione: Dose Massima (Dmax) locale per ciascun volume
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ─── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ─── Configurazione modelli ───────────────────────────────────────────────────
MODELS = {
    "cfm": {
        "checkpoint": "outputs/cfm_130M_rank_final/best_model.pt",
        "stats_json": "outputs/cfm_130M_rank_final/normalization_stats.json",
        "model_type": "cfm",
        "label":      "CFM (Flow Matching + Rank Transform)",
    },
    "nsf": {
        "checkpoint": "outputs/nsf_130mln_rank_final/best_model.pt",
        "stats_json": "outputs/nsf_130mln_rank_final/normalization_stats.json",
        "model_type": "nsf",
        "label":      "NSF (Neural Spline Flow + Rank Transform)",
    },
    "gan": {
        "checkpoint": "outputs/sarrut_pure_replica_run/sarrut_replica_final.pt",
        "stats_json": "outputs/sarrut_pure_replica_run/sarrut_minmax_stats.json",
        "model_type": "gan_sarrut",
        "label":      "GAN (Replica Sarrut 2019)",
    },
}

PHSP2_PATH       = "data/elekta_130mv_eval_completo.h5"
DEFAULT_N        = int(1e8)  # Default impostato a 100M per abbattere il rumore statistico
CHUNK_SIZE       = 250_000
WATER_BOX_CM     = 20.0   # cm — fantoccio cubico come Sarrut
VOXEL_MM         = 4.0    # mm — voxel 4x4x4 mm come Sarrut


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GENERAZIONE CAMPIONI
# ═══════════════════════════════════════════════════════════════════════════════

def _load_cfm(checkpoint_path: str, dim: int, device: str):
    import torch
    from models.cfm import PhaseSpaceCFM, CFMTrainer

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd   = ckpt.get("model") or ckpt

    hidden_dim = 256
    if "velocity_net.input_proj.weight" in sd:
        hidden_dim = sd["velocity_net.input_proj.weight"].shape[0]

    n_layers = 4
    max_idx  = -1
    for k in sd:
        if "velocity_net.res_layers." in k:
            idx = int(k.split("res_layers.")[1].split(".")[0])
            max_idx = max(max_idx, idx)
    if max_idx >= 0:
        n_layers = max_idx + 1

    model = PhaseSpaceCFM(dim=dim, cond_dim=0, hidden_dim=hidden_dim, n_layers=n_layers)
    trainer = CFMTrainer(model, device=device, lr=1e-4)
    trainer.load(checkpoint_path)
    model.eval()
    print(f"    CFM: dim={dim}, hidden_dim={hidden_dim}, n_layers={n_layers}")
    return model


def _load_nsf(checkpoint_path: str, dim: int, device: str):
    import torch
    from models.nsf import PhaseSpaceNSF, NSFTrainer

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd   = ckpt.get("model") or ckpt

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

    n_bins = 8
    for k, v in sd.items():
        if "transform_net.final_layer.bias" in k:
            out_feats = v.shape[0]
            for b in [4, 8, 10, 12, 16, 20]:
                if out_feats % (3 * b - 1) == 0:
                    n_bins = b
                    break
            break

    model = PhaseSpaceNSF(dim=dim, cond_dim=0, n_transforms=n_transforms, hidden_dim=hidden_dim, n_bins=n_bins, tail_bound=7.0)
    trainer = NSFTrainer(model, device=device, lr=1e-4)
    trainer.load(checkpoint_path)
    model.eval()
    print(f"    NSF: dim={dim}, hidden_dim={hidden_dim}, n_transforms={n_transforms}, n_bins={n_bins}")
    return model


def generate_cfm_nsf(model_cfg: dict, n_samples: int, device: str = "cpu", n_ode_steps: int = 100) -> np.ndarray:
    import torch
    from data.synthetic_linac import denormalize_phase_space

    stats_path = model_cfg["stats_json"]
    checkpoint = model_cfg["checkpoint"]
    mtype      = model_cfg["model_type"]

    with open(stats_path) as f:
        stats = json.load(f)

    dim = len(stats.get("col_names", ["x", "y", "theta", "phi", "E"]))

    print(f"  Caricamento {mtype.upper()} da: {checkpoint}")
    if mtype == "cfm":
        model = _load_cfm(checkpoint, dim, device)
    else:
        model = _load_nsf(checkpoint, dim, device)

    print(f"  Generazione {n_samples:,} campioni...")
    chunks_norm = []
    n_done = 0
    t0 = time.time()

    with torch.no_grad():
        while n_done < n_samples:
            n_chunk = min(CHUNK_SIZE, n_samples - n_done)
            chunk = model.sample(n_chunk, n_steps=n_ode_steps) if mtype == "cfm" else model.sample(n_chunk)
            chunks_norm.append(chunk.cpu().numpy())
            n_done += n_chunk
            print(f"\r    {n_done:>12,}/{n_samples:,}  ({100*n_done/n_samples:.1f}%)  {time.time()-t0:.0f}s", end="", flush=True)
    print()

    gen_norm = np.concatenate(chunks_norm, axis=0)
    gen_phys = denormalize_phase_space(gen_norm, stats)
    return gen_phys.astype(np.float32)


def generate_gan_sarrut(model_cfg: dict, n_samples: int) -> np.ndarray:
    import torch
    import torch.nn as nn

    checkpoint = model_cfg["checkpoint"]
    stats_path = model_cfg["stats_json"]

    with open(stats_path) as f:
        stats = json.load(f)

    min_vals       = np.array(stats["min_vals"], dtype=np.float32)
    max_vals       = np.array(stats["max_vals"], dtype=np.float32)
    z_const        = float(stats.get("z_const", 0.0))
    active_indices = stats.get("active_indices", [0, 1, 3, 4, 5, 6])

    class _SarrutG(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(6, 400), nn.ReLU(inplace=True),
                nn.Linear(400, 400), nn.ReLU(inplace=True),
                nn.Linear(400, 400), nn.ReLU(inplace=True),
                nn.Linear(400, 6),   nn.Sigmoid(),
            )
        def forward(self, z): return self.model(z)

    print(f"  Caricamento GAN Sarrut da: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    G = _SarrutG()
    G.load_state_dict(ckpt["generator"])
    G.eval()

    print(f"  Generazione {n_samples:,} campioni...")
    chunks_6d = []
    n_done = 0
    t0 = time.time()

    with torch.no_grad():
        while n_done < n_samples:
            n_chunk = min(CHUNK_SIZE, n_samples - n_done)
            z = torch.randn(n_chunk, 6)
            g_norm = G(z).cpu().numpy()
            g_6d   = g_norm * (max_vals - min_vals) + min_vals
            chunks_6d.append(g_6d)
            n_done += n_chunk
            print(f"\r    {n_done:>12,}/{n_samples:,}  ({100*n_done/n_samples:.1f}%)  {time.time()-t0:.0f}s", end="", flush=True)
    print()

    gen_6d = np.concatenate(chunks_6d, axis=0)
    gen_7d = np.zeros((len(gen_6d), 7), dtype=np.float32)
    for c6, c7 in enumerate(active_indices):
        gen_7d[:, c7] = gen_6d[:, c6]
    gen_7d[:, 2] = z_const
    return gen_7d


def _apply_physical_filters(ps7: np.ndarray, model_name: str) -> np.ndarray:
    E, dx, dy, dz = ps7[:, 6], ps7[:, 3], ps7[:, 4], ps7[:, 5]
    norm_d = np.sqrt(dx**2 + dy**2 + dz**2)
    mask = (E > 0.01) & (norm_d > 0.99) & (norm_d < 1.01) & (np.abs(ps7[:, 0]) < 20.0) & (np.abs(ps7[:, 1]) < 20.0)
    print(f"  Filtro fisico: {mask.sum():,}/{len(ps7):,} mantenuti ({100*mask.sum()/len(ps7):.2f}%)")
    ps7_clean = ps7[mask].copy()
    ps7_clean[:, 3:6] /= np.sqrt(ps7_clean[:,3]**2 + ps7_clean[:,4]**2 + ps7_clean[:,5]**2)[:, np.newaxis]
    return ps7_clean


def generate_and_save(model_name: str, model_cfg: dict, n_samples: int, output_dir: Path, device: str = "cpu", n_ode_steps: int = 100, force: bool = False) -> Path:
    import uproot
    out_root = output_dir / f"gen_{model_name}.root"

    if out_root.exists() and not force:
        with uproot.open(out_root) as f:
            if "PhaseSpace" in f:
                print(f"  [{model_name.upper()}] File ROOT esistente ({f['PhaseSpace'].num_entries:,} campioni).")
                return out_root

    print(f"\n{'─'*55}\n  [{model_name.upper()}] {model_cfg['label']}\n{'─'*55}")
    mtype = model_cfg["model_type"]
    ps7 = generate_gan_sarrut(model_cfg, n_samples) if mtype == "gan_sarrut" else generate_cfm_nsf(model_cfg, n_samples, device, n_ode_steps)
    ps7_clean = _apply_physical_filters(ps7, model_name)

    # Conversione cm -> mm + Blindaggio geometrico Z=0 e dZ > 0
    with uproot.recreate(out_root) as f:
        f["PhaseSpace"] = {
            "X":  ps7_clean[:, 0] * 10.0,
            "Y":  ps7_clean[:, 1] * 10.0,
            "Z":  np.zeros(len(ps7_clean), dtype=np.float32),
            "dX": ps7_clean[:, 3],
            "dY": ps7_clean[:, 4],
            "dZ": np.abs(ps7_clean[:, 5]).astype(np.float32),
            "E":  ps7_clean[:, 6],
        }
    print(f"  Salvati {len(ps7_clean):,} campioni → {out_root}")
    return out_root


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SORGENTE DI RIFERIMENTO (PHSP2)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_reference_h5(phsp2_path: str, output_dir: Path, n_particles: int, seed: int = 42, force: bool = False) -> Path:
    import h5py
    import uproot
    ref_root = output_dir / "gen_reference.root"

    if ref_root.exists() and not force:
        with uproot.open(ref_root) as f:
            if "PhaseSpace" in f:
                print(f"  [REFERENCE] File ROOT esistente ({f['PhaseSpace'].num_entries:,} campioni).")
                return ref_root

    print(f"\n{'─'*55}\n  [REFERENCE] Preparazione sorgente PHSP2 reale\n{'─'*55}")
    with h5py.File(phsp2_path, "r") as f:
        ps_all = f["phase_space"][:]

    if n_particles < len(ps_all):
        ps_ref = ps_all[np.random.default_rng(seed).choice(len(ps_all), size=n_particles, replace=False)]
    else:
        ps_ref = ps_all

    ps_ref = ps_ref.astype(np.float32)

    with uproot.recreate(ref_root) as f:
        f["PhaseSpace"] = {
            "X":  ps_ref[:, 0] * 10.0,
            "Y":  ps_ref[:, 1] * 10.0,
            "Z":  np.zeros(len(ps_ref), dtype=np.float32),
            "dX": ps_ref[:, 3],
            "dY": ps_ref[:, 4],
            "dZ": np.abs(ps_ref[:, 5]).astype(np.float32),
            "E":  ps_ref[:, 6],
        }
    print(f"  Salvate {len(ps_ref):,} particelle di riferimento → {ref_root}")
    return ref_root


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIMULAZIONE GATE 10
# ═══════════════════════════════════════════════════════════════════════════════

def run_gate_dose(source_path: Path, output_dir: Path, run_name: str, n_particles: int, n_threads: int = 4, seed: int = 42, force: bool = False) -> Path:
    import opengate as gate
    from opengate import g4_units

    mm, cm, MeV = g4_units.mm, g4_units.cm, g4_units.MeV
    dose_gate10 = output_dir / f"dose_{run_name}_dose.mhd"
    dose_mhd    = output_dir / f"dose_{run_name}-dose.mhd"
    dose_base   = output_dir / f"dose_{run_name}.mhd"

    if (dose_gate10.exists() or dose_mhd.exists() or dose_base.exists()) and not force:
        return dose_gate10 if dose_gate10.exists() else (dose_mhd if dose_mhd.exists() else dose_base)

    print(f"\n{'─'*55}\n  GATE 10 — Dose: {run_name} | Thread: {n_threads}\n{'─'*55}")
    sim = gate.Simulation()
    sim.g4_verbose, sim.visu, sim.number_of_threads, sim.random_seed = False, False, n_threads, seed
    sim.world.size, sim.world.material = [100 * cm, 100 * cm, 120 * cm], "G4_AIR"
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.set_production_cut("world", "all", 2 * mm)

    water = sim.add_volume("Box", "water_phantom")
    water.size, water.material, water.translation = [WATER_BOX_CM * cm, WATER_BOX_CM * cm, WATER_BOX_CM * cm], "G4_WATER", [0, 0, (WATER_BOX_CM / 2) * cm]

    n_vox = int(WATER_BOX_CM * 10 / VOXEL_MM)
    dose = sim.add_actor("DoseActor", "dose")
    dose.attached_to, dose.size, dose.spacing, dose.output_filename = water.name, [n_vox, n_vox, n_vox], [VOXEL_MM * mm, VOXEL_MM * mm, VOXEL_MM * mm], str(dose_base)
    dose.hit_type, dose.dose.active, dose.dose_uncertainty.active = "random", True, True

    sim.add_actor("SimulationStatisticsActor", "stats").output_filename = str(output_dir / f"stats_{run_name}.txt")

    src = sim.add_source("PhaseSpaceSource", "phsp_src")
    src.phsp_file, src.particle, src.n = str(source_path), "gamma", n_particles
    src.position_key_x, src.position_key_y, src.position_key_z = "X", "Y", "Z"
    src.direction_key_x, src.direction_key_y, src.direction_key_z = "dX", "dY", "dZ"
    src.energy_key, src.primary_PDGCode, src.primary_lower_energy_threshold = "E", 22, 0.01 * MeV

    t0 = time.time()
    sim.run()
    print(f"  ✓ Dose calcolata in {time.time()-t0:.1f}s")
    return dose_gate10 if dose_gate10.exists() else (dose_mhd if dose_mhd.exists() else dose_base)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ANALISI GAMMA-INDEX TRIDIMENSIONALE CLINICA (PyMedPhys)
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_dose(dose_ref_path: Path, dose_model_path: Path, model_name: str, output_dir: Path, dd_pct: float = 2.0, dta_mm: float = 2.0, threshold_pct: float = 0.10) -> dict:
    import SimpleITK as sitk
    import pymedphys

    print(f"\n  [ANALISI CLINICA 3D] Modello: {model_name.upper()}")
    ref_img, model_img = sitk.ReadImage(str(dose_ref_path)), sitk.ReadImage(str(dose_model_path))
    ref, model = sitk.GetArrayFromImage(ref_img).astype(np.float64), sitk.GetArrayFromImage(model_img).astype(np.float64)

    # NORMALIZZAZIONE CLINICA LOCALE (Dmax individuale per correggere fluttuazioni/thread)
    ref_norm   = (ref / ref.max()) * 100.0
    model_norm = (model / model.max()) * 100.0

    # Calcolo delta puro per istogramma statistico (escludendo l'ombra sotto il 10%)
    mask = ref_norm > (threshold_pct * 100.0)
    delta_pct = ref_norm[mask] - model_norm[mask]

    # Assi spaziali 3D richiesti da pymedphys [Z, Y, X]
    axes = tuple(np.arange(s) * VOXEL_MM for s in ref.shape)

    # Esecuzione dell'algoritmo Gamma Index 3D con Ricerca Spaziale DTA
    gamma_map = pymedphys.gamma(
        axes, ref_norm,
        axes, model_norm,
        dose_percent_threshold=dd_pct,
        distance_mm_threshold=dta_mm,
        lower_percent_dose_cutoff=threshold_pct * 100.0,
        quiet=True
    )

    valid_gamma = gamma_map[~np.isnan(gamma_map)]
    pass_rate = float((valid_gamma <= 1.0).mean() * 100)

    print(f"    Voxel valutati (>{threshold_pct*100:.0f}% Dmax): {len(valid_gamma):,}")
    print(f"    Δ medio relativo: {delta_pct.mean():+.4f}%")
    print(f"    💥 GAMMA PASS RATE CLINICO: {pass_rate:.2f}%  [{dd_pct}% / {dta_mm}mm]")

    # Generazione dizionario compatibile con plot_all (vettori pre-normalizzati a scala 100)
    results = {
        "model": model_name, "n_voxels_evaluated": len(valid_gamma), "mean_diff_pct": float(delta_pct.mean()),
        "gamma_pass_rate_pct": pass_rate, "gamma_dd_pct": dd_pct, "gamma_dta_mm": dta_mm,
        "delta_pct_histogram": np.histogram(delta_pct, bins=200, range=(-5, 5))[0].tolist(),
        "pdd_ref": ref_norm.mean(axis=(1, 2)).tolist(), "pdd_model": model_norm.mean(axis=(1, 2)).tolist()
    }

    z_idx = int(20.0 / VOXEL_MM)
    if z_idx < ref.shape[0]:
        cy = ref.shape[1] // 2
        results["transverse_ref"]   = ref_norm[z_idx, cy, :].tolist()
        results["transverse_model"] = model_norm[z_idx, cy, :].tolist()

    with open(output_dir / f"dose_comparison_{model_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def plot_all(all_results: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = {k: v for k, v in all_results.items() if "error" not in v}
    if not valid: return

    COLORS = {"cfm": "#5B9BD5", "nsf": "#E05C5C", "gan": "#F4A460"}
    LSTYLES = {"cfm": "--", "nsf": "-.", "gan": ":"}

    # 1. Delta Hist
    fig, ax = plt.subplots(figsize=(8, 4.5))
    centers = (np.linspace(-5, 5, 201)[:-1] + np.linspace(-5, 5, 201)[1:]) / 2
    for name, res in valid.items():
        hist = np.array(res.get("delta_pct_histogram", []))
        if len(hist) == 200:
            ax.plot(centers, hist, color=COLORS.get(name, "gray"), linestyle=LSTYLES.get(name, "-"), label=f"{name.upper()} (μ={res['mean_diff_pct']:+.2f}%)")
    ax.axvline(0, color="black", alpha=0.4, linestyle=":")
    ax.set_xlabel("ΔD (Relative to D$_{max}$) [%]"); ax.set_ylabel("Voxel Count"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / "dose_delta_distribution.png", dpi=150, bbox_inches="tight"); plt.close()

    # 2. PDD (Normalizzato a 1.0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    z_mm = np.arange(int(WATER_BOX_CM * 10 / VOXEL_MM)) * VOXEL_MM
    ref_plotted = False
    for name, res in valid.items():
        p_ref, p_mod = np.array(res.get("pdd_ref", [])), np.array(res.get("pdd_model", []))
        if len(p_ref) == 0: continue
        if not ref_plotted:
            ax.plot(z_mm[:len(p_ref)], p_ref / p_ref.max(), "k-", linewidth=2.5, label="MC Reference (PHSP2)")
            ref_plotted = True
        ax.plot(z_mm[:len(p_ref)], p_mod / p_mod.max(), color=COLORS.get(name, "gray"), linestyle=LSTYLES.get(name, "--"), label=name.upper())
    ax.set_xlabel("Depth [mm]"); ax.set_ylabel("Relative Dose (D/D$_{max}$)"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / "dose_pdd.png", dpi=150, bbox_inches="tight"); plt.close()

    # 3. Transverse Profile
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n_vox = int(WATER_BOX_CM * 10 / VOXEL_MM)
    x_mm  = (np.arange(n_vox) - n_vox / 2) * VOXEL_MM
    ref_plotted = False
    for name, res in valid.items():
        t_ref, t_mod = np.array(res.get("transverse_ref", [])), np.array(res.get("transverse_model", []))
        if len(t_ref) == 0: continue
        if not ref_plotted:
            ax.plot(x_mm[:len(t_ref)], t_ref / t_ref.max(), "k-", linewidth=2.5, label="MC Reference")
            ref_plotted = True
        ax.plot(x_mm[:len(t_ref)], t_mod / t_mod.max(), color=COLORS.get(name, "gray"), linestyle=LSTYLES.get(name, "--"), label=name.upper())
    ax.set_xlabel("Lateral Position [mm]"); ax.set_ylabel("Relative Dose (D/D$_{max}$)"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / "dose_transverse_profile.png", dpi=150, bbox_inches="tight"); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all",            action="store_true")
    p.add_argument("--model",          choices=["cfm", "nsf", "gan"])
    p.add_argument("--generate_only",  action="store_true")
    p.add_argument("--reference_only", action="store_true")
    p.add_argument("--gamma_only",     action="store_true")
    p.add_argument("--n_particles",  type=int, default=DEFAULT_N)
    p.add_argument("--n_threads",    type=int, default=4)
    p.add_argument("--output_dir",   default="outputs/dose_validation")
    p.add_argument("--phsp2",        default=PHSP2_PATH)
    p.add_argument("--device",       default="cpu")
    p.add_argument("--n_ode_steps",  type=int, default=100)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--force",        action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  🚀 PIPELINE DOSIMETRICA UNIFICATA — PYMEDPHYS 3D\n  N Particelle: {args.n_particles:,}\n{'='*60}")
    models_to_run = {args.model: MODELS[args.model]} if args.model else MODELS

    # 1. Generazione
    gen_root = {}
    if not args.gamma_only and not args.reference_only:
        for name, cfg in models_to_run.items():
            if not Path(cfg["checkpoint"]).exists(): continue
            gen_root[name] = generate_and_save(name, cfg, args.n_particles, out, args.device, args.n_ode_steps, args.force)

    if args.generate_only: return

    # 2. Reference
    ref_root = prepare_reference_h5(args.phsp2, out, args.n_particles, args.seed, args.force) if (args.all or args.reference_only) else None
    if args.reference_only: return

    # 3. Simulazioni GATE 10
    if ref_root and (args.all or args.reference_only):
        run_gate_dose(ref_root, out, "reference", args.n_particles, args.n_threads, args.seed, args.force)

    for name, rpath in gen_root.items():
        run_gate_dose(rpath, out, name, args.n_particles, args.n_threads, args.seed, args.force)

    # 4. Analisi Gamma Index 3D
    print(f"\n[ANALISI] Elaborazione Gamma Index 3D...")
    ref_dose = out / "dose_reference_dose.mhd"
    if not ref_dose.exists(): ref_dose = out / "dose_reference-dose.mhd"

    all_results = {}
    for name in MODELS.keys():
        if args.model and name != args.model: continue
        m_dose = out / f"dose_{name}_dose.mhd"
        if not m_dose.exists(): m_dose = out / f"dose_{name}-dose.mhd"
        
        if m_dose.exists() and ref_dose.exists():
            all_results[name] = analyse_dose(ref_dose, m_dose, name, out)

    if all_results:
        plot_all(all_results, out)
    print(f"\n✅ Pipeline completata con successo! Grafici e report salvati in: {out}")

if __name__ == "__main__":
    main()
