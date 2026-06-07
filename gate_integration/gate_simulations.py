"""
gate_integration/gate_simulations.py
======================================
Script di simulazione GATE 10 per:

    A) Generare dati di training (phase space a un piano isocentrico)
    B) Validazione downstream: simulazione dose in fantoccio d'acqua
       con sorgente = modello generativo, confronto con gold standard MC

Richiede: pip install opengate

Riferimento GATE 10 docs:
    https://opengate-python.readthedocs.io/en/master/user_guide/user_guide_reference_sources_gan_source.html

Uso:
    # A) Genera dati di training (linac Elekta Synergy simulato)
    python gate_integration/gate_simulations.py generate \\
        --n_particles 1e8 \\
        --output_dir data/gate_phsp

    # B) Validazione dose gold standard (reference MC)
    python gate_integration/gate_simulations.py dose_reference \\
        --n_particles 1e7 \\
        --output_dir outputs/dose_reference

    # C) Validazione dose con modello GAN/NSF/CFM
    python gate_integration/gate_simulations.py dose_model \\
        --pth_filename linac_6MV_nsf.pth \\
        --n_particles 1e7 \\
        --output_dir outputs/dose_nsf
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Import GATE 10 ───────────────────────────────────────────────────────────
try:
    import opengate as gate
    from opengate import g4_units
    GATE_AVAILABLE = True
    
    # Riferimento globale statico per isolare il generatore dalle strutture Box
    GLOBAL_REAL_GENERATOR = None
    
    # ─── BYPASS INTEGRALE E MONKEY-PATCH DI GAGA_PHSP ────────────────────────
    try:
        import gaga_phsp
        import torch
        import torch.nn as nn
        import numpy as np
        
        class GATEGeneratedSourceWrapper(nn.Module):
            """
            Intercetta le richieste di GATE, riconosce l'architettura (CFM o NSF),
            campiona dallo spazio latente, applica la denormalizzazione corretta
            e restituisce i vettori fisici pronti per Geant4.
            """
            def __init__(self, checkpoint_path):
                super().__init__()
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                self.stats = ckpt["_phase_space_gen"]
                self.col_names = self.stats["col_names"]
                dim = len(self.col_names)
                
                # Pulizia dei pesi da eventuali prefissi
                state_dict = ckpt["G_model_state"]
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("inner."):
                        cleaned_state_dict[k[6:]] = v
                    else:
                        cleaned_state_dict[k] = v
                
                # ─── RILEVAMENTO DINAMICO DELL'ARCHITETTURA ──────────────────
                if "net.0.weight" in cleaned_state_dict:
                    # Baseline GAN Sarrut 2019: Sequential net.0/2/4/6
                    h_dim = cleaned_state_dict["net.0.weight"].shape[0]
                    z_dim = cleaned_state_dict["net.0.weight"].shape[1]
                    x_dim = cleaned_state_dict["net.6.weight"].shape[0]
                    import torch.nn as _nn
                    class _SarrautGAN(_nn.Module):
                        def __init__(self, zd, hd, xd):
                            super().__init__(); self.z_dim=zd
                            self.net=_nn.Sequential(
                                _nn.Linear(zd,hd),_nn.ReLU(),
                                _nn.Linear(hd,hd),_nn.ReLU(),
                                _nn.Linear(hd,hd),_nn.ReLU(),
                                _nn.Linear(hd,xd))
                        def forward(self,z): return self.net(z)
                        def sample(self,n,device="cpu"):
                            with torch.no_grad():
                                return self.forward(torch.randn(n,self.z_dim,device=device))
                    self.inner_model = _SarrautGAN(z_dim, h_dim, x_dim)
                elif "velocity_net.input_proj.weight" in cleaned_state_dict:
                    # È un modello CFM (Flow Matching)
                    from models.cfm import PhaseSpaceCFM
                    hidden_dim = cleaned_state_dict["velocity_net.input_proj.weight"].shape[0]
                    cond_dim = 3 if any("cond_embed" in k for k in cleaned_state_dict.keys()) else 0
                    print(f"  [INFO Wrapper] Inizializzazione dinamica CFM (hidden_dim={hidden_dim}, cond_dim={cond_dim})")
                    self.inner_model = PhaseSpaceCFM(dim=dim, cond_dim=cond_dim, hidden_dim=hidden_dim)
                else:
                    # È un modello NSF (Normalizing Flow)
                    from models.nsf import PhaseSpaceNSF
                    print(f"  [INFO Wrapper] Inizializzazione dinamica NSF sferico/cartesiano")
                    self.inner_model = PhaseSpaceNSF(dim=dim, cond_dim=0, hidden_dim=256)
                # ─────────────────────────────────────────────────────────────
                
                self.inner_model.load_state_dict(cleaned_state_dict)
                self.inner_model.eval()
                
            def forward(self, z):
                B = z.shape[0]
                device = z.device
                self.inner_model.to(device)
                
                with torch.no_grad():
                    s_norm = self.inner_model.sample(B)
                
                s_norm_np = s_norm.cpu().numpy()
                s_phys = s_norm_np.copy()
                for i, col in enumerate(self.col_names):
                    mu = self.stats["original_mean"][col]
                    sigma = self.stats["original_std"][col]
                    s_phys[:, i] = s_norm_np[:, i] * sigma + mu
                
                gate_out = np.zeros((B, 7), dtype=np.float32)
                
                # Gestione automatica VECCHIO MODELLO (Cartesiano 6D)
                if len(self.col_names) == 6:
                    gate_out[:, 0] = s_phys[:, 5]   # Ekine = E
                    gate_out[:, 1] = s_phys[:, 0]   # X = x
                    gate_out[:, 2] = s_phys[:, 1]   # Y = y
                    gate_out[:, 3] = ckpt.get("params", {}).get("z_const_cm", 0.0)  # 27.21cm for IAEA            # Z = 0
                    gate_out[:, 4] = s_phys[:, 2]   # dX = dx
                    gate_out[:, 5] = s_phys[:, 3]   # dY = dy
                    gate_out[:, 6] = s_phys[:, 4]   # dZ = dz
                # Gestione automatica NUOVO MODELLO (Sferico 5D)
                else:
                    x, y = s_phys[:, 0], s_phys[:, 1]
                    theta, phi = s_phys[:, 2], s_phys[:, 3]
                    E = s_phys[:, 4]
                    
                    sin_theta = np.sin(theta)
                    dx = sin_theta * np.cos(phi)
                    dy = sin_theta * np.sin(phi)
                    dz = np.cos(theta)
                    
                    gate_out[:, 0] = E
                    gate_out[:, 1] = x
                    gate_out[:, 2] = y
                    gate_out[:, 3] = ckpt.get("params", {}).get("z_const_cm", 0.0)  # 27.21cm for IAEA
                    gate_out[:, 4] = dx
                    gate_out[:, 5] = dy
                    gate_out[:, 6] = dz
                    
                return torch.from_numpy(gate_out).to(device)

        def mock_gaga_load(filename, verbose=False):
            """Sostituisce il caricatore originale di gaga_phsp."""
            global GLOBAL_REAL_GENERATOR
            GLOBAL_REAL_GENERATOR = GATEGeneratedSourceWrapper(filename)
            
            ckpt = torch.load(filename, map_location="cpu", weights_only=False)
            params = ckpt["params"]
            
            gate_keys_labels = ["Ekine", "X", "Y", "Z", "dX", "dY", "dZ"]
            params["keys"] = gate_keys_labels
            params["keys_list"] = gate_keys_labels
            params["cond_keys"] = []
            params["gpu_mode"] = "auto"
            
            return params, GLOBAL_REAL_GENERATOR, None, None

        def mock_generate_samples_non_cond(G, params, n, *args, **kwargs):
            """Bypass completo della generazione interna di gaga_phsp."""
            global GLOBAL_REAL_GENERATOR
            import torch
            res = GLOBAL_REAL_GENERATOR(torch.zeros(n, 6))
            if hasattr(res, "cpu"):
                return res.cpu().numpy()
            return res

        # Iniezione globale delle funzioni surrogate
        gaga_phsp.load = mock_gaga_load
        gaga_phsp.generate_samples_non_cond = mock_generate_samples_non_cond
        gaga_phsp.check_input_params = lambda *args, **kwargs: None
        
        import gaga_phsp.gaga_helpers
        gaga_phsp.gaga_helpers.load = mock_gaga_load
        gaga_phsp.gaga_helpers.generate_samples_non_cond = mock_generate_samples_non_cond
        gaga_phsp.gaga_helpers.check_input_params = lambda *args, **kwargs: None
        
        print("  [HACK SUPER] Bypass totale della generazione gaga_phsp attivata.")
    except Exception as e:
        print(f"  [WARNING] Impossibile configurare il super bypass: {e}")
    # ─────────────────────────────────────────────────────────────────────────

except ImportError:
    GATE_AVAILABLE = False
    # Avviso non fatale: lo script può girare senza GATE per ispezione
    print("[WARNING] opengate non installato.")
    print("Installare sulla workstation: pip install opengate")
    print("(Richiede Geant4 compilato — vedi https://opengate-python.readthedocs.io)")


# ─── Costanti fisiche (GATE units) ────────────────────────────────────────────
if GATE_AVAILABLE:
    mm  = g4_units.mm
    cm  = g4_units.cm
    MeV = g4_units.MeV
    MBq = g4_units.Bq * 1e6
    keV = g4_units.keV
    deg = g4_units.deg
    s   = g4_units.s
else:
    mm = 1e-3; cm = 1e-2; MeV = 1.0; MBq = 1e6; keV = 1e-3; deg = 1.0; s = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# A) GENERAZIONE DATI DI TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def create_linac_geometry(sim):
    """
    Geometria semplificata di un linac Elekta Synergy per generare il phase space.

    Per la tesi usiamo la geometria pre-configurata di GATE 10:
        opengate.contrib.linac.elekta_synergy

    Se disponibile, la usiamo; altrimenti creiamo una geometria semplificata.
    """
    try:
        # GATE 10 ha l'Elekta Synergy pre-configurato
        from opengate.contrib.linac import elekta_synergy
        linac = elekta_synergy.add_linac(sim, name="elekta")
        print("  Geometria Elekta Synergy caricata da opengate.contrib")
        return linac
    except (ImportError, AttributeError):
        pass

    # Fallback: sorgente puntiforme con spettro 6MV semplificato
    print("  Usando sorgente semplificata (Elekta contrib non disponibile)")
    world = sim.world
    world.size = [1 * cm, 1 * cm, 3 * cm]

    src = sim.add_source("GenericSource", "linac_simplified")
    src.particle         = "gamma"
    src.n                = 1
    src.position.type    = "point"
    src.energy.type      = "spectrum_lines"
    # Spettro 6MV approssimato con componenti discrete
    src.energy.spectrum_energies = [0.5, 1.0, 2.0, 4.0, 6.0]  # MeV
    src.energy.spectrum_weights  = [0.30, 0.25, 0.20, 0.15, 0.10]
    return src


def run_generate_training_data(args):
    """
    A) Genera il phase space di training salvando le particelle
       che attraversano il piano isocentrico (z = 0, SSD = 100 cm).

    Output: file ROOT (.root) + conversione HDF5 compatibile con il nostro trainer.
    """
    if not GATE_AVAILABLE:
        print("[ERROR] opengate non installato. Impossibile eseguire la simulazione.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  GATE 10 — Generazione Phase Space di Training")
    print(f"  N particelle: {int(args.n_particles):,}")
    print(f"  Output: {output_dir}")
    print(f"{'='*55}")

    sim = gate.Simulation()
    sim.g4_verbose = False
    sim.visu = False
    sim.number_of_threads = args.n_threads
    sim.random_seed = args.seed

    # Geometria
    sim.world.size     = [1 * cm, 1 * cm, 2.5 * cm]
    sim.world.material = "G4_AIR"
    sim.physics_manager.physics_list_name = "QGSP_BIC_EMY"

    # Sorgente
    linac_src = create_linac_geometry(sim)

    # PhaseSpace Actor: registra le particelle sul piano z=0
    # SSD (Source-to-Surface Distance) = 100 cm, piano isocentrico
    phsp_plane = sim.add_volume("Box", "phsp_plane")
    phsp_plane.size     = [30 * cm, 30 * cm, 1 * mm]
    phsp_plane.material = "G4_AIR"

    phsp_actor = sim.add_actor("PhaseSpaceActor", "phsp_actor")
    phsp_actor.attached_to  = phsp_plane.name
    phsp_actor.output_filename = str(output_dir / "linac_phsp.root")
    phsp_actor.attributes = [
        "KineticEnergy",
        "PrePosition",
        "PreDirection",
        "ParticleName",
    ]
    phsp_actor.filters.append(
        sim.add_filter("ParticleFilter", "gamma_only")
    )

    # Statistiche
    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.output_filename = str(output_dir / "stats.txt")

    # Avvia
    sim.run()

    print(f"\n  Phase space salvato: {output_dir}/linac_phsp.root")
    print(f"\n  Prossimo step: convertire in HDF5 per il training:")
    print(f"    python gate_integration/convert_root_to_hdf5.py \\")
    print(f"        --input {output_dir}/linac_phsp.root \\")
    print(f"        --output data/linac_6MV_train.h5")


# ═══════════════════════════════════════════════════════════════════════════════
# B) VALIDAZIONE DOSE — GOLD STANDARD MC
# ═══════════════════════════════════════════════════════════════════════════════

def _build_water_phantom_simulation(sim, n_particles, source_type, **source_kwargs):
    """
    Costruisce una simulazione GATE 10 con fantoccio d'acqua 20×20×20 cm³
    (voxel 4×4×4 mm³, come nel paper Sarrut 2019).

    Usata sia per il gold standard (PhaseSpaceSource) che per il GAN (GANSource).
    """
    sim.g4_verbose = False
    sim.visu       = False
    sim.physics_manager.physics_list_name = "QGSP_BIC_EMY"

    # World
    sim.world.size     = [50 * cm, 50 * cm, 60 * cm]
    sim.world.material = "G4_AIR"

    # Fantoccio d'acqua: 20×20×20 cm³ come nel paper
    water_box = sim.add_volume("Box", "water_box")
    water_box.size       = [20 * cm, 20 * cm, 20 * cm]
    water_box.material   = "G4_WATER"
    water_box.translation = [0, 0, 10 * cm]  # centro a 10 cm dalla sorgente

    # DoseActor: calcolo dose 3D, voxel 4×4×4 mm³ (paper Sarrut 2019)
    dose_actor = sim.add_actor("DoseActor", "dose_actor")
    dose_actor.attached_to    = water_box.name
    dose_actor.size            = [50, 50, 50]        # voxel nel volume 20cm
    dose_actor.spacing         = [4 * mm, 4 * mm, 4 * mm]
    dose_actor.hit_type        = "random"


    dose_actor.dose.active = True
    dose_actor.dose_uncertainty.active = True

    # Statistiche
    stats = sim.add_actor("SimulationStatisticsActor", "stats")

    return water_box, dose_actor, stats


def run_dose_reference(args):
    """
    B) Simulazione gold standard: usa il file di phase space GATE (ROOT)
       come sorgente e calcola la dose nel fantoccio.

    Produce la mappa di dose di riferimento per confronto downstream.
    """
    if not GATE_AVAILABLE:
        print("[ERROR] opengate non installato.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  GATE 10 — Dose Reference (Phase Space Source)")

    sim = gate.Simulation()
    sim.number_of_threads = args.n_threads
    sim.random_seed = args.seed

    water_box, dose_actor, stats = _build_water_phantom_simulation(
        sim, int(args.n_particles), "phsp"
    )

    dose_actor.output_filename = str(output_dir / "dose_reference.mhd")
    stats.output_filename      = str(output_dir / "stats_reference.txt")

    # PhaseSpaceSource: usa il file ROOT generato in step A
    src = sim.add_source("PhaseSpaceSource", "phsp_source")
    src.phsp_file  = args.phsp_file
    src.particle   = "gamma"
    src.n          = int(args.n_particles)

    sim.run()
    print(f"  Dose reference salvata: {output_dir}/dose_reference.mhd")


def run_dose_model(args):
    """
    C) Validazione dose con il modello generativo (GAN/NSF/CFM).

    Usa GANSource di GATE 10 per generare le particelle dal modello .pth
    e calcola la dose nel fantoccio.
    Poi confronta con il gold standard calcolato in B).
    """
    if not GATE_AVAILABLE:
        print("[ERROR] opengate non installato.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(args.pth_filename).stem
    print(f"\n  GATE 10 — Dose con modello: {model_name}")

    sim = gate.Simulation()
    sim.number_of_threads = args.n_threads
    sim.random_seed = args.seed

    water_box, dose_actor, stats = _build_water_phantom_simulation(
        sim, int(args.n_particles), "gan"
    )

    dose_actor.output_filename = str(output_dir / f"dose_{model_name}.mhd")
    stats.output_filename      = str(output_dir / f"stats_{model_name}.txt")

    # GANSource: usa il modello .pth
    # Documentazione: https://opengate-python.readthedocs.io/en/master/
    #   user_guide/user_guide_reference_sources_gan_source.html
    gsource = sim.add_source("GANSource", "gan_source")
    gsource.particle        = "gamma"
    gsource.n               = int(args.n_particles)
    gsource.pth_filename    = args.pth_filename

    # Chiavi nel formato che salviamo con save_for_gate.py
    gsource.position_keys  = ["X",     "Y",     "Z"]
    gsource.direction_keys = ["dX",    "dY",    "dZ"]
    gsource.energy_key     = "Ekine"
    gsource.time_key       = None
    gsource.weight_key     = None

    gsource.energy_min_threshold = 10 * keV
    gsource.skip_policy    = "ZeroEnergy"
    gsource.batch_size     = 50_000
    gsource.gpu_mode       = "auto"

    sim.run()
    print(f"  Dose {model_name} salvata: {output_dir}/dose_{model_name}.mhd")


# ═══════════════════════════════════════════════════════════════════════════════
# D) ANALISI GAMMA-INDEX (Fase 4 della roadmap)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gamma_index(
    dose_reference_path: str,
    dose_model_path: str,
    dd: float = 2.0,   # % dose difference (paper standard: 2%)
    dta: float = 2.0,  # mm distance-to-agreement (paper: 2mm)
    threshold: float = 0.1,  # solo voxel > 10% del massimo
) -> dict:
    """
    Calcola il gamma-index tra dose di riferimento e dose del modello.

    Il gamma-index è la metrica standard di QA in radioterapia:
        γ(r) = min_{r'} sqrt[(ΔD(r,r')/dd)² + (|r-r'|/dta)²]

    Pass rate = frazione di voxel con γ ≤ 1 (obiettivo: > 95%).

    Richiede SimpleITK per leggere i file .mhd:
        pip install SimpleITK

    Parametri
    ---------
    dd  : criterio percentuale sulla dose (default: 2%)
    dta : criterio sulla distanza in mm (default: 2mm)
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        print("[WARNING] SimpleITK non installato. pip install SimpleITK")
        return {}

    ref_img   = sitk.ReadImage(dose_reference_path)
    model_img = sitk.ReadImage(dose_model_path)

    ref   = sitk.GetArrayFromImage(ref_img).astype(np.float64)
    model = sitk.GetArrayFromImage(model_img).astype(np.float64)
    spacing = np.array(ref_img.GetSpacing())  # mm

    D_max  = ref.max()
    mask   = ref > threshold * D_max

    # Gamma locale (approssimazione: confronto voxel-by-voxel senza ricerca spaziale)
    # Per gamma-index rigoroso serve ricerca spaziale (più costoso)
    delta_D_pct = np.abs(ref[mask] - model[mask]) / D_max * 100
    gamma_approx = delta_D_pct / dd  # solo termine dose, approx

    pass_rate = float((gamma_approx <= 1.0).mean() * 100)

    # Confronto punto-a-punto (proxy per fig 6 del paper)
    delta_pct = (ref[mask] - model[mask]) / D_max * 100

    results = {
        "dd_criterion_pct":    dd,
        "dta_criterion_mm":    dta,
        "n_voxels_evaluated":  int(mask.sum()),
        "pass_rate_pct":       pass_rate,
        "mean_diff_pct":       float(delta_pct.mean()),
        "std_diff_pct":        float(delta_pct.std()),
        "max_abs_diff_pct":    float(np.abs(delta_pct).max()),
    }

    print(f"\n  Gamma-index ({dd}%/{dta}mm):")
    print(f"  Pass rate:     {pass_rate:.2f}%  (obiettivo: > 95%)")
    print(f"  Mean diff:     {results['mean_diff_pct']:+.3f}%")
    print(f"  Std diff:      {results['std_diff_pct']:.3f}%")
    print(f"  Max |diff|:    {results['max_abs_diff_pct']:.3f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# E) CONVERSIONE ROOT → HDF5
# ═══════════════════════════════════════════════════════════════════════════════

def convert_root_to_hdf5(root_path: str, hdf5_path: str, max_n: int = None):
    """
    Converte un file ROOT di phase space GATE in HDF5.

    Richiede: uproot (pip install uproot)

    Il file ROOT di GATE contiene le seguenti branches per PhaseSpaceActor:
        KineticEnergy, PrePosition_X, PrePosition_Y, PrePosition_Z,
        PreDirection_X, PreDirection_Y, PreDirection_Z
    """
    try:
        import uproot
    except ImportError:
        print("[ERROR] uproot non installato: pip install uproot awkward")
        return

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.synthetic_linac import save_phase_space_hdf5

    print(f"\n  Conversione ROOT → HDF5: {root_path}")

    with uproot.open(root_path) as f:
        # Il tree si chiama "PhaseSpace" in GATE 10
        tree_name = [k for k in f.keys() if "PhaseSpace" in k or "phsp" in k.lower()]
        tree_name = tree_name[0] if tree_name else list(f.keys())[0]
        print(f"  Tree: {tree_name}")
        tree = f[tree_name]

        branches = tree.keys()
        print(f"  Branches disponibili: {branches}")

        # Carica le colonne necessarie
        E  = tree["KineticEnergy"].array(library="np")
        x  = tree["PrePosition_X"].array(library="np") / 10  # mm → cm
        y  = tree["PrePosition_Y"].array(library="np") / 10
        z  = tree["PrePosition_Z"].array(library="np") / 10
        dx = tree["PreDirection_X"].array(library="np")
        dy = tree["PreDirection_Y"].array(library="np")
        dz = tree["PreDirection_Z"].array(library="np")

    if max_n is not None:
        E  = E[:max_n]; x  = x[:max_n]; y  = y[:max_n]
        z  = z[:max_n]; dx = dx[:max_n]; dy = dy[:max_n]; dz = dz[:max_n]

    ps7 = np.column_stack([x, y, z, dx, dy, dz, E]).astype(np.float32)

    save_phase_space_hdf5(
        ps7, None, hdf5_path,
        metadata={"source": root_path, "n_particles": str(len(ps7))}
    )
    print(f"  Convertiti {len(ps7):,} campioni → {hdf5_path}")
    return hdf5_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="GATE 10 simulation scripts per phase space e validazione dose",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # A) Genera dati di training
    gen_p = sub.add_parser("generate", help="Genera phase space di training con GATE")
    gen_p.add_argument("--n_particles", type=float, default=1e8)
    gen_p.add_argument("--output_dir",  default="data/gate_phsp")
    gen_p.add_argument("--n_threads",   type=int, default=4)
    gen_p.add_argument("--seed",        type=int, default=42)

    # B) Dose reference
    ref_p = sub.add_parser("dose_reference", help="Simulazione dose gold standard")
    ref_p.add_argument("--phsp_file",   required=True,
                       help="File ROOT del phase space GATE (da step generate)")
    ref_p.add_argument("--n_particles", type=float, default=1e7)
    ref_p.add_argument("--output_dir",  default="outputs/dose_reference")
    ref_p.add_argument("--n_threads",   type=int, default=4)
    ref_p.add_argument("--seed",        type=int, default=42)

    # C) Dose con modello generativo
    mod_p = sub.add_parser("dose_model", help="Validazione dose con modello GAN/NSF/CFM")
    mod_p.add_argument("--pth_filename", required=True,
                       help="File .pth del modello (da save_for_gate.py)")
    mod_p.add_argument("--n_particles",  type=float, default=1e7)
    mod_p.add_argument("--output_dir",   default="outputs/dose_model")
    mod_p.add_argument("--n_threads",    type=int, default=4)
    mod_p.add_argument("--seed",         type=int, default=42)

    # D) Gamma-index
    gi_p = sub.add_parser("gamma_index", help="Calcola gamma-index tra due mappe di dose")
    gi_p.add_argument("--reference", required=True,  help="Dose reference .mhd")
    gi_p.add_argument("--model",     required=True,  help="Dose modello .mhd")
    gi_p.add_argument("--dd",        type=float, default=2.0)
    gi_p.add_argument("--dta",       type=float, default=2.0)
    gi_p.add_argument("--out_json",  default=None)

    # E) Conversione ROOT → HDF5
    conv_p = sub.add_parser("convert", help="Converte ROOT phase space → HDF5")
    conv_p.add_argument("--input",   required=True, help="File ROOT GATE")
    conv_p.add_argument("--output",  required=True, help="Output HDF5")
    conv_p.add_argument("--max_n",   type=int, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if   args.command == "generate":
        run_generate_training_data(args)
    elif args.command == "dose_reference":
        run_dose_reference(args)
    elif args.command == "dose_model":
        run_dose_model(args)
    elif args.command == "gamma_index":
        results = compute_gamma_index(
            args.reference, args.model, args.dd, args.dta
        )
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Risultati salvati: {args.out_json}")
    elif args.command == "convert":
        convert_root_to_hdf5(args.input, args.output, args.max_n)
