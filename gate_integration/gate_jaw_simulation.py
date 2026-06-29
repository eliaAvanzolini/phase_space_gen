"""
gate_integration/gate_jaw_simulation.py
========================================
Macro OpenGATE 10 parametrica per generare phase space sotto le jaws.

STRADA B: prende il fascio aperto IAEA (6MV o 10MV), piazza i blocchi
delle ganasce (jaws) a 5×5, 10×10 o 20×20, e registra un nuovo
Phase Space sotto di esse.

Architettura simulazione:
    Z_source (27.21 cm)  →  fascio aperto IAEA
         ↓
    Jaw blocks (tungsteno, ~Z=35-50 cm)  →  collimazione geometrica
         ↓
    PHSP plane (Z=100 cm, SSD)  →  registrazione particelle
         ↓  (output)
    File ROOT con particelle collimate

Uso:
    # Singola configurazione: 6MV con campo 5×5
    python gate_integration/gate_jaw_simulation.py \\
        --phsp_source data/ELEKTA_PRECISE_6mv_part1_big.IAEAphsp \\
        --E_nom 6.0 --jaw_x 2.5 --jaw_y 2.5 \\
        --output_dir outputs/gate_jaw/6mv_5x5

    # Batch: tutte le combinazioni
    python gate_integration/gate_jaw_simulation.py --batch

    # Convertire output ROOT → HDF5 condizionale
    python gate_integration/gate_jaw_simulation.py --convert_all

Requisiti:
    pip install opengate
    (richiede Geant4 compilato sul cluster)
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Costanti fisiche della testata Elekta Precise ──────────────────────────

# Geometria ganasce (da Sarrut 2019 + specifiche Elekta):
# Le jaws sono due coppie di blocchi di tungsteno perpendicolari:
#   - Y jaws (upper): ~Z = 28-36 cm dal target
#   - X jaws (lower): ~Z = 36-44 cm dal target
# Il PHSP IAEA è registrato a Z = 27.21 cm (sopra le jaws)

JAW_GEOMETRY = {
    "material": "G4_W",  # tungsteno
    "Y_jaw": {
        "z_center_cm": 32.0,    # centro delle Y jaws
        "thickness_cm": 7.8,    # spessore lungo Z
        "width_cm": 20.0,       # larghezza fissa (non collimante)
    },
    "X_jaw": {
        "z_center_cm": 40.0,    # centro delle X jaws
        "thickness_cm": 7.8,
        "width_cm": 20.0,
    },
}

# Piano di registrazione PHSP sotto le jaws
PHSP_PLANE_Z_CM = 50.0  # appena sotto la X jaw (Z=44 cm)

# Configurazioni batch da simulare
BATCH_CONFIGS = [
    # (nome, E_nom, jaw_x_cm, jaw_y_cm, file_sorgente)
    ("6mv_5x5",   6.0, 2.5, 2.5, "data/ELEKTA_PRECISE_6mv_part1_big.IAEAphsp"),
    ("6mv_10x10", 6.0, 5.0, 5.0, "data/ELEKTA_PRECISE_6mv_part1_big.IAEAphsp"),
    ("6mv_20x20", 6.0, 10.0, 10.0, "data/ELEKTA_PRECISE_6mv_part1_big.IAEAphsp"),
    ("10mv_5x5",  10.0, 2.5, 2.5, "data/ELEKTA_PRECISE_10mv_part1.IAEAphsp"),
    ("10mv_10x10",10.0, 5.0, 5.0, "data/ELEKTA_PRECISE_10mv_part1.IAEAphsp"),
    ("10mv_20x20",10.0, 10.0, 10.0, "data/ELEKTA_PRECISE_10mv_part1.IAEAphsp"),
]


def create_jaw_simulation(
    phsp_source: str,
    jaw_x_cm: float,
    jaw_y_cm: float,
    E_nom: float,
    output_dir: str,
    n_particles: int = 0,  # 0 = tutte le particelle nel file
    n_threads: int = 1,
    seed: int = 42,
    phsp_plane_z_cm: float = PHSP_PLANE_Z_CM,
) -> str:
    """
    Crea e lancia una simulazione GATE con jaws parametriche.

    L'architettura è:
        1. Sorgente IAEA (PhaseSpaceSource) a Z=27.21 cm
        2. Due coppie di blocchi di tungsteno (Y jaws + X jaws)
        3. Piano di registrazione PHSP sotto le jaws

    Parameters
    ----------
    phsp_source   : path al file .IAEAphsp sorgente
    jaw_x_cm      : semi-apertura X delle jaws [cm]
    jaw_y_cm      : semi-apertura Y delle jaws [cm]
    E_nom         : energia nominale (per etichettatura)
    output_dir    : cartella output
    n_particles   : 0 = usa tutte le particelle nel file
    n_threads     : thread Geant4
    seed          : random seed
    phsp_plane_z_cm : Z del piano di registrazione [cm]

    Returns
    -------
    Path del file ROOT di output
    """
    try:
        import opengate as gate
        from opengate import g4_units
    except ImportError:
        print("[ERROR] opengate non installato.")
        print("Installare sul cluster: pip install opengate")
        print("\nAlternativa: genera lo script macro e lancialo manualmente.")
        _generate_macro_script(
            phsp_source, jaw_x_cm, jaw_y_cm, E_nom,
            output_dir, n_particles, phsp_plane_z_cm
        )
        return None

    mm = g4_units.mm
    cm = g4_units.cm
    MeV = g4_units.MeV

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    field_name = f"{E_nom:.0f}mv_{2*jaw_x_cm:.0f}x{2*jaw_y_cm:.0f}"
    print(f"\n{'='*60}")
    print(f"  GATE Jaw Simulation: {field_name}")
    print(f"  Source: {phsp_source}")
    print(f"  Jaws: X=±{jaw_x_cm} cm, Y=±{jaw_y_cm} cm")
    print(f"  PHSP plane: Z={phsp_plane_z_cm} cm")
    print(f"{'='*60}")

    # ── Simulazione ──────────────────────────────────────────────────────
    sim = gate.Simulation()
    sim.g4_verbose = False
    sim.visu = False
    sim.number_of_threads = n_threads
    sim.random_seed = seed

    # World: abbastanza grande per contenere la testata
    sim.world.size = [60 * cm, 60 * cm, 120 * cm]
    sim.world.material = "G4_AIR"

    # Physics
    sim.physics_manager.physics_list_name = "QGSP_BIC_EMY"
    sim.physics_manager.global_production_cuts.gamma = 1 * mm
    sim.physics_manager.global_production_cuts.electron = 1 * mm
    sim.physics_manager.global_production_cuts.positron = 1 * mm

    # ── Sorgente: PHSP IAEA ──────────────────────────────────────────────
    # La sorgente emette le particelle registrate nel file IAEA
    src = sim.add_source("PhaseSpaceSource", "iaea_source")
    src.phsp_file = phsp_source
    src.particle = "gamma"
    if n_particles > 0:
        src.n = n_particles

    # ── JAWS (blocchi di tungsteno) ──────────────────────────────────────
    jaw_cfg = JAW_GEOMETRY

    # Y Jaws (coppia superiore, collimano lungo Y)
    y_cfg = jaw_cfg["Y_jaw"]
    for side, sign in [("pos", +1), ("neg", -1)]:
        jaw = sim.add_volume("Box", f"y_jaw_{side}")
        jaw.material = jaw_cfg["material"]
        # Il blocco si estende da jaw_y_cm fino al bordo esterno
        jaw_half_size = (y_cfg["width_cm"] - jaw_y_cm) / 2
        jaw.size = [
            y_cfg["width_cm"] * cm,     # X: non collimante, tutta la larghezza
            jaw_half_size * 2 * cm,      # Y: dal bordo del campo al muro
            y_cfg["thickness_cm"] * cm,  # Z: spessore
        ]
        jaw.translation = [
            0,
            sign * (jaw_y_cm + jaw_half_size) * cm,
            y_cfg["z_center_cm"] * cm,
        ]

    # X Jaws (coppia inferiore, collimano lungo X)
    x_cfg = jaw_cfg["X_jaw"]
    for side, sign in [("pos", +1), ("neg", -1)]:
        jaw = sim.add_volume("Box", f"x_jaw_{side}")
        jaw.material = jaw_cfg["material"]
        jaw_half_size = (x_cfg["width_cm"] - jaw_x_cm) / 2
        jaw.size = [
            jaw_half_size * 2 * cm,
            x_cfg["width_cm"] * cm,
            x_cfg["thickness_cm"] * cm,
        ]
        jaw.translation = [
            sign * (jaw_x_cm + jaw_half_size) * cm,
            0,
            x_cfg["z_center_cm"] * cm,
        ]

    # ── Piano di registrazione PHSP ──────────────────────────────────────
    phsp_plane = sim.add_volume("Box", "phsp_plane")
    phsp_plane.size = [40 * cm, 40 * cm, 0.1 * mm]  # sottilissimo
    phsp_plane.material = "G4_AIR"
    phsp_plane.translation = [0, 0, phsp_plane_z_cm * cm]

    root_output = str(output_dir / f"{field_name}_phsp.root")

    phsp_actor = sim.add_actor("PhaseSpaceActor", "phsp_actor")
    phsp_actor.attached_to = phsp_plane.name
    phsp_actor.output_filename = root_output
    phsp_actor.attributes = [
        "KineticEnergy",
        "PrePosition",
        "PreDirection",
        "ParticleName",
    ]

    # Statistiche
    stats = sim.add_actor("SimulationStatisticsActor", "stats")
    stats.output_filename = str(output_dir / f"{field_name}_stats.txt")

    # ── Run ───────────────────────────────────────────────────────────────
    print(f"\n  Avvio simulazione GATE...")
    sim.run()

    # Salva config
    config = {
        "E_nom": E_nom,
        "jaw_x_cm": jaw_x_cm,
        "jaw_y_cm": jaw_y_cm,
        "phsp_source": phsp_source,
        "phsp_plane_z_cm": phsp_plane_z_cm,
        "output_root": root_output,
        "n_particles": n_particles,
        "seed": seed,
    }
    with open(output_dir / f"{field_name}_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  ✓ PHSP salvato: {root_output}")
    print(f"  Config: {output_dir}/{field_name}_config.json")
    print(f"\n  Prossimo step: convertire in HDF5:")
    print(f"    python gate_integration/gate_simulations.py convert \\")
    print(f"        --input {root_output} \\")
    print(f"        --output data/{field_name}.h5")

    return root_output


def _generate_macro_script(
    phsp_source: str,
    jaw_x_cm: float,
    jaw_y_cm: float,
    E_nom: float,
    output_dir: str,
    n_particles: int,
    phsp_plane_z_cm: float,
) -> None:
    """
    Genera uno script Python standalone che può essere lanciato sul cluster
    anche senza questo modulo installato — basta avere opengate.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    field_name = f"{E_nom:.0f}mv_{2*jaw_x_cm:.0f}x{2*jaw_y_cm:.0f}"
    script_path = output_dir / f"run_{field_name}.py"

    script = f'''#!/usr/bin/env python3
"""
Auto-generated GATE jaw simulation script.
{field_name}: E_nom={E_nom} MeV, jaw=({jaw_x_cm}×{jaw_y_cm}) cm

Run on cluster:
    python {script_path}
"""
import opengate as gate
from opengate import g4_units

mm = g4_units.mm
cm = g4_units.cm

sim = gate.Simulation()
sim.g4_verbose = False
sim.visu = False
sim.number_of_threads = 4
sim.random_seed = 42

# World
sim.world.size = [60 * cm, 60 * cm, 120 * cm]
sim.world.material = "G4_AIR"
sim.physics_manager.physics_list_name = "QGSP_BIC_EMY"

# Source: IAEA phase space
src = sim.add_source("PhaseSpaceSource", "iaea_source")
src.phsp_file = "{phsp_source}"
src.particle = "gamma"
{"src.n = " + str(n_particles) if n_particles > 0 else "# src.n = all particles"}

# Y Jaws (upper pair, collimate along Y)
jaw_y = {jaw_y_cm}  # semi-apertura [cm]
y_z = 32.0  # centro Z [cm]
y_thick = 7.8  # spessore [cm]
y_width = 20.0  # larghezza [cm]
y_half = (y_width - jaw_y) / 2

for sign in [+1, -1]:
    name = f"y_jaw_{{'pos' if sign > 0 else 'neg'}}"
    jaw = sim.add_volume("Box", name)
    jaw.material = "G4_W"
    jaw.size = [y_width * cm, y_half * 2 * cm, y_thick * cm]
    jaw.translation = [0, sign * (jaw_y + y_half) * cm, y_z * cm]

# X Jaws (lower pair, collimate along X)
jaw_x = {jaw_x_cm}
x_z = 40.0
x_thick = 7.8
x_width = 20.0
x_half = (x_width - jaw_x) / 2

for sign in [+1, -1]:
    name = f"x_jaw_{{'pos' if sign > 0 else 'neg'}}"
    jaw = sim.add_volume("Box", name)
    jaw.material = "G4_W"
    jaw.size = [x_half * 2 * cm, x_width * cm, x_thick * cm]
    jaw.translation = [sign * (jaw_x + x_half) * cm, 0, x_z * cm]

# PHSP recording plane
phsp_plane = sim.add_volume("Box", "phsp_plane")
phsp_plane.size = [40 * cm, 40 * cm, 0.1 * mm]
phsp_plane.material = "G4_AIR"
phsp_plane.translation = [0, 0, {phsp_plane_z_cm} * cm]

phsp_actor = sim.add_actor("PhaseSpaceActor", "phsp_actor")
phsp_actor.attached_to = phsp_plane.name
phsp_actor.output_filename = "{output_dir / (field_name + '_phsp.root')}"
phsp_actor.attributes = ["KineticEnergy", "PrePosition", "PreDirection", "ParticleName"]

stats = sim.add_actor("SimulationStatisticsActor", "stats")
stats.output_filename = "{output_dir / (field_name + '_stats.txt')}"

print("Starting GATE simulation: {field_name}")
sim.run()
print("Done!")
'''

    with open(script_path, "w") as f:
        f.write(script)

    print(f"\n  Script GATE generato: {script_path}")
    print(f"  Lancialo sul cluster con: python {script_path}")


def run_batch(output_base: str = "outputs/gate_jaw", n_threads: int = 4):
    """Lancia tutte le configurazioni batch."""
    output_base = Path(output_base)

    print(f"\n{'='*60}")
    print(f"  GATE Batch: {len(BATCH_CONFIGS)} configurazioni")
    print(f"{'='*60}")

    for name, E_nom, jaw_x, jaw_y, phsp_file in BATCH_CONFIGS:
        if not Path(phsp_file).exists():
            print(f"\n  [SKIP] {name}: file sorgente non trovato: {phsp_file}")
            continue

        create_jaw_simulation(
            phsp_source=phsp_file,
            jaw_x_cm=jaw_x,
            jaw_y_cm=jaw_y,
            E_nom=E_nom,
            output_dir=str(output_base / name),
            n_threads=n_threads,
        )


def generate_all_scripts(output_base: str = "outputs/gate_jaw"):
    """
    Genera TUTTI gli script Python standalone per ogni configurazione,
    pronti per essere lanciati sul cluster senza dipendere da questo modulo.
    """
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Generazione script GATE per {len(BATCH_CONFIGS)} configurazioni")
    print(f"{'='*60}")

    scripts = []
    for name, E_nom, jaw_x, jaw_y, phsp_file in BATCH_CONFIGS:
        out_dir = output_base / name
        _generate_macro_script(
            phsp_source=phsp_file,
            jaw_x_cm=jaw_x,
            jaw_y_cm=jaw_y,
            E_nom=E_nom,
            output_dir=str(out_dir),
            n_particles=0,
            phsp_plane_z_cm=PHSP_PLANE_Z_CM,
        )
        scripts.append(str(out_dir / f"run_{E_nom:.0f}mv_{2*jaw_x:.0f}x{2*jaw_y:.0f}.py"))

    # Genera anche un launcher script
    launcher = output_base / "run_all.sh"
    with open(launcher, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Lancia tutte le simulazioni GATE con jaws\n")
        f.write("# Generato automaticamente da gate_jaw_simulation.py\n\n")
        for s in scripts:
            f.write(f"echo '=== Running {Path(s).stem} ==='\n")
            f.write(f"python {s}\n\n")

    print(f"\n  ✓ Launcher generato: {launcher}")
    print(f"  Lancia con: bash {launcher}")


def main():
    p = argparse.ArgumentParser(
        description="GATE simulation con jaws parametriche per Strada B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # Singola simulazione
    run_p = sub.add_parser("run", help="Lancia una singola simulazione")
    run_p.add_argument("--phsp_source", required=True)
    run_p.add_argument("--E_nom", type=float, required=True)
    run_p.add_argument("--jaw_x", type=float, required=True)
    run_p.add_argument("--jaw_y", type=float, required=True)
    run_p.add_argument("--output_dir", default="outputs/gate_jaw")
    run_p.add_argument("--n_particles", type=int, default=0)
    run_p.add_argument("--n_threads", type=int, default=4)
    run_p.add_argument("--seed", type=int, default=42)

    # Batch (tutte le configurazioni)
    batch_p = sub.add_parser("batch", help="Lancia batch di tutte le configurazioni")
    batch_p.add_argument("--output_base", default="outputs/gate_jaw")
    batch_p.add_argument("--n_threads", type=int, default=4)

    # Genera script standalone
    gen_p = sub.add_parser("generate_scripts",
                           help="Genera script Python standalone per il cluster")
    gen_p.add_argument("--output_base", default="outputs/gate_jaw")

    args = p.parse_args()

    if args.command == "run":
        create_jaw_simulation(
            phsp_source=args.phsp_source,
            jaw_x_cm=args.jaw_x,
            jaw_y_cm=args.jaw_y,
            E_nom=args.E_nom,
            output_dir=args.output_dir,
            n_particles=args.n_particles,
            n_threads=args.n_threads,
            seed=args.seed,
        )
    elif args.command == "batch":
        run_batch(args.output_base, args.n_threads)
    elif args.command == "generate_scripts":
        generate_all_scripts(args.output_base)
    else:
        # Default: genera gli script
        generate_all_scripts()


if __name__ == "__main__":
    main()
