#!/usr/bin/env python3
import os

classes = [
    {"name": "6mv_5x5",   "energy": "6mv",  "jaw_x": 2.5,  "jaw_y": 2.5},
    {"name": "6mv_10x10", "energy": "6mv",  "jaw_x": 5.0,  "jaw_y": 5.0},
    {"name": "6mv_20x20", "energy": "6mv",  "jaw_x": 10.0, "jaw_y": 10.0},
    {"name": "10mv_5x5",  "energy": "10mv", "jaw_x": 2.5,  "jaw_y": 2.5},
    {"name": "10mv_10x10","energy": "10mv", "jaw_x": 5.0,  "jaw_y": 5.0},
    {"name": "10mv_20x20","energy": "10mv", "jaw_x": 10.0, "jaw_y": 10.0},
]

datasets = [
    {"type": "train", "part": "part1", "out_base": "outputs/benchmarks/train"},
    {"type": "ref",   "part": "part2", "out_base": "outputs/benchmarks/ref"}
]

os.makedirs("scripts_benchmark", exist_ok=True)

for ds in datasets:
    for c in classes:
        folder = os.path.join(ds["out_base"], c["name"])
        os.makedirs(folder, exist_ok=True)

        job_name = f"BM_{'TR' if ds['type']=='train' else 'RF'}_{c['energy']}_{c['name'].split('_')[1]}"

        py_filename = f"scripts_benchmark/run_{ds['type']}_{c['name']}.py"
        pbs_filename = f"scripts_benchmark/submit_{ds['type']}_{c['name']}.pbs"

        # Log dei benchmark
        out_log = f"{job_name}.out"
        err_log = f"{job_name}.err"

        out_root_path = os.path.join(folder, f"{c['name']}_benchmark.root")
        src_phsp_path = f"data/ELEKTA_PRECISE_{c['energy']}_{ds['part']}.root"

        # Python script OpenGATE con benchmark e 100k particelle
        py_code = f"""#!/usr/bin/env python3
import time
import opengate as gate
from opengate import g4_units

mm = g4_units.mm
cm = g4_units.cm

sim = gate.Simulation()
sim.g4_verbose = False
sim.visu = False
sim.number_of_threads = 8
sim.random_seed = 3013

# World
sim.world.size = [60 * cm, 60 * cm, 120 * cm]
sim.world.material = "G4_AIR"
sim.physics_manager.physics_list_name = "QGSP_BIC_EMY"

# Source: IAEA phase space ({ds['part']})
src = sim.add_source("PhaseSpaceSource", "iaea_source")
src.phsp_file = "{src_phsp_path}"
src.particle = "gamma"
src.n = 100000  # 100k particelle per il benchmark

# Mappatura esplicita di POSIZIONI e DIREZIONALITÀ dal file ROOT
src.position_key_x = "PrePosition_X"
src.position_key_y = "PrePosition_Y"
src.position_key_z = "PrePosition_Z"
src.direction_key_x = "PreDirection_X"
src.direction_key_y = "PreDirection_Y"
src.direction_key_z = "PreDirection_Z"

# Y Jaws (upper pair)
jaw_y = {c['jaw_y']}
y_z, y_thick, y_width = 32.0, 7.8, 20.0
y_half = (y_width - jaw_y) / 2

for sign in [+1, -1]:
    name = f"y_jaw_{{'pos' if sign > 0 else 'neg'}}"
    jaw = sim.add_volume("Box", name)
    jaw.material = "G4_W"
    jaw.size = [y_width * cm, y_half * 2 * cm, y_thick * cm]
    jaw.translation = [0, sign * (jaw_y + y_half) * cm, y_z * cm]

# X Jaws (lower pair)
jaw_x = {c['jaw_x']}
x_z, x_thick, x_width = 40.0, 7.8, 20.0
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
phsp_plane.translation = [0, 0, 50.0 * cm]

phsp_actor = sim.add_actor("PhaseSpaceActor", "phsp_actor")
phsp_actor.attached_to = phsp_plane.name
phsp_actor.output_filename = "{out_root_path}"
phsp_actor.attributes = ["KineticEnergy", "PrePosition", "PreDirection", "ParticleName"]

print("Starting GATE benchmark: {ds['type']} - {c['name']} (100,000 particles)...")

t0 = time.time()
sim.run()
t1 = time.time()

elapsed = t1 - t0
throughput = 100000 / elapsed if elapsed > 0 else 0
proj_124m_h = (124000000 / throughput) / 3600 if throughput > 0 else 0

print("\\n" + "=" * 60)
print(f"BENCHMARK RISULTATI: {ds['type'].upper()} - {c['name']}")
print("=" * 60)
print(f"Tempo per 100k particelle : {{elapsed:.2f}} s ({{elapsed/60:.2f}} min)")
print(f"Throughput                : {{throughput:.2f}} particelle/sec")
print(f"PROIEZIONE PER 124M       : {{proj_124m_h:.2f}} ORE")
print("=" * 60 + "\\n")
"""
        with open(py_filename, "w") as f:
            f.write(py_code)

        # PBS script
        pbs_code = f"""#!/bin/bash
#PBS -N {job_name}
#PBS -q shortGPUQ
#PBS -l nodes=1:ppn=8
#PBS -l mem=32gb
#PBS -l walltime=01:00:00
#PBS -o {out_log}
#PBS -e {err_log}
#PBS -m n

cd $PBS_O_WORKDIR
source ~/phase_space_gen-main/env/bin/activate
python {py_filename}
"""
        with open(pbs_filename, "w") as f:
            f.write(pbs_code)

print("✅ Script di Benchmark generati in 'scripts_benchmark/'.")
