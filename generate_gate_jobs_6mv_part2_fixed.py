#!/usr/bin/env python3
"""
generate_gate_jobs_6mv_part2_fixed.py
======================================
Rigenera i job GATE SOLO per PART2 6MV applicando il fix MT-safe (entry_start esplicito).
Determina automaticamente la TTree presente nel file ROOT indipendentemente dal nome della chiave.
"""

import os
import uproot

THREADS_PER_JOB = 4
N_CHUNKS = 30
WALLTIME = "03:30:00"

classes = [
    {"name": "6mv_5x5", "energy": "6mv", "jaw_x": 2.5, "jaw_y": 2.5},
    {"name": "6mv_10x10", "energy": "6mv", "jaw_x": 5.0, "jaw_y": 5.0},
    {"name": "6mv_20x20", "energy": "6mv", "jaw_x": 10.0, "jaw_y": 10.0},
]

out_script_dir = "scripts_gate_parallel_6mv_part2_fixed"
out_base_dir = "outputs/gate_jaw_ref_6mv_part2_fixed"
os.makedirs(out_script_dir, exist_ok=True)

# 1. Ispezione dinamica e robusta del file ROOT sorgente
phsp_file = "data/ELEKTA_PRECISE_6mv_part2.root"
if not os.path.exists(phsp_file):
    raise FileNotFoundError(f"❌ File sorgente non trovato: {phsp_file}")

with uproot.open(phsp_file) as f:
    # Trova qualsiasi TTree presente nel file ROOT
    trees = [k for k, v in f.classnames().items() if v == "TTree"]
    if trees:
        tree_key = trees[0]
    elif f.keys():
        tree_key = f.keys()[0]
    else:
        raise ValueError(f"❌ Il file ROOT {phsp_file} risulta vuoto!")

    total_particles = int(f[tree_key].num_entries)

print(
    f"📊 [6MV part2 sorgente - Key '{tree_key}']: {total_particles:,} particelle trovate"
)

submit_all_lines = ["#!/bin/bash"]
total_jobs = 0
base_chunk_size = total_particles // N_CHUNKS

for c in classes:
    folder = os.path.join(out_base_dir, c["name"])
    os.makedirs(folder, exist_ok=True)

    for chunk_idx in range(N_CHUNKS):
        chunk_start = chunk_idx * base_chunk_size
        chunk_size = (
            (total_particles - chunk_start)
            if chunk_idx == N_CHUNKS - 1
            else base_chunk_size
        )

        n_per_thread = chunk_size // THREADS_PER_JOB
        entry_starts = [
            chunk_start + t * n_per_thread for t in range(THREADS_PER_JOB)
        ]

        job_name = f"P2FIX_6MV_{c['name'].split('_')[1]}_c{chunk_idx:03d}"
        py_filename = (
            f"{out_script_dir}/run_ref_part2_{c['name']}_part{chunk_idx+1}.py"
        )
        pbs_filename = (
            f"{out_script_dir}/submit_ref_part2_{c['name']}_part{chunk_idx+1}.pbs"
        )
        out_log, err_log = f"{job_name}.out", f"{job_name}.err"
        out_root_path = os.path.join(
            folder, f"{c['name']}_phsp_part{chunk_idx+1}.root"
        )

        py_code = f"""#!/usr/bin/env python3
import opengate as gate
from opengate import g4_units

mm = g4_units.mm
cm = g4_units.cm

sim = gate.Simulation()
sim.g4_verbose = False
sim.visu = False
sim.number_of_threads = {THREADS_PER_JOB}
sim.random_seed = {3013 + chunk_idx}

sim.world.size = [60 * cm, 60 * cm, 120 * cm]
sim.world.material = "G4_AIR"
sim.physics_manager.physics_list_name = "QGSP_BIC_EMY"

src = sim.add_source("PhaseSpaceSource", "iaea_source")
src.phsp_file = "{phsp_file}"
src.particle = "gamma"
src.n = {n_per_thread}
src.entry_start = {entry_starts}

src.position_key_x = "PrePosition_X"
src.position_key_y = "PrePosition_Y"
src.position_key_z = "PrePosition_Z"
src.direction_key_x = "PreDirection_X"
src.direction_key_y = "PreDirection_Y"
src.direction_key_z = "PreDirection_Z"

jaw_y = {c['jaw_y']}
y_z, y_thick, y_width = 32.0, 7.8, 20.0
y_half = (y_width - jaw_y) / 2
for sign in [+1, -1]:
    name = f"y_jaw_{{'pos' if sign > 0 else 'neg'}}"
    jaw = sim.add_volume("Box", name)
    jaw.material = "G4_W"
    jaw.size = [y_width * cm, y_half * 2 * cm, y_thick * cm]
    jaw.translation = [0, sign * (jaw_y + y_half) * cm, y_z * cm]

jaw_x = {c['jaw_x']}
x_z, x_thick, x_width = 40.0, 7.8, 20.0
x_half = (x_width - jaw_x) / 2
for sign in [+1, -1]:
    name = f"x_jaw_{{'pos' if sign > 0 else 'neg'}}"
    jaw = sim.add_volume("Box", name)
    jaw.material = "G4_W"
    jaw.size = [x_half * 2 * cm, x_width * cm, x_thick * cm]
    jaw.translation = [sign * (jaw_x + x_half) * cm, 0, x_z * cm]

phsp_plane = sim.add_volume("Box", "phsp_plane")
phsp_plane.size = [40 * cm, 40 * cm, 0.1 * mm]
phsp_plane.material = "G4_AIR"
phsp_plane.translation = [0, 0, 50.0 * cm]

phsp_actor = sim.add_actor("PhaseSpaceActor", "phsp_actor")
phsp_actor.attached_to = phsp_plane.name
phsp_actor.output_filename = "{out_root_path}"
phsp_actor.attributes = ["KineticEnergy", "PrePosition", "PreDirection", "ParticleName"]

print("Starting GATE simulation: 6MV part2 - {c['name']} - chunk {chunk_idx+1}/{N_CHUNKS}")
sim.run()
print("Done!")
"""
        with open(py_filename, "w") as f:
            f.write(py_code)

        pbs_code = f"""#!/bin/bash
#PBS -N {job_name}
#PBS -q shortGPUQ
#PBS -l nodes=1:ppn={THREADS_PER_JOB}
#PBS -l mem=16gb
#PBS -l vmem=16gb
#PBS -l walltime={WALLTIME}
#PBS -o {out_log}
#PBS -e {err_log}
#PBS -m n

cd $PBS_O_WORKDIR
source ~/phase_space_gen-main/env/bin/activate
python {py_filename}
"""
        with open(pbs_filename, "w") as f:
            f.write(pbs_code)

        submit_all_lines.append(f"qsub {pbs_filename}")
        total_jobs += 1

submit_file = f"{out_script_dir}/submit_all.sh"
with open(submit_file, "w") as f:
    f.write("\n".join(submit_all_lines) + "\n")
os.chmod(submit_file, 0o755)

print(f"\n✅ Generati {total_jobs} job solo per 6MV in {out_script_dir}/")
print(f"🚀 Lancio rapido:\n   bash {submit_file}")
