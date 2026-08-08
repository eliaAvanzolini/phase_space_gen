#!/usr/bin/env python3
"""
Generatore di job GATE parallelizzati per la simulazione jaw-collimated.
"""
import os
import math

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

PARTICLE_COUNTS = {
    ("6mv", "part1"): 124_723_612,
    ("6mv", "part2"): 124_726_268,
    ("10mv", "part1"): 124_030_574,
    ("10mv", "part2"): 124_017_250,
}

THREADS_PER_JOB = 4
N_CHUNKS = 30
WALLTIME = "03:30:00"

classes = [
    {"name": "6mv_5x5",    "energy": "6mv",  "jaw_x": 2.5,  "jaw_y": 2.5},
    {"name": "6mv_10x10",  "energy": "6mv",  "jaw_x": 5.0,  "jaw_y": 5.0},
    {"name": "6mv_20x20",  "energy": "6mv",  "jaw_x": 10.0, "jaw_y": 10.0},
    {"name": "10mv_5x5",   "energy": "10mv", "jaw_x": 2.5,  "jaw_y": 2.5},
    {"name": "10mv_10x10", "energy": "10mv", "jaw_x": 5.0,  "jaw_y": 5.0},
    {"name": "10mv_20x20", "energy": "10mv", "jaw_x": 10.0, "jaw_y": 10.0},
]

datasets = [
    {"type": "train", "part": "part1", "out_base": "outputs/gate_jaw"},
    {"type": "ref",   "part": "part2", "out_base": "outputs/gate_jaw_ref"},
]

os.makedirs("scripts_gate_parallel", exist_ok=True)

submit_all_lines = [
    "#!/bin/bash",
    "# Spostati nella cartella contenente gli script PBS",
    'SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"',
    'cd "$SCRIPT_DIR"',
    "",
]

total_jobs = 0

for ds in datasets:
    for c in classes:
        key = (c["energy"], ds["part"])
        total_particles = PARTICLE_COUNTS[key]

        base_chunk_size = total_particles // N_CHUNKS

        folder = os.path.join(ds["out_base"], c["name"])
        os.makedirs(folder, exist_ok=True)

        for chunk_idx in range(N_CHUNKS):
            chunk_start = chunk_idx * base_chunk_size
            if chunk_idx == N_CHUNKS - 1:
                chunk_size = total_particles - chunk_start
            else:
                chunk_size = base_chunk_size

            n_per_thread = chunk_size // THREADS_PER_JOB
            if n_per_thread == 0:
                raise ValueError(
                    f"n_per_thread=0 per {c['name']} chunk {chunk_idx}"
                )

            entry_starts = [chunk_start + t * n_per_thread for t in range(THREADS_PER_JOB)]

            last_thread_end = entry_starts[-1] + n_per_thread
            assert last_thread_end <= total_particles, (
                f"Overrun rilevato per {c['name']} chunk {chunk_idx}"
            )

            job_name = f"{'TR' if ds['type']=='train' else 'RF'}_{c['energy']}_{c['name'].split('_')[1]}_c{chunk_idx:03d}"
            
            py_basename = f"run_{ds['type']}_{c['name']}_part{chunk_idx+1}.py"
            pbs_basename = f"submit_{ds['type']}_{c['name']}_part{chunk_idx+1}.pbs"
            
            py_filename = os.path.join("scripts_gate_parallel", py_basename)
            pbs_filename = os.path.join("scripts_gate_parallel", pbs_basename)

            out_log = f"{job_name}.out"
            err_log = f"{job_name}.err"

            out_root_path = os.path.abspath(os.path.join(folder, f"{c['name']}_phsp_part{chunk_idx+1}.root"))
            src_phsp_path = os.path.abspath(f"data/ELEKTA_PRECISE_{c['energy']}_{ds['part']}.root")

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

# World
sim.world.size = [60 * cm, 60 * cm, 120 * cm]
sim.world.material = "G4_AIR"
sim.physics_manager.physics_list_name = "QGSP_BIC_EMY"

# Source: IAEA phase space ({ds['part']}) - CHUNK {chunk_idx+1}/{N_CHUNKS}
src = sim.add_source("PhaseSpaceSource", "iaea_source")
src.phsp_file = "{src_phsp_path}"
src.particle = "gamma"
src.n = {n_per_thread}
src.entry_start = {entry_starts}

src.position_key_x = "PrePosition_X"
src.position_key_y = "PrePosition_Y"
src.position_key_z = "PrePosition_Z"
src.direction_key_x = "PreDirection_X"
src.direction_key_y = "PreDirection_Y"
src.direction_key_z = "PreDirection_Z"

# Y Jaws
jaw_y = {c['jaw_y']}
y_z, y_thick, y_width = 32.0, 7.8, 20.0
y_half = (y_width - jaw_y) / 2

for sign in [+1, -1]:
    name = f"y_jaw_{{'pos' if sign > 0 else 'neg'}}"
    jaw = sim.add_volume("Box", name)
    jaw.material = "G4_W"
    jaw.size = [y_width * cm, y_half * 2 * cm, y_thick * cm]
    jaw.translation = [0, sign * (jaw_y + y_half) * cm, y_z * cm]

# X Jaws
jaw_x = {c['jaw_x']}
x_z, x_thick, x_width = 40.0, 7.8, 20.0
x_half = (x_width - jaw_x) / 2

for sign in [+1, -1]:
    name = f"x_jaw_{{'pos' if sign > 0 else 'neg'}}"
    jaw = sim.add_volume("Box", name)
    jaw.material = "G4_W"
    jaw.size = [x_half * 2 * cm, x_width * cm, x_thick * cm]
    jaw.translation = [sign * (jaw_x + x_half) * cm, 0, x_z * cm]

# PHSP plane
phsp_plane = sim.add_volume("Box", "phsp_plane")
phsp_plane.size = [40 * cm, 40 * cm, 0.1 * mm]
phsp_plane.material = "G4_AIR"
phsp_plane.translation = [0, 0, 50.0 * cm]

phsp_actor = sim.add_actor("PhaseSpaceActor", "phsp_actor")
phsp_actor.attached_to = phsp_plane.name
phsp_actor.output_filename = "{out_root_path}"
phsp_actor.attributes = ["KineticEnergy", "PrePosition", "PreDirection", "ParticleName"]

print("Starting GATE simulation: {ds['type']} - {c['name']} - chunk {chunk_idx+1}/{N_CHUNKS}")
print(f"  entry_start per thread: {entry_starts}")
print(f"  particelle per thread: {n_per_thread} (totale chunk: {chunk_size})")
sim.run()
print("Done!")
"""
            with open(py_filename, "w") as f:
                f.write(py_code)

            pbs_code = f"""#!/bin/bash
#PBS -N {job_name}
#PBS -q shortGPUQ
#PBS -l nodes=1:ppn={THREADS_PER_JOB}
#PBS -l mem=32gb
#PBS -l walltime={WALLTIME}
#PBS -o {out_log}
#PBS -e {err_log}
#PBS -m n

cd $PBS_O_WORKDIR

# Garantisce di trovarsi sempre nella root di progetto
if [ ! -f "generate_gate_jobs_parallel.py" ] && [ -f "../generate_gate_jobs_parallel.py" ]; then
    cd ..
fi

if [ -f "env/bin/activate" ]; then
    source env/bin/activate
elif [ -f "$HOME/phase_space_gen-main/env/bin/activate" ]; then
    source $HOME/phase_space_gen-main/env/bin/activate
fi

python {py_filename}
"""
            with open(pbs_filename, "w") as f:
                f.write(pbs_code)

            submit_all_lines.append(f"qsub {pbs_basename}")
            total_jobs += 1

with open("scripts_gate_parallel/submit_all.sh", "w") as f:
    f.write("\n".join(submit_all_lines) + "\n")
os.chmod("scripts_gate_parallel/submit_all.sh", 0o755)

print(f"Generati {total_jobs} job PBS in scripts_gate_parallel/")
print(f"  -> {N_CHUNKS} chunk x {len(classes)} classi x {len(datasets)} dataset (train+ref)")
print(f"  -> {THREADS_PER_JOB} thread per job, walltime {WALLTIME}")
