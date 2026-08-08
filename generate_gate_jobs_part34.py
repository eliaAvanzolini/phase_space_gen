#!/usr/bin/env python3
"""
Generatore di job GATE jaw-collimated per part3/part4 (estensione statistica
del reference). Stessa logica di generate_gate_jobs_parallel.py:
- entry_start esplicito per thread (MT-safe, no duplicati interni)
- chunking a livello di job per stare sotto il walltime
- output "{classe}_phsp_part{N}.root" -> compatibile col glob di
  prepare_reference_data.py (che ora va esteso per leggere piu' cartelle,
  vedi prepare_reference_data_v2.py)

IMPORTANTE: PARTICLE_COUNTS_PART34 va aggiornato con i conteggi REALI di
part3/part4 non appena il download e la conversione IAEA->ROOT sono
completati (stesso identico check di intersezione fisica a 5 decimali
che avete gia' fatto tra part1/part2 - fatelo ANCHE tra part2/part3/part4,
vedi verify_and_merge_gate_outputs_v2.py).
"""
import os

# =============================================================================
# CONFIGURAZIONE - AGGIORNARE con i conteggi reali una volta convertiti
# =============================================================================
PARTICLE_COUNTS_PART34 = {
    ("6mv", "part3"): 124_693_356,
    ("6mv", "part4"): 124_682_459,
    ("10mv", "part3"): 124_016_539,
    ("10mv", "part4"): 124_020_556,
}

THREADS_PER_JOB = 8
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

# Le due "part" aggiuntive, ognuna con la propria cartella di output separata
# (mai mescolare part3 e part4 nella stessa cartella prima del merge, cosi'
# il check di duplicati/integrita' resta tracciabile per sorgente)
datasets = [
    {"type": "ref_part3", "part": "part3", "out_base": "outputs/gate_jaw_ref_part3"},
    {"type": "ref_part4", "part": "part4", "out_base": "outputs/gate_jaw_ref_part4"},
]

os.makedirs("scripts_gate_parallel_part34", exist_ok=True)
submit_all_lines = ["#!/bin/bash"]
total_jobs = 0

for ds in datasets:
    for c in classes:
        key = (c["energy"], ds["part"])
        if key not in PARTICLE_COUNTS_PART34:
            print(f"[SKIP] {key}: conteggio mancante in PARTICLE_COUNTS_PART34, "
                  f"compilare prima di generare i job per questa combinazione")
            continue
        total_particles = PARTICLE_COUNTS_PART34[key]

        base_chunk_size = total_particles // N_CHUNKS
        folder = os.path.join(ds["out_base"], c["name"])
        os.makedirs(folder, exist_ok=True)

        for chunk_idx in range(N_CHUNKS):
            chunk_start = chunk_idx * base_chunk_size
            chunk_size = (total_particles - chunk_start) if chunk_idx == N_CHUNKS - 1 else base_chunk_size

            n_per_thread = chunk_size // THREADS_PER_JOB
            if n_per_thread == 0:
                raise ValueError(f"n_per_thread=0 per {c['name']} chunk {chunk_idx}")

            entry_starts = [chunk_start + t * n_per_thread for t in range(THREADS_PER_JOB)]
            assert entry_starts[-1] + n_per_thread <= total_particles

            job_name = f"{'P3' if ds['part']=='part3' else 'P4'}_{c['energy']}_{c['name'].split('_')[1]}_c{chunk_idx:03d}"
            py_filename = f"scripts_gate_parallel_part34/run_{ds['type']}_{c['name']}_part{chunk_idx+1}.py"
            pbs_filename = f"scripts_gate_parallel_part34/submit_{ds['type']}_{c['name']}_part{chunk_idx+1}.pbs"
            out_log, err_log = f"{job_name}.out", f"{job_name}.err"
            out_root_path = os.path.join(folder, f"{c['name']}_phsp_part{chunk_idx+1}.root")
            src_phsp_path = f"data/ELEKTA_PRECISE_{c['energy']}_{ds['part']}.root"

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

print("Starting GATE simulation: {ds['type']} - {c['name']} - chunk {chunk_idx+1}/{N_CHUNKS}")
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
source ~/phase_space_gen-main/env/bin/activate
python {py_filename}
"""
            with open(pbs_filename, "w") as f:
                f.write(pbs_code)

            submit_all_lines.append(f"qsub {pbs_filename}")
            total_jobs += 1

with open("scripts_gate_parallel_part34/submit_all.sh", "w") as f:
    f.write("\n".join(submit_all_lines) + "\n")
os.chmod("scripts_gate_parallel_part34/submit_all.sh", 0o755)

print(f"Generati {total_jobs} job in scripts_gate_parallel_part34/")
print("Lancio: bash scripts_gate_parallel_part34/submit_all.sh")
