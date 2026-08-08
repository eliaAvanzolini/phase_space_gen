#!/usr/bin/env python3
"""
Genera un job PBS per ciascuna combinazione (field, subtask) della dose
validation, usando n_threads=1 (unica modalita' verificata corretta:
il test 1-thread vs 8-thread ha mostrato che con MT ogni worker genera
l'intero conteggio richiesto in modo indipendente, dose finale ~8x quella
attesa con 8 thread). Il parallelismo si ottiene lanciando piu' job PBS
in coda, non con MT dentro il singolo job.
"""
import os

FIELDS = ["6mv_5x5", "6mv_10x10", "6mv_20x20", "10mv_5x5", "10mv_10x10", "10mv_20x20"]
SUBTASKS = ["reference", "cfm", "nsf"]

# Stesso N_PARTICLES per reference/cfm/nsf nella stessa classe (confrontabilita'
# del rumore statistico), tarato sul pool reale disponibile per il reference
# (vedi discussione precedente sui conteggi per classe del dataset ref)
N_PARTICLES_MAP = {
    "6mv_5x5":    200_000,
    "6mv_10x10":  800_000,
    "6mv_20x20":  1_000_000,
    "10mv_5x5":   45_000,
    "10mv_10x10": 100_000,
    "10mv_20x20": 350_000,
}

N_THREADS = 1  # bug MT confermato: entry_start non distribuisce correttamente
WALLTIME = "01:00:00"  # ampio margine rispetto alla stima (~10 min max)

os.makedirs("scripts_dose", exist_ok=True)
submit_lines = ["#!/bin/bash"]

for field in FIELDS:
    n_particles = N_PARTICLES_MAP[field]
    for subtask in SUBTASKS:
        job_name = f"DV_{field}_{subtask}"
        out_dir = f"outputs/dose_validation/{field}"
        pbs_filename = f"scripts_dose/submit_{field}_{subtask}.pbs"

        pbs_code = f"""#!/bin/bash
#PBS -N {job_name}
#PBS -q shortGPUQ
#PBS -l nodes=1:ppn={N_THREADS}
#PBS -l mem=16gb
#PBS -l walltime={WALLTIME}
#PBS -o logs/{job_name}.out
#PBS -e logs/{job_name}.err
#PBS -m n

cd $PBS_O_WORKDIR
mkdir -p logs
source /home/elia.avanzolini/phase_space_gen-main/env/bin/activate

python dose_validation_conditional_PATCHED.py \\
    --subtask {subtask} \\
    --field {field} \\
    --n_particles {n_particles} \\
    --n_threads {N_THREADS} \\
    --voxel_mm 2.0 \\
    --output_dir {out_dir}
"""
        with open(pbs_filename, "w") as f:
            f.write(pbs_code)

        submit_lines.append(f"qsub {pbs_filename}")

with open("scripts_dose/submit_all.sh", "w") as f:
    f.write("\n".join(submit_lines) + "\n")
os.chmod("scripts_dose/submit_all.sh", 0o755)

print(f"Generati {len(FIELDS) * len(SUBTASKS)} job in scripts_dose/")
print("Lancio: bash scripts_dose/submit_all.sh")
