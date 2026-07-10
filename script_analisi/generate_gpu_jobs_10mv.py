import os
import subprocess

for i in range(1, 11):
    pbs_name = f"submit_gpu_10mv_j{i}.pbs"
    content = f"""#!/bin/bash
#PBS -N D10mv_J{i}
#PBS -q shortGPUQ
#PBS -l select=1:ncpus=8:mem=16gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -m n
#PBS -o D10mv_J{i}.o
#PBS -e D10mv_J{i}.e

cd $PBS_O_WORKDIR
source /home/elia.avanzolini/phase_space_gen-main/env/bin/activate

CHUNK_DIR="outputs/dose_validation_10mv_20x20/chunk_{i}"
mkdir -p ${{CHUNK_DIR}}

python3 dose_validation_conditional.py --field 10mv_20x20 --n_particles 10000000 --n_threads 8 --device cuda --subtask reference
mv outputs/dose_validation_10mv_20x20/dose_reference* ${{CHUNK_DIR}}/ 2>/dev/null

python3 dose_validation_conditional.py --field 10mv_20x20 --n_particles 10000000 --n_threads 8 --device cuda --subtask cfm
mv outputs/dose_validation_10mv_20x20/dose_cfm* ${{CHUNK_DIR}}/ 2>/dev/null

python3 dose_validation_conditional.py --field 10mv_20x20 --n_particles 10000000 --n_threads 8 --device cuda --subtask nsf
mv outputs/dose_validation_10mv_20x20/dose_nsf* ${{CHUNK_DIR}}/ 2>/dev/null

python3 dose_validation_conditional.py --field 10mv_20x20 --n_particles 10000000 --n_threads 8 --device cuda --subtask gan
mv outputs/dose_validation_10mv_20x20/dose_gan* ${{CHUNK_DIR}}/ 2>/dev/null
"""
    with open(pbs_name, "w") as f:
        f.write(content)
    subprocess.run(["qsub", pbs_name])
    os.remove(pbs_name)
print("🚀 Sottomessi i 10 job 10mv_20x20 con le specifiche originali!")
