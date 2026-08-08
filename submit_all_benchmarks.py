import os
import subprocess

energies = ["6mv", "10mv"]
field_sizes = ["5x5", "10x10", "20x20"]
n_particles = 1000000  # 100k particelle per test veloci

os.makedirs("outputs/benchmarks/scripts", exist_ok=True)
os.makedirs("outputs/benchmarks/logs", exist_ok=True)

for energy in energies:
    for field in field_sizes:
        job_name = f"BM_{energy}_{field}"
        pbs_file = f"outputs/benchmarks/scripts/run_{job_name}.sh"
        out_log = f"outputs/benchmarks/logs/{job_name}.out"
        err_log = f"outputs/benchmarks/logs/{job_name}.err"

        pbs_content = f"""#!/bin/bash
#PBS -N {job_name}
#PBS -q commonGPUQ
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=01:00:00
#PBS -o {out_log}
#PBS -e {err_log}

cd $PBS_O_WORKDIR

# Decommenta la riga sotto se devi attivare il venv
source ~/phase_space_gen-main/env/bin/activate

echo "Avvio Benchmark per {energy} - {field}..."
python run_gate_benchmark.py --energy {energy} --field {field} --particles {n_particles}
"""

        with open(pbs_file, "w") as f:
            f.write(pbs_content)

        # Sottomissione automatica a PBS
        res = subprocess.run(["qsub", pbs_file], capture_output=True, text=True)
        print(f"Inviato Job: {job_name} -> Job ID: {res.stdout.strip()}")

print("\n🚀 Tutti e 6 i benchmark sono stati sottomessi a commonGPUQ!")
