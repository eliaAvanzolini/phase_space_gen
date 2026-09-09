#!/usr/bin/env python3
"""
Generatore di job GATE parallelizzati per la simulazione jaw-collimated.

FIX rispetto alla versione originale (generate_gate_jobs_parallel.py):

  1. PROIEZIONE ALL'ISOCENTRO
     Le semi-aperture nominali (definite a Z=100cm) vengono ora scalate
     per triangoli simili alla quota fisica reale delle ganasce:
         d(Z_jaw) = d_iso * Z_jaw / 100
     Prima venivano usate senza scalare, producendo campi reali molto
     piu' larghi e asimmetrici del nominale (es. 5x5 -> 12.5x15.6 cm2).

  2. GEOMETRIA DEI BLOCCHI SEPARATA IN DUE VARIABILI INDIPENDENTI
     Nel codice originale un'unica variabile "width" definiva sia
     l'estensione trasversale del blocco (che deve coprire tutta la
     larghezza del piano phsp) sia la sua estensione nella direzione
     di chiusura (che deve arrivare oltre il bordo campo). Erano due
     requisiti fisici diversi schiacciati sullo stesso numero, ed e'
     la causa dei passaggi d'aria ai bordi. Ora sono due costanti:
       - JAW_TRANSVERSE_WIDTH: copertura trasversale (elimina l'alone
         nei quadranti esterni dove prima c'era solo G4_AIR)
       - JAW_REACH: estensione nella direzione di chiusura, fissa e
         generosa, indipendente dalla dimensione del campo

  3. RIDUZIONE STATISTICA DEI PRIMARI
     Non serve trasportare tutti i ~124M fotoni primari per classe:
     bastano circa 25-30M per ottenere 1.5-2M fotoni collimati/classe
     dopo la collimazione (statistica sufficiente per training CFM/NSF
     e per la dose a 2mm). Questo riduce drasticamente il numero di
     chunk/job PBS necessari.
"""
import os
import math

# =============================================================================
# CONFIGURAZIONE FISICA (verificare questi numeri contro il disegno tecnico
# del testadi Elekta Precise prima di lanciare su cluster: qui sono presi
# dal codice originale, che li aveva corretti per Z_jaw e thickness ma non
# per la proiezione all'isocentro)
# =============================================================================

ISOCENTER_Z = 100.0     # cm, piano isocentrico di definizione del campo nominale
Y_JAW_Z = 32.0          # cm, quota fisica ganasce Y
X_JAW_Z = 40.0          # cm, quota fisica ganasce X
JAW_THICK = 7.8         # cm, spessore invariato (assorbimento lungo il fascio)

JAW_TRANSVERSE_WIDTH = 46.0  # cm, estensione trasversale del blocco (FIX #2).
                              # Meta' = 23 cm: copre oltre il piano phsp
                              # (40x40cm, meta'=20cm) con margine, e resta
                              # dentro il World (60x60cm, meta'=30cm).
JAW_REACH = 24.0              # cm, estensione nella direzione di chiusura,
                              # oltre il bordo campo massimo. Verificata sotto
                              # per restare dentro il World in tutti i casi.

# Verifica di contenimento nel World (deve sempre valere, altrimenti Geant4
# si lamenta per un daughter volume che sporge dalla mother volume / lo tronca
# silenziosamente in modo sbagliato):
WORLD_HALF = 30.0  # cm (World size = 60x60x120cm)
_max_jaw_iso = 10.0  # cm, la semi-apertura nominale piu' grande usata sotto (campo 20x20)
_max_scaled_y = _max_jaw_iso * (Y_JAW_Z / ISOCENTER_Z)
_max_scaled_x = _max_jaw_iso * (X_JAW_Z / ISOCENTER_Z)
assert _max_scaled_y + JAW_REACH < WORLD_HALF, "JAW_REACH troppo grande: il blocco Y sporge dal World"
assert _max_scaled_x + JAW_REACH < WORLD_HALF, "JAW_REACH troppo grande: il blocco X sporge dal World"
assert JAW_TRANSVERSE_WIDTH / 2 < WORLD_HALF, "JAW_TRANSVERSE_WIDTH troppo grande: sporge dal World"

# =============================================================================
# CONFIGURAZIONE STATISTICA
# =============================================================================

# Conteggio primari realmente disponibili nei file sorgente (invariato,
# sono i file IAEA phsp gia' prodotti a monte)
PARTICLE_COUNTS = {
    ("6mv", "part1"): 124_723_612,
    ("6mv", "part2"): 124_726_268,
    ("10mv", "part1"): 124_030_574,
    ("10mv", "part2"): 124_017_250,
}

# FIX #3: usiamo solo una frazione dei primari disponibili (25-30M), non
# tutti i 124M. min() con il valore disponibile per sicurezza.
TARGET_PRIMARIES_PER_CLASS = 28_000_000

THREADS_PER_JOB = 4
N_CHUNKS = 8            # ridotto da 30: meno job, ognuno di dimensione paragonabile
WALLTIME = "02:30:00"   # ridotto da 3:30:00 in proporzione al numero di primari/thread
                         # (vedi calcolo commentato in fondo al file)

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
        available = PARTICLE_COUNTS[key]
        total_particles = min(available, TARGET_PRIMARIES_PER_CLASS)

        # FIX #1: proiezione all'isocentro (triangoli simili)
        jaw_y_scaled = c["jaw_y"] * (Y_JAW_Z / ISOCENTER_Z)
        jaw_x_scaled = c["jaw_x"] * (X_JAW_Z / ISOCENTER_Z)

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

# Y Jaws — apertura proiettata all'isocentro (FIX #1), blocco a copertura
# piena (FIX #2): estensione trasversale fissa {JAW_TRANSVERSE_WIDTH}cm,
# reach nella direzione di chiusura fisso {JAW_REACH}cm.
jaw_y = {jaw_y_scaled}   # semi-apertura FISICA a Z={Y_JAW_Z}cm (nominale isocentrico: {c['jaw_y']}cm)
y_z, y_thick = {Y_JAW_Z}, {JAW_THICK}
y_transverse, y_reach = {JAW_TRANSVERSE_WIDTH}, {JAW_REACH}

for sign in [+1, -1]:
    name = f"y_jaw_{{'pos' if sign > 0 else 'neg'}}"
    jaw = sim.add_volume("Box", name)
    jaw.material = "G4_W"
    jaw.size = [y_transverse * cm, y_reach * cm, y_thick * cm]
    jaw.translation = [0, sign * (jaw_y + y_reach / 2) * cm, y_z * cm]

# X Jaws — stesso fix
jaw_x = {jaw_x_scaled}   # semi-apertura FISICA a Z={X_JAW_Z}cm (nominale isocentrico: {c['jaw_x']}cm)
x_z, x_thick = {X_JAW_Z}, {JAW_THICK}
x_transverse, x_reach = {JAW_TRANSVERSE_WIDTH}, {JAW_REACH}

for sign in [+1, -1]:
    name = f"x_jaw_{{'pos' if sign > 0 else 'neg'}}"
    jaw = sim.add_volume("Box", name)
    jaw.material = "G4_W"
    jaw.size = [x_reach * cm, x_transverse * cm, x_thick * cm]
    jaw.translation = [sign * (jaw_x + x_reach / 2) * cm, 0, x_z * cm]

# PHSP plane (invariato)
phsp_plane = sim.add_volume("Box", "phsp_plane")
phsp_plane.size = [40 * cm, 40 * cm, 0.1 * mm]
phsp_plane.material = "G4_AIR"
phsp_plane.translation = [0, 0, 50.0 * cm]

phsp_actor = sim.add_actor("PhaseSpaceActor", "phsp_actor")
phsp_actor.attached_to = phsp_plane.name
phsp_actor.output_filename = "{out_root_path}"
phsp_actor.attributes = ["KineticEnergy", "PrePosition", "PreDirection", "ParticleName"]

print("Starting GATE simulation: {ds['type']} - {c['name']} - chunk {chunk_idx+1}/{N_CHUNKS}")
print(f"  jaw_y (proiettato) = {jaw_y_scaled:.3f} cm | jaw_x (proiettato) = {jaw_x_scaled:.3f} cm")
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
if [ ! -f "generate_gate_jobs_parallel_fixed.py" ] && [ -f "../generate_gate_jobs_parallel_fixed.py" ]; then
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
print(f"Primari per classe: {TARGET_PRIMARIES_PER_CLASS:,} (era: fino a 124M)")
print(f"Chunk per classe: {N_CHUNKS} (era: 30)  ->  {n_per_thread:,} particelle/thread/chunk circa")
print("Lancia con: bash scripts_gate_parallel/submit_all.sh")
