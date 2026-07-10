
import subprocess



# Chunk da recuperare

chunks = [1, 8, 9]

# Forza il percorso del Python corretto con tutti i pacchetti installati

PYTHON_ENV = "/home/elia.avanzolini/phase_space_gen-main/env/bin/python3"



for c in chunks:

    chunk_dir = f"outputs/dose_validation_6mv_5x5/chunk_{c}"

    print(f"\n🔄 [RECUPERO] Completamento automatico del CHUNK {c}...")

    

    # Lanciamo solo NSF (con la patch attiva)

    print(f"  📥 Generazione e simulazione NSF per Chunk {c}...")

    cmd_nsf = [

        PYTHON_ENV, "dose_validation_conditional.py",

        "--field", "6mv_5x5",

        "--n_particles", "10000000",

        "--n_threads", "8",

        "--device", "cuda",

        "--output_dir", chunk_dir,

        "--subtask", "nsf"

    ]

    subprocess.run(cmd_nsf, check=True)

    

    # Lanciamo la GAN finale

    print(f"  📥 Generazione e simulazione GAN per Chunk {c}...")

    cmd_gan = [

        PYTHON_ENV, "dose_validation_conditional.py",

        "--field", "6mv_5x5",

        "--n_particles", "10000000",

        "--n_threads", "8",

        "--device", "cuda",

        "--output_dir", chunk_dir,

        "--subtask", "gan"

    ]

    subprocess.run(cmd_gan, check=True)



print("\n🚀 Chunk 1 e 8 recuperati e completati al 100%!")

