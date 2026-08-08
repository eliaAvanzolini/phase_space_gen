import argparse
import time
import opengate as gate

def run_benchmark(energy, field_size, n_particles):
    print("=" * 60)
    print(f"=== BENCHMARK START: {energy.upper()} - {field_size} | {n_particles:,} particelle ===")
    print("=" * 60)

    sim = gate.Simulation()

    # 1. --- DEFINIZIONE GEOMETRIA / PHANTOM / JAWS ---
    # (Inserisci qui o importa la tua geometria reale)
    # es: setup_geometry(sim, field_size)

    # 2. --- DEFINIZIONE DELLA SORGENTE ---
    # Esempio se usi una sorgente da file Phase Space o generica:
    source = sim.add_source("GenericSource", "my_source")
    source.particle = "gamma"  # o e-, ecc.
    source.n = n_particles    # ⚠️ FONDAMENTALE: collega il parametro del benchmark
    
    # Se la tua sorgente legge da un file Phase Space d'ingresso:
    # source.type = "PhaseSpace"
    # source.pth_filename = f"path/to/source_{energy}.root"

    # 3. --- ATTORE DI OUTPUT (PhaseSpaceActor / Dose) ---
    # ⚠️ Assicurati di includere anche l'attore di output per misurare
    # l'impatto reale dell'I/O su disco!
    # es: add_phase_space_actor(sim)

    # --- BENCHMARK RUN ---
    t0_run = time.time()
    sim.run()
    t1_run = time.time()
    run_duration = t1_run - t0_run

    throughput = n_particles / run_duration if run_duration > 0 else 0
    proj_124M_hours = (124_000_000 / throughput) / 3600 if throughput > 0 else 0

    print("\n" + "=" * 60)
    print(f"               RISULTATI BENCHMARK [{energy.upper()} - {field_size}]")
    print("=" * 60)
    print(f"Tempo totale simulazione     : {run_duration:.2f} s ({run_duration / 60:.2f} min)")
    print(f"Throughput effettivo         : {throughput:.2f} particelle/sec")
    print("-" * 60)
    print(f"⏱️  TEMPO STIMATO PER 124M   : {proj_124M_hours:.2f} ORE")
    print("=" * 60 + "\n")
