#!/usr/bin/env python3
import opengate as gate
from opengate import g4_units

cm = g4_units.cm

sim = gate.Simulation()
sim.g4_verbose = False
sim.visu = False
sim.number_of_threads = 2

sim.world.size = [60 * cm, 60 * cm, 120 * cm]
sim.world.material = "G4_AIR"
sim.physics_manager.physics_list_name = "QGSP_BIC_EMY"

src = sim.add_source("PhaseSpaceSource", "iaea_source")
src.phsp_file = "data/ELEKTA_PRECISE_6mv_part1.root"
src.particle = "gamma"
src.n = 1000
src.entry_start = [0, 1000]

src.position_key_x = "PrePosition_X"
src.position_key_y = "PrePosition_Y"
src.position_key_z = "PrePosition_Z"
src.direction_key_x = "PreDirection_X"
src.direction_key_y = "PreDirection_Y"
src.direction_key_z = "PreDirection_Z"

# Attore per contare le particelle primarie generate
stats = sim.add_actor("SimulationStatisticsActor", "stats")
stats.output_filename = "stats.txt"

sim.run()

print("\n" + "=" * 50)
print(f"EVENTI GENERATI (TOTALE JOB): {stats.counts.event_count}")
print("=" * 50 + "\n")
