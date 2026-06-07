"""
gate_integration/
==================
Integrazione con GATE 10 (opengate).

Moduli:
    save_for_gate.py     -- salva modelli in formato .pth per GANSource
    gate_simulations.py  -- simulazioni GATE: generazione PS, dose, conversione
    workflow.py          -- pipeline end-to-end completa

Installazione GATE 10 (sulla workstation):
    pip install opengate
    # (richiede Geant4 compilato — ~30 min)
    # Guida: https://opengate-python.readthedocs.io/en/master/user_guide/user_guide_installation.html

Workflow completo:
    1. Genera phase space con GATE:
       python gate_simulations.py generate --n_particles 1e8

    2. Converti ROOT → HDF5:
       python gate_simulations.py convert --input phsp.root --output phsp_train.h5

    3. Addestra modello (dalla root del progetto):
       python train.py --model nsf --data_path phsp_train.h5 --epochs 200

    4. Salva in formato GATE:
       python gate_integration/save_for_gate.py --checkpoint outputs/nsf*/best_model.pt
                                                 --model nsf --out nsf_gate.pth

    5. Simula dose e valida:
       python gate_simulations.py dose_model --pth_filename nsf_gate.pth
       python gate_simulations.py gamma_index --reference dose_ref.mhd --model dose_nsf.mhd
"""
