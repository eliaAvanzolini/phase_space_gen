"""
Virtual Source Modeling: il CFM energia-soltanto genera il fascio aperto,
la collimazione dei jaw e' applicata analiticamente (ray-tracing), non
appresa da una rete neurale.

Perche' questo approccio, e non il transfer learning diretto sui jaw:
- Nessun gradino netto da apprendere (niente sharp discontinuity per un CNF,
  il taglio e' geometrico esatto, non stimato dalla rete)
- Nessun tetto statistico sul lato "generato": il CFM campiona quanti fotoni
  sintetici servono, gratis (nessuna nuova simulazione GATE)
- Riusa la stessa formula di proiezione all'isocentro e lo stesso controllo
  su entrambe le facce del blocco jaw gia' verificati in questa conversazione
  (diagnostica_trasmissione_v3.py, confermato su 3 campi indipendenti)

IMPORTANTE - verificato nel codice reale prima di scrivere questo script:
- dim=6 (non 7): Z e' droppato di default (drop_z=True), la rappresentazione
  di training e' [x, y, dx, dy, dz, E]. Verificato in train.py riga 351.
- Il modello energia-soltanto ha z_const=27.209993... cm (verificato in
  final_result/cfm_energy_only/normalization_stats.json) - la stessa quota
  Z=27.21cm del file sorgente IAEA grezzo. Coerenza confermata in modo
  indipendente rispetto a quanto trovato analizzando ELEKTA_PRECISE_*.root
  in questa conversazione.
- condition_stats.json: mu=15.5, sigma=9.5 per l'energia (energy-only).
"""
import json
import torch
import numpy as np
from models.cfm import PhaseSpaceCFM
from data.synthetic_linac import denormalize_phase_space

CKPT_PATH = "outputs/cfm_energy_only/best_model.pt"
STATS_PATH = "outputs/cfm_energy_only/normalization_stats.json"
COND_STATS_PATH = "outputs/cfm_energy_only/condition_stats.json"

# Geometria jaw, isocentro Z=100cm - stessa fisica gia' verificata in
# generate_gate_jobs_parallel_fixed.py e diagnostica_trasmissione_v3.py
Y_JAW_Z, JAW_THICK = 32.0, 7.8
X_JAW_Z = 40.0
ISOCENTER_Z = 100.0


def load_energy_only_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = PhaseSpaceCFM(dim=6, cond_dim=1, hidden_dim=128, n_layers=4)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def sample_open_field_physical(model, energy_mv: float, n_samples: int, n_steps: int = 100):
    """Campiona n_samples fotoni dal CFM energia-soltanto, condizionati su
    energy_mv (in MV, unita' fisiche). Restituisce un array (n_samples, 7)
    in UNITA' FISICHE: X, Y, Z, dX, dY, dZ, E (Z ricostruito da z_const)."""
    with open(STATS_PATH) as f:
        ps_stats = json.load(f)
    with open(COND_STATS_PATH) as f:
        cond_stats = json.load(f)

    # Normalizza la condizione con le stesse mu/sigma usate in training
    c_mu, c_sigma = cond_stats["mu"][0], cond_stats["sigma"][0]
    c_norm_value = (energy_mv - c_mu) / c_sigma
    c = torch.full((n_samples, 1), c_norm_value, dtype=torch.float32)

    with torch.no_grad():
        x1_norm = model.sample(n_samples=n_samples, c=c, n_steps=n_steps)  # (N, 6), normalizzato

    # FIX: denormalize_phase_space ricostruisce GIA' l'array completo a 7
    # colonne internamente (Z reinserito da z_const, vedi righe finali della
    # funzione in data/synthetic_linac.py) - NON serve nessuna ricostruzione
    # manuale, che duplicherebbe/shifterebbe le colonne se aggiunta qui.
    ps_phys_7 = denormalize_phase_space(x1_norm.numpy(), ps_stats)  # (N, 7): X,Y,Z,dX,dY,dZ,E
    return ps_phys_7


def apply_analytical_jaws(phase_space: np.ndarray, jaw_x_iso: float, jaw_y_iso: float):
    """Filtra analiticamente un array (N,7) [X,Y,Z,dX,dY,dZ,E] tenendo solo
    i fotoni che passerebbero geometricamente attraverso jaw proiettati
    all'isocentro con semiapertura jaw_x_iso/jaw_y_iso."""
    x0, y0, z0 = phase_space[:, 0], phase_space[:, 1], phase_space[:, 2]
    dx, dy, dz = phase_space[:, 3], phase_space[:, 4], phase_space[:, 5]

    jaw_y_scaled = jaw_y_iso * (Y_JAW_Z / ISOCENTER_Z)
    jaw_x_scaled = jaw_x_iso * (X_JAW_Z / ISOCENTER_Z)

    y_near, y_far = Y_JAW_Z - JAW_THICK / 2, Y_JAW_Z + JAW_THICK / 2
    x_near, x_far = X_JAW_Z - JAW_THICK / 2, X_JAW_Z + JAW_THICK / 2

    def pos_at(z_target, start, slope):
        return start + (z_target - z0) * slope

    y_at_near = pos_at(y_near, y0, dy / dz)
    y_at_far = pos_at(y_far, y0, dy / dz)
    x_at_near = pos_at(x_near, x0, dx / dz)
    x_at_far = pos_at(x_far, x0, dx / dz)

    passes = (
        (np.abs(y_at_near) < jaw_y_scaled) & (np.abs(y_at_far) < jaw_y_scaled) &
        (np.abs(x_at_near) < jaw_x_scaled) & (np.abs(x_at_far) < jaw_x_scaled)
    )
    return phase_space[passes]


if __name__ == "__main__":
    model = load_energy_only_model()
    
    print("Campionamento di 2.000.000 fotoni in corso (ci vorra' un po')...")
    raw = sample_open_field_physical(model, energy_mv=6.0, n_samples=2_000_000)

    print("\nRisultati della collimazione analitica:")
    for name, jx, jy in [('5x5', 2.5, 2.5), ('10x10', 5.0, 5.0), ('20x20', 10.0, 10.0)]:
        coll = apply_analytical_jaws(raw, jaw_x_iso=jx, jaw_y_iso=jy)
        print(f'{name}: {len(coll)} fotoni ({len(coll)/len(raw)*100:.5f}%)')
