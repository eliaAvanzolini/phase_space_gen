import uproot
import numpy as np

SRC_FILE = "data/ELEKTA_PRECISE_6mv_part1.root"
N_CHECK = 3_500_000

Y_JAW_Z, jaw_y_scaled, JAW_THICK = 32.0, 0.8, 7.8   # campo 5x5
X_JAW_Z, jaw_x_scaled = 40.0, 1.0

with uproot.open(SRC_FILE) as f:
    tree = f[f.keys()[0]]

    x0 = tree["PrePosition_X"].array(library="np", entry_stop=N_CHECK)
    y0 = tree["PrePosition_Y"].array(library="np", entry_stop=N_CHECK)
    z0 = tree["PrePosition_Z"].array(library="np", entry_stop=N_CHECK)

    dx = tree["PreDirection_X"].array(library="np", entry_stop=N_CHECK)
    dy = tree["PreDirection_Y"].array(library="np", entry_stop=N_CHECK)
    dz = tree["PreDirection_Z"].array(library="np", entry_stop=N_CHECK)

    z_start = z0.mean()

    # FIX: il blocco jaw non e' un piano infinitamente sottile a Z_jaw, ha
    # spessore JAW_THICK. Per attraversarlo il fotone deve restare dentro
    # l'apertura sia all'ingresso (faccia vicina alla sorgente) sia
    # all'uscita (faccia lontana) del blocco - quest'ultima e' il vincolo
    # piu' restrittivo per una traiettoria rettilinea con angolo grande.
    y_near = Y_JAW_Z - JAW_THICK / 2
    y_far  = Y_JAW_Z + JAW_THICK / 2
    x_near = X_JAW_Z - JAW_THICK / 2
    x_far  = X_JAW_Z + JAW_THICK / 2

    def pos_at(z_target, start, slope):
        return start + (z_target - z_start) * slope

    y_at_near = pos_at(y_near, y0, dy / dz)
    y_at_far  = pos_at(y_far,  y0, dy / dz)
    x_at_near = pos_at(x_near, x0, dx / dz)
    x_at_far  = pos_at(x_far,  x0, dx / dz)

    passes_y = (np.abs(y_at_near) < jaw_y_scaled) & (np.abs(y_at_far) < jaw_y_scaled)
    passes_x = (np.abs(x_at_near) < jaw_x_scaled) & (np.abs(x_at_far) < jaw_x_scaled)
    passes_both = passes_y & passes_x

    n_pass = passes_both.sum()
    print(f"--- Stima raffinata (vincolo su ENTRAMBE le facce del blocco, spessore {JAW_THICK}cm) ---")
    print(f"Fotoni stimati passare: {n_pass:,} su {N_CHECK:,}  ({n_pass/N_CHECK*100:.5f}%)")
    print(f"Fotoni reali (GATE): 85")
    if n_pass > 0:
        print(f"Rapporto stima/osservato: {n_pass/85:.2f}x")

    # Diagnostica aggiuntiva: quanti passavano al centro ma NON su entrambe le facce
    # (i "falsi positivi" del modello precedente, dovuti a ignorare lo spessore)
    y_at_center = pos_at(Y_JAW_Z, y0, dy / dz)
    x_at_center = pos_at(X_JAW_Z, x0, dx / dz)
    passed_center_only = (np.abs(y_at_center) < jaw_y_scaled) & (np.abs(x_at_center) < jaw_x_scaled) & (~passes_both)
    print(f"\nFotoni che sembravano passare guardando solo il centro del blocco, ma in realta' no: {passed_center_only.sum():,}")
    print("(questa e' la sovrastima del modello precedente, diagnostica_trasmissione_v2.py)")
