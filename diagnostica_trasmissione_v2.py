import uproot
import numpy as np

SRC_FILE = "data/ELEKTA_PRECISE_6mv_part1.root"
N_CHECK = 3_500_000  # stesso numero di primari usati nel chunk c000

# Geometria dei jaw (proiettata all'isocentro, gia' nota dal fix precedente)
Y_JAW_Z, jaw_y_scaled = 32.0, 0.8   # campo 5x5
X_JAW_Z, jaw_x_scaled = 40.0, 1.0

with uproot.open(SRC_FILE) as f:
    tree = f[f.keys()[0]]

    # FIX: NESSUNA divisione per 10. I file IAEA standard registrano le
    # posizioni gia' in cm (a differenza dei file scritti da OpenGATE/Geant4,
    # che usano mm). Il file precedente applicava una conversione mm->cm che
    # non serviva, introducendo un fattore 10 di errore (confermato: il
    # vecchio risultato "Z media = 2.72cm" era esattamente 27.21/10).
    x0 = tree["PrePosition_X"].array(library="np", entry_stop=N_CHECK)
    y0 = tree["PrePosition_Y"].array(library="np", entry_stop=N_CHECK)
    z0 = tree["PrePosition_Z"].array(library="np", entry_stop=N_CHECK)

    dx = tree["PreDirection_X"].array(library="np", entry_stop=N_CHECK)
    dy = tree["PreDirection_Y"].array(library="np", entry_stop=N_CHECK)
    dz = tree["PreDirection_Z"].array(library="np", entry_stop=N_CHECK)

    print(f"--- Verifica unita' ---")
    print(f"Z: min={z0.min():.2f} max={z0.max():.2f} media={z0.mean():.2f} cm")
    print(f"(atteso, da Gemini/check_z_stats.py: 27.21 cm)")
    print(f"X0: percentili |X| -> 50%={np.percentile(np.abs(x0),50):.2f} 90%={np.percentile(np.abs(x0),90):.2f} cm")
    print(f"Y0: percentili |Y| -> 50%={np.percentile(np.abs(y0),50):.2f} 90%={np.percentile(np.abs(y0),90):.2f} cm")

    z_start = z0.mean()  # tutte le entries condividono lo stesso Z (piano di scoring)

    # FIX PRINCIPALE: propagazione reale dalla vera quota di partenza fino a
    # ciascun piano dei jaw, non da un'ipotetica sorgente puntiforme a Z=0.
    # tan(theta) = componente trasversale della direzione / componente Z
    # (assumendo (dx,dy,dz) siano coseni direttori, dz>0 per fotoni in avanti)
    dist_to_y_jaw = Y_JAW_Z - z_start
    dist_to_x_jaw = X_JAW_Z - z_start
    print(f"\nDistanza reale dal piano di partenza ai jaw:")
    print(f"  -> ai jaw Y (Z={Y_JAW_Z}cm): {dist_to_y_jaw:.2f} cm")
    print(f"  -> ai jaw X (Z={X_JAW_Z}cm): {dist_to_x_jaw:.2f} cm")

    # Posizione proiettata di ciascun fotone alla quota di ciascun jaw
    y_at_jaw = y0 + dist_to_y_jaw * (dy / dz)
    x_at_jaw = x0 + dist_to_x_jaw * (dx / dz)

    passes_y = np.abs(y_at_jaw) < jaw_y_scaled
    passes_x = np.abs(x_at_jaw) < jaw_x_scaled
    passes_both = passes_y & passes_x

    n_pass = passes_both.sum()
    print(f"\n--- Stima per-particella (posizione reale + propagazione corretta) ---")
    print(f"Fotoni che geometricamente passerebbero entrambi i jaw: {n_pass:,} su {N_CHECK:,}")
    print(f"Tasso stimato: {n_pass/N_CHECK*100:.5f}%")
    print(f"Fotoni realmente prodotti da GATE per questo chunk: 85")
    if n_pass > 0:
        print(f"Rapporto stima/osservato: {n_pass/85:.2f}x")
    else:
        print("Nessun fotone stimato passare -> il numero osservato (85) sarebbe interamente scattering nel tungsteno, non trasmissione diretta")
