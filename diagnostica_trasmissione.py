import uproot
import numpy as np

SRC_FILE = "data/ELEKTA_PRECISE_6mv_part1.root"
N_CHECK = 3_500_000  # stesso numero di primari usati nel chunk c000

with uproot.open(SRC_FILE) as f:
    tree = f[f.keys()[0]]
    print(f"Branch disponibili: {tree.keys()}")

    x = tree["PrePosition_X"].array(library="np", entry_stop=N_CHECK) / 10.0  # cm
    y = tree["PrePosition_Y"].array(library="np", entry_stop=N_CHECK) / 10.0
    z = tree["PrePosition_Z"].array(library="np", entry_stop=N_CHECK) / 10.0

    dx = tree["PreDirection_X"].array(library="np", entry_stop=N_CHECK)
    dy = tree["PreDirection_Y"].array(library="np", entry_stop=N_CHECK)
    dz = tree["PreDirection_Z"].array(library="np", entry_stop=N_CHECK)

    print(f"\n--- Posizione di partenza dei primari (prime {N_CHECK:,}) ---")
    print(f"Z: min={z.min():.2f} max={z.max():.2f} media={z.mean():.2f} cm")
    print(f"  (se non e' vicino a 0, i jaw a Z=32/40cm potrebbero essere nel posto sbagliato)")
    print(f"X: percentili |X| -> 50%={np.percentile(np.abs(x),50):.2f} 90%={np.percentile(np.abs(x),90):.2f} 99%={np.percentile(np.abs(x),99):.2f} cm")
    print(f"Y: percentili |Y| -> 50%={np.percentile(np.abs(y),50):.2f} 90%={np.percentile(np.abs(y),90):.2f} 99%={np.percentile(np.abs(y),99):.2f} cm")

    # Angolo rispetto all'asse Z (in gradi), assumendo direzione normalizzata
    theta_x = np.degrees(np.arctan2(dx, dz))
    theta_y = np.degrees(np.arctan2(dy, dz))
    print(f"\n--- Distribuzione angolare rispetto all'asse ---")
    print(f"theta_x: percentili |.| -> 50%={np.percentile(np.abs(theta_x),50):.2f} 90%={np.percentile(np.abs(theta_x),90):.2f} 99%={np.percentile(np.abs(theta_x),99):.2f} gradi")
    print(f"theta_y: percentili |.| -> 50%={np.percentile(np.abs(theta_y),50):.2f} 90%={np.percentile(np.abs(theta_y),90):.2f} 99%={np.percentile(np.abs(theta_y),99):.2f} gradi")

    # Angolo richiesto per passare i jaw del campo 5x5 (proiezione dal fuoco Z=0)
    # NB: valido solo se z.mean() e' vicino a 0 (fuoco della sorgente)
    y_jaw_z, jaw_y_scaled = 32.0, 0.8
    x_jaw_z, jaw_x_scaled = 40.0, 1.0
    ang_y_needed = np.degrees(np.arctan2(jaw_y_scaled, y_jaw_z))
    ang_x_needed = np.degrees(np.arctan2(jaw_x_scaled, x_jaw_z))
    print(f"\nAngolo necessario per passare i jaw (campo 5x5): |theta_x|<{ang_x_needed:.2f} gradi, |theta_y|<{ang_y_needed:.2f} gradi")

    # Frazione di primari che geometricamente rientrano in questo cono
    # (ATTENZIONE: approssimazione valida solo se z.mean() e' vicino a 0;
    # se z.mean() e' lontano da 0 questa stima non ha senso e va rifatta
    # calcolando la traiettoria fino a Z=32/40 partendo dalla vera posizione)
    within_cone = (np.abs(theta_x) < ang_x_needed) & (np.abs(theta_y) < ang_y_needed)
    frac = within_cone.sum() / len(theta_x) * 100
    print(f"\nFrazione di primari geometricamente nel cono 5x5: {frac:.4f}%")
    print(f"  -> su {N_CHECK:,} primari, ci si aspetterebbero circa {within_cone.sum():,} fotoni")
    print(f"  -> il file GATE reale ne ha prodotti: 85")
    print(f"  -> rapporto atteso/osservato: {within_cone.sum()/85:.1f}x" if within_cone.sum() > 0 else "  -> nessuno nel cono, serve altra indagine")
