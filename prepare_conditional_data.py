import os
import glob
import h5py
import numpy as np
import uproot
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_balance", action="store_true", help="Disabilita il bilanciamento delle classi")
    args = parser.parse_args()

    print("\n==================================================")
    print("🗂️ AVVIO CREAZIONE DATASET CONDIZIONALE PURIFICATO")
    print("==================================================")

    # Configurazione delle etichette GATE attese: Cartella -> [E_nom, jaw_x, jaw_y]
    gate_configs = {
        "6mv_5x5": [6.0, 2.5, 2.5],
        "6mv_10x10": [6.0, 5.0, 5.0],
        "6mv_20x20": [6.0, 10.0, 10.0],
        "10mv_5x5": [10.0, 2.5, 2.5],
        "10mv_10x10": [10.0, 5.0, 5.0],
        "10mv_20x20": [10.0, 10.0, 10.0],
    }

    all_ps = []
    all_cond = []
    counts = {}

    # Elenco dei branch ufficiali generati dal PhaseSpaceActor di GATE
    gate_branches = [
        "PrePosition_X", "PrePosition_Y", "PrePosition_Z",
        "PreDirection_X", "PreDirection_Y", "PreDirection_Z",
        "KineticEnergy"
    ]

    # 1. CARICAMENTO RIGOROSO DEI FILE SORGENTE GATE
    for folder, cond_vec in gate_configs.items():
        path_pattern = f"outputs/gate_jaw/{folder}/{folder}_phsp_part*.root"
        files = sorted(glob.glob(path_pattern))
        if not files:
            print(f"⚠️ Nessun file trovato per {folder} al percorso {path_pattern}")
            continue

        folder_ps = []
        for fpath in files:
            with uproot.open(fpath) as f:
                keys = f.keys()
                if not keys:
                    continue
                # Identificazione dinamica dell'albero phsp_actor
                main_key = keys[0]
                tree = f[main_key]
                
                # Estrazione delle sottomatrici dell'albero di fase usando i branch reali di GATE
                arrays = tree.arrays(gate_branches, library="np")
                n_events = len(arrays["KineticEnergy"])
                
                # Costruzione matrice 7D standard del progetto (Posizioni convertite mm -> cm)
                chunk = np.zeros((n_events, 7), dtype=np.float32)
                chunk[:, 0] = arrays["PrePosition_X"] / 10.0   # mm -> cm
                chunk[:, 1] = arrays["PrePosition_Y"] / 10.0   # mm -> cm
                chunk[:, 2] = arrays["PrePosition_Z"] / 10.0   # mm -> cm
                chunk[:, 3] = arrays["PreDirection_X"]
                chunk[:, 4] = arrays["PreDirection_Y"]
                chunk[:, 5] = arrays["PreDirection_Z"]
                chunk[:, 6] = arrays["KineticEnergy"]
                
                folder_ps.append(chunk)

        if folder_ps:
            folder_ps = np.concatenate(folder_ps, axis=0)
            cond_matrix = np.tile(cond_vec, (len(folder_ps), 1))
            
            all_ps.append(folder_ps)
            all_cond.append(cond_matrix)
            counts[tuple(cond_vec)] = len(folder_ps)
            print(f"  ✓ Caricato GATE {folder} -> {len(folder_ps)} particelle grezze registrate.")

    # 2. ISPEZIONE FILE IAEA ESTERNI E APPLICAZIONE FILTRO ANTI-COLLISIONE
    iaea_files = sorted(glob.glob("data/ELEKTA_PRECISE_10mv_part*.IAEAphsp"))
    if iaea_files:
        print("\n📦 Rilevati file IAEA esterni nel disco. Verifica collisioni in corso...")
        iaea_cond = [10.0, 5.0, 5.0]
        
        if tuple(iaea_cond) in counts:
            print(f"  🛡️ [FILTRO ATTIVO] La condizione {iaea_cond} è già coperta dalle simulazioni GATE.")
            print("     I file IAEA vengono scartati per impedire la contaminazione fisica del dataset.")
        else:
            print("  ✓ Nessuna collisione rilevata per i file IAEA esterni.")

    if not all_ps:
        print("❌ Errore: Nessun dato estratto. Processo interrotto.")
        return

    # 3. BILANCIAMENTO DELLE CLASSI SUL MINIMO COMUNE DENOMINATORE
    final_ps = []
    final_cond = []
    
    if not args.no_balance and len(counts) > 1:
        np.random.seed(42)  # riproducibilita' del sottocampionamento
        min_count = min(counts.values())
        print(f"\n⚖️ Bilanciamento attivo. Target di campionamento: {min_count} eventi stabili per classe.")
        
        for ps_mat, cond_mat in zip(all_ps, all_cond):
            # Selezione casuale ma senza reinserimento (replace=False) per preservare l'unicità
            indices = np.random.choice(len(ps_mat), size=min_count, replace=False)
            final_ps.append(ps_mat[indices])
            final_cond.append(cond_mat[indices])
    else:
        print("\n⚠️ Bilanciamento disattivato o classe singola. Unione asimmetrica delle matrici.")
        final_ps = all_ps
        final_cond = all_cond

    final_ps = np.concatenate(final_ps, axis=0)
    final_cond = np.concatenate(final_cond, axis=0)

    # 4. SCRITTURA FILE HDF5 COMPRESSO FINALE
    h5_path = "data/conditional_jaws_dataset.h5"
    with h5py.File(h5_path, "w") as h5_f:
        h5_f.create_dataset("phase_space", data=final_ps, compression="gzip")
        h5_f.create_dataset("conditions", data=final_cond, compression="gzip")

    print("\n==================================================")
    print("🎉 DATASET CONDIZIONALE RIGENERATO CON SUCCESSO")
    print("==================================================")
    print(f" -> File d'uscita: {h5_path}")
    print(f" -> Dimensioni finali spazio delle fases: {final_ps.shape}")
    print(f" -> Dimensioni finali condizioni:         {final_cond.shape}")
    
    print("\n📊 Verifica finale della distribuzione dei vettori h5:")
    for c in np.unique(final_cond, axis=0):
        mask = np.all(final_cond == c, axis=1)
        print(f"    Configurazione {c.tolist()} -> {mask.sum()} fotoni unici.")

if __name__ == "__main__":
    main()
