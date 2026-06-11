# Phase Space Generative Models
### Generative AI per simulazioni Monte Carlo di Fisica Medica

Implementazione di **Neural Spline Flows (NSF)**, **Conditional Flow Matching (CFM)** e **WGAN-GP** per il phase space modeling di sorgenti di fasci medici (linac, SPECT).

---
<!--
## Struttura del progetto

```
phase_space_gen/
├── data/
│   ├── synthetic_linac.py   # generatore sintetico (fisicamente motivato)
│   └── dataset.py           # PyTorch Dataset wrapper + normalizzazione
├── models/
│   ├── nsf.py               # Neural Spline Flow (nflows)
│   ├── cfm.py               # Conditional Flow Matching (zuko + torchdiffeq)
│   └── gan.py               # WGAN-GP baseline (Sarrut 2019)
├── utils/
│   └── plot_training.py     # curve di training, tabella comparazione
├── configs/
│   ├── cfm_default.yaml     # CFM condizionato (Fase 3 roadmap)
│   ├── nsf_default.yaml     # NSF singola config (Fase 2)
│   └── gan_baseline.yaml    # GAN baseline (Fase 1)
├── train.py                 # training unificato con CLI
├── generate.py              # inferenza → file HDF5 per GATE
├── evaluate.py              # W1, MMD, separability, plot
├── demo.py                  # demo numpy-only (no PyTorch)
└── requirements.txt
```

---

## Quick start

### 1. Demo (nessun PyTorch richiesto)

Verifica che la pipeline funzioni e genera i plot delle distribuzioni:

```bash
pip install scipy scikit-learn matplotlib h5py
python demo.py
# Output in: outputs/demo/
```

### 2. Installazione completa

```bash
# CPU (sviluppo locale)
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install nflows zuko torchdiffeq h5py scipy scikit-learn matplotlib

# GPU (workstation laboratorio)
pip install torch nflows zuko torchdiffeq h5py scipy scikit-learn matplotlib
```

### 3. Training

**Fase 1 — Baseline GAN** (riproduce Sarrut 2019):
```bash
python train.py --model gan --n_samples 1000000 --epochs 300
```

**Fase 2 — Neural Spline Flow** (primo confronto):
```bash
python train.py --model nsf --n_samples 1000000 --epochs 200
```

**Fase 3 — Conditional Flow Matching** (modello condizionato):
```bash
python train.py --model cfm --conditional --epochs 300
```

**Con dati GATE reali** (invece dei dati sintetici):
```bash
python train.py --model cfm --conditional \
                --data_path /path/to/linac_6MV.h5 \
                --epochs 300
```

### 4. Generazione per GATE

```bash
# Genera 1M campioni con il modello CFM addestrato
python generate.py \
    --checkpoint outputs/cfm_run/best_model.pt \
    --model cfm --conditional \
    --E_nom 6.0 --jaw_x 5.0 --jaw_y 5.0 \
    --n_samples 1000000 \
    --validate \
    --out linac_6MV_generated.h5
```

**Uso in GATE (Python API)**:
```python
source = sim.add_source("PhaseSpaceSource", "beam")
source.phsp_file = "linac_6MV_generated.h5"
source.particle  = "gamma"
```

---

## Il Phase Space

Ogni particella è rappresentata da un vettore 7D:

```
s = (x, y, z, dx, dy, dz, E)
    ─────────────────────────
    posizione [cm]: x, y, z
    direzione (||d||=1): dx, dy, dz  ∈ S²
    energia [MeV]: E > 0
```

Il modello lavora in **6D** (z è costante = 0 al piano isocentrico) e reinserisce z=0 in post-processing.

**Condizionamento**: `c = [E_nom, jaw_x, jaw_y]` — un unico modello copre tutte le configurazioni del linac senza rigenerare file PS separati.

---

## Metriche di validazione

| Metrica | Significato | Target |
|---------|-------------|--------|
| W1 (mean) | Wasserstein-1 medio su 6 dimensioni | < 0.01 |
| W1 (E) | Fidelità spettro energetico | < 0.01 |
| MMD² | Distanza distribuzione congiunta 6D | < 0.001 |
| Separability | Accuracy RF (0.5=ottimo, 1.0=fail) | ≈ 0.50 |
| γ-index 2%/2mm | Metrica clinica downstream | > 95% |

---

## Perché NSF e CFM invece delle GAN

| Problema GAN | NSF/CFM |
|---|---|
| Mode collapse (code mancanti) | No: NLL penalizza tutto uniformemente |
| No likelihood esatta | NSF: log p(s) esatta per ogni campione |
| Training instabile (G vs D) | Loss MSE diretta, convergenza monotona |
| Vincolo `||d||=1` non nativo | Reparametrizzazione (θ, φ) integrata |
| Non invertibile | NSF: invertibile per design |

---

## Paper di riferimento

| Modello | Paper chiave |
|---------|-------------|
| **Baseline** | Sarrut et al. 2019 — `doi:10.1088/1361-6560/ab3fc3` |
| **NSF** | Durkan et al. 2019 — `arXiv:1906.04032` |
| **CFM** | Lipman et al. 2022 — `arXiv:2210.02747` |
| **Blueprint fisico** | Farmer et al. 2025 — `arXiv:2512.13965` |
| **Review campo** | Sarrut et al. 2021 — `doi:10.3389/fphy.2021.738899` |

---

## Roadmap di sviluppo

- [x] **Fase 1**: generazione dati sintetici + pipeline di valutazione
- [x] **Fase 1**: baseline WGAN-GP (Sarrut 2019)
- [x] **Fase 2**: Neural Spline Flow 6D
- [x] **Fase 3**: Conditional Flow Matching multi-config
- [ ] **Fase 4**: validazione fisica downstream (dose in fantoccio GATE)
- [ ] **Fase 5**: estensione a sorgenti SPECT
-->
