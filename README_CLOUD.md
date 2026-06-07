# Phase Space Generative Models — Guida Cloud

## Prima di caricare sul cloud

```bash
# Verifica che tutto funzioni in locale (dati sintetici, ~5 min)
python check_pipeline.py
```

Se tutti i check sono ✓, sei pronto.

---

## Cosa caricare

```bash
tar -czf phase_space_gen_cloud.tar.gz \
    phase_space_gen/ \
    data/elekta_6mv_train.h5 \
    data/elekta_6mv_eval.h5
```

Dimensione stimata: ~3.5 GB (2 file HDF5 + codice).

---

## Scegliere il server cloud

| Provider | GPU | Costo/h | Costo totale* | Note |
|---|---|---|---|---|
| **Vast.ai** | RTX 3090 24GB | ~$0.30 | ~$30 | Più economico |
| **Lambda Labs** | A10 24GB | ~$0.60 | ~$55 | Più stabile |
| **RunPod** | RTX 3090 | ~$0.35 | ~$35 | Buon equilibrio |
| AWS p3.2xlarge | V100 16GB | ~$3.00 | ~$250 | Troppo caro |

*Training completo: GAN 2h + NSF 60h + CFM 25h = ~87h

**Raccomandazione**: Vast.ai con RTX 3090, almeno 40 GB disco, Ubuntu 22.04.

---

## Setup sul server

```bash
# 1. Estrai e vai nella directory
tar -xzf phase_space_gen_cloud.tar.gz
cd phase_space_gen

# 2. Installa dipendenze
bash run_cloud.sh setup

# 3. Verifica dati
python -c "import h5py; f=h5py.File('../data/elekta_6mv_train.h5'); print(f['phase_space'].shape)"
```

---

## Training

```bash
# Tutto in sequenza (consigliato, ~90h totali)
bash run_cloud.sh all

# Oppure uno alla volta in sessioni separate (usa tmux o screen)
tmux new -s gan
bash run_cloud.sh gan   # ~2h

tmux new -s nsf
bash run_cloud.sh nsf   # ~60h

tmux new -s cfm
bash run_cloud.sh cfm   # ~25h
```

---

## Parametri ottimali (letteratura)

### GAN — replica esatta Sarrut 2019
```
h_dim=400, z_dim=6, lr=1e-5, batch=10000, n_critic=4, 80k iterazioni
```
Non modificare: l'obiettivo è replicare il paper, non ottimizzare.

### NSF — per battere GAN
```
n_transforms=12, n_bins=16, hidden_dim=256, lr=3e-5
tail_bound=7.0, spherical=True, epochs=1500
```
Da letteratura CaloFlow (Krause 2021): n_bins=16 è ottimale per 5D,
più bin su spazio a bassa dimensionalità causa overfitting nelle regioni sparse.
tail_bound=7.0 copre i ±5.3σ della distribuzione x/y IAEA.

### CFM — per battere GAN e comparare con NSF
```
n_layers=6, hidden_dim=512, lr=5e-5, spherical=True, epochs=800
```
Da Bothmann et al. 2025 (flow matching per LHC): MLP residuale con
6 layer da 512 neuroni è il punto di ottimo per distribuzioni fisiche 5-7D.

---

## Dopo il training

### 1. Esporta i modelli per GATE
```bash
bash run_cloud.sh export
# Produce: outputs/gate_models/{gan,nsf,cfm}_elekta_6mv.pth
```

### 2. Scarica gli output
```bash
# Dal tuo computer locale:
scp -r user@server:~/phase_space_gen/outputs/gate_models/ .
scp -r user@server:~/phase_space_gen/outputs/ .
```

### 3. Valutazione distribuzionale (in locale o sul server)
```bash
# Tabella comparativa GAN vs NSF vs CFM
python eval_conditional.py table \
    --gan_report  outputs/baseline_GAN_iaea/metrics.json \
    --nsf_report  outputs/nsf_iaea_spherical*/eval/nsf_report.json \
    --cfm_report  outputs/cfm_iaea_spherical*/eval/cfm_report.json
```

### 4. Validazione downstream con GATE 10 (in locale con GATE installato)
```bash
# Gold standard (phase space GATE come sorgente)
python gate_integration/gate_simulations.py dose_reference \
    --phsp_file data/elekta_6mv_eval.h5 \
    --n_particles 10000000 \
    --output_dir outputs/dose_reference

# Ogni modello generativo
for model in gan nsf cfm; do
    python gate_integration/gate_simulations.py dose_model \
        --pth_filename outputs/gate_models/${model}_elekta_6mv.pth \
        --n_particles 10000000 \
        --output_dir outputs/dose_${model}
done

# Confronto e gamma-index (Fig. 6-7 del paper)
python validate_gate_output.py \
    --reference outputs/dose_reference/dose_reference_dose.mhd \
    --compare \
    --models outputs/dose_gan/dose_gan_dose.mhd \
             outputs/dose_nsf/dose_nsf_dose.mhd \
             outputs/dose_cfm/dose_cfm_dose.mhd \
    --labels "GAN (Sarrut)" "NSF" "CFM" \
    --output_dir outputs/final_comparison
```

---

## Struttura finale degli output attesi

```
outputs/
├── baseline_GAN_iaea/
│   ├── best_model.pt
│   ├── metrics.json          ← W1, separability del GAN
│   └── fig2_marginal_distributions.png
├── nsf_iaea_spherical_*/
│   ├── best_model.pt
│   ├── normalization_stats.json
│   └── eval/nsf_report.json  ← W1, MMD, separability NSF
├── cfm_iaea_spherical_*/
│   └── eval/cfm_report.json  ← W1, MMD, separability CFM
├── gate_models/
│   ├── gan_elekta_6mv.pth
│   ├── nsf_elekta_6mv.pth
│   └── cfm_elekta_6mv.pth
└── final_comparison/
    ├── dose_profiles.png       ← Fig. 7 paper (depth dose + profilo)
    ├── comparison_diff_hist.png ← Fig. 6 paper (istogramma differenze)
    └── gamma_*.json            ← pass rate 2%/2mm per ogni modello
```

---

## Metriche target (da confrontare col paper Sarrut 2019)

| Metrica | GAN (paper) | NSF (target) | CFM (target) |
|---|---|---|---|
| W1 medio | ~0.04 | < 0.015 | < 0.015 |
| Separability | ~0.78 | < 0.55 | < 0.55 |
| Gamma 2%/2mm | > 95% | > 97% | > 97% |
| Max Δ% | < 4% | < 2% | < 2% |
| Mean Δ% | < 0.03% | < 0.02% | < 0.02% |
