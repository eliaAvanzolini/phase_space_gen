import time
import torch
import numpy as np
import json

# Importiamo sia le funzioni di generazione sia il dizionario delle configurazioni reali
from dose_validation import MODELS, generate_cfm_nsf, generate_gan_sarrut

# Configurazione del Benchmark Ottimizzata
N_PARTICELLE = 1000000  # 1 Milione di particelle (veloce ma statisticamente stabile)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("\n=============================================================")
print(" ⏱️  BENCHMARK SCIENTIFICO: VELOCITÀ DI INFERENZA PYTORCH")
print(f" 📊 Campione: {N_PARTICELLE:,} particelle su dispositivo: {DEVICE.upper()}")
print("=============================================================\n")

def misura_tempo_modello(nome_modello, funzione_generazione, **kwargs):
    print(f"⏳ Avvio generazione con {nome_modello.upper()}...")
    
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        
    t0 = time.time()
    
    # Esecuzione della generazione pura
    _ = funzione_generazione(**kwargs)
    
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        
    t1 = time.time()
    tempo_totale = t1 - t0
    velocita = N_PARTICELLE / tempo_totale
    
    print(f"  ✓ Completato in: {tempo_totale:.3f} secondi")
    print(f"  🚀 Velocità: {velocita:,.0f} particelle/secondo\n")
    return tempo_totale, velocita

risultati = {}

# 1. BENCHMARK NEURAL SPLINE FLOW (NSF)
try:
    t, v = misura_tempo_modello(
        "Neural Spline Flow (NSF)", 
        generate_cfm_nsf, 
        model_cfg=MODELS["nsf"], 
        n_samples=N_PARTICELLE, 
        device=DEVICE
    )
    risultati["NSF"] = (t, v)
except Exception as e:
    print(f"❌ Impossibile testare NSF: {e}\n")

# 2. BENCHMARK CONDITIONAL FLOW MATCHING (CFM) - 100 STEPS
try:
    t, v = misura_tempo_modello(
        "CFM (100 Steps ODE)", 
        generate_cfm_nsf, 
        model_cfg=MODELS["cfm"], 
        n_samples=N_PARTICELLE, 
        device=DEVICE, 
        n_ode_steps=100
    )
    risultati["CFM (100 steps)"] = (t, v)
except Exception as e:
    print(f"❌ Impossibile testare CFM 100: {e}\n")

# 3. BENCHMARK CONDITIONAL FLOW MATCHING (CFM) - 500 STEPS (Il tuo upgrade!)
try:
    t, v = misura_tempo_modello(
        "CFM (500 Steps ODE)", 
        generate_cfm_nsf, 
        model_cfg=MODELS["cfm"], 
        n_samples=N_PARTICELLE, 
        device=DEVICE, 
        n_ode_steps=500
    )
    risultati["CFM (500 steps)"] = (t, v)
except Exception as e:
    print(f"❌ Impossibile testare CFM 500: {e}\n")

# 4. BENCHMARK GAN (SARRUT REPLICA)
try:
    t, v = misura_tempo_modello(
        "Wasserstein GAN", 
        generate_gan_sarrut, 
        model_cfg=MODELS["gan"], 
        n_samples=N_PARTICELLE
    )
    risultati["GAN"] = (t, v)
except Exception as e:
    print(f"❌ Impossibile testare GAN: {e}\n")

# STAMPA DELLA TABELLA COMPRENSIVA PER LA TESI
print("=============================================================")
print(" 📊 TABELLA RIASSUNTIVA DI EFFICIENZA COMPUTAZIONALE")
print("=============================================================")
print(f"{'Modello':<22} | {'Tempo (s)':<12} | {'Velocità (particelle/s)':<25}")
print("-" * 65)
for modello, (tempo, velocita) in risultati.items():
    print(f"{modello:<22} | {tempo:<12.3f} | {velocita:,.0f}")
print("=============================================================\n")
