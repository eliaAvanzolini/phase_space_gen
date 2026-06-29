# 📑 Report Dosimetrico: Formule di Confronto e Risultati Finali (100M Storie)

Questo documento riassume le metriche matematiche utilizzate per l'analisi del profilo di dose tridimensionale e compila i risultati definitivi di Gamma Index e scostamento relativo ($\Delta\%$) per i modelli generativi analizzati.

---

## 1. Formule di Normalizzazione della Dose

Per il calcolo del delta medio relativo ($\Delta\%$) all'interno del volume del fantoccio (sotto la maschera di sbarramento al $10\%$ della dose massima), sono state impiegate due diverse strategie di normalizzazione.

### 1.1 Formula A: Normalizzazione Locale Separata (Profili di Forma)
Questa metrica effettua una conversione scalare indipendente per ciascun volume sul rispettivo picco di dose massima locale. Rileva puramente le discrepanze nella forma geometrica del profilo, ma risente dell'instabilità numerica o statistica del singolo voxel di picco ($\max(D_{\text{mod}})$).

La formula applicata per il calcolo del $\Delta_{\text{locale}}\%$ sul singolo voxel è:

$$D_{\text{diff, locale}}(x,y,z) = \left( \frac{D_{\text{mod}}(x,y,z)}{\max(D_{\text{mod}})} - \frac{D_{\text{ref}}(x,y,z)}{\max(D_{\text{ref}})} \right) \times 100$$

Il valore medio aggregato calcolato sui soli $N$ voxel validi appartenenti alla maschera è:

$$\Delta_{\text{medio, locale}}\% = \frac{1}{N} \sum_{x,y,z \in \text{mask}} \left( \frac{D_{\text{mod}}(x,y,z)}{\max(D_{\text{mod}})} - \frac{D_{\text{ref}}(x,y,z)}{\max(D_{\text{ref}})} \right) \times 100$$

### 1.2 Formula B: Normalizzazione Ancorata al Reference (Bilancio Fisico Assoluto)
Questa metrica ancora stabilmente entrambi i volumi di dose allo stesso identico denominatore, rappresentato dal valore di picco del Gold Standard Monte Carlo ($\max(D_{\text{ref}})$). Elimina gli artefatti causati dalle fluttuazioni dei massimi locali dei modelli generativi, esprimendo il reale bilancio energetico depositato nel bulk del fantoccio.

La formula applicata per il calcolo del $\Delta_{\text{ancorato}}\%$ sul singolo voxel è:

$$D_{\text{diff, ancorata}}(x,y,z) = \left( \frac{D_{\text{mod}}(x,y,z) - D_{\text{ref}}(x,y,z)}{\max(D_{\text{ref}})} \right) \times 100$$

Il valore medio aggregato finale ancorato è:

$$\Delta_{\text{medio, ancorato}}\% = \frac{1}{N} \sum_{x,y,z \in \text{mask}} \left( \frac{D_{\text{mod}}(x,y,z) - D_{\text{ref}}(x,y,z)}{\max(D_{\text{ref}})} \right) \times 100$$

---

## 2. Tabella Riassuntiva dei Risultati Clinici

I valori di **Gamma Pass Rate** sono stati calcolati sulle matrici di dose assoluta non normalizzata tramite la libreria `pymedphys`, applicando una soglia di sbarramento (Cutoff) al $10\%$ di $D_{\text{max}}$ secondo le linee guida internazionali **AAPM TG-218**.

| Modello Generativo | Configurazione Inferenza | Gamma Pass Rate (2% / 2mm) | Gamma Pass Rate (3% / 3mm) | Δ Medio (Formula A - Locale) | Δ Medio (Formula B - Ancorata) | Esito Clinico (Soglia ≥ 95%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Neural Spline Flow (NSF)** | 100M storie (16 Bins) | 95.24%  | 99.94%  | +0.7563%  | +0.1428% | 🟢 **Approvato** (Eccellente accuratezza nativa) |
| **Conditional Flow Matching (CFM)** | 100M storie (500 steps ODE) | 99.77%  | 99.99%  | -1.2529%  | +0.0283% | 🟢 **Approvato** (Massima precisione dosimetrica) |
| *CFM (Pre-Upgrade)* | *100M storie (100 steps ODE)* | *86.68%*  | *99.99%* | +1.3159% | — | ⚠️ **Respinto a 2%** (Drift numerico di Eulero) |
| **Wasserstein GAN (Sarrut)** | 100M storie (Replica) | 48.27% | 85.92% | +3.3239%  | +1.6801% | 🔴 **Respinto** (Fallimento e distorsione spettrale) |

---

## 3. Considerazioni Chiave per la Discussione

1. **Ottimizzazione del CFM:** Il passaggio da 100 a 500 step nell'integratore del CFM ha innalzato il pass rate dal 86.68%  al 99.77%  sotto i criteri stringenti del 2%/2mm, azzerando di fatto il $\Delta$ medio complessivo (+0.0283%). Questo attesta che l'architettura neurale possedeva una corretta comprensione intrinseca della fisica, limitata in precedenza solo dall'errore di discretizzazione del solutore di Eulero.
2. **Superiorità dei Modelli a Flusso:** Sia l'NSF che il CFM ottimizzato superano abbondantemente i requisiti di accettabilità clinica ospedaliera, consolidandosi come candidati ideali per la sostituzione dei file dello Spazio delle Fasi.
3. **Inadeguatezza della GAN:** La GAN fallisce drasticamente entrambi i criteri e accumula un bias positivo reale di +1.6801% nel volume, ascrivibile a una distorsione sistematica dello spettro energetico generato (eccesso di fotoni a bassa energia).

### 4 . Efficienza Computazionale e Velocità di Inferenza

I tempi e le velocità di generazione si riferiscono a un campione di benchmark standardizzato di **1.000.000 di particelle** (spazio delle fasi sintetico) campionato interamente su GPU dedicata CUDA.

| Modello Generativo | Tempo di Inferenza (s) | Velocità (particelle/s) |
| :--- | :---: | :---: |
| **Wasserstein GAN** (Sarrut) | 2.021 | 494,841 |
| **Neural Spline Flow (NSF)** | 12.089 | 82,723 |
| **CFM** (100 steps ODE) | 184.468 | 5,421 |
| **CFM** (500 steps ODE) | 189.306 | 5,282 |

---
