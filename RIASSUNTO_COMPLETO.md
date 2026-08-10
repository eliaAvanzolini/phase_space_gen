# Riassunto completo del progetto — Modelli generativi condizionati per phase space linac

*Versione arricchita con i risultati numerici (report JSON e log) inseriti nei punti della
narrazione a cui si riferiscono.*

Questo documento copre due capitoli distinti del lavoro di tesi:
- **Parte 1**: Modelli condizionati su energia + apertura jaws, validazione statistica e dosimetrica (gamma index)
- **Parte 2**: Interpolazione a sola energia (6MV+25MV di training, 10MV nascosto come ground truth)

---

# PARTE 1 — Modelli condizionati su energia e jaws, validazione dosimetrica

## 1. Punto di partenza

Obiettivo: addestrare tre modelli generativi condizionati (GAN, NSF, CFM) a stimare il phase space
di un linac Elekta Precise per 6 configurazioni energia/campo (6MV e 10MV, campi 5x5/10x10/20x20 cm),
usando come sorgente i file IAEA ufficiali (part1 per il training, part2 come held-out per la
valutazione finale e il gamma index). L'esperimento condizionato presentava risultati inizialmente
scorretti a causa di un problema di duplicati (part1 era stato convertito solo parzialmente in ROOT
in precedenza, causando il riciclo degli stessi eventi sia in training che in eval).

Primo intervento: riconversione completa di part1 in ROOT (~124M particelle per file/energia) e
verifica di intersezione fisica a 5 cifre decimali tra part1 e part2 grezzi — risultato: zero
duplicati, dataset di partenza pulito.

## 2. Come si ottengono le 6 classi campo/energia

Le 6 classi derivano da simulazioni GATE: i fotoni primari di part1 vengono trasportati attraverso
una geometria di jaws in tungsteno (aperture 2.5/5/10 cm semi-apertura) e registrati su un piano di
phase space a valle.



## 3. Parallelizzazione delle simulazioni GATE-jaw

Un benchmark iniziale a 100k particelle ha mostrato tempi proiettati di 75-101 ore per classe — non
praticabile nei limiti di walltime del cluster.

Soluzione: script di generazione job a due livelli — 30 chunk indipendenti per classe (per stare
sotto il walltime) e, dentro ciascun chunk, un entry_start esplicito per thread.

Tutti i 30x6 job sono stati eseguiti con successo, integrita' e assenza di duplicati confermate.

## 4. Dataset

**Sorgente**: `outputs/gate_jaw/{classe}/{classe}_phsp_part*.root` — cioè le simulazioni GATE (non i vettori IAEA grezzi direttamente), dove i fotoni primari di **part1** vengono trasportati attraverso la geometria dei jaws in tungsteno per ciascuna delle 6 combinazioni energia/campo.

**Costruzione** (`prepare_conditional_data_PATCHED.py`): per ciascuna delle 6 classi, legge tutti i file `_phsp_partN.root` di quella cartella, li concatena, poi **bilancia** sottocampionando ogni classe al conteggio della classe più piccola (`6mv_5x5`, 48.397 particelle) con seed fisso 42 — dataset finale: **48.397 × 6 = 290.382 particelle totali**, equamente distribuite tra le 6 classi.

**Colonne**: `[X, Y, Z, dX, dY, dZ, E]` (7 colonne, cm/adimensionale/MeV) + `conditions = [E_nom, jaw_x, jaw_y]` ripetuto per ogni particella.

**Split**: 70/15/15 (train/val/test) fatto internamente da `train.py` con seed 42, sullo stesso file h5.

## 5. Scelta di voxel size e criteri gamma

Un voxel richiede 1/2-1/3 del DTA: la scelta iniziale di 4mm era troppo grossolana per i criteri
gamma usati (3%/3mm, 2%/2mm clinico). Deciso 2mm come compromesso.

## 6. Bug MT nella simulazione di dose

Test diagnostico: stessa richiesta di particelle, confronto n_threads=1 vs 8 -> dose 8.09x piu' alta
a 8 thread. Stesso bug entry_start di prima, ma nello script di validazione dosimetrica. Fix
verificato con test di linearita' (rapporto osservato 1.085 contro atteso 1.084).

## 7. Prima analisi gamma index completa: risultati scarsi

Con voxel 2mm, pass rate insufficienti ovunque, bias medio sistematicamente negativo (-10.6% a
-15.7%) e quasi identico tra CFM e NSF in ogni classe, nonostante le metriche a livello di phase
space mostrassero CFM nettamente superiore. Segnale tipico di artefatto di normalizzazione/rumore.

## 8. Caccia al rumore di normalizzazione

Il massimo grezzo usato per normalizzare le mappe e' instabile con statistica MC limitata (un
piccolo cluster di voxel puo' catturare per caso il deposito di un elettrone secondario energetico
raro). Confermato con rapporto P99.9/Dmax basso (40-67%) e incertezza statistica elevata al voxel
del massimo. Applicata normalizzazione robusta (percentile 99.9).

## 9. Noise floor test

Split del reference in due meta' indipendenti, gamma index di A contro B. Per 5 classi su 6, CFM/NSF
risultavano statisticamente indistinguibili dal rumore. Unica eccezione: 6mv_20x20, dove emerse una
vera discrepanza fisica. Conclusione: il gamma index era data-limited per la maggior parte delle
classi.

## 10. Aumento di statistica: part3/part4

Reperiti online altri due file IAEA ufficiali (part3, part4) per entrambe le energie — statistica
realmente indipendente, non un espediente statistico (il recycling introduce varianza latente senza
vera nuova informazione oltre un certo fattore di riuso).

## 11. Bug storico specifico nel reference 6MV

Confrontando i conteggi GATE-jaw di part3/part4 contro il vecchio reference part2: per le sole
classi 6MV, part2 aveva conteggi 4.5-7.4x superiori a part1/part3/part4. Causa: il gate_jaw_ref
(part2) originale per le 6MV risaliva a prima della correzione entry_start. Fix: rigenerate le 3
classi 6MV. Un tentativo successivo ha causato OOM — causa isolata: file sorgente non compresso
(4.9GB invece di 2.6GB).

## 12. Fusione finale del reference e secondo giro di regressioni

Reference finale: part2 (fixed per le 6MV) + part3 + part4 per classe. Una riscrittura dello script
di dose validation aveva perso silenziosamente il fix entry_start — individuata e ripristinata prima
del rilancio.

## 13. Rilancio completo e secondo giro di gamma index

Con statistica molto piu' ampia, il bias sistematico non e' migliorato — segnale che il problema non
era (solo) rumore riducibile con piu' campioni. Un tentativo di normalizzazione simmetrica robusta ha
persino peggiorato il bias, spiegabile dal fatto che il rapporto max/P99.9 non e' uguale tra
reference e modello.

### Risultati numerici — secondo giro (normalizzazione P99.9 standard, 2mm voxel)

Fonte: `dose_validation/gamma_summary_all_classes.json`. Δ medio negativo = dose del modello
sistematicamente sotto-stimata rispetto al reference in quel range dinamico.

| Classe | Modello | Δ medio | γ 3%/3mm (pass rate) | γ 2%/2mm clinico (pass rate) |
|---|---|---:|---:|---:|
| 6mv_5x5 | CFM | -15.8% | 74.6% | 56.2% |
| 6mv_5x5 | NSF | -15.3% | 75.6% | 57.3% |
| 6mv_10x10 | CFM | -18.8% | 61.9% | 41.6% |
| 6mv_10x10 | NSF | -18.6% | 64.8% | 44.6% |
| 6mv_20x20 | CFM | -17.9% | 64.8% | 44.9% |
| 6mv_20x20 | NSF | -18.4% | 66.2% | 46.2% |
| 10mv_5x5 | CFM | -16.1% | 70.0% | 52.5% |
| 10mv_5x5 | NSF | -16.0% | 72.4% | 55.2% |
| 10mv_10x10 | CFM | -14.3% | 80.6% | 64.2% |
| 10mv_10x10 | NSF | -15.1% | 78.3% | 61.1% |
| 10mv_20x20 | CFM | -11.6% | 87.7% | 74.0% |
| 10mv_20x20 | NSF | -12.1% | 87.3% | 73.4% |

Si conferma quanto descritto nel testo: bias quasi identico tra CFM e NSF in ogni classe (differenze
tipicamente < 1 punto percentuale), nonostante a livello di phase space CFM risulti nettamente
migliore (vedi tabella di sintesi al punto 16) — la firma tipica di un problema di normalizzazione
o di statistica condivisa piuttosto che di fedelta' del modello.

### Il tentativo di normalizzazione simmetrica robusta (peggiorativo)

Fonte: `dose_validation/gamma_summary_robust.json`. Il bias medio peggiora in quasi tutte le classi
(fino a -26.3% per 10mv_5x5/CFM), confermando che il rapporto max/P99.9 non e' uguale tra reference
e modello e che una correzione simmetrica introduce una distorsione ulteriore invece di rimuoverla:

| Classe | Δ medio CFM (robusto) | Δ medio NSF (robusto) |
|---|---:|---:|
| 6mv_5x5 | -16.9% | -17.4% |
| 6mv_10x10 | -14.7% | -14.1% |
| 6mv_20x20 | -20.7% | -21.5% |
| 10mv_5x5 | -26.2% | -26.3% |
| 10mv_10x10 | -25.3% | -26.2% |
| 10mv_20x20 | -23.4% | -23.7% |

## 14. La vera causa: granularita' della griglia insufficiente

Estratti i profili di dose (PDD + trasversali). I profili a singolo voxel erano puro rumore anche nel
reference stesso, nessuna forma fisica riconoscibile: conteggio medio per voxel illuminato
nell'ordine di poche unita'. Con media laterale su finestra 7x7 voxel, emerse forme fisicamente
sensate ovunque. Confermata buona fedelta' spaziale; osservata possibile discrepanza in profondita'
non approfondita ulteriormente (fuori scope del capitolo).

## 15. Ultimo tentativo, time-boxed: gamma 1D su profili mediati

Concordato come "bounded": qualunque esito, si chiude il capitolo. Pass rate ancora modesti e
variabili, nessuna classifica coerente tra CFM e NSF tra le classi — firma di un confronto dominato
dal rumore.

### Risultati numerici — gamma 1D su profili mediati (PDD e trasversali a z=3cm e z=10cm)

Fonte: `dose_validation/gamma_summary_1d_profiles.json`, criterio clinico 2%/2mm.

| Classe | Modello | PDD | Trasversale z=3cm | Trasversale z=10cm |
|---|---|---:|---:|---:|
| 6mv_5x5 | CFM | 12.7% | 28.0% | 17.6% |
| 6mv_5x5 | NSF | 41.8% | 48.0% | 35.3% |
| 6mv_10x10 | CFM | 0.0% | 21.1% | 6.5% |
| 6mv_10x10 | NSF | 7.3% | 12.3% | 10.9% |
| 6mv_20x20 | CFM | 0.0% | 24.7% | 6.4% |
| 6mv_20x20 | NSF | 29.1% | 32.3% | 29.8% |
| 10mv_5x5 | CFM | 21.2% | 64.5% | 44.4% |
| 10mv_5x5 | NSF | 47.0% | 45.2% | 50.0% |
| 10mv_10x10 | CFM | 57.1% | 66.1% | 55.6% |
| 10mv_10x10 | NSF | 44.3% | 39.0% | 63.9% |
| 10mv_20x20 | CFM | 49.3% | 53.3% | 60.3% |
| 10mv_20x20 | NSF | 76.1% | 51.1% | 54.0% |

Come descritto nel testo: nessuna classifica coerente — CFM vince in alcune classi (es. 10mv_10x10
PDD), NSF in altre (es. 6mv_5x5 e 10mv_20x20 PDD), spesso con scarti ampi e non sistematici tra
classi affini. Questo pattern, combinato col noise floor test del punto 9, e' l'evidenza principale
usata per concludere che il gamma a livello di dose e' data-limited.

## 16. Decisione finale 

Il gamma index a livello di dose risulta limitato dalla statistica Monte Carlo disponibile. Anche
riducendo il confronto a profili 1D con integrazione laterale, i pass rate mostrano elevata
variabilita' e nessuna classificazione coerente tra i modelli, indicando che il segnale e' dominato
dal rumore statistico. Le metriche a livello di distribuzione di phase space (separability,
Wasserstein, MMD) restano il criterio primario e piu' affidabile, dove CFM mostra fedelta'
sistematicamente superiore a NSF. GAN inclusa come baseline di confronto coerente con Sarrut et al.,
non ottimizzata ulteriormente essendo strutturalmente meno stabile (WGAN-GP).

### Risultati numerici — metriche a livello di phase space (criterio primario)

Fonte: `run_cfm_conditional/eval/cfm_report.json`, `run_nsf_conditional/eval/nsf_report.json`,
`run_gan_conditional/eval/gan_report.json` (valutazione sulle 6 classi condizionate energia+jaws).

| Modello | W1 medio (7 variabili) | MMD | Separability accuracy (0.5 = indistinguibile) |
|---|---:|---:|---:|
| **CFM** | **0.0727** | **0.0000** | **0.553** |
| NSF | 0.2320 | 0.00047 | 0.624 |
| GAN | 0.8266 | 0.02194 | 0.907 |

I numeri confermano quantitativamente l'affermazione del testo: CFM ha W1 medio ~3x piu' basso di
NSF e ~11x piu' basso di GAN, MMD sostanzialmente nullo, e una separability accuracy piu' vicina al
50% (indistinguibilita' statistica da un classificatore reale-vs-generato) rispetto a NSF e
soprattutto a GAN, che risulta facilmente distinguibile dal reale (90.7%) — coerente con la sua
esclusione da ulteriori ottimizzazioni come baseline strutturalmente meno stabile.

---

# PARTE 2 — Interpolazione a sola energia (6MV+25MV -> 10MV nascosto)

## 1. La richiesta

Testare la capacita' del modello generativo condizionato di interpolare nello spazio latente la
fisica del fascio a un'energia mai vista durante il training.

## 2. Decisioni di design

- Sola energia come condizione (non piu' energia+jaws): isola la variabile, e i fasci aperti hanno
  piena statistica IAEA invece dei pool limitati post-jaws del capitolo precedente
- Design principale: training su 6MV + 25MV (stesso acceleratore Elekta Precise), 10MV tenuto
  nascosto come vero ground truth
- Test preliminare gratuito: query rapida ai modelli jaws-condizionati gia' esistenti a E=8 (nessun
  ground truth, ma check di monotonia ed endpoint bremsstrahlung) — risultato incoraggiante

### Risultati numerici — test preliminare a E=8 (nessun ground truth)

Fonte: `energy_interpolation/spectrum_stats_summary.json`. Statistiche dello spettro energetico
generato a E=8 MeV, confrontate con i due estremi reali (6MV e 10MV) come check di monotonia — la
media generata deve cadere tra la media del 6MV reale e quella del 10MV reale:

| Dataset | Media E | Mediana E | Std E |
|---|---:|---:|---:|
| Reale 6MV | 1.701 | 1.382 | 1.209 |
| **Generato CFM a E=8** | **2.072** | **1.635** | **1.530** |
| Generato NSF a E=8 | 2.025 | 1.646 | 1.437 |
| Reale 10MV | 2.652 | 2.131 | 1.884 |

Il check di monotonia e' verificato per entrambi i modelli: la media generata a E=8 cade
correttamente tra 6MV (1.70) e 10MV (2.65) reali, risultato coerente con quanto descritto come
"incoraggiante" nel testo.

## 3. Bug nel file 25MV: split a meta' record

Il dataset 25MV era diviso in due file (part1_a, part1_b). Il taglio tra i due non cadeva su un
confine di record (size_a % 33 = 17, size_b % 33 = 16, somma = 33). Fix: skip dei primi 16 byte del
file _b, con perdita di 1 solo record su 141M.

## 4. Il bug critico delle unita' di misura

Nella funzione di caricamento ROOT (usata sia per il training 6MV sia per il reference 10MV),
applicata erroneamente una divisione per 10 (mm->cm) alle posizioni. Corretta per i ROOT prodotti da
GATE (mm), sbagliata per i ROOT convertiti direttamente da IAEA (gia' in cm).

Conseguenza: il 6MV di training aveva Z circa 2.72cm invece di 27.21cm, mentre il 25MV (caricato
direttamente da IAEA grezzo) era rimasto corretto — training set con due energie in due scale
spaziali diverse. Sulle condizioni viste i numeri sembravano ottimi (train/test dello stesso h5
bacato), ma il confronto con il reference 10MV rivelo' un enorme scarto sulla coordinata Z.

Fix: rimossa la divisione per 10 in entrambi gli script coinvolti. Verificato che entrambe le classi
di training tornano a Z circa 27.21cm.

Costo: rifare da zero il dataset di training e riaddestrare tutti e tre i modelli, questo ha portato via molto tempo.

### Risultati numerici — metriche sulle condizioni viste (6MV+25MV, post-fix)

Fonte: `cfm_energy_only/eval/cfm_report.json`, `nsf_energy_only/eval/nsf_report.json`,
`gan_energy_only/eval/gan_report.json` (n_real = n_gen = 500.000 particelle per modello).

| Modello | W1 medio | MMD | Separability accuracy |
|---|---:|---:|---:|
| **CFM** | **0.0224** | **0.0000** | **0.516** |
| NSF | 0.0783 | 0.00095 | 0.565 |
| GAN | 0.1823 | 0.01918 | 0.894 |

Questi sono proprio i numeri "ottimi sulle condizioni viste" citati nel testo: separability quasi al
50% per CFM (0.516) e W1 molto basso — ma, come spiegato nel paragrafo, erano calcolati train/test
sullo stesso h5 con la scala Z bacata, quindi non intercettavano il bug: il problema emerse solo nel
confronto con il reference 10MV indipendente (vedi §5).

## 5. Secondo bug: una delle 4 parti del reference 10MV con Z diversa

Dopo il fix, lo scarto su Z restava enorme. Verifica per-parte: part1 del 10MV aveva Z=0.0, mentre
part2/3/4 avevano tutti Z=27.21 — piano di registrazione fisicamente diverso, risalente a una
conversione precedente a questa sessione. Fix: escluso part1 dal reference 10MV (restano
part2+part3+part4, circa 372M particelle).

### Verifica numerica del bug e del fix

Fonte: `logs/check_z.out` e `logs/prep_10mv_ref.out`.

| File | N particelle | Z medio (cm) | Z std |
|---|---:|---:|---:|
| 10mv_part1 (escluso) | 124.030.574 | **0.000000** | 0.000000 |
| 10mv_part2 | 124.017.250 | 27.209993 | 0.000006 |
| 10mv_part3 | 124.016.539 | 27.209995 | 0.000004 |
| 10mv_part4 | 124.020.556 | 27.209993 | 0.000006 |

Reference finale ricostruito da part2+part3+part4 (spot-check di duplicati tra le coppie di part
negativo, campione 200.000 eventi ciascuna): **372.054.345 particelle**, coerente con quanto
riportato nel testo ("circa 372M particelle").

## 6. Dimensionamento del training

Con dataset completo (249M totali), CFM comodo (~30h) ma NSF e GAN avrebbero richiesto 120-130+ ore
— troppo anche per la coda con walltime piu' lungo trovata (100h, non le 48h assunte inizialmente).
Tagliato a 80M particelle/classe (160M totali), dimensionato per dare margine a tutti e tre i modelli
per 200 epoche piene sugli stessi dati.

## 7. Crash NSF nella valutazione finale su larga scala

Generando milioni di campioni per la valutazione finale, NSF crashava con instabilita' numerica nelle
spline razionali quadratiche. Fix: valutazione finale limitata a 500k campioni con retry automatico
su eventuali fallimenti, invece di far crashare l'intero job.

## 8. Resume da checkpoint

Implementato --resume_from per riprendere il training da un checkpoint invece di ripartire da zero,
necessario perche' NSF e GAN rischiavano di non finire le 200 epoche nel walltime disponibile.


## 9. Bug minore nella history della GAN

Il salvataggio del trainer GAN includeva la history di training per intero ad ogni checkpoint, ma
questa cresce di una entry per ogni singolo batch (decine di migliaia di volte per epoca), non per
epoca. Checkpoint passati da decine a centinaia di MB nel giro di poche epoche. Fix: troncata a un
massimo di 5000 entry recenti.

## 10. Risultati finali dell'interpolazione a 10MV

Test principale: separability/W1/MMD/KS tra 10MV generato (E=10, mai vista) e 10MV reale. Risultato
iniziale sorprendentemente buono per CFM — ma con un "problema" emerso durante l'interpretazione:

- La separability di CFM era statisticamente vicina a quella di una baseline ingenua (miscela lineare
  di vettori reali 6MV+25MV, senza alcun modello, pesata secondo la posizione di 10MV
  nell'intervallo) — serviva un test piu' mirato per distinguere vera interpolazione da semplice
  media pesata

### Risultati numerici — metriche a livello di phase space, 10MV interpolato

Fonte: `interpolation_10mv_eval/{cfm,nsf,gan,naive_baseline}/*_report.json`, n_real = n_gen = 500.000
particelle per ciascun confronto.

| Dataset generato | W1 medio | MMD | Separability accuracy |
|---|---:|---:|---:|
| **CFM (E=10, mai vista)** | **0.0654** | **0.00066** | **0.541** |
| Baseline ingenua (mix lineare 6+25MV) | 0.0951 | 0.00349 | 0.559 |
| NSF (E=10, mai vista) | 0.1726 | 0.00868 | 0.614 |
| GAN (E=10, mai vista) | 0.1449 | 0.01597 | 0.861 |

Questa tabella e' il punto di partenza descritto nel testo: CFM ha il W1 medio piu' basso e una
separability (0.541) molto vicina a quella della baseline ingenua (0.559) — differenza minima, non
sufficiente da sola a dimostrare vera interpolazione fisica invece di una semplice media pesata.
Da qui la necessita' del test decisivo sull'endpoint energetico.

### Il test decisivo: endpoint energetico

Un vero fascio a 10MV non puo' fisicamente contenere fotoni sopra circa 9.4-9.5 MeV. La baseline
ingenua importa letteralmente fotoni reali del 25MV (fino a 18.8 MeV) dentro il finto 10MV — 1.64% di
violazioni fisiche sopra soglia. CFM rispetta il vincolo quasi perfettamente (4 violazioni su 500k campioni).
Prova quantitativa che il modello ha imparato una vera dipendenza funzionale dall'energia, non una
miscela.

#### Tabella 1 — Endpoint energetico (vincolo fisico bremsstrahlung, cutoff 10.2 MeV)

Fonte: `logs/inter_report.out`.

| Dataset | E_max (MeV) | E al 99.9° percentile | E media | Eventi > 10.2 MeV | Δ E_max vs reale |
|---|---:|---:|---:|---:|---:|
| Reference reale (10MV) | 9.38 | 8.85 | 2.08 | 0 (0.000%) | — |
| **CFM interpolato** | **11.13** | **8.48** | **1.91** | **4 (0.001%)** | **1.75** |
| NSF interpolato | 16.95 | 14.43 | 1.47 | 1.812 (0.362%) | 7.57 |
| GAN interpolato | 5.51 | 5.48 | 1.97 | 0 (0.000%) | 3.87 |
| Baseline ingenua | 18.83 | 16.58 | 1.86 | 8.185 (1.637%) | 9.45 |

Nota sul GAN: "vince" sulle violazioni (0%) solo perche' genera uno spettro troppo stretto e
troncato (E_max = 5.51 MeV, ben sotto il vero endpoint di 9.38 MeV) — collasso di modalita', non un
successo, come sottolineato nel testo. CFM e' il modello che si avvicina di piu' al vero endpoint
pur restando fisicamente plausibile (0.001% di violazioni, contro l'1.637% della baseline ingenua e
lo 0.362% di NSF).

### Secondo test: correlazioni fisiche note (E-dz, x-dx)

Confermato lo stesso pattern su corr(E,dz): CFM piu' vicino al reale della baseline. Su corr(x,dx)
sostanziale pareggio, spiegabile perche' quella correlazione dipende poco dall'energia.

#### Tabella 2 — Correlazioni fisiche (struttura congiunta)

Fonte: `logs/inter_report.out`.

| Dataset | Corr(E, dz) | Δ vs reale | Corr(x, dx) | Δ vs reale |
|---|---:|---:|---:|---:|
| Reference reale | 0.2079 | — | 0.9057 | — |
| **CFM interpolato** | **0.1936** | **0.0143** | 0.9021 | 0.0035 |
| NSF interpolato | 0.1174 | 0.0904 | 0.8901 | 0.0155 |
| GAN interpolato | 0.1344 | 0.0735 | 0.9181 | 0.0124 |
| Baseline ingenua | 0.1524 | 0.0555 | 0.9082 | 0.0026 |

Su corr(E,dz) CFM e' nettamente il piu' vicino al reale (Δ=0.014, circa 4x meglio della baseline
ingenua e 6x meglio di NSF). Su corr(x,dx) la baseline ingenua e' marginalmente la piu' vicina
(Δ=0.0026 contro 0.0035 di CFM), coerente con la nota del testo: quella correlazione dipende poco
dall'energia, quindi non e' un test discriminante.

### Tabella finale a 3 modelli

| Test | Vincitore | Nota |
|---|---|---|
| Endpoint E_max | CFM | GAN "vince" sulle violazioni ma solo perche' genera uno spettro troppo stretto (collasso di modalita', non successo) |
| Corr(E,dz) | CFM | NSF il piu' debole |
| Corr(x,dx) | Baseline ingenua | Non discriminante (poco energia-dipendente) |

Lezione metodologica: una singola percentuale di violazioni puo' ingannare se non accompagnata da una
misura di quanto il modello si avvicini al vero valore limite — un modello troppo stretto sembra
"sicuro" ma sta sbagliando nella direzione opposta.

## 11. Conclusione — Parte 2

CFM interpola la fisica dell'energia in modo genuino e verificabile, non banalmente riconducibile a
una miscela lineare dei due estremi di training — dimostrato con due test fisici indipendenti
(endpoint energetico, correlazione E-dz). NSF e GAN restano piu' deboli, coerentemente con quanto
osservato nel capitolo precedente.

Limiti onesti da dichiarare:
- Solo due energie di training (6, 25 MV) — con due soli punti non si puo' escludere del tutto
  un'interpolazione sofisticata ma pur sempre "tra due ancore"
- Nessun dataset IAEA pubblico trovato per energie intermedie per una verifica quantitativa
  aggiuntiva
- GAN non ottimizzata oltre la baseline Sarrut
