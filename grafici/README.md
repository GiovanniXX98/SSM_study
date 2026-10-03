# Archivio Grafici e Telemetria — Progetto 21.0

Questa cartella raccoglie i grafici analitici ad alta risoluzione relativi all'addestramento e ai benchmark di **Progetto 21.0 (FAM-LLM 30M)**.

---

## 1. `training30M_su_openwebtext.png`
Grafico a 4 pannelli ad alta fedeltà che documenta la sessione completa di addestramento su **OpenWebText (87.891 Step di gradient descent)** del modello FAM-LLM a 30M di parametri:
1. **Cross-Entropy Loss:** Curva istantanea e media mobile (MA 150 step) che mostra la discesa progressiva della loss.
2. **Perplexity (Scala Logaritmica):** Andamento della perplexity durante tutte le epoche di addestramento.
3. **Learning Rate Schedule:** Profilo del Cosine Annealing con warmup lineare e decadimento controllato.
4. **Throughput (Tokens/s):** Velocità di elaborazione media della pipeline memmap.

- 🖼️ **Immagine:**

![Telemetria Addestramento OpenWebText](training30M_su_openwebtext.png)

---

## 2. `benchmark_scalabilita_memoria_e_velocita_30M.png`
Grafico comparativo a 2 pannelli che confronta le curve asintotiche tra **FSTLLM 2.0 (Olografico $\mathcal{O}(1)$)** e il **Transformer Standard Baseline (KV-Cache $\mathcal{O}(S)$)** per finestre di contesto da $S = 128$ a $S = 2048$ token:
1. **Impronta di Memoria Cache (MB):** Mostra la linea rigidamente piatta a **63.0 KB** di FSTLLM contro l'esplosione lineare del Transformer a **73.7 MB** a 2048 token (**abbattimento di 1170.3 volte**).
2. **Velocità di Decodifica (token/s):** Mostra il throughput stabile di FSTLLM (**27.3 tok/s**) contro il crollo a **0.4 tok/s** del Transformer (**68 volte più veloce** a contesto esteso).

- 🖼️ **Immagine:**

![Scalabilità Memoria e Velocità](benchmark_scalabilita_memoria_e_velocita_30M.png)

---

## 3. `benchmark_vera_memoria_processo_30M.png`
Grafico a 3 pannelli ad altissima risoluzione basato sul benchmark scientifico imparziale ed isolato (processi Linux separati, pulizia heap `glibc`, vera KV-Cache SDPA C++):
1. **Memoria Fisica Totale di Processo (OS RSS RAM):** 
   - A contesti brevi ($S = 128$), il **Transformer vince** consumando meno RAM iniziale (**367.7 MB vs 390.7 MB**).
   - A contesti lunghi ($S = 2048$), **FSTLLM 2.0 vince** rimanendo piatto a **408.1 MB** contro l'esplosione del Transformer a **532.1 MB** (risparmio netto di 124 MB di RAM di processo).
2. **Dimensione Cache di Inferenza (KB - Scala Log):**
   - Lo stato ricorrente di FSTLLM è rigorosamente costante a **67.5 KB** ($\mathcal{O}(1)$).
   - La KV-Cache del Transformer scala linearmente da **5.3 MB** a **74.4 MB** ($1.102\times$ più grande a 2048 token).
3. **Throughput di Decodifica Reale (Token/s):**
   - A contesti brevi ($S=128$), il **Transformer domina nettamente** (**13.7 tok/s vs 2.4 tok/s**) grazie ai kernel vettorializzati C++ di PyTorch SDPA.
   - A contesti lunghi ($S=2048$), **FSTLLM sorpassa il Transformer** (**4.0 tok/s vs 2.49 tok/s**), poiché la sua complessità di inferenza è invariante rispetto alla storia passata.

- 🖼️ **Immagine:**

![Benchmark Vera Memoria Processo](benchmark_vera_memoria_processo_30M.png)


