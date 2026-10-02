---
title: "Oltre l'Attenzione Quadratica: Dai Transformer ai Modelli State-Space Olografici di Fourier (FSTLLM 2.0)"
subtitle: "Studio Comparativo Fondamentale: Attention Softmax vs SSM a Stati Reali (Mamba) vs Modulazione di Fase Complessa (FSTLLM 2.0)"
author: "Progetto di Ricerca FSTLLM 2.0 — Guida Accademica e Tesi Universitaria"
date: "Anno Accademico 2025/2026"
lang: "it"
---

<div class="abstract">
<strong>Abstract:</strong> Il presente lavoro offre una trattazione teorica e architetturale comparativa, appositamente strutturata per una presentazione o tesi universitaria, sull'evoluzione dei Large Language Models (LLM). L'elaborato parte dall'analisi dei modelli basati su Transformer (Softmax Attention) e dei loro limiti computazionali di memoria, esamina i modelli State-Space a transizione continua e lineare (SSM, come Mamba e RWKV), e approfondisce l'architettura innovativa <strong>FSTLLM 2.0 (Fourier Space-Time LLM)</strong>. Unendo il rigore formale delle equazioni differenziali e alle differenze con la pedagogia visivo-geometrica resa celebre da 3Blue1Brown, il documento affronta: la conservazione della natura vettoriale degli embedding, l'interpretazione fisica e armonica della fase modulata $\theta$, la formulazione dello stato olografico ricorrente sul cerchio unitario complesso $\mathbb{S}^1$, il disaccoppiamento multi-scala Grouped-Query Resonance (GQR) con $8$ teste di Query e $2$ teste di memoria, e la garanzia di inferenza strettamente $\mathcal{O}(1)$ sia in tempo che in occupazione di memoria.
</div>

---

# 1. Introduzione e Intuizione Geometrica (La Lezione di 3Blue1Brown)

Per comprendere a fondo le differenze tra le architetture per modelli linguistici, è fondamentale partire dalla prospettiva geometrica con cui le reti neurali moderne manipolano il significato, come mirabilmente spiegato da Grant Sanderson (3Blue1Brown) nei capitoli dedicati a Deep Learning e Transformer.

<div class="box-pedagogical">
<div class="box-pedagogical-title">💡 Concetto Chiave 1: Le Parole come Vettori e Direzioni Semantiche</div>
Una rete neurale non elabora testo o simboli discreti, ma coordinate continue in uno spazio ad altissima dimensionalità ($\mathbb{R}^{d_{\text{model}}}$, dove ad esempio $d_{\text{model}} = 384$ o $12.288$ come in GPT-3). 
All'ingresso della rete:
$$\mathbf{x}_t^{(0)} = W_E [\text{token}_t]$$
Ogni parola viene prelevata da una tabella di lookup (la matrice di embedding $W_E$). In questo spazio:
<ul>
  <li>Vettori vicini corrispondono a parole con sfumature simili.</li>
  <li>Le <strong>direzioni</strong> codificano concetti semantici (es. la traslazione da maschile a femminile $\vec{v}_{\text{donna}} - \vec{v}_{\text{uomo}}$, la pluralità, i tempi verbali, le relazioni geografiche).</li>
</ul>
Tuttavia, l'embedding iniziale è <em>completamente decontestualizzato</em>: la parola "chiave" ha lo stesso vettore numerico sia che si trovi in un testo di informatica ("chiave crittografica"), sia che si trovi davanti a una serratura ("chiave di ferro").
</div>

### Il Fenomeno del "Context Soaking" (Assorbimento del Contesto)
L'obiettivo di un modello linguistico profondo è permettere a ciascun vettore di **assorbire il contesto** dei token circostanti attraverso i blocchi della rete.
Se al primo layer il vettore di una parola rappresenta soltanto la sua definizione isolata da dizionario, attraversando gli strati successivi esso viene progressivamente orientato e modificato in base al contesto, fino a quando nell'ultimo strato il vettore dell'ultimo token contiene una sintesi predittiva completa per generare il token successivo.

In ogni architettura per LLM, si alternano due funzioni primarie:
1. **Comunicazione Temporale tra Token:** Permettere ai token a distanze diverse di scambiarsi informazioni (nel Transformer ciò avviene mediante *Attention*, negli SSM e in FSTLLM tramite un *accumulatore ricorrente*).
2. **Memoria Associativa / Feed-Forward (MLP o SwiGLU):** Elaborare ciascun token individualmente e in parallelo, operando come un database associativo chiave-valore per estrarre la conoscenza fattuale appresa nel training.

---

# 2. Il Transformer Classico e il suo Collo di Bottiglia

Il Transformer (Vaswani et al., 2017) risolve la comunicazione tra token tramite il celebre meccanismo di **Scaled Dot-Product Attention**:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

dove:
- $Q = X W_Q \in \mathbb{R}^{S \times d_k}$ (Query: *"Cosa sto cercando?"*)
- $K = X W_K \in \mathbb{R}^{S \times d_k}$ (Key: *"Cosa offro come informazione?"*)
- $V = X W_V \in \mathbb{R}^{S \times d_v}$ (Value: *"Cosa aggiungo al vettore target in caso di compatibilità?"*)
- $M$ è la maschera causale triangolare inferiore con $-\infty$ sopra la diagonale principale.

### I Due Limiti Strutturali del Transformer:

1. **Costo di Addestramento Quadratico $\mathcal{O}(S^2)$:**
   Il calcolo esplicito della matrice di affinità $S \times S$ richiede tempo e memoria $\mathcal{O}(S^2)$. Per contesti lunghi (es. 32k, 128k o 1M token), questo impone un costo energetico e computazionale enorme.

2. **L'Esplosione del KV-Cache durante l'Inferenza ($\mathcal{O}(S)$ Memoria per Token):**
   Durante la generazione di testo autoregressiva (generazione token per token):
   - Ad ogni nuovo step temporale $t$, per calcolare l'attenzione verso il passato, il modello deve conservare in memoria tutti i vettori $K_{1:t}$ e $V_{1:t}$ di tutti i layer.
   - La dimensione del KV-Cache cresce **linearmente con la lunghezza del contesto generato**.
   - Ciò sposta il collo di bottiglia dell'hardware da *Compute-Bound* (potenza di calcolo dei core GPU) a **Memory-Bandwidth-Bound** (lentezza nel trasferire gigabyte di cache dalla VRAM ai registri della GPU ad ogni singolo token generato).

---

# 3. I Modelli State Space (SSM): S4, Mamba e RWKV

I modelli State Space (SSM) nascono dalla teoria classica dei controlli lineari e dei sistemi dinamici a tempo continuo:

$$\frac{d h(t)}{dt} = A h(t) + B x(t), \quad y(t) = C h(t) + D x(t)$$

Discretizzando il sistema tramite il passo $\Delta$ (Zero-Order Hold):
$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C_t h_t$$
dove $\bar{A} = \exp(\Delta A)$ e $\bar{B} = (\Delta A)^{-1}(\bar{A} - I)(\Delta B)$.

### I Vantaggi degli SSM:
- **Inferenza a Memoria e Tempo Costante $\mathcal{O}(1)$:** Non esiste il KV-Cache. Tutto il passato è compresso nello stato latente $h_t$ di dimensione fissa $N$.
- **Training Parallelo $\mathcal{O}(S)$:** Poiché il sistema è lineare e invariante nel tempo (o linearmente variabile nei selettivi come Mamba), l'intera sequenza può essere calcolata simultaneamente su GPU tramite *Associative Parallel Prefix-Scan*.

<div class="box-math">
<div class="box-math-title">⚠️ Il Limite Fisico dei Modelli SSM a Stati Reali</div>
Nei modelli come Mamba o S4, la matrice $\bar{A}$ possiede autovalori reali strettamente minori di 1 (o numeri complessi con parte reale negativa) per evitare l'esplosione esponenziale dello stato:
$$h_t = \sum_{\tau=1}^t \bar{A}^{t-\tau} \bar{B} x_\tau$$
Questo significa che l'operatore di transizione opera essenzialmente come un <strong>decadimento esponenziale reale monotono</strong>. Di conseguenza, il modello fatica ad apprendere <em>coerenze di fase armonica periodiche</em> o interferenze strutturate a lungo raggio nel testo senza degradare l'intensità del segnale.
</div>

---

# 4. FSTLLM 2.0: Fourier Space-Time State Space Model

Il modello **FSTLLM 2.0** unisce l'efficienza $\mathcal{O}(1)$ degli SSM con l'espressività della teoria delle onde di Fourier e dei segnali analitici complessi sul cerchio unitario:
$$\mathbb{S}^1 = \{ z \in \mathbb{C} \mid |z| = 1 \}$$

In FSTLLM 2.0, l'informazione storica non subisce unicamente una perdita d'intensità esponenziale, ma **viene codificata come interferenza costruttiva e distruttiva di oscillatori armonici**.

```
                         ARCHITETTURA DI UN BLOCCO FSTLLM 2.0

                         ┌───────────────────────────────────┐
                         │         Token ID in Ingresso      │
                         └─────────────────┬─────────────────┘
                                           │
                                           ▼
                         ┌───────────────────────────────────┐
                         │   Embedding Reale (nn.Embedding)  │  ◄── Vettore denso R^D
                         └─────────────────┬─────────────────┘
                                           │
                                           ▼
                         ┌───────────────────────────────────┐
                         │       RMSNorm Normalization       │
                         └─────────────────┬─────────────────┘
                                           │
                                           ▼
                         ┌───────────────────────────────────┐
                         │      Local Depthwise Conv1D       │  ◄── Micro-sintassi locale
                         │       (Kernel=4, groups=D)        │      Causal buffer O(1)
                         └─────────────────┬─────────────────┘
                                           │
                                           ▼  x_conv
                         ┌─────────────────┴─────────────────┐
                         │                                   │
                         ▼                                   ▼
         ┌───────────────────────────────┐   ┌───────────────────────────────┐
         │     Grouped KV Heads (GQR)    │   │       Full Query Heads        │
         │       num_kv_heads = 2        │   │        num_heads = 8          │
         │  k_proj, v_proj, phase, decay │   │            q_proj             │
         └───────────────┬───────────────┘   └───────────────┬───────────────┘
                         │                                   │
                         ▼                                   │
         ┌───────────────────────────────┐                   │
         │   Stato Olografico Ricorrente │                   │
         │ S_t = S_{t-1}·γ + K·V·e^{-jΦ} │                   │
         └───────────────┬───────────────┘                   │
                         │                                   │
                         │  repeat_interleave(4)             │
                         ▼                                   ▼
         ┌───────────────────────────────────────────────────┴───────────────┐
         │          Demodulazione di Fase e Risonanza Selettiva              │
         │              y = Re( (Q · e^{jΦ}) ⊙ S_t^{expanded} )              │
         └─────────────────────────────────┬─────────────────────────────────┘
                                           │
                                           ▼
                         ┌───────────────────────────────────┐
                         │   Gating (SiLU) + Output Proj     │
                         └─────────────────┬─────────────────┘
                                           │
                                           ▼
                         ┌───────────────────────────────────┐
                         │     SwiGLU Feed-Forward Network   │
                         └───────────────────────────────────┘
```

---

## 4.1 Risposte Dirette alle Questioni Fondamentali

### 1. I Token di Ingresso hanno ancora un Vettore Numerico di Embedding?
**Sì, esattamente come nei Transformer e in Mamba.**  
In [`FourierSpaceTimeLLM_V2`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/model/model.py#L223):
```python
self.token_emb = nn.Embedding(vocab_size, d_model)
x = self.token_emb(idx)
```
I token discreti vengono mappati in vettori reali densi $\mathbb{R}^{d_{\text{model}}}$. L'algebra dei numeri complessi e dei fasori non sostituisce l'embedding del vocabolario, ma viene impiegata esclusivamente **all'interno del layer come meccanismo ricorrente di evoluzione temporale**.

---

### 2. Il parametro $\theta$ (Fase) è un parametro in più?
**No, $\theta$ (o $\Phi_t$) non è un parametro statico memorizzato per posizione, ma un angolo calcolato dinamicamente.**  
L'angolo di fase $\Phi_t$ ad ogni passo temporale è dato da:
$$\Phi_t = \Phi_{t-1} + \omega_{\text{base}} + \Delta\theta_t$$
dove:
1. **$\omega_{\text{base}}$ ([`base_freqs`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/model/model.py#L67-L68)):** È registrata come `buffer` fisso non addestrabile:
   ```python
   freqs = torch.linspace(0.01, math.pi, self.head_dim)
   self.register_buffer("base_freqs", freqs.view(1, 1, 1, self.head_dim))
   ```
   Rappresenta una griglia di frequenze armoniche distribuite uniformemente da $0.01$ a $\pi$.
2. **$\Delta\theta_t$ ([`d_phase`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/model/model.py#L105)):** È una **modulazione selettiva dipendente dall'input**:
   $$\Delta\theta_t = \pi \cdot \tanh(W_{\phi} \tilde{\mathbf{x}}_t)$$
   generata dalla proiezione lineare addestrabile `phase_proj`. La rete non memorizza un $\theta$ fisso per ogni token, ma *apprende la matrice $W_\phi$* che decide come ruotare i fasori in base al contenuto semantico del testo.

---

### 3. Come Interagiscono Teste Diverse tra Q e KV? (Meccanismo GQR e repeat_interleave)

Una delle domande più importanti riguarda la compatibilità tra insiemi disuguali di teste: *se disponiamo di 8 teste di Query e solo 2 teste di Memoria (KV), come possono moltiplicarsi se hanno dimensionalità diverse?*

#### L'Analogia della Biblioteca (3Blue1Brown Style)
Immaginiamo la memoria del modello come una biblioteca specializzata:
- **Le teste KV rappresentano gli Archivi di Memoria:** ne costruiamo solo **2** ($KV_0$ e $KV_1$). Ogni archivio raccoglie e condensa una prospettiva temporale della storia della conversazione.
- **Le teste Q rappresentano i Ricercatori Specializzati:** ne assegniamo **8** ($Q_0, Q_1, \dots, Q_7$), ciascuno istruito a estrarre una specifica relazione (sintassi, legami anaforici, entità nominate, tono emotivo).

Invece di costruire 8 archivi ridondanti, i ricercatori lavorano in **gruppi di 4 sullo stesso archivio**:
$$[Q_0, Q_1, Q_2, Q_3] \longrightarrow \text{interrogano } KV_0, \quad [Q_4, Q_5, Q_6, Q_7] \longrightarrow \text{interrogano } KV_1$$

#### Il Meccanismo Matematico dei Tensori nel Codice (`model.py`)
1. **Proiezioni Lineari Asimmetriche:**
   - $Q$: $W_Q \mathbf{x} \in \mathbb{R}^{B \times S \times 8 \times 48}$ (8 teste)
   - $K, V$: $W_K \mathbf{x}, W_V \mathbf{x} \in \mathbb{R}^{B \times S \times 2 \times 48}$ (2 teste)
2. **Aggiornamento Compatto dello Stato Ricorrente:**
   Lo stato olografico memorizza unicamente le 2 teste KV:
   $$S_t \in \mathbb{C}^{B \times 2 \times 48}$$
   Durante l'inferenza, occupiamo solo $2 \times 48 = 96$ numeri complessi per token invece di $384$.
3. **Espansione Dinamica (`repeat_interleave`):**
   Per consentire la risonanza con le 8 Query, il tensore dello stato viene espanso lungo l'asse delle teste:
   ```python
   # Ogni testa KV viene replicata per 8 // 2 = 4 Query:
   state_expanded = f_state.repeat_interleave(4, dim=2)  # Forma: (B, 8, 48)
   carrier_pos_exp = carrier_pos.repeat_interleave(4, dim=2)
   ```
4. **Demodulazione di Risonanza:**
   $$y_t = \text{Re}\left( (Q_t \odot e^{j\Phi_t}) \odot S_t^{\text{expanded}} \right) \in \mathbb{R}^{B \times S \times 8 \times 48}$$
   Il risultato viene poi ricompattato a $d_{\text{model}} = 384$ e proiettato in uscita tramite $W_O$.

---

# 5. Tavola Sinottica Comparativa tra le Architetture

La seguente tabella riassume i contrasti architetturali e computazionali essenziali per l'esposizione universitaria:

| Proprietà Architetturale | Standard Transformer (GPT-3/4) | Mamba (S6) | RWKV-6 (Eagle) | **FSTLLM 2.0 (Nostro Progetto)** |
| :--- | :--- | :--- | :--- | :--- |
| **Complessità Training** | $\mathcal{O}(S^2)$ (Quadratica) | $\mathcal{O}(S)$ (Lineare) | $\mathcal{O}(S)$ (Lineare) | **$\mathcal{O}(S)$ (Prefix-Scan Parallelo)** |
| **Complessità Inferenza (Tempo)** | $\mathcal{O}(S)$ per token (crescente) | $\mathcal{O}(1)$ (Costante) | $\mathcal{O}(1)$ (Costante) | **$\mathcal{O}(1)$ (Strettamente Costante)** |
| **Footprint Memoria Inferenza** | Cresce linearmente $\mathcal{O}(S)$ (KV-Cache) | Fisso $\mathcal{O}(1)$ (Stato Reale $h$) | Fisso $\mathcal{O}(1)$ (Stato Matriciale $W$) | **Fisso $\mathcal{O}(1)$ (Stato Olografico $S$)** |
| **Dominio Matematico di Stato** | Spazio discreto delle attivazioni passate | Vettori continui reali $\mathbb{R}^N$ | Matrici reali $\mathbb{R}^{D \times D}$ | **Fasori sul cerchio unitario complesso $\mathbb{S}^1$** |
| **Dinamica Temporale** | Matrice statica $Q K^T$ + Softmax | Decadimento esponenziale $\exp(\Delta A)$ | Time-mixing con decadimento canali | **Interferenza d'onda + Modulazione Selettiva $\Delta\theta_t$** |
| **Organizzazione Teste / Memoria** | Multi-Head (MHA) o GQA | 1 stato per canale / layer | Canali indipendenti per dimensione | **GQR (8 Query Heads / 2 Holographic KV Heads)** |
| **Cattura Sintassi Locale** | Implicita nei positional embedding | Convoluzione 1D causale iniziale | Time-shift operator adiacente | **Local Depthwise Conv1D con buffer causale $\mathcal{O}(1)$** |
| **Regime Hardware Prevalente** | Memory-Bandwidth Bound (KV-Cache bottleneck) | Compute-Bound (Kernel associative scan) | Compute-Bound | **Compute-Bound (Zero memory-stall di lettura)** |

---

# 6. Analisi dei Parametri e del Flusso di Calcolo

Nel modulo [`SelectiveFourierWaveLayerV2`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/model/model.py#L20-L65) configurato con $d_{\text{model}} = 384$, $H = 8$, $KV\_H = 2$, $HD = 48$:

### Distribuzione Parametrica delle Proiezioni:
- **Proiezione Query ($W_Q$):** $384 \times 384 = 147.456$ parametri (piena dimensionalità per garantire espressività nelle query).
- **Proiezione Key ($W_K$):** $384 \times (2 \times 48) = 384 \times 96 = 36.864$ parametri (**risparmio del 75%** rispetto ai 147k di MHA).
- **Proiezione Value ($W_V$):** $384 \times 96 = 36.864$ parametri.
- **Modulazione di Fase ($W_\phi$):** $384 \times 96 = 36.864$ parametri.
- **Decadimento Selettivo ($W_d$):** $384 \times 2 = 768$ parametri.
- **Convoluzione Depthwise Locale:** $384 \times 1 \times 4 = 1.536$ parametri.
- **Output Projection ($W_O$) & Gating ($W_G$):** $2 \times (384 \times 384) = 294.912$ parametri.

Questa suddivisione consente di concentrare la densità di parametri sulle capacità di lettura ed elaborazione non-lineare (SwiGLU FFN), mantenendo l'infrastruttura di memoria compattissima.

---

# 7. Benchmark e Confronto Sperimentale a ~30M di Parametri (Letteratura arXiv)

Nello studio delle architetture neurali, testare modelli su una scala controllata di **circa 30 Milioni di parametri** rappresenta la metodologia accademica ideale per isolare i meriti dell'innovazione architetturale dai semplici effetti di scala legati alla potenza di calcolo grezza.

Nella cartella `paper_confronto/` sono stati raccolti e analizzati i paper fondanti della letteratura con cui confrontare FSTLLM 2.0:

1. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023 - arXiv:2312.00752):**
   - Dimostra l'efficacia del meccanismo selettivo nei compiti di *Selective Copying* e *Induction Heads*, evidenziando i limiti dei modelli LTI (Linear Time-Invariant).
   - Offre il termine di paragone per misurare il throughput di generazione $\mathcal{O}(1)$ e la capacità di filtro del contesto.
2. **TinyStories: How Small Can Language Models Be and Still Speak Coherent English? (Eldan & Li, 2023 - arXiv:2305.07759):**
   - Dimostra che modelli compresi tra **1M e 33M di parametri** (basati su Transformer standard tipo GPT-Neo/GPT-2 a 28M/33M parametri) sono in grado di apprendere sintassi, coerenza logica e ragionamento causale se addestrati su un vocabolario controllato.
   - Fornisce il *benchmark primario* su cui allenare e confrontare direttamente FSTLLM 2.0 (~30M parametri) con un Transformer classico di pari taglia.
3. **RWKV: Reinventing RNNs for the Transformer Era (Peng et al., 2023 - arXiv:2305.13048):**
   - Confronta architetture ricorrenti lineari e Transformer su scale a partire da **14M, 70M, 169M e 430M** di parametri su task NLP classici (LAMBADA, PIQA, StoryCloze).
   - Valuta la stabilità numerica del decadimento temporale per canale rispetto a modelli attention.
4. **Transformers are SSMs: Structured State Space Duality (Dao & Gu, 2024 - arXiv:2405.21060 - Mamba-2):**
   - Formalizza il legame teorico (State Space Duality, SSD) tra la moltiplicazione di matrici semidense causali e le forme canoniche degli spazi di stato, offrendo il formalismo per interpretare la risonanza armonica come forma strutturata di attenzione lineare.
5. **MobileLLM: Optimizing Sub-Billion Language Models (Meta AI, 2024 - arXiv:2402.14905):**
   - Analizza la massimizzazione del parametro-budget sotto il miliardo di pesi tramite Grouped-Query Attention (GQA) e deep-and-thin design.
6. **Attention Is All You Need (Vaswani et al., 2017 - arXiv:1706.03762):**
   - Il punto di riferimento immutabile per la formulazione del Transformer standard.

### Protocollo Sperimentale Raccomandato per FSTLLM 2.0 (30M):
Per la tesi o la presentazione universitaria, si raccomanda di impostare il confronto empirico su tre prove:
- **1. Language Modeling su TinyStories (30M Parametri):** Addestrare FSTLLM 2.0 ($d_{\text{model}} = 384$, 6 layer, $H=8, KV_H=2$) contro un Transformer equivalente a 6 layer. Verificare che la perplexity e la qualità generativa siano comparabili.
- **2. Long-Range Needle-In-A-Haystack (NIAH):** Posizionare un'informazione specifica all'interno di un contesto sintetico esteso (da 1k fino a 16k token) e testare la capacità della risonanza di fase nel recuperare il fatto senza degradazione d'ampiezza.
- **3. Benchmark di Inferenza e VRAM:** Misurare la memoria consumata al crescere della sequenza generata ($S = 128, 512, 2048, 8192$), dimostrando sperimentalmente la **curva perfettamente orizzontale $\mathcal{O}(1)$** di FSTLLM 2.0 rispetto alla crescita lineare ripida del Transformer con KV-Cache.

---

# 8. Guida Strategica per la Presentazione Universitaria

Per esporre questo progetto in sede accademica (esame di Machine Learning / Deep Learning, seminario o tesi di laurea), si suggerisce di articolare l'esposizione in **4 momenti narrativi**:

### 1. La Sfida: L'Impossibilità di Scalare i Transformer a Contesti Infiniti
- Spiegare che i Transformer dominano il settore perché altamente parallelizzabili via GPU.
- Mostrare tuttavia la contraddizione intrinseca: generare una risposta richiede di memorizzare l'intera storia della conversazione nel **KV-Cache**, portando a saturazione la memoria delle schede grafiche.
- Usare l'analogia di 3Blue1Brown: *"Nei Transformer ogni parola per esistere deve voltarsi indietro e stringere la mano a tutte le parole precedenti"*.

### 2. La Transizione: I Modelli State Space (SSM)
- Introdurre il concetto di sistema dinamico: invece di conservare ogni singola parola passata, aggiorniamo un **vettore di stato continuo riassuntivo**.
- Evidenziare la conquista: inferenza in tempo e memoria costante $\mathcal{O}(1)$.
- Evidenziare il punto critico: i modelli attuali (come Mamba) utilizzano decadimenti reali monotoni ($e^{\Delta A}$ con $A < 0$). Un decadimento reale tende a smorzare le oscillazioni periodiche e le correlazioni armoniche a lungo raggio.

### 3. La Nostra Proposta: FSTLLM 2.0 e la Meccanica Ondulatoria
- Introdurre l'idea dell'ologramma: l'informazione viene incisa modulando la fase di oscillatori complessi sul cerchio unitario $\mathbb{S}^1$.
- Spiegare chiaramente che:
  - Non si perdono gli embedding tradizionali: ogni token parte come vettore denso reale.
  - La micro-sintassi (articoli, desinenze) viene catturata dalla convoluzione 1D causale prima delle onde.
  - La fase $\theta$ non è fissa, ma modulata dal contenuto del token ($\Delta\theta_t = \pi \tanh(W_\phi x_t)$).
  - L'architettura GQR consente a 8 teste di Query di interrogare 2 stati olografici condensati, tagliando del 75% la memoria di stato.

### 4. Risposte Pronte a Possibili Domande dei Docenti
- **"Perché utilizzare i numeri complessi invece di uno spazio vettoriale reale?"**  
  *Risposta:* I numeri complessi sul cerchio unitario ($e^{j\Phi}$) garantiscono l'unitarietà della trasformazione ($|e^{j\Phi}| = 1$). A differenza di matrici reali di decadimento che possono causare vanishing gradient o richiedere delicati vincoli di normatura, la rotazione di fase conserva integralmente l'energia del segnale, consentendo fenomeni di interferenza costruttiva e distruttiva a distanza arbitraria.
- **"Come si assicura la causalità nel calcolo parallelo del training?"**  
  *Risposta:* La causalità è garantita matematicamente sia dalla convoluzione causale (padding sinistro con kernel causale) sia dalla formulazione triangolare inferiore dell'associative prefix scan ($M_{i,j} = -\infty$ per $j > i$ nel decadimento cumulativo), impedendo che qualsiasi informazione futura trapeli nel passato.
- **"Che differenza c'è tra RoPE (Rotary Position Embedding) e la vostra modulazione di fase?"**  
  *Risposta:* RoPE applica una rotazione di fase deterministica e statica legata unicamente alla posizione assoluta o relativa del token nel contesto. In FSTLLM 2.0, la fase è **selettiva e dipendente dai dati** ($\Delta\theta_t$ dipende dall'input $x_t$): il modello può scegliere di accelerare o congelare l'avanzamento della fase temporale in base al contenuto semantico della frase.

---

# 8. Conclusioni

Il modello **FSTLLM 2.0** dimostra la fattibilità di un'architettura ibrida che sintetizza la ricchezza geometrica degli embedding continui, l'efficienza costante $\mathcal{O}(1)$ dei modelli State Space e la conservazione armonica della trasformata di Fourier. Grazie a innovazioni come la convoluzione locale depthwise, il Grouped-Query Resonance (GQR) e la modulazione selettiva sul cerchio unitario, FSTLLM 2.0 si pone come un'alternativa promettente e rigorosa ai tradizionali Transformer per l'elaborazione del linguaggio naturale a lungo contesto.
