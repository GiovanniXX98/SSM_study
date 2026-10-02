#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.2cm, right: 2.2cm),
  header: locate(loc => {
    if loc.page() > 1 [
      #grid(
        columns: (1fr, 1fr),
        align: (left, right),
        text(size: 8pt, fill: rgb("#64748b"), font: "DejaVu Serif", style: "italic")[Fourier Space-Time State Space Model (FSTLLM 2.0)],
        text(size: 8pt, fill: rgb("#64748b"), font: "DejaVu Serif")[Studio Comparativo: Transformer vs SSM vs FSTLLM]
      )
      #line(length: 100%, stroke: 0.5pt + rgb("#cbd5e1"))
    ]
  }),
  footer: locate(loc => [
    #line(length: 100%, stroke: 0.5pt + rgb("#cbd5e1"))
    #v(2pt)
    #align(center)[
      #text(size: 8.5pt, fill: rgb("#64748b"), font: "DejaVu Serif")[Pagina #loc.page() di #counter(page).final(loc).at(0)]
    ]
  ])
)

#set text(
  font: ("DejaVu Serif", "Linux Libertine"),
  size: 10pt,
  lang: "it"
)
#set par(justify: true, leading: 0.68em)

// Styling funzioni
#let callout(title: "", body, color: rgb("#3b82f6"), bg: rgb("#eff6ff")) = {
  block(
    fill: bg,
    inset: 11pt,
    radius: 4pt,
    stroke: (left: 4pt + color, rest: 0.5pt + color.lighten(60%)),
    width: 100%,
    breakable: false,
    [
      #text(weight: "bold", fill: color.darken(20%), size: 10pt)[#title]
      #v(4pt)
      #body
    ]
  )
}

#let mathbox(title: "", body) = {
  callout(title: title, body, color: rgb("#ca8a04"), bg: rgb("#fefce8"))
}

// -------------------------------------------------------------
// TITOLO & FRONTESPIZIO
// -------------------------------------------------------------
#align(center)[
  #v(0.5cm)
  #text(size: 20pt, weight: "bold", fill: rgb("#0f172a"))[Oltre l'Attenzione Quadratica: Dai Transformer ai Modelli State-Space Olografici di Fourier]
  #v(0.3cm)
  #text(size: 12pt, fill: rgb("#334155"), weight: "medium")[Analisi Teorica e Architetturale Comparativa: Attention Softmax vs SSM a Stati Reali (Mamba) vs Modulazione di Fase Complessa (FSTLLM 2.0)]
  #v(0.4cm)
  #text(size: 10.5pt, fill: rgb("#475569"))[
    *Progetto di Ricerca FSTLLM 2.0* — Documento Guida per Esposizione Accademica e Tesi Universitaria \
    Anno Accademico 2025/2026
  ]
  #v(0.6cm)
]

#block(
  fill: rgb("#f8fafc"),
  inset: 12pt,
  radius: 4pt,
  stroke: (left: 4pt + rgb("#1e293b")),
  width: 100%,
  [
    #text(weight: "bold", size: 10pt, fill: rgb("#0f172a"))[Sommario Esecutivo (Abstract)] \
    #v(3pt)
    #text(size: 9.5pt, style: "italic", fill: rgb("#334155"))[
      Questo documento fornisce una disamina teorico-architetturale completa, appositamente strutturata per una presentazione o discussione d'esame universitario, sulle architetture avanzate per l'elaborazione del linguaggio naturale. A partire dalle intuizioni geometriche rese celebri da 3Blue1Brown, viene formalizzato il meccanismo di Softmax Attention nei Transformer e ne vengono dimostrati i limiti computazionali: la complessità quadratica $cal(O)(S^2)$ in fase di addestramento e l'esplosione lineare del KV-Cache durante l'inferenza autoregressiva ($cal(O)(S)$ di footprint di memoria). Vengono poi analizzati i modelli State-Space continui e discreti (S4, Mamba, RWKV) e i limiti fisici derivanti dai decadimenti monotoni ad autovalori reali. Infine, viene dettagliata l'architettura proposta *FSTLLM 2.0 (Fourier Space-Time LLM)*: la persistenza degli embedding densi in $bb(R)^D$, l'introduzione della convoluzione locale depthwise causale per la micro-sintassi, la modulazione selettiva della fase angolare sul cerchio unitario $bb(S)^1$, la memoria olografica complessa ad inferenza strettamente $cal(O)(1)$, e il paradigma Grouped-Query Resonance (GQR) con rapporto 4:1 tra Query Heads e Memory Heads.
    ]
  ]
)

#v(0.5cm)

// -------------------------------------------------------------
// SEZIONE 1
// -------------------------------------------------------------
= 1. Fondamenti Geometrici e la Lezione di 3Blue1Brown

Per spiegare con efficacia le architetture neurali in sede accademica, è essenziale iniziare dalla prospettiva fondante del Deep Learning applicato al linguaggio: *la geometria degli spazi vettoriali ad alta dimensionalità*, così come illustrata da Grant Sanderson (3Blue1Brown) nei capitoli 5, 6 e 7 della serie _Neural Networks_.

#callout(title: "💡 L'Intuizione di 3Blue1Brown: Le Parole come Vettori e Direzioni Semantiche")[
  Una rete neurale non opera direttamente su stringhe o simboli grafici discreti. Il primissimo passo consiste nel mappare ogni token di indice $t$ in un vettore numerico reale continuo ad altissima dimensionalità ($d_"model"$, tipicamente tra $384$ e $12.288$):
  $ bold(x)_t^((0)) = W_E [ "token"_t ] in bb(R)^(d_"model") $
  dove $W_E in bb(R)^(V times d_"model")$ è la matrice di _embedding_. In questo spazio:
  - *Vettori vicini* rappresentano parole con affinità semantiche o contestuali simili.
  - *Le direzioni* (differenze vettoriali) incarnano specifici concetti (ad es. il vettore $arrow(v)_"donna" - arrow(v)_"uomo"$ è quasi parallelo a $arrow(v)_"regina" - arrow(v)_"re"$, così come esistono direzioni dedicate alla pluralità, alla desinenza verbale o ai ruoli grammaticali).
  
  Tuttavia, l'embedding iniziale è un vettore *statico e decontestualizzato*: la parola _"banca"_ assume lo stesso identico vettore sia che si parli di finanza sia che si parli dell'argine di un fiume.
]

=== Il Principio del "Context Soaking" (Assorbimento del Contesto)
Come spiegato nel Capitolo 5 di 3Blue1Brown, il compito dell'intera rete non è altro che permettere a ciascun vettore di *assorbire il contesto circostante* man mano che attraversa i diversi strati (layers).
All'ingresso della rete il vettore esprime solo il significato isolato da vocabolario; all'uscita dall'ultimo blocco, lo stesso vettore è stato deformato, ruotato e arricchito per codificare l'intero scenario narrativo, grammaticale e logico indispensabile a predire il token successivo.

Indipendentemente dalla specifica architettura adottata, un blocco per Large Language Model moderno deve sempre assolvere a due ruoli distinti:
1. *Scambio di Informazione Temporale:* Permettere ai token distanti nella frase di dialogare e influenzarsi a vicenda.
2. *Memoria Associativa Non-Lineare:* Un modulo feed-forward (MLP o SwiGLU) che, applicato a ciascun token singolarmente e in parallelo, agisce come una memoria associativa chiave-valore per recuperare informazioni fattuali e pattern astratti appresi durante il training.

#v(0.3cm)

// -------------------------------------------------------------
// SEZIONE 2
// -------------------------------------------------------------
= 2. Il Transformer Classico e il Collo di Bottiglia dell'Attenzione

Il Transformer standard (Vaswani et al., 2017) implementa la comunicazione tra token per mezzo dell'operatore di *Scaled Dot-Product Attention*:

$ "Attention"(Q, K, V) = "Softmax"( (Q K^T) / sqrt(d_k) + M ) V $

dove $Q = X W_Q$, $K = X W_K$, $V = X W_V$ e $M$ rappresenta la matrice di mascheramento causale triangolare inferiore:
$ M_(i,j) = cases(0 & "se" j <= i, -infinity & "se" j > i) $

#align(center)[
  #block(
    fill: rgb("#f1f5f9"),
    inset: 8pt,
    radius: 4pt,
    stroke: 0.5pt + rgb("#cbd5e1"),
    ```
Matrice di Attenzione Causale Softmax (S x S):
          Token 1   Token 2   Token 3  ...  Token S
Token 1  [  1.0       0         0            0   ]
Token 2  [  0.3      0.7        0            0   ]
Token 3  [  0.1      0.4       0.5           0   ]
  ...    [  ...      ...       ...          ...  ]
Token S  [ a_{S,1}  a_{S,2}   a_{S,3}  ...  a_{S,S}]
                               ▲
                 Costo Computazionale: O(S^2)
    ```
  )
]

=== La Metafora delle Query, Key e Value
Nella formulazione pedagogica di 3Blue1Brown:
- *Query ($Q$):* Rappresenta la domanda che il token corrente pone all'ambiente circostante (_"Sono un sostantivo singolare, cerco aggettivi o articoli che mi precedono"_).
- *Key ($K$):* Rappresenta l'etichetta descrittiva che ciascun token espone (_"Sono l'aggettivo 'blu' posizionato a $t-2$”_).
- *Dot-Product ($Q K^T$):* Il prodotto scalare quantifica geometricamente la congruenza tra la domanda del primo token e l'offerta del secondo.
- *Softmax:* Normalizza la riga affinché la somma dei coefficienti valga $1$, trasformando le affinità in percentuali di attenzione.
- *Value ($V$):* Contiene il reale pacchetto informativo da sommare al vettore target per arricchirne il significato.

=== I Due Limiti Strutturali Insuperabili del Transformer

1. *Costo di Addestramento Quadratico $cal(O)(S^2)$:*
   La costruzione della matrice di attenzione $S times S$ richiede un numero di operazioni e un'allocazione di memoria che scalano con il quadrato della sequenza. Addestrare modelli con contesti di 64k o 1M token comporta una spesa computazionale gigantesca.

2. *Il Muro del KV-Cache durante l'Inferenza ($cal(O)(S)$ Memoria per Token):*
   Nella generazione autoregressiva (generazione un token alla volta):
   - Ad ogni nuovo step temporale $t$, per consentire alla Query del nuovo token di confrontarsi con tutte le parole del passato, il sistema deve conservare in VRAM l'intera storia dei vettori $K_(1:t)$ e $V_(1:t)$ per ogni livello della rete.
   - La dimensione di questo buffer (il *KV-Cache*) cresce linearmente con la lunghezza della conversazione ($cal(O)(S)$).
   - In produzione, il fattore limitante non è la potenza di calcolo (TFLOPS della GPU), bensì la *banda di memoria (Memory-Bandwidth Bound)*: per generare anche un solo singolo carattere, la GPU deve ricaricare decine di gigabyte di cache dalla VRAM ai registri.

#v(0.3cm)

// -------------------------------------------------------------
// SEZIONE 3
// -------------------------------------------------------------
= 3. Modelli State Space Lineari (SSM): S4, Mamba e RWKV

Per abbattere la barriera quadratica $cal(O)(S^2)$, la letteratura recente ha riscoperto la teoria classica dei sistemi dinamici a tempo continuo, introducendo i modelli *State Space (SSM)*:

$ frac(d h(t), d t) = A h(t) + B x(t), quad y(t) = C h(t) + D x(t) $

Discretizzando il sistema continuo con un passo temporale $Delta$ (mediante trasformazione Zero-Order Hold):
$ macron(A) = exp(Delta A), quad macron(B) = (Delta A)^(-1) (macron(A) - I) (Delta B) $
ottenendo le equazioni ricorrenti discrete:
$ h_t = macron(A) h_(t-1) + macron(B) x_t, quad y_t = C_t h_t $

=== Vantaggi Rivoluzionari degli SSM
- *Inferenza a Memoria e Tempo Rigorosamente Costante $cal(O)(1)$:* Non esiste alcun KV-Cache. La storia passata viene compressa all'interno del vettore di stato latente $h_t in bb(R)^N$. Generare il token successivo costa sempre lo stesso tempo e occupa sempre gli stessi byte, sia al token 10 che al token 100.000.
- *Addestramento Parallelo $cal(O)(S)$:* Grazie alla linearità delle equazioni differenziali, la sequenza temporale può essere calcolata simultaneamente su GPU mediante l'algoritmo di _Associative Parallel Prefix Scan_ in tempo $cal(O)(S)$.

#mathbox(title: "⚠️ Il Limite Fisico degli SSM a Stati Reali")[
  Nei modelli a stati reali (quali Mamba/S6 o S4), gli autovalori della matrice di transizione $macron(A)$ devono essere strettamente minori di $1$ per garantire la stabilità asintotica del sistema dinamico:
  $ h_t = sum_(tau=1)^t macron(A)^(t - tau) macron(B) x_tau $
  L'operatore di transizione $macron(A)$ agisce quindi primariamente come un *filtro a decadimento esponenziale reale*.
  Questo comportamento smorza monotonamente il segnale nel passato: risulta estremamente complesso per il modello apprendere dinamiche di _coerenza di fase armonica periodica_ o mantenere correlazioni sintattiche a lungo raggio che richiederebbero interferenze costruttive e distruttive stabili nel tempo.
]

#v(0.3cm)

// -------------------------------------------------------------
// SEZIONE 4
// -------------------------------------------------------------
= 4. FSTLLM 2.0: Fourier Space-Time State Space Model

L'architettura *FSTLLM 2.0 (Fourier Space-Time LLM)* nasce specificamente per superare i limiti dei decadimenti reali monotoni, introducendo la *meccanica ondulatoria complessa* all'interno dello spazio di stato.
I vettori di memoria non decadono unicamente per smorzamento, ma vengono incisi e modulati come *fasori complessi sul cerchio unitario*:
$ bb(S)^1 = { z in bb(C) mid(|) |z| = 1 } $

#align(center)[
  #block(
    fill: rgb("#f8fafc"),
    inset: 7pt,
    radius: 4pt,
    stroke: 0.5pt + rgb("#94a3b8"),
    breakable: false,
    text(size: 7.5pt)[
    ```
                     PIPELINE ARCHITETTURALE FSTLLM 2.0
                     
                     ┌───────────────────────────────────┐
                     │         Token ID in Ingresso      │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │   Embedding Reale (nn.Embedding)  │ ◄── Vettore denso R^D
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │      Local Depthwise Conv1D       │ ◄── Micro-sintassi locale
                     │       (Kernel=4, groups=D)        │     Buffer causale O(1)
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
    ]
  )
]

=== 4.1 I Token di Ingresso mantengono il Vettore di Embedding Reale?
*Sì, in modo categorico.* All'ingresso del modello, il token discreto non viene convertito in entità esoteriche:
$ bold(x) = "Embedding"("token"_t) in bb(R)^(d_"model") $
Nel codice di `model.py` (`FourierSpaceTimeLLM_V2`):
```python
self.token_emb = nn.Embedding(vocab_size, d_model)
x = self.token_emb(idx)
```
I token vivono nello spazio vettoriale reale classico $bb(R)^D$. I numeri complessi e i fasori non sostituiscono la rappresentazione semantica del vocabolario, ma entrano in gioco *esclusivamente all'interno del layer ricorrente come operatore di transizione dinamica temporale*.

=== 4.2 La Convoluzione Locale Depthwise 1D (Kernel 4)
I modelli spettrali globali puri rischiano di non distinguere accuratamente relazioni di prossimità immediata (ad esempio la correlazione stretta tra un articolo e il sostantivo subito adiacente).
In FSTLLM 2.0, prima delle proiezioni, l'input viene filtrato da una convoluzione 1D causale depthwise con kernel $K=4$:
$ bold(x)_"conv" = "SiLU" ( "Conv1D"(bold(x)) ) $
- In addestramento viene calcolata causalmente in parallelo su tutta la sequenza.
- In inferenza step-by-step $cal(O)(1)$, il modello mantiene un micro-buffer di soli $K-1 = 3$ vettori, garantendo un costo computazionale costante indipendente dalla lunghezza del testo.

=== 4.3 Il Parametro $theta$ (Fase) è un Parametro Fisso in Più?
*No, $theta$ (o $Phi_t$) non è un peso statico memorizzato per posizione, ma un angolo cumulativo modulato dinamicamente.*
La fase totale al passo $t$ è data dall'integrazione:
$ Phi_t = Phi_(t-1) + omega_"base" + Delta theta_t $
dove:
1. *$omega_"base"$ (Buffer Statico non Apprendibile):* 
   È una griglia di frequenze armoniche precalcolata e fissata da $0.01$ a $pi$ lungo le dimensioni interne della testa (`head_dim`):
   ```python
   freqs = torch.linspace(0.01, math.pi, self.head_dim)
   self.register_buffer("base_freqs", freqs.view(1, 1, 1, self.head_dim))
   ```
2. *$Delta theta_t$ (Modulazione di Fase Selettiva):* 
   È calcolata all'istante dall'input corrente tramite la proiezione lineare apprendibile `phase_proj`:
   $ Delta theta_t = pi tanh(W_phi bold(x)_"conv") $
   La rete impara la matrice $W_phi$, che stabilisce *come accelerare o rallentare la rotazione di fase* in funzione del significato semantico del token (meccanismo analogo alla modulazione di $Delta$ in Mamba).

=== 4.4 Come interagiscono Teste Diverse tra Q e KV? (Meccanismo GQR e repeat_interleave)

Una delle domande più frequenti in sede di discussione riguarda l'interazione tra insiemi disuguali di teste: *se disponiamo di 8 teste di Query e solo 2 teste di Memoria (KV), come possono moltiplicarsi se hanno dimensionalità diverse?*

#callout(title: "💡 L'Analogia della Biblioteca: 2 Archivi di Memoria e 8 Ricercatori")[
  Immaginiamo la memoria del modello come una biblioteca specializzata:
  - *Le teste KV rappresentano gli Archivi di Memoria:* ne costruiamo solo *2* ($"KV"_0$ e $"KV"_1$), riducendo l'impronta di memoria di un fattore $4$. Ciascun archivio condensa una prospettiva temporale della storia della conversazione.
  - *Le teste Q rappresentano i Ricercatori Specializzati:* ne assegniamo *8* ($Q_0, Q_1, dots, Q_7$), ciascuno istruito a estrarre una specifica relazione (sintassi, legami anaforici, entità nominate, tono emotivo).
  
  Invece di costruire 8 mastodontici archivi ridondanti, i ricercatori lavorano in *gruppi di 4 sullo stesso archivio*:
  $ [Q_0, Q_1, Q_2, Q_3] arrow.long "interrogano" "KV"_0, quad [Q_4, Q_5, Q_6, Q_7] arrow.long "interrogano" "KV"_1 $
]

==== Il Meccanismo Matematico dei Tensori nel Codice (`model.py`)
Nel forward di `SelectiveFourierWaveLayerV2`, il mismatch dimensionale viene risolto elegantemente tramite la duplicazione causale broadcasted (*repeat_interleave*):

1. *Proiezioni Lineari Asimmetriche:*
   $ q = W_Q bold(x)_"conv" in bb(R)^(B times S times 8 times 48) $
   $ k = W_K bold(x)_"conv" in bb(R)^(B times S times 2 times 48), quad v = W_V bold(x)_"conv" in bb(R)^(B times S times 2 times 48) $
2. *Aggiornamento Compatto dello Stato Ricorrente:*
   Lo stato olografico memorizza unicamente le 2 teste KV:
   $ S_t in bb(C)^(B times 2 times 48) $
   Durante l'inferenza, occupiamo solo $2 times 48 = 96$ numeri complessi per token invece di $384$.
3. *Espansione Dinamica (`repeat_interleave`):*
   Per consentire la risonanza con le 8 Query, il tensore dello stato viene espanso lungo la dimensione delle teste:
   ```python
   # Ogni testa KV viene replicata per 8 // 2 = 4 Query:
   state_expanded = f_state.repeat_interleave(4, dim=2)  # Forma: (B, 8, 48)
   carrier_pos_exp = carrier_pos.repeat_interleave(4, dim=2)
   ```
4. *Demodulazione di Risonanza Elemento-per-Elemento:*
   $ y_t = "Re"( (Q_t dot.circle e^(j Phi_t)) dot.circle S_t^"expanded" ) in bb(R)^(B times S times 8 times 48) $
   Il risultato viene poi compattato a $d_"model" = 384$ e proiettato in uscita tramite $W_O$.

#callout(title: "Perché questa asimmetria è lo Standard Moderno (LLaMA 3, Mistral, FSTLLM 2.0)?")[
  La ricerca empirica su larga scala (Ainslie et al., GQA) ha dimostrato che la memoria del testo passata possiede una ridondanza intrinseca: mantenere decine di matrici di chiave e valore duplicate spreca banda di memoria senza aumentare l'accuratezza. Al contrario, mantenere una *molteplicità di Query* permette al modello di estrarre rappresentazioni eterogenee dallo stesso compatto substrato di memoria.
]

=== 4.5 Aggiornamento dello Stato Olografico Ricorrente
L'equazione che regola l'evoluzione temporale della memoria è:
$ S_t = S_(t-1) dot.circle gamma_t + (1 - gamma_t) ( K_t dot.circle V_t dot.circle e^(-j Phi_t) ) $
dove:
- $gamma_t = sigma(W_d bold(x)_"conv")$ è il coefficiente di decadimento selettivo.
- $e^(-j Phi_t)$ è l'onda portante coniugata che incide spettralmente l'informazione $K_t dot.circle V_t$.
- In inferenza l'operazione costa un singolo prodotto elemento-per-elemento in tempo $cal(O)(1)$. In addestramento si calcola in parallelo tramite una matrice triangolare inferiore associativa.

#v(0.3cm)

// -------------------------------------------------------------
// SEZIONE 5
// -------------------------------------------------------------
= 5. Tavola Sinottica Comparativa tra i Modelli

La seguente tabella offre la sintesi comparativa ideale da esporre alla commissione o proiettare durante l'esame:

#align(center)[
#table(
  columns: (1.8fr, 2.2fr, 2.2fr, 2.2fr, 2.4fr),
  fill: (col, row) => if row == 0 { rgb("#0f172a") } else if calc.even(row) { rgb("#f8fafc") } else { none },
  stroke: (x, y) => if y == 0 { (bottom: 1.5pt + rgb("#0f172a")) } else { 0.5pt + rgb("#e2e8f0") },
  inset: 6pt,
  align: (col, row) => if row == 0 { center + horizon } else { left + horizon },
  
  // Header
  text(weight: "bold", fill: white, size: 8pt)[Proprietà],
  text(weight: "bold", fill: white, size: 8pt)[Transformer (GPT-3/4)],
  text(weight: "bold", fill: white, size: 8pt)[Mamba (S6)],
  text(weight: "bold", fill: white, size: 8pt)[RWKV-6 (Eagle)],
  text(weight: "bold", fill: white, size: 8pt)[FSTLLM 2.0 (Nostro)],

  // Rows
  [Complessità Training], [$cal(O)(S^2)$ (Quadratica)], [$cal(O)(S)$ (Lineare)], [$cal(O)(S)$ (Lineare)], [*$cal(O)(S)$ (Prefix-Scan)*],
  [Complessità Inferenza], [$cal(O)(S)$ (Crescente)], [$cal(O)(1)$ (Costante)], [$cal(O)(1)$ (Costante)], [*$cal(O)(1)$ (Strettamente Costante)*],
  [Memoria Inferenza], [Esplosione KV-Cache ($cal(O)(S)$)], [Fisso $cal(O)(1)$ (Vettore $h$)], [Fisso $cal(O)(1)$ (Matrice $W$)], [*Fisso $cal(O)(1)$ (Ologramma $S$)*],
  [Dominio dello Stato], [Attivazioni passate discrete], [Vettori continui reali $bb(R)^N$], [Matrici reali $bb(R)^(D times D)$], [*Fasori sul cerchio $bb(S)^1$*],
  [Dinamica Temporale], [Allineamento $Q K^T$ + Softmax], [Decadimento $exp(Delta A)$ reale], [Time-mixing esponenziale], [*Interferenza d'onda + $Delta theta_t$*],
  [Configurazione Teste], [Multi-Head (es. 96 teste)], [1 stato per canale], [Canali indipendenti], [*GQR (8 Query / 2 Memoria)*],
  [Sintassi Locale], [Positional embedding], [Conv1D causale iniziale], [Time-shift operator], [*Depthwise Conv1D causale*],
  [Vincolo Hardware], [Memory-Bandwidth Bound], [Compute-Bound], [Compute-Bound], [*Compute-Bound (Zero-Cache)*]
)
]

#v(0.3cm)

// -------------------------------------------------------------
// SEZIONE 6
// -------------------------------------------------------------
= 6. Analisi dei Parametri del Layer FSTLLM 2.0

Nel blocco `SelectiveFourierWaveLayerV2` (configurazione di riferimento con $d_"model" = 384$, $H = 8$, $"KV"_H = 2$, $"HD" = 48$):

- *Proiezione Query ($W_Q$):* $384 times 384 = 147.456$ parametri.
- *Proiezione Key ($W_K$):* $384 times (2 times 48) = 384 times 96 = 36.864$ parametri ($-75%$ rispetto a MHA).
- *Proiezione Value ($W_V$):* $384 times 96 = 36.864$ parametri.
- *Modulazione di Fase ($W_phi$):* $384 times 96 = 36.864$ parametri.
- *Decadimento Selettivo ($W_d$):* $384 times 2 = 768$ parametri.
- *Convoluzione Depthwise Locale:* $384 times 1 times 4 = 1.536$ parametri.
- *Proiezione di Uscita ($W_O$) e Gate ($W_G$):* $2 times (384 times 384) = 294.912$ parametri.

*Risultato:* L'approccio GQR libera parametri dall'infrastruttura di conservazione del contesto per reinvestirli nelle Query e nella rete non-lineare SwiGLU FFN, massimizzando il potere espressivo a parità di budget computazionale.

#v(0.3cm)

// -------------------------------------------------------------
// SEZIONE 7
// -------------------------------------------------------------
= 7. Benchmark e Confronto Sperimentale a ~30M di Parametri (Letteratura arXiv)

Nello studio delle architetture neurali, testare modelli su una scala controllata di *circa 30 Milioni di parametri* rappresenta la metodologia accademica ideale per isolare i meriti dell'innovazione architetturale dai semplici effetti di scala legati alla potenza di calcolo grezza.

Nella cartella `paper_confronto/` sono stati raccolti e analizzati i paper fondanti della letteratura con cui confrontare FSTLLM 2.0:

1. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023 - arXiv:2312.00752):*
   - Dimostra l'efficacia del meccanismo selettivo nei compiti di *Selective Copying* e *Induction Heads*, evidenziando i limiti dei modelli LTI (Linear Time-Invariant).
   - Offre il termine di paragone per misurare il throughput di generazione $cal(O)(1)$ e la capacità di filtro del contesto.
2. *TinyStories: How Small Can Language Models Be and Still Speak Coherent English? (Eldan & Li, 2023 - arXiv:2305.07759):*
   - Dimostra che modelli compresi tra *1M e 33M di parametri* (basati su Transformer standard tipo GPT-Neo/GPT-2 a 28M/33M parametri) sono in grado di apprendere sintassi, coerenza logica e ragionamento causale se addestrati su un vocabolario controllato.
   - Fornisce il *benchmark primario* su cui allenare e confrontare direttamente FSTLLM 2.0 ($~30$M parametri) con un Transformer classico di pari taglia.
3. *RWKV: Reinventing RNNs for the Transformer Era (Peng et al., 2023 - arXiv:2305.13048):*
   - Confronta architetture ricorrenti lineari e Transformer su scale a partire da *14M, 70M, 169M e 430M* di parametri su task NLP classici (LAMBADA, PIQA, StoryCloze).
   - Valuta la stabilità numerica del decadimento temporale per canale rispetto a modelli attention.
4. *Transformers are SSMs: Structured State Space Duality (Dao & Gu, 2024 - arXiv:2405.21060 - Mamba-2):*
   - Formalizza il legame teorico (State Space Duality, SSD) tra la moltiplicazione di matrici semidense causali e le forme canoniche degli spazi di stato, offrendo il formalismo per interpretare la risonanza armonica come forma strutturata di attenzione lineare.
5. *MobileLLM: Optimizing Sub-Billion Language Models (Meta AI, 2024 - arXiv:2402.14905):*
   - Analizza la massimizzazione del parametro-budget sotto il miliardo di pesi tramite Grouped-Query Attention (GQA) e deep-and-thin design.
6. *Attention Is All You Need (Vaswani et al., 2017 - arXiv:1706.03762):*
   - Il punto di riferimento immutabile per la formulazione del Transformer standard.

=== Protocollo Sperimentale Raccomandato per FSTLLM 2.0 (30M)
Per la tesi o la presentazione universitaria, si raccomanda di impostare il confronto empirico su tre prove:
- *1. Language Modeling su TinyStories (30M Parametri):* Addestrare FSTLLM 2.0 ($d_"model" = 384$, $6$ layer, $H=8, "KV"_H=2$) contro un Transformer equivalente a 6 layer. Verificare che la perplexity e la qualità generativa siano comparabili.
- *2. Long-Range Needle-In-A-Haystack (NIAH):* Posizionare un'informazione specifica all'interno di un contesto sintetico esteso (da 1k fino a 16k token) e testare la capacità della risonanza di fase nel recuperare il fatto senza degradazione d'ampiezza.
- *3. Benchmark di Inferenza e VRAM:* Misurare la memoria consumata al crescere della sequenza generata ($S = 128, 512, 2048, 8192$), dimostrando sperimentalmente la *curva perfettamente orizzontale $cal(O)(1)$* di FSTLLM 2.0 rispetto alla crescita lineare ripida del Transformer con KV-Cache.

#v(0.3cm)

// -------------------------------------------------------------
// SEZIONE 8
// -------------------------------------------------------------
= 8. Guida Strategica per la Presentazione Orale all'Università

Per condurre una discussione accademica impeccabile, si raccomanda di strutturare il discorso in *quattro passaggi sequenziali*:

1. *Il Paradosso del Transformer:*
   Lodare la capacità dei Transformer nel catturare il contesto, ma evidenziare subito il collo di bottiglia reale: l'attenzione $S times S$ diventa ingestibile per contesti di milioni di parole e il KV-Cache satura la memoria delle schede grafiche. Usare la metafora di 3Blue1Brown: _"Nei Transformer ogni parola, per essere interpretata, deve stringere la mano a tutte le parole del passato"_.
2. *La Promessa degli SSM e il loro Limite Reale:*
   Mostrare come i modelli a spazio di stato comprimano la storia in un vettore compatto ad inferenza $cal(O)(1)$. Evidenziare tuttavia che matrici reali $macron(A)$ con autovalori $<1$ equivalgono a un decadimento esponenziale monotono, incapace di preservare la periodicità e le interferenze di fase armonica a lungo raggio.
3. *L'Innovazione di FSTLLM 2.0:*
   Presentare la modulazione di fasori sul cerchio unitario $bb(S)^1$. Spiegare che:
   - I token rimangono vettori continui reali.
   - La convoluzione 1D locale cattura la sintassi immediata (articoli e desinenze).
   - La fase non è statica ma guidata selettivamente dal contenuto del testo ($Delta theta_t$).
   - Il Grouped-Query Resonance (GQR) permette a 4 teste di Query di interrogare simultaneamente lo stesso ologramma di memoria ($"KV"_0, "KV"_1$).
4. *Risposte Chiare alle Domande Tipiche della Commissione:*
   - *Domanda:* _"Perché usare numeri complessi sul cerchio unitario?"_ \
     *Risposta:* Il fasore $e^(j Phi)$ possiede modulo unitario $|e^(j Phi)| = 1$. A differenza dei coefficienti reali che smorzano il segnale portando a vanishing gradient, la rotazione di fase preserva rigorosamente l'energia, consentendo fenomeni di interferenza costruttiva e distruttiva a distanza temporale arbitraria.
   - *Domanda:* _"Come garantite la causalità se il training avviene in parallelo?"_ \
     *Risposta:* La causalità è garantita a monte dalla convoluzione causale (con padding asimmetrico a sinistra) e a valle dalla struttura triangolare inferiore della matrice di decadimento spettrale, impedendo qualunque trapelamento di informazione futura.
   - *Domanda:* _"Che differenza c'è tra RoPE (Rotary Position Embedding) e FSTLLM 2.0?"_ \
     *Risposta:* RoPE applica una rotazione deterministica fissa basata sulla posizione aritmetica assoluta del token. In FSTLLM 2.0, l'avanzamento di fase $Delta theta_t$ è un parametro *selettivo dipendente dai dati* appreso dalla rete, consentendo al modello di accelerare o congelare la frequenza in base alla semantica della frase.

#v(0.3cm)

// -------------------------------------------------------------
// SEZIONE 9
// -------------------------------------------------------------
= 9. Conclusioni

Il modello *FSTLLM 2.0* coniuga la solidità teorica dell'analisi armonica di Fourier con le esigenze architetturali dell'intelligenza artificiale generativa moderna. Garantendo inferenza strettamente $cal(O)(1)$, addestramento parallelo $cal(O)(S)$ e un'espressività ondulatoria complessa, l'architettura si candida come alternativa concreta ed elegante ai tradizionali meccanismi di Softmax Attention per l'elaborazione di sequenze a lunghissimo contesto.
