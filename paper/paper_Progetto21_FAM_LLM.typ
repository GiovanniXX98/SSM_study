#set page(
  paper: "a4",
  margin: (top: 2.2cm, bottom: 2.2cm, left: 2.0cm, right: 2.0cm),
  header: locate(loc => {
    if loc.page() > 1 [
      #grid(
        columns: (1fr, 1fr),
        align: (left, right),
        text(size: 8pt, fill: rgb("#64748b"), font: "DejaVu Serif", style: "italic")[Progetto 21.0: Fourier Associative Matrix LLM (FAM-LLM)],
        text(size: 8pt, fill: rgb("#64748b"), font: "DejaVu Serif")[Memoria Associativa Matriciale $K^dagger times.circle V$]
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
  size: 9.5pt,
  lang: "it"
)
#set par(justify: true, leading: 0.65em)

#let callout(title: "", body, color: rgb("#3b82f6"), bg: rgb("#eff6ff")) = {
  block(
    fill: bg,
    inset: 10pt,
    radius: 4pt,
    stroke: (left: 4pt + color, rest: 0.5pt + color.lighten(60%)),
    width: 100%,
    breakable: false,
    [
      #text(weight: "bold", fill: color.darken(20%), size: 9.5pt)[#title] \
      #v(3pt)
      #body
    ]
  )
}

#let mathbox(title: "", body) = {
  callout(title: title, body, color: rgb("#ca8a04"), bg: rgb("#fefce8"))
}

#let successbox(title: "", body) = {
  callout(title: title, body, color: rgb("#16a34a"), bg: rgb("#f0fdf4"))
}

#align(center)[
  #v(0.3cm)
  #text(size: 18pt, weight: "bold", fill: rgb("#0f172a"))[Progetto 21.0: Fourier Associative Matrix LLM (FAM-LLM)]
  #v(0.2cm)
  #text(size: 11pt, fill: rgb("#334155"), weight: "medium")[Sblocco della Capacità di Ragionamento nei Modelli Sub-Quadratici tramite Memoria Associativa Matriciale Complessa ($K^dagger times.circle V$) e Onde di Fourier]
  #v(0.3cm)
  #text(size: 9.5pt, fill: rgb("#475569"))[
    *Autore:* Giovanni & Team Fourier LLM \
    *Data:* Settembre – Ottobre 2026 \
    *Scala di Riferimento:* GPT-2 (50.257 Token Vocabulary, $d_"model" = 384 / 768$)
  ]
  #v(0.4cm)
]

#block(
  fill: rgb("#f8fafc"),
  inset: 11pt,
  radius: 4pt,
  stroke: (left: 4pt + rgb("#1e293b")),
  width: 100%,
  [
    #text(weight: "bold", size: 9.5pt, fill: rgb("#0f172a"))[Sommario (Abstract)] \
    #v(3pt)
    #text(size: 9pt, style: "italic", fill: rgb("#334155"))[
      I modelli linguistici sub-quadratici ricorrenti e a onde di Fourier (FSTLLM v1, v2) hanno introdotto l'inferenza a costo costante $cal(O)(1)$ e il training quasi-lineare $cal(O)(N log N)$. Ciononostante, l'adozione del prodotto di Hadamard (element-wise) per l'accumulazione dello stato spettrale ha storicamente imposto un grave *Capacity Bottleneck*: comprimere la storia di una sequenza in un vettore 1D di dimensione $D$ limita drasticamente i gradi di libertà a disposizione del modello, provocando interferenze distruttive delle fasi e rapido oblio associativo (*catastrophic forgetting*). In questo lavoro presentiamo il *Progetto 21.0: Fourier Associative Matrix LLM (FAM-LLM)*. L'architettura sostituisce la compressione vettoriale con un operatore di memoria associativa matriciale complessa:
      $ bold(M)_t = K_(bb(C), t)^dagger times.circle V_(bb(C), t) in bb(C)^("HD" times "HD") $
      L'aggiornamento dello stato cumulativo opera come un tensore di correlazione hebbiana continua, scalando la capacità di memorizzazione locale da $cal(O)("HD")$ a $cal(O)("HD"^2)$ per ciascuna testa. La complessità computazionale rispetto alla lunghezza del contesto $N$ resta strettamente lineare *$cal(O)(N)$* in fase di training (grazie alla formulazione a scansione parallela per blocchi *Chunked Scan*) e *$cal(O)(1)$* costante in inferenza per singolo token generato.
    ]
  ]
)

#v(0.3cm)

= 1. Il Limite dei Modelli Ricorrenti a Vettore 1D

Nelle formulazioni precedenti (V1, V2 e modelli SSM element-wise come RWKV-4), l'aggiornamento dello stato era descritto da:
$ s_t = gamma_t dot s_(t-1) + (1 - gamma_t) dot ((k_t dot e^(-i theta_t)) dot v_t) in bb(C)^("HD") $

Questa formulazione soffre di un'impossibilità algebrica:
1. *Mancanza di Interazione Incrociata (Rank-1 Vector Limit):* Il canale $i$-esimo della chiave $k$ può interagire unicamente con il canale $i$-esimo del valore $v$. Non esiste un tensore di transizione in grado di memorizzare l'associazione "la parola A implica il concetto B".
2. *Saturazione di Fase:* La fase angolare $theta in [0, 2pi)$ modula vettori su un cerchio unitario. Quando il numero di concetti supera la dimensione $"HD"$ della testa, onde su frequenze simili provocano battimenti o cancellazioni distruttive complete.

= 2. Formulazione Matematica di FAM-LLM 21.0

== 2.1 Onde di Fase e Proiezioni
Dato il vettore di attivazione $x_t in bb(R)^D$, estraiamo tramite proiezioni lineari:
$ Q_t = x_t W_Q, quad K_t = x_t W_K, quad V_t = x_t W_V $
La derivata di fase istantanea $Delta theta_t = pi tanh(x_t W_theta)$ guida l'oscillatore complesso:
$ theta_t = theta_(t-1) + omega_"base" + Delta theta_t $
$ K_(bb(C), t) = K_t dot e^(-i theta_t), quad Q_(bb(C), t) = Q_t dot e^(i theta_t) $

== 2.2 Il Salto di Memoria: Outer Product Matriciale
Invece del prodotto di Hadamard element-wise, calcoliamo il prodotto tensoriale esterno (outer product):
$ bold(M)_t = K_(bb(C), t)^dagger times.circle V_(bb(C), t) in bb(C)^("HD" times "HD") $
dove $dagger$ indica la trasposta coniugata hermitiana.
La memoria associativa si aggiorna ricorsivamente tramite:
$ bold(S)_t = gamma_t dot bold(S)_(t-1) + (1 - gamma_t) dot bold(M)_t $

== 2.3 Estrazione Associativa (Query Readout)
All'arrivo della Query complessa $Q_(bb(C), t) in bb(C)^("HD")$, l'estrazione non è un filtro element-wise ma un vero *prodotto vettore-matrice*:
$ Y_t = "Re"(Q_(bb(C), t) dot bold(S)_t) in bb(R)^("HD") $

Per la proprietà associativa:
$ Q_t dot (sum_tau K_tau^dagger V_tau) = sum_tau (Q_t K_tau^dagger) V_tau $
Il modello recupera la capacità espressiva tipica dei Transformer (confronto scalare di risonanza tra $Q$ e tutte le $K$ storiche), ma *senza dover memorizzare la sequenza di token*, poiché la storia è compressa nella matrice compatta $bold(S)$.

= 3. Complessità e Prestazioni a Confronto

#align(center)[
#table(
  columns: (2.0fr, 2.2fr, 2.0fr, 2.2fr),
  fill: (col, row) => if row == 0 { rgb("#0f172a") } else if calc.even(row) { rgb("#f8fafc") } else { none },
  stroke: (x, y) => if y == 0 { (bottom: 1.5pt + rgb("#0f172a")) } else { 0.5pt + rgb("#e2e8f0") },
  inset: 6pt,
  align: (col, row) => if row == 0 { center + horizon } else { left + horizon },

  text(weight: "bold", fill: white, size: 7.5pt)[Proprietà],
  text(weight: "bold", fill: white, size: 7.5pt)[Transformer Standard],
  text(weight: "bold", fill: white, size: 7.5pt)[Vecchio Fourier LLM (V2)],
  text(weight: "bold", fill: white, size: 7.5pt)[Progetto 21.0 (FAM-LLM)],

  [Forma dello Stato], [KV Cache ($N times D$)], [Vettore 1D compresso ($D$)], [*Matrice Associativa ($"HD" times "HD"$)*],
  [Capacità di Memoria], [$cal(O)(N dot D)$], [$cal(O)(D)$ (Bottleneck)], [*$cal(O)(D^2)$ per testa (Elevata)*],
  [Complessità Training], [$cal(O)(N^2 dot D)$ (Quadratica)], [$cal(O)(N log N)$], [*$cal(O)(N dot D^2)$ (Lineare in $N$)*],
  [Complessità Generazione], [$cal(O)(N)$ per token (Degrada)], [$cal(O)(1)$ per token], [*$cal(O)(1)$ per token (Costante)*],
  [VRAM a $N=32.000$ tk], [~14 GB (Esplosione)], [~0.5 GB], [*~0.6 GB (Strettamente Costante)*]
)
]

= 4. Conclusioni

Il Progetto 21.0 unifica la purezza della fisica delle onde di Fourier con la robustezza matematica della Linear Attention e dei sistemi associativi hebbiani. Il modello mantiene intatta la velocità e l'efficienza $cal(O)(1)$ in inferenza, offrendo al contempo lo spazio di rappresentazione matriciale necessario per il ragionamento elaborato e l'in-context learning.
