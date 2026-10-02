#set page(
  paper: "a4",
  margin: (top: 2.2cm, bottom: 2.2cm, left: 2.0cm, right: 2.0cm),
  header: locate(loc => {
    if loc.page() > 1 [
      #grid(
        columns: (1fr, 1fr),
        align: (left, right),
        text(size: 8pt, fill: rgb("#64748b"), font: "DejaVu Serif", style: "italic")[Piano Industriale FSTLLM 2.0: Suite Agenti Coding (1B)],
        text(size: 8pt, fill: rgb("#64748b"), font: "DejaVu Serif")[Startup Bootstrapped & Segreto Industriale]
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
  #text(size: 18pt, weight: "bold", fill: rgb("#0f172a"))[Piano Industriale e Architettura Software: Suite di Agenti AI di Coding (FSTLLM 2.0)]
  #v(0.2cm)
  #text(size: 11pt, fill: rgb("#334155"), weight: "medium")[Strategia Bootstrapped da Casa a 5 € / Mese con Segreto Industriale e Hardware da 16GB VRAM]
  #v(0.3cm)
  #text(size: 9.5pt, fill: rgb("#475569"))[
    *Documento Strategico, Architetturale ed Esecutivo* \
    Progetto FSTLLM 2.0 (Progetto 21) — Autonomous AI Startup \
    Autore: Giovanni — Anno 2026
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
    #text(weight: "bold", size: 9.5pt, fill: rgb("#0f172a"))[Executive Summary] \
    #v(3pt)
    #text(size: 9pt, style: "italic", fill: rgb("#334155"))[
      Il presente piano delinea la roadmap tecnologica e commerciale per il lancio indipendente di una suite di Agenti di Intelligenza Artificiale per il coding basati sull'architettura proprietaria *Fourier Space-Time LLM (FSTLLM 2.0)* scalata a *1 Miliardo di parametri*. Sfruttando la memoria ricorrente rigorosamente $cal(O)(1)$ e un addestramento ultra-rapido con AdamW potenziato su singola GPU da 16GB VRAM, il progetto adotta una strategia "Black Box" a segreto industriale. L'offerta a *5 € al mese* (contro i 20-24 € di Copilot e Cursor) crea un vantaggio competitivo incolmabile, consentendo margini operativi netti superiori al *90%* fin dai primi 50 clienti.
    ]
  ]
)

#v(0.3cm)

= 1. Visione Strategica e Proposta di Valore (Value Proposition)

== 1.1 Il Problema del Mercato Attuale
I programmatori oggi si affidano a strumenti generalisti costosi e pesanti:
- *GitHub Copilot:* 10 \$ – 19 \$ al mese.
- *Cursor Pro:* 20 \$ al mese (~24 €).
- *ChatGPT Plus / Claude Pro:* 20 \$ + IVA (~24 € al mese).

Questi strumenti utilizzano modelli generalisti basati su Transformer tradizionali. Quando il programmatore lavora su repository complessi (documentazione estesa, decine di file aperti, file sorgente da migliaia di righe), la memoria KV-Cache dei Transformer subisce un'esplosione quadratica/lineare $cal(O)(S)$, provocando rallentamenti drastici, troncamento arbitrario del contesto e costi di esercizio insostenibili.

== 1.2 La Nostra Soluzione: Suite di Agenti Specializzati a 5 € / Mese
Invece di un unico modello generalista lento, offriamo una *Suite di Agenti Iper-Specializzati da 1B di parametri*, ciascuno verticalizzato al 100% su un singolo linguaggio di programmazione (Python, Rust, TypeScript, Security/Bug-Hunting), basati sull'architettura proprietaria *FSTLLM 2.0*.

#successbox(title: "I Due Vantaggi Competitivi Incolmabili (Il Moat Tecnologico)")[
  1. *Memoria di Stato Rigidamente $cal(O)(1)$:* L'agente può analizzare interi repository di codice mantenendo l'impronta di memoria a *pochi megabyte*, senza saturare la RAM del server e garantendo una latenza per-token fissa.
  2. *Prezzo "No-Brainer" (5 € / mese):* Meno di un quarto del prezzo di Copilot e Cursor. Il costo equivale a una colazione: scatta l'acquisto impulsivo e il tasso di cancellazione (churn) si azzera.
]

= 2. Fase 1: Data Strategy & Dataset di "Oro Puro" (Textbooks Are All You Need)

Per addestrare un modello da 1B che scriva codice impeccabile senza passare mesi a scaricare petabyte di dati, applichiamo il principio di Microsoft Research (_Phi-1, Gunasekar et al._): *la qualità batte la quantità di 100 a 1*.

- *Non scaricare codice grezzo da GitHub:* Il 90% del codice open-source pubblico è privo di tipi, mal documentato e pieno di bug.
- *Le 4 Fonti a Densità Massima per il primo Agente (Python Specialist):*
  1. *Documentazione Ufficiale & PEP:* Documentazione Python 3.12+, tutorial ufficiali, standard library commentata riga per riga.
  2. *Top-Tier Repositories:* Repository di riferimento ingegneristico assoluto (`requests`, `fastapi`, `pydantic`, `pytorch`).
  3. *Coppie Istruzione-Codice Sintetico:* Esercizi algoritmici con spiegazione passo-passo della logica, docstring, typing completo e test unitari `pytest`.
  4. *Refactoring & Bug Fixing:* Esempi "prima e dopo" che mostrano codice errato o inefficiente corretto e ottimizzato.
- *Volume Target:* 5 – 8 Miliardi di token (~15 – 20 GB di testo) con Tokenizer BPE dedicato da 32.000 token orientato all'indentazione e parole chiave.

= 3. Fase 2: Architettura FSTLLM 1B & Fast Training con AdamW Potenziato

== 3.1 Specifiche del Modello FSTLLM 1B (Specialized Code Agent)
- *Parametri Totali:* ~1.10 Miliardi di parametri.
- *Dimensione Nascosta ($d_"model"$):* 2048 | *Numero di Strati:* 24 blocchi risonanti.
- *Grouped Query Resonance (GQR):* 16 teste Query vs 4 teste Memoria KV ($"HD" = 128$).
- *Operatore di Fase:* Fasori $e^(i phi) in bb(S)^1$ con spettro armonico log-spaziato da $0.001$ a $pi$.
- *Non-Linearità:* SwiGLU FFN (dimensione intermedia 5504) con Conv1D causale ($K=4$).

== 3.2 Il Curriculum di Training Rapido da Casa (GPU 16GB VRAM)
Grazie alla capacità analitica dei fasori di agganciare le frequenze sintattiche del codice molto più rapidamente dell'attenzione Softmax, e impiegando un *ottimizzatore AdamW potenziato* (warmup lineare al 3%, cosine learning rate decay, gradient clipping a 1.0 e weight decay disaccoppiato a 0.1):

#align(center)[
#table(
  columns: (2.2fr, 2.0fr, 2.0fr, 2.2fr),
  fill: (col, row) => if row == 0 { rgb("#0f172a") } else if calc.even(row) { rgb("#f8fafc") } else { none },
  stroke: (x, y) => if y == 0 { (bottom: 1.5pt + rgb("#0f172a")) } else { 0.5pt + rgb("#e2e8f0") },
  inset: 6pt,
  align: (col, row) => if row == 0 { center + horizon } else { left + horizon },

  text(weight: "bold", fill: white, size: 8pt)[Stadio Curriculum],
  text(weight: "bold", fill: white, size: 8pt)[Dimensione Pesi],
  text(weight: "bold", fill: white, size: 8pt)[Tempo Training Stimato],
  text(weight: "bold", fill: white, size: 8pt)[Obiettivo],

  [Step 2.1: Prototipo 300M], [~600 MB], [*3 – 5 ore*], [Verifica convergenza e sintassi Python],
  [Step 2.2: Agente 500M], [~1.0 GB], [*8 – 12 ore* (overnight)], [Padroneggia Standard Library e test],
  [Step 2.3: Agente 1B Definitivo], [~2.0 GB], [*24 – 36 ore* (1.5 giorni)], [Ragionamento multi-file e refactoring]
)
]

Tutto l'addestramento viene eseguito sulla GPU domestica da 16GB con consumo elettrico inferiore a *10 – 15 € di bolletta totale*.

= 4. Fase 3: Hardening e Protezione del Segreto Industriale

Il vantaggio competitivo non deve mai essere divulgato. Il modello viene distribuito e servito come una *"Scatola Nera" (Black Box)* inespugnabile:

1. *Compilazione dei Moduli Critici in C++ / Rust (`.so`):* I file dell'architettura FSTLLM (il calcolo della fase, la rotazione dei fasori e la risonanza) vengono compilati in binari macchina condivisi (`.so`) tramite `pybind11` o Cython, rendendo impossibile il reverse engineering del codice matematico.
2. *Cifratura dei Pesi del Modello (AES-256):* Il file dei pesi `.pt` viene cifrato con chiave simmetrica AES-256. All'avvio del container, una chiave master (iniettata via variabile d'ambiente protetta) decifra i tensori *esclusivamente nella RAM volatile della GPU*, senza scrivere mai pesi in chiaro su disco.
3. *Distribuzione Docker Sigillata:* Creazione di un container Docker minimale (Debian slim / Alpine) privo di compilatori, strumenti di debug o shell esposte.

= 5. Fase 4: Architettura di Serving, Hosting e Infrastruttura Client-Server

Il server non ricalcola il passato: sfrutta la memoria $cal(O)(1)$ di FSTLLM con Continuous Batching.

#align(center)[
#table(
  columns: (2.5fr, 2.5fr, 2.5fr),
  fill: (col, row) => if row == 0 { rgb("#0f172a") } else if calc.even(row) { rgb("#f8fafc") } else { none },
  stroke: (x, y) => if y == 0 { (bottom: 1.5pt + rgb("#0f172a")) } else { 0.5pt + rgb("#e2e8f0") },
  inset: 6pt,
  align: (col, row) => if row == 0 { center + horizon } else { left + horizon },

  text(weight: "bold", fill: white, size: 8pt)[Opzione Hosting],
  text(weight: "bold", fill: white, size: 8pt)[Costo Mensile],
  text(weight: "bold", fill: white, size: 8pt)[Capacità e Note],

  [Opzione 1: PC da Casa con Cloudflare Tunnel], [*~50 € – 60 € / mese* (corrente)], [Zero porte aperte, IP mascherato, regge 50-100 utenti],
  [Opzione 2: Cloud Serverless (RunPod/Modal)], [*~15 € – 25 € / mese*], [Paga solo al secondo di calcolo effettivo, zero costi a riposo],
  [Opzione 3: Server Dedicato 24/7 (RunPod)], [*~100 € – 120 € / mese*], [1x RTX 3090/4070 (24GB), uptime 99.99%, per 100-300 utenti]
)
]

= 6. Fase 5: Modello Economico, Unit Economics e Fair Use Policy a 5 € / Mese

== 6.1 La Matematica di 1 Utente a 5 € / Mese
- *Incasso Lordo:* *5,00 €* | *Commissione Stripe:* *- 0,33 €*
- *Costo Calcolo GPU (50k token/mese):* *- 0,005 €* | *Quota Hosting:* *- 0,05 €*
- *UTILE NETTO PER UTENTE:* *~ 4,60 € al mese (Margine > 90%)*

== 6.2 La Regola della "Fair Use Policy" (Anti-Bot)
L'abbonamento include *1.000.000 di token al mese* (~1.500 pagine di codice). Un programmatore umano non li esaurirà mai in un mese di lavoro normale. Oltre 1M di token: blocco o addebito di 1 € ogni 500.000 token extra.

#align(center)[
#table(
  columns: (2.0fr, 2.0fr, 2.0fr, 2.0fr),
  fill: (col, row) => if row == 0 { rgb("#0f172a") } else if calc.even(row) { rgb("#f8fafc") } else { none },
  stroke: (x, y) => if y == 0 { (bottom: 1.5pt + rgb("#0f172a")) } else { 0.5pt + rgb("#e2e8f0") },
  inset: 6pt,
  align: (col, row) => if row == 0 { center + horizon } else { center + horizon },

  text(weight: "bold", fill: white, size: 8pt)[Scaglione Clienti],
  text(weight: "bold", fill: white, size: 8pt)[Ricavo Lordo],
  text(weight: "bold", fill: white, size: 8pt)[Costi Operativi],
  text(weight: "bold", fill: white, size: 8pt)[UTILE NETTO MENSILE],

  [50 programmatori], [250 € / mese], [~35 € (Serverless)], [*~ 215 € / mese*],
  [100 programmatori], [500 € / mese], [~50 € (Serverless)], [*~ 450 € / mese*],
  [300 programmatori], [1.500 € / mese], [~150 € (Dedicato)], [*~ 1.350 € / mese*],
  [500 programmatori], [2.500 € / mese], [~160 € (Dedicato)], [*~ 2.340 € / mese*],
  [1.000 programmatori], [5.000 € / mese], [~250 € (Dedicato)], [*~ 4.750 € / mese*],
  [3.000 programmatori], [15.000 € / mese], [~550 € (2x GPU)], [*~ 14.450 € / mese*]
)
]

= 7. Fase 6: Tabella di Marcia Operativa (Roadmap Settimanale)

- *Settimana 1 (Data & Training):* Curatela dataset Python (5B token). Addestramento FSTLLM 500M (8-12 ore) e scaling a 1B (24-36 ore) con AdamW potenziato.
- *Settimana 2 (Hardening & Backend):* Compilazione binaria C++ (`.so`) e crittografia AES-256 dei pesi. Creazione server FastAPI asincrono con streaming SSE.
- *Settimana 3 (Piattaforma Web & Pagamenti):* Landing page essenziale, integrazione Stripe a 5 €/mese con generazione API Key e Cloudflare Tunnel.
- *Settimana 4 (Go-to-Market & Primi Clienti):* Distribuzione gratuita di prova 7 giorni nei gruppi dev (Reddit, Discord, università). Conversione nei primi 20 abbonamenti a 5 €/mese e reinvestimento degli utili nei successivi agenti (Rust, TypeScript).
