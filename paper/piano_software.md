# Piano Industriale e Architettura Software: Suite di Agenti AI di Coding (FSTLLM 2.0)
### Modello di Business Bootstrapped da Casa a 5 € / Mese con Segreto Industriale e Hardware da 16GB VRAM

**Documento Strategico, Architetturale ed Esecutivo**  
*Progetto FSTLLM 2.0 (Progetto 21) — Autonomous AI Startup*  
*Autore: Giovanni — Anno 2026*

---

## 1. Visione Strategica e Proposta di Valore (Value Proposition)

### 1.1 Il Problema del Mercato Attuale
I programmatori oggi si affidano a strumenti generalisti costosi e pesanti:
- **GitHub Copilot:** 10 $ – 19 $ al mese.
- **Cursor Pro:** 20 $ al mese (~24 €).
- **ChatGPT Plus / Claude Pro:** 20 $ + IVA (~24 € al mese).

Questi strumenti utilizzano modelli generalisti monolitici basati su Transformer tradizionali. Quando il programmatore lavora su repository complessi (documentazione estesa, decine di file aperti, file sorgente da migliaia di righe), la memoria KV-Cache dei Transformer subisce un'esplosione quadratica/lineare $\mathcal{O}(S)$, provocando:
1. Rallentamenti drastici della risposta.
2. Taglio arbitrario del contesto passato.
3. Costi di esercizio astronomici per i provider (che scaricano sull'utente con abbonamenti cari).

### 1.2 La Nostra Soluzione: Suite di Agenti Specializzati a 5 € / Mese
Invece di un unico modello generalista lento, offriamo una **Suite di Agenti Iper-Specializzati da 1B di parametri**, ciascuno verticalizzato al 100% su un singolo linguaggio di programmazione (Python, Rust, TypeScript, Security/Bug-Hunting), basati sull'architettura proprietaria **Fourier Space-Time LLM (FSTLLM 2.0)**.

#### I Due Vantaggi Competitivi Incolmabili (Il Moat Tecnologico):
1. **Memoria di Stato Rigidamente $\mathcal{O}(1)$:** L'agente può analizzare interi repository di codice mantenendo l'impronta di memoria a **pochi megabyte**, senza saturare la RAM del server e garantendo una latenza per-token fissa.
2. **Prezzo "No-Brainer" (5 € / mese):** Meno di un quarto del prezzo di Copilot e Cursor. Il costo equivale a una colazione: scatta l'acquisto impulsivo e il tasso di cancellazione (churn) si azzera.

---

## 2. Fase 1: Data Strategy & Dataset di "Oro Puro" (Textbooks Are All You Need)

Per addestrare un modello da 1B che scriva codice impeccabile senza passare mesi a scaricare petabyte di dati, applichiamo il principio di Microsoft Research (*Phi-1, Gunasekar et al.*): **la qualità batte la quantità di 100 a 1**.

### 2.1 Composizione del Dataset (5 – 8 Miliardi di Token = ~15 – 20 GB)
* **Non scaricare codice grezzo da GitHub:** Il 90% del codice open-source pubblico è privo di tipi, mal documentato e pieno di bug.
* **Le 4 Fonti a Densità Massima per il primo Agente (Python Specialist):**
  1. *Documentazione Ufficiale & PEP:* Documentazione Python 3.12+, tutorial ufficiali, standard library commentata riga per riga.
  2. *Top-Tier Repositories:* Repository di riferimento ingegneristico assoluto (es. `requests`, `fastapi`, `pydantic`, `pytorch`).
  3. *Coppie Istruzione-Codice Sintetico:* Esercizi algoritmici con spiegazione passo-passo della logica, docstring, typing completo e test unitari `pytest`.
  4. *Refactoring & Bug Fixing:* Esempi "prima e dopo" che mostrano codice errato o inefficiente corretto e ottimizzato.

### 2.2 Tokenizer Ottimizzato per il Codice
- Utilizzo di un tokenizer Byte-Pair Encoding (BPE) da **32.000 – 64.000 token** specializzato sulla conservazione dell'indentazione (4 spazi, tabulazioni) e delle parole chiave di programmazione (`def`, `class`, `async`, `await`, `return`, `lambda`).

---

## 3. Fase 2: Architettura FSTLLM 1B & Fast Training con AdamW Potenziato

### 3.1 Specifiche del Modello FSTLLM 1B (Specialized Code Agent)
* **Parametri Totali:** ~1.10 Miliardi di parametri.
* **Dimensione Nascosta ($d_{\text{model}}$):** 2048.
* **Numero di Strati:** 24 blocchi risonanti.
* **Grouped Query Resonance (GQR):** 16 teste Query vs 4 teste Memoria KV ($HD = 128$).
* **Operatore di Fase:** Fasori $e^{i \phi} \in \mathbb{S}^1$ con spettro armonico log-spaziato da $0.001$ a $\pi$.
* **Non-Linearità:** SwiGLU FFN (dimensione intermedia 5504).
* **Accoppiamento Locale:** Depthwise Conv1D causale con kernel $K=4$.

### 3.2 Il Curriculum di Training Rapido da Casa (GPU 16GB VRAM)
Grazie alla capacità analitica dei fasori di agganciare le frequenze sintattiche del codice molto più rapidamente dell'attenzione Softmax, e impiegando un **ottimizzatore AdamW potenziato** (warmup lineare al 3%, cosine learning rate decay, gradient clipping a 1.0 e weight decay disaccoppiato a 0.1):

```
[ Step 2.1: Prototipo 300M ] ──► Training in 3 – 5 ore  (Verifica convergenza e sintassi)
           │
           ▼
[ Step 2.2: Agente 500M ]   ──► Training in 8 – 12 ore (Overnight training: pronto al mattino)
           │
           ▼
[ Step 2.3: Agente 1B ]     ──► Training in 24 – 36 ore (Modello definitivo pronto al serving)
```

Tutto l'addestramento viene eseguito sulla GPU domestica da 16GB con consumo elettrico inferiore a **10 – 15 € di corrente totale**.

---

## 4. Fase 3: Hardening e Protezione del Segreto Industriale

Il vantaggio competitivo non deve mai essere divulgato. Il modello viene distribuito e servito come una **"Scatola Nera" (Black Box)** inespugnabile:

1. **Compilazione dei Moduli Critici in C++ / Rust (`.so`):**
   - I file dell'architettura FSTLLM (il calcolo della fase, la rotazione dei fasori e la risonanza) non restano script Python in chiaro.
   - Vengono compilati in binari macchina condivisi (`.so`) tramite `pybind11` o Cython, rendendo impossibile il reverse engineering del codice sorgente matematico.
2. **Cifratura dei Pesi del Modello (AES-256):**
   - Il file dei pesi `.pt` o safetensors viene cifrato con chiave simmetrica AES-256 a 256 bit.
   - All'avvio del container, una chiave master (iniettata via variabile d'ambiente protetta) decifra i tensori **esclusivamente nella RAM volatile della GPU**, senza scrivere mai pesi in chiaro su disco fisso.
3. **Distribuzione Docker Sigillata:**
   - Creazione di un container Docker minimale (Debian slim / Alpine) privo di compilatori, strumenti di debug o shell esposte.

---

## 5. Fase 4: Architettura di Serving, Hosting e Infrastruttura Client-Server

### 5.1 Lo Stack di Serving Asincrono ad Altissima Efficienza
Il server non ricalcola il passato: sfrutta la memoria $\mathcal{O}(1)$ di FSTLLM con Continuous Batching.

```
[ Client: VS Code Extension / Web App ]
                 │ (Richiesta HTTPS con API Key)
                 ▼
      [ Cloudflare Tunnel ]  <─── Zero porte aperte sul router, SSL gratis, Anti-DDoS
                 │
                 ▼
[ NGINX / Traefik Reverse Proxy ] (Rate Limiting, Autenticazione Stripe)
                 │
                 ▼
     [ FastAPI Backend (Python) ] (Gestione code asincrone con Redis)
                 │
                 ▼
 [ FSTLLM 2.0 Inference Engine (.so) ] (GPU 16GB - Paged State Buffer)
                 │
                 ▼ (Risposta in Streaming Token-by-Token via SSE)
      [ Client Riceve il Codice ]
```

### 5.2 Strategia di Hosting Scalabile per Costi

* **Opzione 1 (Da Casa a Costo Zero):** 
  - Il tuo PC attuale con GPU da 16GB serve i primi 50-100 clienti con un consumo di appena 50 €/mese di bolletta.
  - Connesso tramite **Cloudflare Tunnel**, nessuno conosce l'IP di casa tua.
* **Opzione 2 (Cloud Serverless a 15 € – 25 € / mese):**
  - Con **RunPod Serverless** o **Modal**, la GPU cloud si attiva solo per i 2 secondi in cui l'utente genera codice. Per 50-100 utenti spendi circa **0.50 € al giorno**.
* **Opzione 3 (Server Dedicato Cloud 24/7 a 100 € – 120 € / mese):**
  - 1x NVIDIA RTX 3090 / 4070 da 24GB su RunPod Secure Cloud, sempre accesa, con IP dedicato e uptime 99.99%. Si attiva quando gli abbonati superano quota 30-40.

---

## 6. Fase 5: Modello Economico, Unit Economics e Fair Use Policy a 5 € / Mese

### 6.1 La Matematica di 1 Utente a 5 € / Mese
* **Incasso Lordo:** **5,00 €**
* **Commissione Stripe (1.5% + 0.25€):** **- 0,33 €**
* **Costo Elettricità / Compute GPU (50k token/mese):** **- 0,005 €**
* **Quota Hosting Server ripartita:** **- 0,05 €**
* **UTILE NETTO PER UTENTE:** **~ 4,60 € al mese (Margine > 90%)**

### 6.2 La Regola della "Fair Use Policy" (Anti-Bot)
Per evitare che un utente colleghi script automatici che saturano la GPU:
- L'abbonamento da 5 €/mese include **1.000.000 di token al mese** (~1.500 pagine di codice).
- Un programmatore umano non li esaurirà mai in un mese di lavoro normale.
- Oltre 1M di token: blocco o addebito di 1 € ogni 500.000 token extra.

### 6.3 Proiezioni Finanziarie di Crescita

| Scaglione Clienti | Ricavo Lordo Mensile | Costi Operativi Totali | UTILE NETTO IN TASCA |
| :---: | :---: | :---: | :---: |
| **50 programmatori** | 250 € / mese | ~35 € (Serverless) | **~ 215 € / mese** |
| **100 programmatori** | 500 € / mese | ~50 € (Serverless) | **~ 450 € / mese** |
| **300 programmatori** | 1.500 € / mese | ~150 € (Server dedicato) | **~ 1.350 € / mese** |
| **500 programmatori** | 2.500 € / mese | ~160 € (Server dedicato) | **~ 2.340 € / mese** |
| **1.000 programmatori** | 5.000 € / mese | ~250 € (Server dedicato) | **~ 4.750 € / mese** |
| **3.000 programmatori** | 15.000 € / mese | ~550 € (Server 2x GPU) | **~ 14.450 € / mese** |

---

## 7. Fase 6: Tabella di Marcia Operativa (Roadmap Settimanale)

* **Settimana 1 (Data & Training):** 
  - Curatela del dataset Python ad altissima qualità (5B token).
  - Addestramento del modello FSTLLM 500M (8-12 ore) e scaling a 1B (24-36 ore) con AdamW potenziato.
  - Salvataggio e verifica del checkpoint con benchmark di coding.
* **Settimana 2 (Hardening & Backend):**
  - Compilazione in modulo binario C++ (`.so`) e crittografia AES-256 dei pesi.
  - Creazione del server FastAPI con streaming SSE e code asincrone.
* **Settimana 3 (Piattaforma Web & Pagamenti):**
  - Landing page essenziale e moderna con interfaccia chat/playground per testare il modello.
  - Integrazione abbonamento Stripe a 5 €/mese con generazione automatica di API Key.
  - Configurazione Cloudflare Tunnel.
* **Settimana 4 (Go-to-Market & Primi 20 Clienti Pilota):**
  - Distribuzione gratuita per 7 giorni nei gruppi di programmatori (Reddit r/Python, Discord dev, università).
  - Raccolta feedback e conversione nei primi abbonamenti paganti a 5 €/mese.
  - Reinvestimento del 100% degli utili nei successivi agenti specializzati (Rust Specialist, TypeScript Specialist).
