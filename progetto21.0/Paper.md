# Progetto 21.0: Fourier Associative Matrix LLM (FAM-LLM)
## Sblocco della Capacità di Ragionamento nei Modelli Sub-Quadratici tramite Memoria Associativa Matriciale Complessa ($K^\dagger \otimes V$) e Onde di Fourier

**Autore:** Giovanni & Team Fourier LLM  
**Data:** Settembre 2026  
**Scala di Riferimento:** GPT-2 (50.257 Token Vocabulary, $d_{model} = 384 / 768$)

---

### Abstract
I modelli linguistici sub-quadratici ricorrenti e a onde di Fourier (FSTLLM v1, v2) hanno introdotto l'inferenza a costo costante $O(1)$ e il training quasi-lineare $O(N \log N)$. Ciononostante, l'adozione del prodotto di Hadamard (element-wise) per l'accumulazione dello stato spettrale ha storicamente imposto un grave **Capacity Bottleneck**: comprimere la storia di una sequenza in un vettore 1D di dimensione $D$ limita drasticamente i gradi di libertà a disposizione del modello, provocando interferenze distruttive delle fasi e rapido oblio associativo (*catastrophic forgetting*).

In questo lavoro presentiamo il **Progetto 21.0: Fourier Associative Matrix LLM (FAM-LLM)**. L'architettura sostituisce la compressione vettoriale con un operatore di memoria associativa matriciale complessa:
$$\mathbf{M}_t = K_{\mathbb{C}, t}^\dagger \otimes V_{\mathbb{C}, t} \in \mathbb{C}^{HD \times HD}$$
L'aggiornamento dello stato cumulativo opera come un tensore di correlazione hebbiana continua, scalando la capacità di memorizzazione locale da $O(HD)$ a $O(HD^2)$ per ciascuna testa. La complessità computazionale rispetto alla lunghezza del contesto $N$ resta strettamente lineare **$O(N)$** in fase di training (grazie alla formulazione a scansione parallela per blocchi *Chunked Scan*) e **$O(1)$** costante in inferenza per singolo token generato.

---

## 1. Il Limite dei Modelli Ricorrenti a Vettore 1D

Nelle formulazioni precedenti (V1, V2 e modelli SSM element-wise come RWKV-4), l'aggiornamento dello stato era descritto da:
$$s_t = \gamma_t \odot s_{t-1} + (1 - \gamma_t) \odot \left( (k_t \cdot e^{-i \theta_t}) \odot v_t \right) \in \mathbb{C}^{HD}$$

Questa formulazione soffre di un'impossibilità algebrica:
1. **Mancanza di Interazione Incrociata (Rank-1 Vector Limit):**  
   Il canale $i$-esimo della chiave $k$ può interagire unicamente con il canale $i$-esimo del valore $v$. Non esiste un tensore di transizione in grado di memorizzare l'associazione "la parola A implica il concetto B".
2. **Saturazione di Fase:**  
   La fase angolare $\theta \in [0, 2\pi)$ modula vettori su un cerchio unitario. Quando il numero di concetti supera la dimensione $HD$ della testa, onde su frequenze simili provocano battimenti o cancellazioni distruttive complete.

---

## 2. Formulazione Matematica di FAM-LLM 21.0

### 2.1 Onde di Fase e Proiezioni
Dato il vettore di attivazione $x_t \in \mathbb{R}^D$, estraiamo tramite proiezioni lineari:
$$Q_t = x_t W_Q, \quad K_t = x_t W_K, \quad V_t = x_t W_V$$
La derivata di fase istantanea $\Delta \theta_t = \pi \tanh(x_t W_\theta)$ guida l'oscillatore complesso:
$$\theta_t = \theta_{t-1} + \omega_{\text{base}} + \Delta \theta_t$$
$$K_{\mathbb{C}, t} = K_t \odot e^{-i \theta_t}, \quad Q_{\mathbb{C}, t} = Q_t \odot e^{i \theta_t}$$

### 2.2 Il Salto di Memoria: Outer Product Matriciale
Invece del prodotto di Hadamard, calcoliamo il prodotto tensoriale esterno (outer product):
$$\mathbf{M}_t = K_{\mathbb{C}, t}^\dagger \otimes V_{\mathbb{C}, t} \in \mathbb{C}^{HD \times HD}$$
dove $\dagger$ indica la trasposta coniugata.
La memoria associativa si aggiorna ricorsivamente tramite:
$$\mathbf{S}_t = \gamma_t \odot \mathbf{S}_{t-1} + (1 - \gamma_t) \odot \mathbf{M}_t$$

### 2.3 Estrazione Associativa (Query Readout)
All'arrivo della Query complessa $Q_{\mathbb{C}, t} \in \mathbb{C}^{HD}$, l'estrazione non è un filtro element-wise ma un vero **prodotto vettore-matrice**:
$$Y_t = \text{Re}\left( Q_{\mathbb{C}, t} \cdot \mathbf{S}_t \right) \in \mathbb{R}^{HD}$$

Per la proprietà associativa:
$$Q_t \cdot \left( \sum_\tau K_\tau^\dagger V_\tau \right) = \sum_\tau \left( Q_t K_\tau^\dagger \right) V_\tau$$
Il modello recupera la capacità espressiva tipica dei Transformer (confronto scalare di risonanza tra $Q$ e tutte le $K$ storiche), ma **senza dover memorizzare la sequenza di token**, poiché la storia è compressa nella matrice compatta $\mathbf{S}$.

---

## 3. Complessità e Prestazioni

| Proprietà | Transformer Standard (GPT-2 / Llama) | Vecchio Fourier LLM (V2) | **Progetto 21.0 (FAM-LLM)** |
| :--- | :--- | :--- | :--- |
| **Forma dello Stato** | KV Cache illimitata ($N \times D$) | Vettore 1D compresso ($D$) | **Matrice Associativa ($HD \times HD$)** |
| **Capacità di Memoria** | $O(N \cdot D)$ | $O(D)$ (molto debole) | **$O(D^2)$ per testa (elevatissima)** |
| **Complessità Training** | $O(N^2 \cdot D)$ (Quadratica) | $O(N \log N)$ | **$O(N \cdot D^2)$ (Lineare in $N$)** |
| **Complessità Generazione**| $O(N)$ per token (degradante) | $O(1)$ per token | **$O(1)$ per token (costante)** |
| **VRAM a $N=32.000$ tk** | ~14 GB (esplosione) | ~0.5 GB | **~0.6 GB (costante)** |

---

## 4. Conclusioni
Il Progetto 21.0 unifica la purezza della fisica delle onde di Fourier con la robustezza matematica della Linear Attention e dei sistemi associativi hebbiani. Il modello mantiene intatta la velocità e l'efficienza $O(1)$ in inferenza, offrendo al contempo lo spazio di rappresentazione matriciale necessario per il ragionamento elaborato e l'in-context learning.
