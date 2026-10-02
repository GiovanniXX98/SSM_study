# Archivio Ufficiale Paper — Progetto 21.0 & FSTLLM

Questo archivio raccoglie tutti i paper accademici, i report sperimentali, i piani industriali e la letteratura di riferimento relativi a **Progetto 21.0 (Fourier Associative Matrix LLM)** e **FSTLLM 2.0**.

Tutti i documenti sono disponibili sia in formato sorgente **Markdown (.md)** che in formato **PDF vettoriale (.pdf)** ad alta definizione tipografica, compilati nativamente con il motore scientifico **Typst**.

---

## 📚 Indice dei Documenti di Progetto 21.0

### 1. Paper Teorico Fondativo: Progetto 21.0 (FAM-LLM)
*Memoria Associativa Matriciale Complessa ($K^\dagger \otimes V$) e Onde di Fourier per il superamento del Capacity Bottleneck nei modelli sub-quadratici.*
- 📄 **PDF:** [`paper_Progetto21_FAM_LLM.pdf`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/paper_Progetto21_FAM_LLM.pdf)
- 📝 **Markdown:** [`Paper_Progetto21_Originale.md`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/Paper_Progetto21_Originale.md)
- ⚙️ **Typst:** [`paper_Progetto21_FAM_LLM.typ`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/paper_Progetto21_FAM_LLM.typ)

---

### 2. Master Paper Architetturale: FSTLLM 2.0 vs Transformers & SSM
*Analisi comparativa approfondita tra Attention Softmax $\mathcal{O}(S^2)$, State Space Models (Mamba) e Fourier Space-Time con Grouped Query Resonance (GQR).*
- 📄 **PDF:** [`paper_FSTLLM_vs_Transformers_SSM.pdf`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/paper_FSTLLM_vs_Transformers_SSM.pdf)
- 📝 **Markdown:** [`paper_FSTLLM_vs_Transformers_SSM.md`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/paper_FSTLLM_vs_Transformers_SSM.md)
- ⚙️ **Typst:** [`paper_FSTLLM_vs_Transformers_SSM.typ`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/paper_FSTLLM_vs_Transformers_SSM.typ)

---

### 3. Report Sperimentale: Valutazione Empirica su TinyStories (30M)
*Benchmark quantitativo di convergenza, throughput di decodifica e scalabilità di memoria con abbattimento certificato di 1170.3x della cache di inferenza e confronto puntuale con il paper di Microsoft Research.*
- 📄 **PDF:** [`paper_FSTLLM_30M_TinyStories_Empirical_Evaluation.pdf`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/paper_FSTLLM_30M_TinyStories_Empirical_Evaluation.pdf)
- 📝 **Markdown:** [`paper_FSTLLM_30M_TinyStories_Empirical_Evaluation.md`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/paper_FSTLLM_30M_TinyStories_Empirical_Evaluation.md)
- ⚙️ **Typst:** [`paper_FSTLLM_30M_TinyStories_Empirical_Evaluation.typ`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/paper_FSTLLM_30M_TinyStories_Empirical_Evaluation.typ)

---

### 4. Piano Industriale & Software: Suite di Agenti AI di Coding a 5 €/Mese
*Strategia di go-to-market bootstrapped da casa, hardening del segreto industriale con moduli C++/.so e cifratura AES-256, architettura di serving su Cloudflare Tunnel / Serverless e unit economics scalabili.*
- 📄 **PDF:** [`piano_software.pdf`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/piano_software.pdf)
- 📝 **Markdown:** [`piano_software.md`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/piano_software.md)
- ⚙️ **Typst:** [`piano_software.typ`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/piano_software.typ)

---

## 🏛️ Letteratura Scientifica di Confronto (nella sottocartella `letteratura_confronto/`)
I paper originali di riferimento scaricati da arXiv:
- [`TinyStories_Small_Language_Models_30M.pdf`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/letteratura_confronto/TinyStories_Small_Language_Models_30M.pdf) (Eldan & Li, Microsoft Research, arXiv:2305.07759)
- [`Mamba_Linear_Time_Sequence_Modeling.pdf`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/letteratura_confronto/Mamba_Linear_Time_Sequence_Modeling.pdf) (Gu & Dao, arXiv:2312.00752)
- [`Mamba2_State_Space_Duality.pdf`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/letteratura_confronto/Mamba2_State_Space_Duality.pdf) (Dao & Gu, arXiv:2405.21060)
- [`RWKV_Reinventing_RNNs_Transformer_Era.pdf`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/letteratura_confronto/RWKV_Reinventing_RNNs_Transformer_Era.pdf) (Peng et al., arXiv:2305.13048)
- [`MobileLLM_Sub_Billion_Models.pdf`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/letteratura_confronto/MobileLLM_Sub_Billion_Models.pdf) (Liu et al., arXiv:2402.14905)
- [`Attention_Is_All_You_Need_Transformer.pdf`](file:///home/giovanni/Desktop/progetti/LLM/SSM_study/paper/letteratura_confronto/Attention_Is_All_You_Need_Transformer.pdf) (Vaswani et al., arXiv:1706.03762)
