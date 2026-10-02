"""
Fourier Associative Matrix Language Model (FAM-LLM) — Progetto 21.0
====================================================================
Superamento definitivo del Memory Bottleneck nei modelli sub-quadratici:
1. Memoria Associativa Matriciale Complessa (K^T * V in C^{HD x HD}) per ciascuna testa.
2. Modulazione Spazio-Temporale delle Fasi di Fourier (Carrier e^{i theta}).
3. Dual-Mode Execution:
   - Training: Chunked Parallel Scan ultra-rapido (O(N) anziché O(N^2)).
   - Inferenza: Recurrent Step O(1) costante tramite Matrix State Caching.
4. Gated SwiGLU Feed-Forward Network per massima capacità espressiva.
5. Vocabolario standard GPT-2 (50.257 token).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLUFeedForward(nn.Module):
    """FFN SwiGLU (standard LLaMA/Mistral) per il channel mixing semantico."""

    def __init__(self, d_model: int, expansion: float = 2.67):
        super().__init__()
        inner = int(d_model * expansion)
        self.w1 = nn.Linear(d_model, inner, bias=False)
        self.w2 = nn.Linear(d_model, inner, bias=False)  # Gating
        self.w3 = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ---------------------------------------------------------------------------
# Layer Fourier Associative Matrix (FAM)
# ---------------------------------------------------------------------------

class FourierAssociativeMatrixLayer(nn.Module):
    """
    Core Layer del Progetto 21.0:
    - Espande lo stato ricorsivo da un semplice vettore 1D a una vera MATRICE associativa
      K^T * V in C^{HD x HD}.
    - Modula le chiavi e le query con le onde di fase di Fourier (e^{i theta}).
    - Supporta inferenza O(1) continua e training chunked parallelo.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 6,
        num_kv_heads: int = 2,
        chunk_size: int = 32,
        conv_kernel: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.chunk_size = chunk_size

        assert d_model % num_heads == 0, f"d_model ({d_model}) deve essere divisibile per num_heads ({num_heads})"
        assert num_heads % num_kv_heads == 0, f"num_heads ({num_heads}) deve essere multiplo di num_kv_heads ({num_kv_heads})"
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim

        # 1. Depthwise Causal Conv1D per mescolamento sintattico locale
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel,
            groups=d_model,
            padding=conv_kernel - 1,
        )

        # 2. Proiezioni Lineari (Q, K, V, Gate, Out)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # 3. Parametri Spazio-Temporali di Fourier (Frequenze e Decadimento)
        self.phase_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.decay_proj = nn.Linear(d_model, self.num_kv_heads, bias=True)
        nn.init.constant_(self.decay_proj.bias, 4.0)  # Decay a lungo termine (~0.982 per ritenere contesti estesi)

        # Frequenze armoniche base distribuite logaritmicamente
        freqs = torch.linspace(0.01, math.pi, self.head_dim)
        self.register_buffer("base_freqs", freqs.view(1, 1, 1, self.head_dim))

    def forward(self, x: torch.Tensor, past_state: dict = None) -> tuple[torch.Tensor, dict]:
        """
        Supporta due modalità:
        - Se past_state è fornito (Inference O(1)): calcolo ricorsivo step-by-step su 1 token.
        - Se past_state è None (Training): calcolo parallelo a blocchi (Chunked Scan).
        """
        B, S, D = x.shape
        H = self.num_heads
        KV_H = self.num_kv_heads
        HD = self.head_dim
        k_len = self.conv1d.kernel_size[0] if isinstance(self.conv1d.kernel_size, (tuple, list)) else self.conv1d.kernel_size

        # 1. Convoluzione Causale 1D Locale
        if past_state is not None:
            conv_state = past_state['conv_state']  # (B, D, K-1)
            x_cat = torch.cat([conv_state, x.transpose(1, 2)], dim=-1)
            x_conv = F.silu(F.conv1d(x_cat, self.conv1d.weight, self.conv1d.bias, groups=self.conv1d.groups))
            new_conv_state = x_cat[..., 1:]
            x_conv = x_conv.transpose(1, 2)  # (B, 1, D)
        else:
            x_conv = F.silu(self.conv1d(x.transpose(1, 2))[..., :S]).transpose(1, 2)
            if S >= k_len - 1:
                new_conv_state = x.transpose(1, 2)[..., -(k_len - 1):]
            else:
                new_conv_state = F.pad(x.transpose(1, 2), (k_len - 1 - S, 0))

        # 2. Proiezioni
        q = self.q_proj(x_conv).view(B, S, H, HD)
        k = self.k_proj(x_conv).view(B, S, KV_H, HD)
        v = self.v_proj(x_conv).view(B, S, KV_H, HD)
        gate = F.silu(self.gate_proj(x_conv))

        d_phase = torch.tanh(self.phase_proj(x_conv).view(B, S, KV_H, HD)) * math.pi
        decay = torch.sigmoid(self.decay_proj(x_conv)).view(B, S, KV_H, 1, 1)

        # 3. Onde di Fase di Fourier
        if past_state is not None:
            phase = past_state['phase_cum'] + self.base_freqs + d_phase
            new_phase_cum = phase
        else:
            phase = torch.cumsum(self.base_freqs + d_phase, dim=1)
            new_phase_cum = phase[:, -1:]

        carrier_pos = torch.polar(torch.ones_like(phase), phase)
        carrier_neg = torch.conj(carrier_pos)

        # Modulazione spettrale delle chiavi e dei valori
        k_c = k.to(torch.cfloat) * carrier_neg
        v_c = v.to(torch.cfloat)

        carrier_pos_exp = carrier_pos.repeat_interleave(self.num_queries_per_kv, dim=2)
        q_c = q.to(torch.cfloat) * carrier_pos_exp

        # 4. Memoria Associativa Matriciale (K^T * V in C^{HD x HD})
        if past_state is not None:
            # -------------------------------------------------------------
            # MODALITÀ INFERENZA O(1): Recurrent Matrix Update
            # -------------------------------------------------------------
            matrix_state = past_state['matrix_state']  # (B, KV_H, HD, HD)
            
            # Outer Product: K^T * V (B, KV_H, HD, HD)
            m_t = torch.einsum('bhi,bhj->bhij', k_c.squeeze(1), v_c.squeeze(1))
            
            # Aggiornamento matriciale della memoria associativa
            d_step = decay.squeeze(1)  # (B, KV_H, 1, 1)
            new_matrix_state = matrix_state * d_step + (1.0 - d_step) * m_t

            # Grouped Query Expansion dello stato matriciale
            state_exp = new_matrix_state.repeat_interleave(self.num_queries_per_kv, dim=1)  # (B, H, HD, HD)

            # Query Readout: Q * S -> Y (B, H, HD)
            y_head = torch.einsum('bhi,bhij->bhj', q_c.squeeze(1), state_exp).real
            y = y_head.unsqueeze(1).contiguous().view(B, 1, D)

        else:
            # -------------------------------------------------------------
            # MODALITÀ TRAINING: Chunked Parallel Scan (100% Unificato all'Inferenza O(1))
            # -------------------------------------------------------------
            C = self.chunk_size
            pad_len = (C - (S % C)) % C
            if pad_len > 0:
                k_c_pad = F.pad(k_c, (0, 0, 0, 0, 0, pad_len))
                v_c_pad = F.pad(v_c, (0, 0, 0, 0, 0, pad_len))
                q_c_pad = F.pad(q_c, (0, 0, 0, 0, 0, pad_len))
                decay_pad = F.pad(decay, (0, 0, 0, 0, 0, 0, 0, pad_len))
            else:
                k_c_pad, v_c_pad, q_c_pad, decay_pad = k_c, v_c, q_c, decay

            S_pad = k_c_pad.shape[1]
            M = S_pad // C

            k_chunks = k_c_pad.view(B, M, C, KV_H, HD).permute(0, 3, 1, 2, 4)
            v_chunks = v_c_pad.view(B, M, C, KV_H, HD).permute(0, 3, 1, 2, 4)
            q_chunks = q_c_pad.view(B, M, C, H, HD).permute(0, 3, 1, 2, 4)
            decay_chunks = decay_pad.view(B, M, C, KV_H).permute(0, 3, 1, 2)  # (B, KV_H, M, C)

            mat_state = torch.zeros(B, KV_H, HD, HD, device=x.device, dtype=torch.cfloat)
            y_chunks_list = []

            k_exp = k_chunks.repeat_interleave(self.num_queries_per_kv, dim=1)
            v_exp = v_chunks.repeat_interleave(self.num_queries_per_kv, dim=1)

            for m in range(M):
                # 1. Matrice di decadimento intra-chunk per ciascun token j >= i: prod_{r=i+1}^j gamma_r
                d_m = decay_chunks[:, :, m]  # (B, KV_H, C)
                log_gamma = torch.log(d_m.clamp(min=1e-6))
                log_cum = torch.cumsum(log_gamma, dim=-1)  # (B, KV_H, C)
                log_diff = log_cum.unsqueeze(-1) - log_cum.unsqueeze(-2)  # (B, KV_H, C, C) [j, i]
                causal_mask = torch.tril(torch.ones(C, C, device=x.device, dtype=torch.bool))
                decay_matrix = torch.exp(log_diff).masked_fill(~causal_mask, 0.0)  # (B, KV_H, C, C)

                decay_matrix_exp = decay_matrix.repeat_interleave(self.num_queries_per_kv, dim=1)  # (B, H, C, C)
                one_minus_d = (1.0 - d_m).unsqueeze(-2).repeat_interleave(self.num_queries_per_kv, dim=1)  # (B, H, 1, C)

                # A) Intra-chunk Readout: Q_j * [ sum_{i=1}^j (K_i^T V_i) (1-gamma_i) prod_{r=i+1}^j gamma_r ]
                sim = (torch.matmul(q_chunks[:, :, m], k_exp[:, :, m].transpose(-2, -1))).real  # (B, H, C, C)
                weighted_sim = sim * one_minus_d * decay_matrix_exp  # (B, H, C, C)
                y_intra_m = torch.matmul(weighted_sim.to(torch.cfloat), v_exp[:, :, m]).real  # (B, H, C, HD)

                # B) Inter-chunk Readout: Q_j * [ S_{m-1} * prod_{r=1}^j gamma_r ]
                gamma_cum_j = torch.exp(log_cum).repeat_interleave(self.num_queries_per_kv, dim=1)  # (B, H, C)
                state_exp = mat_state.repeat_interleave(self.num_queries_per_kv, dim=1)  # (B, H, HD, HD)
                q_s = torch.matmul(q_chunks[:, :, m], state_exp).real
                y_inter_m = q_s * gamma_cum_j.unsqueeze(-1)

                y_m = y_intra_m + y_inter_m
                y_chunks_list.append(y_m)

                # Aggiornamento mat_state per il chunk successivo m
                d_m_kv = d_m.unsqueeze(-1)  # (B, KV_H, C, 1)
                km = k_chunks[:, :, m]
                vm = v_chunks[:, :, m]
                decay_last = decay_matrix[:, :, -1, :].unsqueeze(-1)  # (B, KV_H, C, 1)
                weight_i = (1.0 - d_m_kv) * decay_last
                delta_mat = torch.matmul((km * weight_i).transpose(-2, -1), vm)  # (B, KV_H, HD, HD)
                prod_gamma_all = decay_matrix[:, :, -1, 0:1].unsqueeze(-1)  # prod_{r=1}^C gamma_r
                mat_state = mat_state * prod_gamma_all + delta_mat

            new_matrix_state = mat_state
            y_fused = torch.stack(y_chunks_list, dim=2).permute(0, 2, 3, 1, 4).reshape(B, S_pad, D)
            y = y_fused[:, :S, :] if pad_len > 0 else y_fused

        # Output Gating e Proiezione
        out = self.out_proj(y * gate)

        next_state = {
            'conv_state': new_conv_state,
            'matrix_state': new_matrix_state,
            'phase_cum': new_phase_cum
        }
        return out, next_state


# ---------------------------------------------------------------------------
# Blocco Integrato FAM (RMSNorm + FAM Layer + RMSNorm + SwiGLU)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class FAMBlock(nn.Module):
    """Un blocco unificato residuo con RMSNorm, FAM Layer e SwiGLU FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 6,
        num_kv_heads: int = 2,
        chunk_size: int = 32,
        expansion: float = 2.67,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.fam = FourierAssociativeMatrixLayer(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            chunk_size=chunk_size,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, expansion=expansion)

    def forward(self, x: torch.Tensor, past_state: dict = None) -> tuple[torch.Tensor, dict]:
        # 1. Fourier Associative Matrix Layer
        h, next_state = self.fam(self.norm1(x), past_state=past_state)
        x = x + h
        # 2. SwiGLU FFN
        x = x + self.ffn(self.norm2(x))
        return x, next_state


# ---------------------------------------------------------------------------
# Modello Linguistico Completo: FourierAssociativeMatrixLLM
# ---------------------------------------------------------------------------

class FourierAssociativeMatrixLLM(nn.Module):
    """
    Architettura Completa Progetto 21.0:
    - Embedding token con vocabolario GPT-2 (50.257)
    - N layer FAMBlock con memoria associativa matriciale e modulazione d'onda
    - RMSNorm finale e LM Head per predizione dei token
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 512,
        n_layers: int = 8,
        num_heads: int = 8,
        num_kv_heads: int = 2,
        chunk_size: int = 32,
    ):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": n_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "chunk_size": chunk_size,
        }
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        # Token Embedding
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # Stack di Blocchi FAM
        self.layers = nn.ModuleList([
            FAMBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                chunk_size=chunk_size,
            )
            for _ in range(n_layers)
        ])

        # Normalizzazione Finale e Testa di Predizione
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight Tying (come nei moderni LLM per ottimizzare la convergenza)
        self.lm_head.weight = self.tok_emb.weight

        # Inizializzazione pesi
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
        past_states: list = None,
        return_hidden_states: bool = False,
    ) -> tuple:
        """
        Forward pass per training ed inferenza.
        """
        B, S = idx.shape
        x = self.tok_emb(idx)

        hidden_states = [x] if return_hidden_states else None
        new_states = []
        for i, layer in enumerate(self.layers):
            layer_past = past_states[i] if past_states is not None else None
            x, next_layer_state = layer(x, past_state=layer_past)
            new_states.append(next_layer_state)
            if return_hidden_states:
                hidden_states.append(x)

        x_norm = self.norm_f(x)
        logits = self.lm_head(x_norm)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if return_hidden_states:
            return logits, loss, new_states, hidden_states
        return logits, loss, new_states

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_k: int = 40,
        eos_token_id: int | None = 50256,
    ) -> list[int]:
        """
        Generazione O(1) State-Space: prefill del prompt ed emissione token-per-token
        sfruttando la Matrix State Cache senza mai ricalcolare il passato.
        """
        self.eval()
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)

        logits, _, past_states = self(prompt_tokens)
        generated_ids = []

        # 2. Generazione Recurrente O(1)
        curr_logits = logits[0, -1, :]
        for _ in range(max_new_tokens):
            next_token_logits = curr_logits / max(0.01, temperature)

            # Filtro Top-K
            if top_k > 0:
                vals, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < vals[-1]] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            token_val = next_token.item()
            if eos_token_id is not None and token_val == eos_token_id:
                break
            generated_ids.append(token_val)
            curr_token = next_token.unsqueeze(0)

            logits, _, past_states = self(curr_token, past_states=past_states)
            curr_logits = logits[0, -1, :]

        return generated_ids

    @torch.no_grad()
    def generate_stream(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 40,
        eos_token_id: int | None = 50256,
    ):
        """
        Generatore Streamer O(1): yielda i token generati uno per volta in tempo reale.
        """
        self.eval()
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)

        logits, _, past_states = self(prompt_tokens)
        curr_logits = logits[0, -1, :]

        for _ in range(max_new_tokens):
            next_token_logits = curr_logits / max(0.01, temperature)
            if top_k > 0:
                vals, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < vals[-1]] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_val = next_token.item()

            if eos_token_id is not None and token_val == eos_token_id:
                break

            yield token_val

            curr_token = next_token.unsqueeze(0)
            logits, _, past_states = self(curr_token, past_states=past_states)
            curr_logits = logits[0, -1, :]
