"""
Fourier Space-Time LLM (FSTLLM) 2.0
===================================
Versione 2.0 con migliorie architetturali massive:
1. Caching Olografico per Inferenza O(1) autoregressiva.
2. Memoria Locale (Depthwise Conv1D) pre-proiezione.
3. Grouped-Query Resonance (GQR) per alta efficienza parametrica.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FSTLLM 2.0 - Selective Fourier Wave Layer
# ---------------------------------------------------------------------------

class SelectiveFourierWaveLayerV2(nn.Module):
    """
    FSTLLM V2 Layer: include Conv1D locale, GQR e Caching per inferenza.
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int = 256,
        num_heads: int = 8,
        num_kv_heads: int = 2,  # GQR: meno teste per Memoria/Phase rispetto alle Query
        decay_init: float = 0.5,
        conv_kernel: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        
        assert num_heads % num_kv_heads == 0, "num_heads deve essere divisibile per num_kv_heads"
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim

        # 1. Convoluzione Locale Depthwise
        self.conv1d = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=conv_kernel, 
            groups=d_model, 
            padding=conv_kernel - 1
        )
        #qui genero le projezioni
        # 2. Proiezioni GQR
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Modulazione Phase & Decay basata sulle KV Heads
        self.phase_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.decay_proj = nn.Linear(d_model, self.num_kv_heads, bias=True)
        nn.init.constant_(self.decay_proj.bias, 2.0)

        # Frequenze armoniche pari al numero di teste
        freqs = torch.linspace(0.01, math.pi, self.head_dim)
        self.register_buffer("base_freqs", freqs.view(1, 1, 1, self.head_dim))

    def forward(self, x: torch.Tensor, past_state: dict = None) -> tuple[torch.Tensor, dict]:
        """
        past_state dict keys:
        - 'conv_state': buffer (B, D, kernel_size-1)
        - 'fourier_state': accumulatore complesso (B, KV_H, HD)
        - 'phase_cum': fase cumulativa precedente (B, KV_H, 1, HD)
        """
        B, S, D = x.shape
        H = self.num_heads
        KV_H = self.num_kv_heads
        HD = self.head_dim

        # 1. Depthwise Conv1D
        k_len = self.conv1d.kernel_size[0] if isinstance(self.conv1d.kernel_size, (tuple, list)) else self.conv1d.kernel_size
        if past_state is not None:
            # Inferenza step-by-step (S=1)
            conv_state = past_state['conv_state'] # (B, D, K-1)
            x_conv = torch.cat([conv_state, x.transpose(1, 2)], dim=-1) # (B, D, K)
            x_out = F.silu(F.conv1d(x_conv, self.conv1d.weight, self.conv1d.bias, groups=self.conv1d.groups)) # (B, D, 1)
            new_conv_state = x_conv[..., 1:] # Aggiorna la coda (B, D, K-1)
            x_conv = x_out.transpose(1, 2) # (B, 1, D)
        else:
            # Training parallelo (Prefix-scan)
            x_conv = F.silu(self.conv1d(x.transpose(1, 2))[..., :S]).transpose(1, 2)
            if S >= k_len - 1:
                new_conv_state = x.transpose(1, 2)[..., -(k_len - 1):]
            else:
                new_conv_state = F.pad(x.transpose(1, 2), (k_len - 1 - S, 0))

        # 2. Proiezioni (Q è su H teste, K/V su KV_H teste)
        q = self.q_proj(x_conv).view(B, S, H, HD)
        k = self.k_proj(x_conv).view(B, S, KV_H, HD)
        v = self.v_proj(x_conv).view(B, S, KV_H, HD)
        gate = F.silu(self.gate_proj(x_conv))

        d_phase = torch.tanh(self.phase_proj(x_conv).view(B, S, KV_H, HD)) * math.pi
        decay = torch.sigmoid(self.decay_proj(x_conv)).view(B, S, KV_H, 1)

        # 3. Gestione della Fase
        if past_state is not None:
            phase = past_state['phase_cum'] + self.base_freqs + d_phase
            new_phase_cum = phase
        else:
            phase = torch.cumsum(self.base_freqs + d_phase, dim=1)
            new_phase_cum = phase[:, -1:]

        carrier_pos = torch.polar(torch.ones_like(phase), phase)
        carrier_neg = torch.conj(carrier_pos)

        # 4. Modulazione KV e Aggiornamento Stato
        #puoi usare i complessi a 32bit
        k_complex = k.to(torch.cfloat) * carrier_neg
        v_complex = v.to(torch.cfloat)
        kv = k_complex * v_complex

        if past_state is not None:
            # INFERENZA O(1): Recurrent Step
            f_state = past_state['fourier_state']
            f_state = f_state * decay.squeeze(1) + (1.0 - decay.squeeze(1)) * kv.squeeze(1)
            new_fourier_state = f_state
            
            # GQR: Espansione dello stato per pareggiare le Query
            state_expanded = f_state.unsqueeze(1).repeat_interleave(self.num_queries_per_kv, dim=2)
            carrier_pos_expanded = carrier_pos.repeat_interleave(self.num_queries_per_kv, dim=2)
            
            q_c_p = (q.to(torch.cfloat) * carrier_pos_expanded)
            y = (q_c_p * state_expanded).real
        else:
            # TRAINING O(N log N): Parallel Prefix-Scan Matrix (PyTorch sim)
            log_d = torch.log(decay.clamp(min=1e-4, max=1.0 - 1e-4)).squeeze(-1).permute(0, 2, 1)
            cum_log_d = torch.cumsum(log_d, dim=-1)
            
            log_decay_mat = cum_log_d.unsqueeze(-1) - cum_log_d.unsqueeze(-2)
            mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
            log_decay_mat = log_decay_mat.masked_fill(~mask, -1e9)
            decay_mat = torch.exp(log_decay_mat).to(torch.cfloat)
            
            kv_weighted = ((1.0 - decay) * kv).permute(0, 2, 1, 3)
            f_state = torch.matmul(decay_mat, kv_weighted) # (B, KV_H, S, HD)
            
            new_fourier_state = f_state[:, :, -1, :] # Ultimo step
            
            # GQR: Espansione per Query
            state_expanded = f_state.permute(0, 2, 1, 3).repeat_interleave(self.num_queries_per_kv, dim=2)
            carrier_pos_expanded = carrier_pos.repeat_interleave(self.num_queries_per_kv, dim=2)
            
            q_c_p = (q.to(torch.cfloat) * carrier_pos_expanded)
            y = (q_c_p * state_expanded).real

        # Formattazione Output
        y = y.contiguous().view(B, S, D)
        out = self.out_proj(y * gate)

        next_state = {
            'conv_state': new_conv_state,
            'fourier_state': new_fourier_state,
            'phase_cum': new_phase_cum
        }
        return out, next_state


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: float = 2.67):
        super().__init__()
        inner = int(d_model * expansion)
        self.w1 = nn.Linear(d_model, inner, bias=False)
        self.w2 = nn.Linear(d_model, inner, bias=False)
        self.w3 = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ---------------------------------------------------------------------------
# Blocco Fourier V2
# ---------------------------------------------------------------------------

class FourierSpaceTimeBlockV2(nn.Module):
    def __init__(self, d_model: int, seq_len: int, num_heads: int, num_kv_heads: int, conv_kernel: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.fourier = SelectiveFourierWaveLayerV2(
            d_model=d_model,
            seq_len=seq_len,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            conv_kernel=conv_kernel
        )
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model=d_model)

    def forward(self, x: torch.Tensor, past_state: dict = None) -> tuple[torch.Tensor, dict]:
        res, next_state = self.fourier(self.norm1(x), past_state)
        x = x + res
        x = x + self.ffn(self.norm2(x))
        return x, next_state


# ---------------------------------------------------------------------------
# Modello Principale FSTLLM 2.0
# ---------------------------------------------------------------------------

class FourierSpaceTimeLLM_V2(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 384, n_layers: int = 6, 
                 seq_len: int = 256, num_heads: int = 8, num_kv_heads: int = 2, conv_kernel: int = 4):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            FourierSpaceTimeBlockV2(d_model, seq_len, num_heads, num_kv_heads, conv_kernel)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.RMSNorm(d_model)
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, past_states: list = None) -> tuple[torch.Tensor, torch.Tensor, list]:
        x = self.token_emb(idx)
        next_states = []
        
        for i, block in enumerate(self.blocks):
            p_state = past_states[i] if past_states is not None else None
            x, n_state = block(x, p_state)
            next_states.append(n_state)

        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss, next_states

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def from_config(config: dict) -> "FourierSpaceTimeLLM_V2":
        return FourierSpaceTimeLLM_V2(
            vocab_size=config["vocab_size"],
            d_model=config.get("d_model", 384),
            n_layers=config.get("n_layers", 6),
            seq_len=config.get("seq_len", 256),
            num_heads=config.get("num_heads", 8),
            num_kv_heads=config.get("num_kv_heads", 2),
            conv_kernel=config.get("conv_kernel", 4)
        )
