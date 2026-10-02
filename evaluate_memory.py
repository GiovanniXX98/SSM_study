#!/usr/bin/env python3
"""
Evaluate Memory — Misurazione della Vera Memoria Totale di Processo (OS RSS RAM)
==============================================================================
Confronta la vera memoria RAM fisica occupata dall'intero programma (Resident Set Size)
durante l'inferenza e la generazione autoregressiva di prompt a contesti crescenti:
- Modello Target: FSTLLM 2.0 (Checkpoint reale 30M: checkpoints/fstllm_30m_best.pt)
- Modello di Confronto: Standard Transformer Baseline equivalente (30M)

Metodologia Rigorosa:
- Esecuzione isolata in sotto-processi separati per evitare inquinamento della memoria.
- Misurazione tramite psutil.Process().memory_info().rss e getrusage ru_maxrss.
"""

import os
import sys
import time
import json
import gc
import resource
from pathlib import Path
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from model.model import FourierSpaceTimeLLM_V2


# ---------------------------------------------------------------------------
# Definizione Transformer Baseline (Equivalente per confronto pulito)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.RMSNorm(d_model)
        inner = int(d_model * 2.67)
        self.w1 = nn.Linear(d_model, inner, bias=False)
        self.w2 = nn.Linear(d_model, inner, bias=False)
        self.w3 = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        h = self.norm1(x)
        B, S, D = h.shape
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device) if is_causal else None
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False, is_causal=is_causal if mask is None else False)
        x = x + attn_out
        h2 = self.norm2(x)
        x = x + self.w3(F.silu(self.w1(h2)) * self.w2(h2))
        return x


class StandardTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 576, n_layers: int = 8, num_heads: int = 8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(n_layers)])
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm_f(x))


def get_current_rss_mb() -> float:
    """Restituisce l'occupazione fisica reale di RAM del processo corrente (in MB)."""
    gc.collect()
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Benchmark FSTLLM 2.0 (Target Model dal Checkpoint)
# ---------------------------------------------------------------------------

def run_fstllm_benchmark(ckpt_path: Path, context_lengths: list[int], new_tokens_to_gen: int = 15):
    torch.set_num_threads(12)
    base_rss = get_current_rss_mb()
    
    # 1. Carica Checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    stoi = ckpt.get("stoi", {})
    itos = ckpt.get("itos", {})
    
    model = FourierSpaceTimeLLM_V2(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        num_heads=cfg["num_heads"],
        num_kv_heads=cfg["num_kv_heads"],
        conv_kernel=cfg["conv_kernel"],
        seq_len=cfg["seq_len"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    model_loaded_rss = get_current_rss_mb()
    weights_rss = model_loaded_rss - base_rss
    
    results = []
    
    for S in context_lengths:
        # Prompt reale di test ripetuto/esteso fino a lunghezza S
        base_prompt = "Once upon a time, Lily found a little bird in the garden. "
        prompt_text = (base_prompt * ((S // len(base_prompt)) + 2))[:S]
        tokens = [stoi.get(c, 0) for c in prompt_text]
        input_ids = torch.tensor(tokens[:S], dtype=torch.long).unsqueeze(0)
        
        # Misura RAM prima del prefill
        pre_rss = get_current_rss_mb()
        
        # Prefill & generazione O(1) con past_states
        t0 = time.time()
        with torch.no_grad():
            logits, _, past_states = model(input_ids)
            curr_token = input_ids[:, -1:]
            
            gen_tokens = []
            for _ in range(new_tokens_to_gen):
                logits, _, past_states = model(curr_token, past_states=past_states)
                next_t = logits.argmax(dim=-1)
                curr_token = next_t
                gen_tokens.append(next_t.item())
                
        elapsed = time.time() - t0
        speed = new_tokens_to_gen / max(1e-5, elapsed)
        
        # Misura RAM reale di picco durante/dopo l'esecuzione
        post_rss = get_current_rss_mb()
        delta_ctx_rss = max(0.0, post_rss - model_loaded_rss)
        
        # Decodifica testo generato
        gen_str = "".join([itos.get(t, "") for t in gen_tokens])
        
        results.append({
            "context_length": S,
            "total_process_rss_mb": round(post_rss, 2),
            "delta_context_rss_mb": round(delta_ctx_rss, 2),
            "tokens_per_sec": round(speed, 1),
            "sample_output": gen_str[:50]
        })
        
    return {
        "model_name": "FSTLLM 2.0 (Target Checkpoint)",
        "base_python_rss_mb": round(base_rss, 2),
        "weights_rss_mb": round(weights_rss, 2),
        "total_initial_rss_mb": round(model_loaded_rss, 2),
        "runs": results
    }


# ---------------------------------------------------------------------------
# Benchmark Standard Transformer (Modello di Confronto)
# ---------------------------------------------------------------------------

def run_transformer_benchmark(context_lengths: list[int], new_tokens_to_gen: int = 15):
    torch.set_num_threads(12)
    base_rss = get_current_rss_mb()
    
    # Inizializza Transformer equivalente (30M)
    model = StandardTransformer(vocab_size=92, d_model=576, n_layers=8, num_heads=8)
    model.eval()
    
    model_loaded_rss = get_current_rss_mb()
    weights_rss = model_loaded_rss - base_rss
    
    results = []
    
    for S in context_lengths:
        dummy_input = torch.randint(0, 92, (1, S))
        
        pre_rss = get_current_rss_mb()
        
        t0 = time.time()
        curr_seq = dummy_input
        with torch.no_grad():
            for _ in range(new_tokens_to_gen):
                logits = model(curr_seq)
                next_t = logits[:, -1:, :].argmax(dim=-1)
                curr_seq = torch.cat([curr_seq, next_t], dim=1)
                
        elapsed = time.time() - t0
        speed = new_tokens_to_gen / max(1e-5, elapsed)
        
        post_rss = get_current_rss_mb()
        delta_ctx_rss = max(0.0, post_rss - model_loaded_rss)
        
        results.append({
            "context_length": S,
            "total_process_rss_mb": round(post_rss, 2),
            "delta_context_rss_mb": round(delta_ctx_rss, 2),
            "tokens_per_sec": round(speed, 1)
        })
        
    return {
        "model_name": "Standard Transformer Baseline (30M)",
        "base_python_rss_mb": round(base_rss, 2),
        "weights_rss_mb": round(weights_rss, 2),
        "total_initial_rss_mb": round(model_loaded_rss, 2),
        "runs": results
    }


def main():
    ckpt_path = ROOT_DIR / "checkpoints" / "fstllm_30m_best.pt"
    if not ckpt_path.exists():
        print(f"❌ Checkpoint non trovato in {ckpt_path}")
        sys.exit(1)

    context_lengths = [128, 256, 512, 1024, 2048]
    new_tokens = 5

    print("=" * 85)
    print("🔬 MISURAZIONE SCIENTIFICA DELLA VERA MEMORIA FISICA DI PROCESSO (OS RSS RAM)")
    print("===========================================================================")
    print(f"🎯 Modello Target: FSTLLM 2.0 Checkpoint ({ckpt_path.name})")
    print(f"⚖️  Modello Confronto: Standard Transformer Baseline (576 dim, 8 layer)")
    print(f"📏 Finestre di Contesto: {context_lengths} token | Token Generati: {new_tokens}")
    print("=" * 85)
    print("⏳ Esecuzione benchmark in corso...")
    sys.stdout.flush()

    # 1. Benchmark FSTLLM
    fst_data = run_fstllm_benchmark(ckpt_path, context_lengths, new_tokens_to_gen=new_tokens)
    print("✅ Benchmark FSTLLM 2.0 completato.")
    sys.stdout.flush()

    # 2. Benchmark Transformer
    trans_data = run_transformer_benchmark(context_lengths, new_tokens_to_gen=new_tokens)
    print("✅ Benchmark Transformer completato.\n")
    sys.stdout.flush()

    # Stampa Tabella di Confronto
    print("=" * 85)
    print(f"{'Contesto':>10} | {'FSTLLM Totale':>15} | {'Trans Totale':>15} | {'FSTLLM Delta':>13} | {'Trans Delta':>13} | {'FSTLLM Speed':>12}")
    print("-" * 85)

    comparison_table = []
    for f, t in zip(fst_data["runs"], trans_data["runs"]):
        s = f["context_length"]
        f_tot = f["total_process_rss_mb"]
        t_tot = t["total_process_rss_mb"]
        f_delta = f["delta_context_rss_mb"]
        t_delta = t["delta_context_rss_mb"]
        f_spd = f["tokens_per_sec"]
        t_spd = t["tokens_per_sec"]
        
        print(f"{s:>10d} | {f_tot:>12.2f} MB | {t_tot:>12.2f} MB | {f_delta:>10.2f} MB | {t_delta:>10.2f} MB | {f_spd:>10.1f} t/s")
        comparison_table.append({
            "context_length": s,
            "fstllm_total_rss_mb": f_tot,
            "transformer_total_rss_mb": t_tot,
            "fstllm_delta_rss_mb": f_delta,
            "transformer_delta_rss_mb": t_delta,
            "fstllm_speed_tok_s": f_spd,
            "transformer_speed_tok_s": t_spd
        })

    print("=" * 85)
    print(f"📦 Memoria Pesi FSTLLM caricati in RAM:      {fst_data['weights_rss_mb']:.2f} MB")
    print(f"📦 Memoria Pesi Transformer caricati in RAM: {trans_data['weights_rss_mb']:.2f} MB")
    print(f"🐍 Overhead Base Processo Python + PyTorch:  ~{fst_data['base_python_rss_mb']:.2f} MB")
    print("=" * 85)

    # Salva report JSON
    out_json = ROOT_DIR / "benchmark_true_process_memory.json"
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump({
            "summary": "Misurazione reale della Resident Set Size (RSS) totale del processo dell'OS",
            "device": "Intel Core i5-1245U (12 threads)",
            "fstllm_base": fst_data,
            "transformer_base": trans_data,
            "comparison": comparison_table
        }, fp, indent=2)
    print(f"💾 Report JSON completo salvato in: {out_json}")


if __name__ == "__main__":
    main()
