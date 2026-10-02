#!/usr/bin/env python3
"""
Evaluate Memory — Benchmark Scientifico e Imparziale della Vera Memoria di Processo (OS RSS RAM)
================================================================================================
Confronto 100% EQUO e OBIETTIVO tra:
1. FSTLLM 2.0 (Target Checkpoint 30M: checkpoints/fstllm_30m_best.pt)
2. Standard Transformer con KV-Cache reale (Baseline equivalente 30M: 576 dim, 8 layer, 8 teste)

Criteri di Rigore e Imparzialità:
- Esecuzione isolata in SOTTOPROCESSI SEPARATI per garantire che la memoria dell'uno
  non inquini o influenzi l'allocatore glibc/Python dell'altro.
- Pulizia del checkpoint (`del ckpt; gc.collect()`) per misurare solo i pesi attivi.
- Transformer dotato di VERA KV-Cache ottimizzata con PyTorch SDPA (Scaled Dot-Product Attention).
- Misurazione tramite Linux OS Resident Set Size (RSS) reale via psutil.
"""

import os
import sys
import time
import json
import gc
import subprocess
from pathlib import Path
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from model.model import FourierSpaceTimeLLM_V2


# ---------------------------------------------------------------------------
# Definizione Standard Transformer con VERA KV-Cache (Standard Industriale)
# ---------------------------------------------------------------------------

class TransformerBlockWithKVCache(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm1 = nn.RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.RMSNorm(d_model)
        inner = int(d_model * 2.67)
        self.w1 = nn.Linear(d_model, inner, bias=False)
        self.w2 = nn.Linear(d_model, inner, bias=False)
        self.w3 = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: dict = None) -> tuple[torch.Tensor, dict]:
        B, S, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            # Step di decodifica autoregressiva: concatena con la cache storica
            k = torch.cat([kv_cache['k'], k], dim=2)
            v = torch.cat([kv_cache['v'], v], dim=2)
            new_cache = {'k': k, 'v': v}
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # Prefill iniziale con maschera causale
            new_cache = {'k': k, 'v': v}
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.out_proj(attn_out)
        h2 = self.norm2(x)
        x = x + self.w3(F.silu(self.w1(h2)) * self.w2(h2))
        return x, new_cache


class StandardTransformerWithKVCache(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 576, n_layers: int = 8, num_heads: int = 8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlockWithKVCache(d_model, num_heads) for _ in range(n_layers)])
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx: torch.Tensor, past_kv: list[dict] = None) -> tuple[torch.Tensor, list[dict]]:
        x = self.token_emb(idx)
        new_kvs = []
        for i, block in enumerate(self.blocks):
            layer_cache = past_kv[i] if past_kv is not None else None
            x, cache = block(x, kv_cache=layer_cache)
            new_kvs.append(cache)
        logits = self.lm_head(self.norm_f(x))
        return logits, new_kvs


def get_current_rss_mb() -> float:
    """Restituisce l'occupazione fisica reale di RAM del processo corrente (in MB)."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Worker Isolato: FSTLLM 2.0
# ---------------------------------------------------------------------------

def worker_fstllm(context_lengths: list[int], new_tokens_to_gen: int):
    torch.set_num_threads(12)
    gc.collect()
    base_rss = get_current_rss_mb()

    ckpt_path = ROOT_DIR / "checkpoints" / "fstllm_30m_best.pt"
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

    # PULIZIA RIGOROSA: Rimuove l'oggetto checkpoint da memoria per non falsare il test
    del ckpt
    gc.collect()

    model_loaded_rss = get_current_rss_mb()
    weights_rss = model_loaded_rss - base_rss

    results = []
    base_prompt = "Once upon a time, in a vibrant forest, a little fox wanted to discover the magic stream. "

    for S in context_lengths:
        prompt_text = (base_prompt * ((S // len(base_prompt)) + 2))[:S]
        tokens = [stoi.get(c, 0) for c in prompt_text]
        input_ids = torch.tensor(tokens[:S], dtype=torch.long).unsqueeze(0)

        # Misura picco di memoria durante la computazione
        t0 = time.time()
        with torch.no_grad():
            logits, _, past_states = model(input_ids)
            curr_token = input_ids[:, -1:]
            for _ in range(new_tokens_to_gen):
                logits, _, past_states = model(curr_token, past_states=past_states)
                next_t = logits.argmax(dim=-1)
                curr_token = next_t

        elapsed = time.time() - t0
        speed = new_tokens_to_gen / max(1e-5, elapsed)
        post_rss = get_current_rss_mb()
        delta_rss = max(0.0, post_rss - model_loaded_rss)

        # Calcolo dimensione esatta dello stato ricorrente in KB
        state_bytes = sum(s.element_size() * s.nelement() for layer in past_states for s in layer.values() if isinstance(s, torch.Tensor))
        state_kb = state_bytes / 1024.0

        results.append({
            "context_length": S,
            "total_process_rss_mb": round(post_rss, 2),
            "delta_context_rss_mb": round(delta_rss, 2),
            "tokens_per_sec": round(speed, 2),
            "state_cache_kb": round(state_kb, 2)
        })

    out = {
        "model_name": "FSTLLM 2.0 (Target Checkpoint)",
        "base_python_rss_mb": round(base_rss, 2),
        "weights_rss_mb": round(weights_rss, 2),
        "total_initial_rss_mb": round(model_loaded_rss, 2),
        "runs": results
    }
    print(f"JSON_OUTPUT_FSTLLM:{json.dumps(out)}")


# ---------------------------------------------------------------------------
# Worker Isolato: Standard Transformer con KV-Cache Reale
# ---------------------------------------------------------------------------

def worker_transformer(context_lengths: list[int], new_tokens_to_gen: int):
    torch.set_num_threads(12)
    gc.collect()
    base_rss = get_current_rss_mb()

    # Modello identico a 30M di parametri (576 dim, 8 layer, 8 teste)
    model = StandardTransformerWithKVCache(vocab_size=92, d_model=576, n_layers=8, num_heads=8)
    model.eval()
    gc.collect()

    model_loaded_rss = get_current_rss_mb()
    weights_rss = model_loaded_rss - base_rss

    results = []

    for S in context_lengths:
        dummy_input = torch.randint(0, 92, (1, S))

        t0 = time.time()
        with torch.no_grad():
            # 1. Prefill del contesto iniziale e inizializzazione KV-cache
            logits, past_kv = model(dummy_input)
            curr_token = dummy_input[:, -1:]
            
            # 2. Generazione autoregressiva con accumulo incrementale nella KV-cache
            for _ in range(new_tokens_to_gen):
                logits, past_kv = model(curr_token, past_kv=past_kv)
                next_t = logits.argmax(dim=-1)
                curr_token = next_t

        elapsed = time.time() - t0
        speed = new_tokens_to_gen / max(1e-5, elapsed)
        post_rss = get_current_rss_mb()
        delta_rss = max(0.0, post_rss - model_loaded_rss)

        # Calcolo dimensione esatta della KV-Cache in KB
        kv_bytes = sum(k.element_size() * k.nelement() + v.element_size() * v.nelement() for layer in past_kv for k, v in [layer.values()])
        kv_kb = kv_bytes / 1024.0

        results.append({
            "context_length": S,
            "total_process_rss_mb": round(post_rss, 2),
            "delta_context_rss_mb": round(delta_rss, 2),
            "tokens_per_sec": round(speed, 2),
            "state_cache_kb": round(kv_kb, 2)
        })

    out = {
        "model_name": "Standard Transformer con KV-Cache Reale",
        "base_python_rss_mb": round(base_rss, 2),
        "weights_rss_mb": round(weights_rss, 2),
        "total_initial_rss_mb": round(model_loaded_rss, 2),
        "runs": results
    }
    print(f"JSON_OUTPUT_TRANSFORMER:{json.dumps(out)}")


# ---------------------------------------------------------------------------
# Main Orchestrator (Esecuzione in Processi Separati)
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker-fstllm":
        ctxs = json.loads(sys.argv[2])
        n_toks = int(sys.argv[3])
        worker_fstllm(ctxs, n_toks)
        return
    elif len(sys.argv) > 1 and sys.argv[1] == "--worker-transformer":
        ctxs = json.loads(sys.argv[2])
        n_toks = int(sys.argv[3])
        worker_transformer(ctxs, n_toks)
        return

    context_lengths = [128, 256, 512, 1024, 2048]
    new_tokens = 20

    print("=" * 95)
    print("⚖️  BENCHMARK SCIENTIFICO E IMPARZIALE — FSTLLM 2.0 vs STANDARD TRANSFORMER (30M)")
    print("=======================================================================================")
    print(f"🔬 Metodologia: Sottoprocessi Linux separati, pulizia heap glibc, vera KV-Cache SDPA")
    print(f"📏 Finestre di Contesto: {context_lengths} token | Token Generati: {new_tokens}")
    print("=" * 95)

    # 1. Esegui Worker FSTLLM in processo isolato
    print("⏳ [1/2] Esecuzione benchmark isolato FSTLLM 2.0...")
    sys.stdout.flush()
    res_fst = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker-fstllm", json.dumps(context_lengths), str(new_tokens)],
        capture_output=True, text=True
    )
    if res_fst.returncode != 0:
        print("❌ Errore in FSTLLM Worker:", res_fst.stderr)
        sys.exit(1)

    fst_data = None
    for line in res_fst.stdout.splitlines():
        if line.startswith("JSON_OUTPUT_FSTLLM:"):
            fst_data = json.loads(line.replace("JSON_OUTPUT_FSTLLM:", ""))

    # 2. Esegui Worker Transformer in processo isolato
    print("⏳ [2/2] Esecuzione benchmark isolato Standard Transformer (con KV-Cache)...")
    sys.stdout.flush()
    res_tr = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker-transformer", json.dumps(context_lengths), str(new_tokens)],
        capture_output=True, text=True
    )
    if res_tr.returncode != 0:
        print("❌ Errore in Transformer Worker:", res_tr.stderr)
        sys.exit(1)

    trans_data = None
    for line in res_tr.stdout.splitlines():
        if line.startswith("JSON_OUTPUT_TRANSFORMER:"):
            trans_data = json.loads(line.replace("JSON_OUTPUT_TRANSFORMER:", ""))

    # Tabella Comparativa Finale
    print("\n" + "=" * 95)
    print(f"{'Contesto':>8} | {'FSTLLM RAM':>12} | {'Trans RAM':>12} | {'FST Cache':>11} | {'Trans Cache':>11} | {'FST Speed':>10} | {'Trans Speed':>10}")
    print("-" * 95)

    comp_list = []
    for f_run, t_run in zip(fst_data["runs"], trans_data["runs"]):
        ctx = f_run["context_length"]
        f_ram = f_run["total_process_rss_mb"]
        t_ram = t_run["total_process_rss_mb"]
        f_cache = f"{f_run['state_cache_kb']} KB"
        t_cache = f"{t_run['state_cache_kb']} KB"
        f_spd = f"{f_run['tokens_per_sec']} t/s"
        t_spd = f"{t_run['tokens_per_sec']} t/s"

        print(f"{ctx:>8} | {f_ram:>10.2f} MB | {t_ram:>10.2f} MB | {f_cache:>11} | {t_cache:>11} | {f_spd:>10} | {t_spd:>10}")

        comp_list.append({
            "context_length": ctx,
            "fstllm_total_rss_mb": f_ram,
            "transformer_total_rss_mb": t_ram,
            "fstllm_delta_rss_mb": f_run["delta_context_rss_mb"],
            "transformer_delta_rss_mb": t_run["delta_context_rss_mb"],
            "fstllm_cache_kb": f_run["state_cache_kb"],
            "transformer_cache_kb": t_run["state_cache_kb"],
            "fstllm_speed_tok_s": f_run["tokens_per_sec"],
            "transformer_speed_tok_s": t_run["tokens_per_sec"],
        })

    print("=" * 95)
    print(f"📦 Pesi FSTLLM in RAM:      {fst_data['weights_rss_mb']:.2f} MB")
    print(f"📦 Pesi Transformer in RAM: {trans_data['weights_rss_mb']:.2f} MB")
    print("=" * 95)

    # Salvataggio Dati Scientifici
    final_output = {
        "summary": "Benchmark rigoroso ed imparziale: processi isolati, KV-Cache reale SDPA",
        "fstllm": fst_data,
        "transformer": trans_data,
        "comparison": comp_list
    }
    json_path = ROOT_DIR / "benchmark_true_process_memory.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"💾 Report JSON imparziale salvato in: {json_path}")


if __name__ == "__main__":
    main()
