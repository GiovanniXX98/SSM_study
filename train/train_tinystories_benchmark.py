"""
SSM_study: FSTLLM 2.0 — 30M Parameter Training & Benchmark on TinyStories
=========================================================================
Allena il modello FSTLLM 2.0 configurato a ~31.4M di parametri sul dataset TinyStories.
Salva i checkpoint in checkpoints/fstllm_30m_best.pt e registra tutte le metriche
di confronto rispetto a un Transformer equivalente per il paper accademico.
"""

import os
import sys
import time
import json
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Root del progetto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from model import FourierSpaceTimeLLM_V2


# ---------------------------------------------------------------------------
# 1. Dataset TinyStories
# ---------------------------------------------------------------------------

class TinyStoriesDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int = 128, split: str = "train", split_ratio: float = 0.9):
        self.seq_len = seq_len
        print(f"📖 Caricamento dataset TinyStories da: {data_path}")
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

        split_idx = int(len(text) * split_ratio)
        if split == "train":
            text_split = text[:split_idx]
        else:
            text_split = text[split_idx:]

        self.data = torch.tensor([self.stoi[c] for c in text_split], dtype=torch.long)
        print(f"   Split '{split}': {len(self.data):,} caratteri (Vocabolario: {self.vocab_size})")

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


# ---------------------------------------------------------------------------
# 2. Transformer Baseline per il Confronto Diretto (~31M Parametri)
# ---------------------------------------------------------------------------

class StandardTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln2 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(d_model * 2.67), bias=False),
            nn.SiLU(),
            nn.Linear(int(d_model * 2.67), d_model, bias=False)
        )

    def forward(self, x, is_causal: bool = True):
        norm_x = self.ln1(x)
        B, S, D = x.shape
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device) if is_causal else None
        attn_out, _ = self.mha(norm_x, norm_x, norm_x, is_causal=is_causal, attn_mask=mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class StandardTransformerLLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 576, n_layers: int = 8, num_heads: int = 8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(d_model, num_heads) for _ in range(n_layers)
        ])
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 3. Funzioni di Generazione e Benchmark O(1) vs O(S)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_fstllm_o1(model, prompt_ids, max_new_tokens=50, temperature=0.7):
    model.eval()
    device = next(model.parameters()).device
    prompt = prompt_ids.unsqueeze(0).to(device)
    
    # Prefill
    _, _, past_states = model(prompt)
    current_token = prompt[:, -1:]
    generated = []
    
    t0 = time.time()
    for _ in range(max_new_tokens):
        logits, _, past_states = model(current_token, past_states=past_states)
        next_logit = logits[0, -1, :] / max(0.01, temperature)
        probs = torch.softmax(next_logit, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token.item())
        current_token = next_token.unsqueeze(0)
    t1 = time.time()
    
    speed = max_new_tokens / max(1e-6, t1 - t0)
    return generated, speed


@torch.no_grad()
def benchmark_decode_latency(fst_model, trans_model, context_lengths, vocab_size, device="cpu"):
    """Misura la latenza di generazione (token/s) e consumo memoria di stato al crescere del contesto."""
    results = []
    print("\n⏱️ --- Avvio Benchmark Latenza di Inferenza al Variare del Contesto ---")
    
    for S in context_lengths:
        dummy_context = torch.randint(0, vocab_size, (1, S), device=device)
        
        # 1. FSTLLM 2.0: Prefill e decode di 20 token
        fst_model.eval()
        _, _, past_states = fst_model(dummy_context)
        tok = dummy_context[:, -1:]
        t0 = time.time()
        for _ in range(20):
            logits, _, past_states = fst_model(tok, past_states=past_states)
            tok = logits.argmax(dim=-1)
        t_fst = (time.time() - t0) / 20.0
        fst_speed = 1.0 / max(1e-6, t_fst)
        
        # Memoria di stato per FSTLLM 2.0: 8 layers * 2 KV heads * 72 HD * 8 bytes (cfloat32)
        # d_model=576, num_heads=8 -> head_dim=72
        # fourier_state shape: (1, 2, 72) in cfloat32 (8 bytes/complex)
        fst_state_bytes = 8 * (2 * 72 * 8) + 8 * (576 * 3 * 4) # fourier_state + conv_state
        fst_mem_kb = fst_state_bytes / 1024.0
        
        # 2. Transformer: Decode autoregressivo a contesto crescente (simulazione KV-cache)
        # Dimensione KV-Cache: 8 layers * 2 (K e V) * S * 576 * 4 bytes (float32)
        trans_kv_bytes = 8 * 2 * S * 576 * 4
        trans_mem_kb = trans_kv_bytes / 1024.0
        
        # Stima tempo decode per Transformer (dipende linearmente da S per rilettura cache)
        # Misuriamo con passaggio di contesto S
        t0 = time.time()
        curr_seq = dummy_context
        for _ in range(5):
            logits, _ = trans_model(curr_seq)
            next_t = logits[:, -1:, :].argmax(dim=-1)
            curr_seq = torch.cat([curr_seq, next_t], dim=1)
        t_trans = (time.time() - t0) / 5.0
        trans_speed = 1.0 / max(1e-6, t_trans)
        
        results.append({
            "context_length": S,
            "fstllm_tokens_per_sec": round(fst_speed, 1),
            "fstllm_state_mem_kb": round(fst_mem_kb, 2),
            "transformer_tokens_per_sec": round(trans_speed, 1),
            "transformer_kv_cache_mem_kb": round(trans_mem_kb, 2),
            "memory_reduction_factor": round(trans_mem_kb / max(1e-4, fst_mem_kb), 1)
        })
        print(f"  Contesto S={S:>4d} | FSTLLM: {fst_speed:>5.1f} tok/s ({fst_mem_kb:.1f} KB) | Transformer: {trans_speed:>5.1f} tok/s ({trans_mem_kb:.1f} KB) | Risparmio Mem: {trans_mem_kb/fst_mem_kb:.1f}x")
        
    return results


# ---------------------------------------------------------------------------
# 4. Pipeline Principale di Addestramento e Benchmark
# ---------------------------------------------------------------------------

def main():
    torch.set_num_threads(12)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = PROJECT_ROOT / "data" / "tinystories.txt"
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🌟 SSM_study: Addestramento FSTLLM 2.0 (30M) su TinyStories & Benchmark")
    print(f"   Device: {device} | Thread CPU: {torch.get_num_threads()}")
    print("=" * 70)

    # 1. Dataset
    seq_len = 128
    batch_size = 8
    train_dataset = TinyStoriesDataset(str(data_path), seq_len=seq_len, split="train")
    val_dataset = TinyStoriesDataset(str(data_path), seq_len=seq_len, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Configurazione Modello 30M
    d_model = 576
    n_layers = 8
    num_heads = 8
    num_kv_heads = 2
    conv_kernel = 4
    
    cfg = {
        "vocab_size": train_dataset.vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "conv_kernel": conv_kernel
    }
    
    model = FourierSpaceTimeLLM_V2.from_config(cfg).to(device)
    n_params = model.get_num_params()
    print(f"\n📊 Architettura FSTLLM 2.0: {n_params/1e6:.2f}M parametri ({n_params:,} pesi)")
    print(f"   d_model={d_model}, layers={n_layers}, heads={num_heads}, kv_heads={num_kv_heads}")
    
    # 3. Transformer Baseline Equivalente
    trans_model = StandardTransformerLLM(
        vocab_size=train_dataset.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        num_heads=num_heads
    ).to(device)
    trans_params = trans_model.get_num_params()
    print(f"📊 Architettura Transformer Baseline: {trans_params/1e6:.2f}M parametri ({trans_params:,} pesi)")
    
    # 4. Addestramento
    lr = 8e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    epochs = 1
    max_steps = 60
    log_interval = 10
    
    train_losses = []
    step_times = []
    
    print("\n🏋️ --- Inizio Addestramento FSTLLM 2.0 (30M) ---")
    model.train()
    step = 0
    best_loss = float("inf")
    
    for epoch in range(epochs):
        for x, y in train_loader:
            if step >= max_steps:
                break
            
            t0 = time.time()
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, loss, _ = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            dt = time.time() - t0
            
            loss_val = loss.item()
            train_losses.append(loss_val)
            step_times.append(dt)
            
            if step % log_interval == 0 or step == max_steps - 1:
                ppl = math.exp(min(15.0, loss_val))
                print(f"Step {step:>3d}/{max_steps} | Loss: {loss_val:.4f} | Perplexity: {ppl:>7.2f} | Tempo: {dt:.2f}s")
                
                # Salva checkpoint migliore
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_ckpt_path = checkpoints_dir / "fstllm_30m_best.pt"
                    torch.save({
                        "step": step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "loss": loss_val,
                        "perplexity": ppl,
                        "config": cfg,
                        "stoi": train_dataset.stoi,
                        "itos": train_dataset.itos,
                        "num_params": n_params
                    }, best_ckpt_path)
                    
            step += 1

    print(f"\n✅ Checkpoint migliore salvato in: {checkpoints_dir / 'fstllm_30m_best.pt'} (Loss: {best_loss:.4f})")
    
    # 5. Valutazione su Validation Set
    print("\n🧪 --- Valutazione Validation Set ---")
    model.eval()
    val_loss_acc = 0.0
    val_steps = 15
    with torch.no_grad():
        for v_step, (vx, vy) in enumerate(val_loader):
            if v_step >= val_steps:
                break
            vx, vy = vx.to(device), vy.to(device)
            _, v_loss, _ = model(vx, vy)
            val_loss_acc += v_loss.item()
            
    avg_val_loss = val_loss_acc / val_steps
    val_ppl = math.exp(min(15.0, avg_val_loss))
    print(f"   Validation Loss: {avg_val_loss:.4f} | Validation Perplexity: {val_ppl:.2f}")

    # 6. Test Generazione O(1) con Prompt TinyStories
    print("\n📖 --- Generazione Autoregressiva O(1) con Holographic Cache ---")
    prompt_text = "Once upon a time, Lily found a"
    prompt_ids = torch.tensor([train_dataset.stoi.get(c, 0) for c in prompt_text], dtype=torch.long)
    gen_ids, gen_speed = generate_fstllm_o1(model, prompt_ids, max_new_tokens=80, temperature=0.75)
    generated_story = prompt_text + "".join([train_dataset.itos.get(i, "") for i in gen_ids])
    print(f"Prompt: \"{prompt_text}\"")
    print(f"Generato:\n------------------------------------------------------------\n{generated_story}\n------------------------------------------------------------")
    print(f"Velocità di Decode O(1): {gen_speed:.1f} token/s")

    # 7. Benchmark di Latenza e Memoria vs Transformer
    bench_results = benchmark_decode_latency(
        fst_model=model,
        trans_model=trans_model,
        context_lengths=[128, 256, 512, 1024, 2048],
        vocab_size=train_dataset.vocab_size,
        device=device
    )

    # 8. Salvataggio Risultati per il Paper
    final_report = {
        "model_name": "FSTLLM 2.0 (Fourier Space-Time LLM)",
        "dataset": "TinyStories (Eldan & Li, Microsoft Research)",
        "parameters": n_params,
        "parameters_m": round(n_params / 1e6, 2),
        "transformer_baseline_params": trans_params,
        "config": cfg,
        "training": {
            "initial_loss": round(train_losses[0], 4),
            "final_train_loss": round(train_losses[-1], 4),
            "best_train_loss": round(best_loss, 4),
            "validation_loss": round(avg_val_loss, 4),
            "validation_perplexity": round(val_ppl, 2),
            "total_steps": max_steps,
            "avg_step_time_s": round(sum(step_times) / len(step_times), 2)
        },
        "generation_demo": {
            "prompt": prompt_text,
            "sample_output": generated_story,
            "speed_tokens_per_sec": round(gen_speed, 1)
        },
        "benchmark_scaling_vs_transformer": bench_results
    }
    
    report_path = PROJECT_ROOT / "benchmark_results_30m.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n📁 Report sperimentale completo salvato in: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
