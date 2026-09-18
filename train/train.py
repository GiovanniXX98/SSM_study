"""
SSM_study: Fourier Space-Time State Space Model (FSTLLM 2.0) — Training & Inference
=====================================================================================
Script di addestramento e generazione autoregressiva a complessità costante O(1).
Sfrutta la Holographic State Cache per generare token senza ricalcolo della storia passata.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

# Aggiunge la root del repository a sys.path per importare model
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import FourierSpaceTimeLLM_V2


class TextDataset(Dataset):
    """Dataset a livello di caratteri con fallback automatico a testo sintetico demo."""
    def __init__(self, file_path: str, seq_len: int):
        self.seq_len = seq_len
        loaded = False

        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                if len(text) > seq_len + 1:
                    loaded = True
            except Exception as e:
                print(f"⚠️ Impossibile leggere {file_path}: {e}")

        if not loaded:
            print(f"ℹ️ File dataset non trovato o non specificato. Utilizzo del dataset demo interno.")
            text = (
                "Fourier Space-Time State Space Model (FSTLLM 2.0). "
                "Studying wave-theoretic state space models for competitive sub-quadratic NLP. "
                "Constant O(1) inference with Holographic State Cache and Grouped-Query Resonance. "
                "Fusing continuous spectral modulation with discrete state recurrence. "
            ) * 150

        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


@torch.no_grad()
def generate_text_o1(model, prompt_ids, max_new_tokens=60, temperature=0.7):
    """
    Generazione State-Space O(1): sfrutta la Holographic Cache per 
    processare solo l'ULTIMO token a ogni step, senza ricalcolare la storia passata.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Prefill Stage (elaborazione parallela del prompt)
    prompt = prompt_ids.unsqueeze(0).to(device)
    _, _, past_states = model(prompt)
    
    # 2. Decode Stage (inferenza ricorsiva O(1))
    current_token = prompt[:, -1:]
    generated = []
    
    t0 = time.time()
    for _ in range(max_new_tokens):
        # Si passa SOLO 1 token e la cache olografica: tempo per token O(1)
        logits, _, past_states = model(current_token, past_states=past_states)
        
        next_logit = logits[0, -1, :] / max(0.01, temperature)
        probs = torch.softmax(next_logit, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated.append(next_token.item())
        current_token = next_token.unsqueeze(0)
        
    t1 = time.time()
    speed = max_new_tokens / max(1e-6, t1 - t0)
    return generated, speed


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print(f"🚀 SSM_study — Fourier Space-Time State Space Model (FSTLLM 2.0)")
    print(f"   Device: {device}")
    print("=" * 65)

    dataset = TextDataset(args.data_path, args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    cfg = {
        "vocab_size": dataset.vocab_size,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "seq_len": args.seq_len,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "conv_kernel": args.conv_kernel
    }
    
    model = FourierSpaceTimeLLM_V2.from_config(cfg).to(device)
    n_params = model.get_num_params()
    print(f"📊 Parametri totali modello: {n_params / 1e6:.2f}M ({n_params:,} pesi)")
    print(f"⚙️ Configurazione: d_model={args.d_model}, layers={args.n_layers}, heads={args.num_heads}, kv_heads={args.num_kv_heads}")
    print("-" * 65)
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        steps = 0
        for step, (x, y) in enumerate(loader):
            if step >= args.max_steps_per_epoch:
                break
            
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, loss, _ = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            epoch_loss += loss.item()
            steps += 1
            
            if step % 10 == 0:
                print(f"Epoca {epoch + 1}/{args.epochs} | Step {step:>3d} | Loss: {loss.item():.4f}")
                
        # Test generazione O(1) al termine dell'epoca
        print("\n🔮 --- Test Generazione O(1) con Holographic Cache ---")
        prompt_str = "Fourier "
        prompt_ids = torch.tensor([dataset.stoi.get(c, 0) for c in prompt_str], dtype=torch.long)
        gen_ids, speed = generate_text_o1(model, prompt_ids, max_new_tokens=args.gen_tokens)
        output_text = prompt_str + "".join([dataset.itos.get(i, "") for i in gen_ids])
        print(f"Testo generato: \"{output_text}\"")
        print(f"Velocità Decode: {speed:.1f} token/s (complessità costante O(1) per step)\n")
        print("-" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Addestramento e test di SSM_study (FSTLLM 2.0)")
    parser.add_argument("--data-path", type=str, default="", help="Percorso del dataset testuale (.txt)")
    parser.add_argument("--seq-len", type=int, default=128, help="Lunghezza sequenza di addestramento")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Numero di epoche")
    parser.add_argument("--max-steps-per-epoch", type=int, default=50, help="Max step per epoca (demo)")
    parser.add_argument("--d-model", type=int, default=256, help="Dimensione del modello d_model")
    parser.add_argument("--n-layers", type=int, default=4, help="Numero di layer")
    parser.add_argument("--num-heads", type=int, default=8, help="Numero di teste Query")
    parser.add_argument("--num-kv-heads", type=int, default=2, help="Numero di teste KV per GQR")
    parser.add_argument("--conv-kernel", type=int, default=4, help="Kernel size per la convoluzione locale 1D")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gen-tokens", type=int, default=50, help="Numero di token da generare nel test O(1)")
    args = parser.parse_args()
    train(args)
