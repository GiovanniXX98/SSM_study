"""
Training Pipeline — Progetto 21.0: Fourier Associative Matrix LLM (FAM-LLM) su TinyStories
========================================================================================
- Dataset reale TinyStories tokenizzato in GPT-2 BPE (50.257 token) tramite numpy.memmap.
- Architettura a 29.94M di parametri (d_model=384, n_layers=6, heads=6, kv_heads=2).
- Supporto Hardware Ibrido: Auto-detect CUDA (GPU NVIDIA) con BF16/FP16 oppure CPU multicore.
- Checkpoint salvati ESCLUSIVAMENTE in: progetto21.0/checkpoints/tinystories/
- Nessun checkpoint pregresso viene mai sovrascritto.
"""

import os
import sys
import time
import math
import csv
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tiktoken

sys.path.insert(0, str(Path(__file__).parent))
from model import FourierAssociativeMatrixLLM


def ensure_bin_dataset(bin_path: Path, txt_path: Path) -> Path:
    """Verifica e genera automaticamente il file binario BPE se manca."""
    if bin_path.exists() and bin_path.stat().st_size > 1000:
        return bin_path

    print(f"📦 File binario non trovato in {bin_path}. Generazione istantanea da {txt_path}...")
    if not txt_path.exists():
        # Cerca nei percorsi alternativi
        workspace_root = bin_path.parent.parent.parent
        cands = [workspace_root / "data" / "tinystories.txt", bin_path.parent / "tinystories.txt"]
        for c in cands:
            if c.exists():
                txt_path = c
                break

    if not txt_path.exists():
        raise FileNotFoundError(f"Impossibile trovare il testo sorgente TinyStories in {txt_path}")

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
    arr = np.array(tokens, dtype=np.uint16)
    with open(bin_path, "wb") as f:
        f.write(arr.tobytes())

    print(f"✅ Tokenizzati {len(tokens):,} token GPT-2 in: {bin_path} ({bin_path.stat().st_size / (1024*1024):.2f} MB)")
    return bin_path


class TinyStoriesMemMapDataset(Dataset):
    """Dataset MemMap per TinyStories con split Train/Validation."""
    def __init__(self, bin_path: Path, seq_len: int = 128, split: str = "train", split_ratio: float = 0.92):
        self.seq_len = seq_len
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab  # 50257

        total_tokens = bin_path.stat().st_size // 2  # uint16
        train_tokens = int(total_tokens * split_ratio)

        self.full_data = np.memmap(bin_path, dtype=np.uint16, mode="r", shape=(total_tokens,))
        if split == "train":
            self.data = self.full_data[:train_tokens]
        else:
            self.data = self.full_data[train_tokens:]

        self.n_tokens = len(self.data)
        self.num_samples = (self.n_tokens - 1) // self.seq_len
        print(f"📖 Split [{split.upper()}]: {self.n_tokens:,} token GPT-2 | {self.num_samples:,} finestre (seq_len={seq_len})")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine Annealing con Linear Warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def evaluate_val_loss(model, val_loader, device, max_batches: int = 15) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, targets=y)
            total_loss += loss.item()
            count += 1
    model.train()
    avg_loss = total_loss / max(1, count)
    ppl = math.exp(min(20.0, avg_loss))
    return avg_loss, ppl


def parse_args():
    parser = argparse.ArgumentParser(description="Addestramento TinyStories su FAM-LLM Progetto 21.0")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4 su CPU, 8-16 su GPU)")
    parser.add_argument("--seq_len", type=int, default=128, help="Lunghezza sequenza di training (default: 128)")
    parser.add_argument("--max_steps", type=int, default=1000, help="Numero totale di step (default: 1000)")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate massimo")
    parser.add_argument("--min_lr", type=float, default=2e-5, help="Learning rate minimo")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Step di warmup")
    parser.add_argument("--eval_interval", type=int, default=50, help="Intervallo di valutazione e test generazione")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Rilevamento automatico hardware
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cpu":
        num_threads = os.cpu_count() or 12
        torch.set_num_threads(num_threads)
    else:
        num_threads = "GPU CUDA Cores"

    proj_dir = Path(__file__).resolve().parent.parent
    workspace_root = proj_dir.parent
    data_bin = proj_dir / "data" / "tinystories_gpt2.bin"
    data_txt = workspace_root / "data" / "tinystories.txt"

    # Assicura la presenza del dataset
    data_bin = ensure_bin_dataset(data_bin, data_txt)

    # Cartella dedicata sicura per i checkpoint di TinyStories
    ckpt_dir = proj_dir / "checkpoints" / "tinystories"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    telemetry_file = proj_dir / "tinystories_bpe_telemetry.csv"

    # Iperparametri
    d_model = 384
    n_layers = 6
    num_heads = 6
    num_kv_heads = 2
    seq_len = args.seq_len
    batch_size = args.batch_size
    max_steps = args.max_steps
    lr = args.lr
    min_lr = args.min_lr
    warmup_steps = args.warmup_steps
    eval_interval = args.eval_interval

    print("=" * 80)
    print("🚀 AVVIO TRAINING SCIENTIFICO: FAM-LLM Progetto 21.0 (30M) su TinyStories BPE")
    print(f"💻 Device attivo: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else f'CPU Threads: {num_threads}'})")
    print(f"📁 Dataset MemMap: {data_bin} (Zero RAM overhead)")
    print(f"💾 Checkpoints isolati in: {ckpt_dir} (Nessun checkpoint pregresso verrà toccato!)")
    print(f"📊 Telemetria CSV: {telemetry_file}")
    print(f"🎯 Step Totali: {max_steps} | Batch: {batch_size} | SeqLen: {seq_len}")
    print("=" * 80)
    sys.stdout.flush()

    # 1. Carica Dataset Train e Val
    train_dataset = TinyStoriesMemMapDataset(data_bin, seq_len=seq_len, split="train", split_ratio=0.92)
    val_dataset = TinyStoriesMemMapDataset(data_bin, seq_len=seq_len, split="val", split_ratio=0.92)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    tokenizer = train_dataset.tokenizer

    # 2. Inizializza Modello
    model = FourierAssociativeMatrixLLM(
        vocab_size=train_dataset.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        chunk_size=32,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Parametri Totali del Modello: {total_params:,} ({total_params/1e6:.2f}M)")
    sys.stdout.flush()

    # 3. Ottimizzatore AdamW potenziato
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)

    # Inizializza CSV se nuovo
    if not telemetry_file.exists() or telemetry_file.stat().st_size == 0:
        with open(telemetry_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss", "train_ppl", "val_loss", "val_ppl", "lr", "tok_per_sec", "elapsed_s"])
            f.flush()

    best_val_loss = float("inf")
    start_time = time.time()
    train_iter = iter(train_loader)
    tokens_per_step = batch_size * seq_len

    print("\n⏳ Inizio loop di addestramento autoregressivo...")
    sys.stdout.flush()

    for step in range(1, max_steps + 1):
        step_start = time.time()

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        curr_lr = get_lr(step, warmup_steps, max_steps, lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

        optimizer.zero_grad(set_to_none=True)
        logits, loss, _ = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step_elapsed = time.time() - step_start
        tok_per_sec = tokens_per_step / max(1e-5, step_elapsed)
        train_loss = loss.item()
        train_ppl = math.exp(min(20.0, train_loss))

        # Log periodico
        if step % eval_interval == 0 or step == 1 or step == max_steps:
            val_loss, val_ppl = evaluate_val_loss(model, val_loader, device)
            total_elapsed = time.time() - start_time

            with open(telemetry_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([step, f"{train_loss:.4f}", f"{train_ppl:.2f}", f"{val_loss:.4f}", f"{val_ppl:.2f}", f"{curr_lr:.6f}", f"{tok_per_sec:.1f}", f"{total_elapsed:.1f}"])
                f.flush()

            print(f"Step {step:4d}/{max_steps} | Train Loss: {train_loss:.4f} (ppl: {train_ppl:.2f}) | Val Loss: {val_loss:.4f} (ppl: {val_ppl:.2f}) | Speed: {tok_per_sec:.1f} tok/s | Elapsed: {total_elapsed:.1f}s")
            sys.stdout.flush()

            # Test di generazione
            prompt_str = "Once upon a time, Lily found a"
            prompt_tokens = torch.tensor(tokenizer.encode(prompt_str), dtype=torch.long, device=device)
            gen_ids = model.generate(prompt_tokens, max_new_tokens=25, temperature=0.7, top_k=40)
            gen_text = tokenizer.decode(prompt_tokens.tolist() + gen_ids)
            print(f"   📝 [Step {step}] Generazione: \"{gen_text.strip()}\"")
            sys.stdout.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = ckpt_dir / "fam_tinystories_best.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "num_heads": num_heads,
                    "num_kv_heads": num_kv_heads,
                    "vocab_size": train_dataset.vocab_size,
                }, best_ckpt_path)
                print(f"   💾 Nuovo record! Checkpoint salvato: {best_ckpt_path} (Val Loss: {val_loss:.4f})")
                sys.stdout.flush()

    final_ckpt_path = ckpt_dir / "fam_tinystories_final.pt"
    torch.save({
        "step": max_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
    }, final_ckpt_path)
    print(f"\n🎉 TRAINING COMPLETATO CON SUCCESSO!")
    print(f"🏆 Best Validation Loss: {best_val_loss:.4f} (Perplexity: {math.exp(min(20.0, best_val_loss)):.2f})")
    print(f"📦 Checkpoint custodito in: {ckpt_dir / 'fam_tinystories_best.pt'}")


if __name__ == "__main__":
    main()
