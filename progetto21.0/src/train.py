"""
Training Pipeline — Progetto 21.0: Fourier Associative Matrix LLM (FAM-LLM)
===========================================================================
Addestramento autoregressivo Next-Token Prediction su testo reale GPT-2 (primi 100.000 token).
- Ottimizzatore AdamW con Weight Decay disaccoppiato e schedule Cosine Annealing con Warmup.
- Memoria Associativa Matriciale K^T * V in C^{HD x HD} con lettura Q * S via Matmul.
- Monitoraggio della Loss e Perplexity con telemetria CSV a scrittura immediata e generazione O(1).
- Piena compatibilità con esecuzione in background tramite nohup.
"""

import os
import sys
import time
import math
import csv
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))
from model import FourierAssociativeMatrixLLM
from dataset import get_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Addestramento Fourier Associative Matrix LLM (Progetto 21.0)")
    parser.add_argument("--data_path", type=str, default=None, help="Percorso del dataset (.bin o .txt)")
    parser.add_argument("--max_tokens", type=int, default=100_000, help="Numero di token GPT-2 da caricare (default: 100.000)")
    parser.add_argument("--d_model", type=int, default=384, help="Dimensione embedding (384 standard, 768 GPT-2)")
    parser.add_argument("--n_layers", type=int, default=6, help="Numero di blocchi FAM")
    parser.add_argument("--num_heads", type=int, default=6, help="Numero teste per la memoria")
    parser.add_argument("--num_kv_heads", type=int, default=2, help="Numero teste Key-Value (GQR)")
    parser.add_argument("--chunk_size", type=int, default=32, help="Dimensione chunk locale")
    parser.add_argument("--seq_len", type=int, default=256, help="Lunghezza sequenza di training")
    parser.add_argument("--batch_size", type=int, default=4, help="Dimensione batch")
    parser.add_argument("--max_steps", type=int, default=500, help="Numero totale di step di addestramento")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate massimo per AdamW")
    parser.add_argument("--min_lr", type=float, default=2e-5, help="Learning rate minimo")
    parser.add_argument("--warmup_steps", type=int, default=40, help="Step di warmup lineare")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Decadimento pesi AdamW")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Clip della norma del gradiente")
    parser.add_argument("--save_interval", type=int, default=100, help="Intervallo salvataggio checkpoint")
    parser.add_argument("--sample_interval", type=int, default=50, help="Intervallo generazione testo di prova")
    parser.add_argument("--resume", action="store_true", help="Riprendi l'addestramento dall'ultimo checkpoint se presente")
    return parser.parse_args()


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine Annealing con Linear Warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    args = parse_args()
    proj_dir = Path(__file__).parent.parent
    checkpoints_dir = proj_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    telemetry_file = proj_dir / "training_telemetry.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"🚀 AVVIO ADDESTRAMENTO: Progetto 21.0 — Fourier Associative Matrix LLM")
    print(f"💻 Device attivo: {device}")
    print(f"📐 Architettura: d_model={args.d_model}, n_layers={args.n_layers}, heads={args.num_heads}, kv_heads={args.num_kv_heads}")
    print(f"📦 Batch: {args.batch_size} | Seq: {args.seq_len} token | Chunk: {args.chunk_size}")
    print(f"🎯 Target Token: {args.max_tokens:,} token GPT-2 | Max Step: {args.max_steps}")
    print("=" * 75)
    sys.stdout.flush()

    # 1. Caricamento Dataset e Tokenizer
    dataloader, dataset = get_dataloader(
        data_path=args.data_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        shuffle=True,
        max_tokens=args.max_tokens,
    )
    data_iter = iter(dataloader)
    tokenizer = dataset.tokenizer

    tokens_per_step = args.batch_size * args.seq_len
    steps_per_epoch = max(1, dataset.n_tokens // tokens_per_step)
    total_epochs = (args.max_steps * tokens_per_step) / max(1, dataset.n_tokens)
    print(f"🔄 Dinamica Training: 1 epoca = {steps_per_epoch} step | {args.max_steps} step equivalgono a {total_epochs:.2f} epoche sui {dataset.n_tokens:,} token attivi.")
    sys.stdout.flush()

    # 2. Inizializzazione Modello
    model = FourierAssociativeMatrixLLM(
        vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        chunk_size=args.chunk_size,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tied_emb_params = model.tok_emb.weight.numel()
    non_emb_params = sum(p.numel() for n, p in model.named_parameters() if "tok_emb" not in n)
    print(f"🧠 Parametri Totali: {total_params:,} (Pesi Core Non-Embedding: {non_emb_params:,} | Tied Embedding: {tied_emb_params:,})")
    sys.stdout.flush()

    # 3. Ottimizzatore AdamW ad alte prestazioni
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    start_step = 1
    # Possibile ripristino checkpoint
    latest_ckpt = checkpoints_dir / "fam_latest.pt"
    if args.resume and latest_ckpt.exists():
        print(f"🔄 Ripristino checkpoint da: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0) + 1
        print(f"⏩ Ripreso da step {start_step}")
        sys.stdout.flush()
    else:
        # Inizializza telemetria CSV
        with open(telemetry_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "perplexity", "lr", "tokens_per_sec", "elapsed_sec", "epoch"])
            f.flush()

    # 4. Training Loop
    model.train()
    start_time = time.time()
    tokens_processed = 0

    print(f"\n⏳ Avvio ciclo di ottimizzazione AdamW per {args.max_steps} step...")
    sys.stdout.flush()

    for step in range(start_step, args.max_steps + 1):
        step_t0 = time.time()

        # Aggiornamento Learning Rate
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Prelievo Batch
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Forward & Backward Pass
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(x, targets=y)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # Metriche
        step_time = time.time() - step_t0
        tokens_in_batch = x.numel()
        tokens_processed += tokens_in_batch
        tokens_per_sec = tokens_in_batch / max(1e-5, step_time)
        ppl = math.exp(min(loss.item(), 20.0))
        current_epoch = (step * tokens_per_step) / max(1, dataset.n_tokens)
        elapsed_total = time.time() - start_time

        # Log a console
        if step % 10 == 0 or step == start_step or step == args.max_steps:
            print(
                f"Step {step:4d}/{args.max_steps} | "
                f"Epoch: {current_epoch:5.2f} | "
                f"Loss: {loss.item():.4f} | "
                f"PPL: {ppl:8.2f} | "
                f"LR: {lr:.2e} | "
                f"Speed: {tokens_per_sec:5.0f} tok/s"
            )
            sys.stdout.flush()

        # Salvataggio Telemetria CSV immediato con flush
        with open(telemetry_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                f"{loss.item():.4f}",
                f"{ppl:.2f}",
                f"{lr:.2e}",
                f"{tokens_per_sec:.1f}",
                f"{elapsed_total:.2f}",
                f"{current_epoch:.3f}",
            ])
            f.flush()

        # Generazione di Controllo Periodica
        if step % args.sample_interval == 0 or step == args.max_steps:
            model.eval()
            prompt_text = "The"
            prompt_ids = torch.tensor(tokenizer.encode(prompt_text), dtype=torch.long, device=device)
            gen_ids = model.generate(prompt_ids, max_new_tokens=25, temperature=0.7, top_k=40)
            sample_decoded = tokenizer.decode(gen_ids)
            print("=" * 70)
            print(f"🔮 [Step {step} | Generazione O(1)] '{prompt_text}' -> {sample_decoded.strip()}")
            print("=" * 70)
            sys.stdout.flush()
            model.train()

        # Salvataggio Checkpoint
        if step % args.save_interval == 0 or step == args.max_steps:
            ckpt_latest = checkpoints_dir / "fam_latest.pt"
            ckpt_step = checkpoints_dir / f"fam_step_{step}.pt"
            checkpoint_data = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
                "config": model.config,
                "epoch": current_epoch,
            }
            torch.save(checkpoint_data, ckpt_latest)
            torch.save(checkpoint_data, ckpt_step)
            print(f"💾 Checkpoint salvato: {ckpt_latest.name} (step {step})")
            sys.stdout.flush()

    total_time = time.time() - start_time
    print("=" * 75)
    print(f"🎉 ADDESTRAMENTO COMPLETATO CON SUCCESSO!")
    print(f"⏱️ Tempo totale: {total_time:.2f} s | Token totali processati: {tokens_processed:,}")
    print(f"📁 Checkpoint finale salvato in: {checkpoints_dir / 'fam_latest.pt'}")
    print(f"📊 Telemetria salvata in: {telemetry_file}")
    print("=" * 75)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
