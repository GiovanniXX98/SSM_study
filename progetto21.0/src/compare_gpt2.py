"""
Confronto Inferenza: Progetto 21.0 (FAM-LLM) vs GPT-2 (HuggingFace)
===================================================================
Confronta in tempo reale le generazioni token-per-token di:
1. Fourier Associative Matrix LLM (Progetto 21.0) con inferenza O(1) Matrix State
2. GPT-2 Standard (Pre-trained HuggingFace o baseline)
"""

import sys
import argparse
from pathlib import Path
import torch
import tiktoken
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from model import FourierAssociativeMatrixLLM


def main():
    parser = argparse.ArgumentParser(description="Confronto Progetto 21.0 vs GPT-2")
    parser.add_argument("--checkpoint", type=str, default=None, help="Percorso checkpoint .pt (default: checkpoints/fam_latest.pt)")
    parser.add_argument("--max_new_tokens", type=int, default=40, help="Token da generare")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperatura di campionamento")
    parser.add_argument("--top_k", type=int, default=40, help="Filtro Top-K")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("🔬 CONFRONTO INFERENZA: Progetto 21.0 (FAM-LLM) vs GPT-2")
    print(f"💻 Device: {device}")
    print("=" * 70)

    # 1. Carica Modello Custom (Progetto 21.0)
    tokenizer_custom = tiktoken.get_encoding("gpt2")
    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
    ckpt_path = Path(args.checkpoint) if args.checkpoint else checkpoints_dir / "fam_latest.pt"

    if not ckpt_path.exists():
        print(f"⚠️ Checkpoint '{ckpt_path}' non trovato.")
        print("   Avvia prima il training per generare i pesi.")
        return

    print(f"📦 Caricamento FAM-LLM da: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {
        "vocab_size": 50257,
        "d_model": 384,
        "n_layers": 6,
        "num_heads": 6,
        "num_kv_heads": 2,
        "chunk_size": 32,
    })

    model_custom = FourierAssociativeMatrixLLM(**config).to(device)
    model_custom.load_state_dict(ckpt["model_state_dict"])
    model_custom.eval()

    total_params = sum(p.numel() for p in model_custom.parameters())
    print(f"🧠 FAM-LLM Parametri: {total_params:,} | Step addestrati: {ckpt.get('step', 'N/A')}")

    # 2. Carica GPT-2
    print("📦 Caricamento GPT-2 da HuggingFace...")
    try:
        tokenizer_hf = GPT2Tokenizer.from_pretrained("gpt2")
        model_hf = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        model_hf.eval()
        has_gpt2_hf = True
    except Exception as e:
        print(f"⚠️ Impossibile caricare GPT-2 HuggingFace ({e}). Solo modello locale.")
        has_gpt2_hf = False

    prompts = [
        "The earthquake victims were",
        "The scientific breakthrough in Fourier analysis",
        "Artificial intelligence will transform",
        "In modern computer architectures",
    ]

    print("\n" + "=" * 70)
    print("🚀 INIZIO TEST COMPARATIVO")
    print("=" * 70)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- [Test {i}/{len(prompts)}] Prompt: \"{prompt}\" ---")

        # FAM-LLM O(1)
        prompt_ids = torch.tensor(tokenizer_custom.encode(prompt), dtype=torch.long, device=device)
        with torch.no_grad():
            gen_ids = model_custom.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        output_fam = tokenizer_custom.decode(gen_ids).strip()
        print(f"🔮 FAM-LLM (O(1)): {prompt} {output_fam}")

        # GPT-2 HF
        if has_gpt2_hf:
            input_hf = tokenizer_hf.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen_hf = model_hf.generate(
                    input_hf,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    do_sample=True,
                    pad_token_id=tokenizer_hf.eos_token_id,
                )
            output_gpt2 = tokenizer_hf.decode(gen_hf[0][input_hf.shape[1]:], skip_special_tokens=True).strip()
            print(f"🤖 GPT-2 (HF):    {prompt} {output_gpt2}")


if __name__ == "__main__":
    main()
