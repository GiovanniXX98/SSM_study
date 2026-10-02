"""
Chat Interattiva & Generazione O(1) — Progetto 21.0 (FAM-LLM)
==============================================================
Consente di dialogare in tempo reale con il modello Fourier Associative Matrix LLM.
Sfrutta la Matrix State Cache complessa per un'emissione a velocità strettamente costante O(1).
"""

import sys
import argparse
from pathlib import Path
import torch
import tiktoken

sys.path.insert(0, str(Path(__file__).parent))
from model import FourierAssociativeMatrixLLM


def main():
    parser = argparse.ArgumentParser(description="Chat interattiva con Progetto 21.0 (FAM-LLM)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Percorso checkpoint .pt specifico")
    parser.add_argument("--max_tokens", type=int, default=120, help="Numero massimo di token da generare (default: 120)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperatura di campionamento (0.1 - 1.2)")
    parser.add_argument("--top_k", type=int, default=40, help="Filtro Top-K")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    # Ricerca checkpoint
    proj_dir = Path(__file__).parent.parent
    checkpoints_dir = proj_dir / "checkpoints"
    ckpt_path = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        # Seleziona automaticamente il checkpoint con lo step più alto tra fam_step_*.pt
        step_files = list(checkpoints_dir.glob("fam_step_*.pt")) if checkpoints_dir.exists() else []
        if step_files:
            def extract_step(p):
                try:
                    return int(p.stem.split("_")[-1])
                except Exception:
                    return -1
            ckpt_path = max(step_files, key=extract_step)
        elif (checkpoints_dir / "fam_latest.pt").exists():
            ckpt_path = checkpoints_dir / "fam_latest.pt"

    if ckpt_path and ckpt_path.exists():
        print(f"📦 Caricamento pesi dal checkpoint: {ckpt_path.name}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        config = checkpoint.get("config", {
            "vocab_size": 50257,
            "d_model": 384,
            "n_layers": 6,
            "num_heads": 6,
            "num_kv_heads": 2,
            "chunk_size": 32,
        })
        model = FourierAssociativeMatrixLLM(**config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        step = checkpoint.get("step", "?")
        loss = checkpoint.get("loss", 0.0)
        print(f"✅ Modello caricato con successo (Step {step} - Loss {loss:.4f})")
    else:
        print("⚠️ Nessun checkpoint trovato. Inizializzazione modello base non addestrato per test immediato.")
        model = FourierAssociativeMatrixLLM(
            vocab_size=50257,
            d_model=384,
            n_layers=6,
            num_heads=6,
            num_kv_heads=2,
            chunk_size=32,
        ).to(device)

    model.eval()

    current_max_tokens = args.max_tokens
    current_temp = args.temperature
    current_top_k = args.top_k

    print("\n" + "=" * 68)
    print("💬 CHAT AVVIATA — Fourier Associative Matrix LLM (Progetto 21.0)")
    print(f"⚙️ Config: Temp={current_temp} | Top-K={current_top_k} | MaxTokens={current_max_tokens}")
    print("Comandi disponibili:")
    print("  • /tokens <N>   -> Modifica lunghezza max risposta (es. /tokens 200)")
    print("  • /temp <float> -> Modifica temperatura (es. /temp 0.5)")
    print("  • exit / quit   -> Chiudi la chat")
    print("=" * 68 + "\n")

    while True:
        try:
            user_input = input("Tu: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "esci"]:
                print("Chiusura sessione chat. Arrivederci!")
                break

            # Comandi di configurazione rapida
            if user_input.startswith("/tokens "):
                try:
                    current_max_tokens = int(user_input.split()[1])
                    print(f"⚙️ Max tokens impostato a: {current_max_tokens}\n")
                except ValueError:
                    print("⚠️ Uso: /tokens <numero_intero>\n")
                continue

            if user_input.startswith("/temp "):
                try:
                    current_temp = float(user_input.split()[1])
                    print(f"⚙️ Temperatura impostata a: {current_temp}\n")
                except ValueError:
                    print("⚠️ Uso: /temp <valore_decimale>\n")
                continue

            # Prompt naturale per continuazione coerente
            prompt_ids = torch.tensor(tokenizer.encode(user_input), dtype=torch.long, device=device)

            print(f"\n🤖 FAM-LLM 21.0:\n\033[96m{user_input}\033[0m", end="", flush=True)

            token_count = 0
            reached_limit = True
            for token_id in model.generate_stream(
                prompt_ids,
                max_new_tokens=current_max_tokens,
                temperature=current_temp,
                top_k=current_top_k,
                eos_token_id=50256,
            ):
                token_count += 1
                piece = tokenizer.decode([token_id])
                sys.stdout.write(piece)
                sys.stdout.flush()
            else:
                if token_count < current_max_tokens:
                    reached_limit = False

            if reached_limit and token_count >= current_max_tokens:
                print(f"\n\n\033[90m[... Fine generazione: raggiunto limite di {current_max_tokens} token. Usa /tokens {current_max_tokens + 80} per risposte più lunghe]\033[0m")
            else:
                print()

            print("\n" + "-" * 68)

        except (KeyboardInterrupt, EOFError):
            print("\nInterruzione rilevata. Chiusura...")
            break


if __name__ == "__main__":
    main()
