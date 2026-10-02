"""
Valutazione & Benchmark Comparativo — Progetto 21.0: Fourier Associative Matrix LLM (FAM-LLM)
=============================================================================================
1. Calcolo Perplexity su contesti estesi.
2. Benchmark comparativo con GPT-2 Small (HuggingFace 124M) su molteplici prompt.
3. Test di Memoria Associativa (Needle-in-a-Haystack / MQAR): verifica empirica
   della capacità della matrice K^T * V di preservare coppie chiave-valore nel tempo.
4. Generazione Report Markdown automatico in 'reports/'.
"""

import sys
import os
import math
import argparse
from datetime import datetime
from pathlib import Path
import torch
import torch.nn.functional as F
import tiktoken

sys.path.insert(0, str(Path(__file__).parent))
from model import FourierAssociativeMatrixLLM
from dataset import get_dataloader

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_GPT2_HF = True
except ImportError:
    HAS_GPT2_HF = False


def test_associative_recall(model, tokenizer, device, distance: int = 128) -> tuple[str, str]:
    """
    Test Needle-in-a-Haystack per la Matrice degli Stati:
    Inietta una chiave segreta all'inizio del testo, inserisce 'distance' token di rumore
    e verifica il recupero O(1) tramite la matrice K^T * V.
    """
    key_phrase = "The secret access code is ALPHA_99."
    query_phrase = " What was the secret access code? The secret access code is"

    filler = " The continuous wave dynamics propagate through four dimensional spacetime." * (distance // 10)
    full_prompt = key_phrase + filler + query_phrase

    prompt_ids = torch.tensor(tokenizer.encode(full_prompt), dtype=torch.long, device=device)
    with torch.no_grad():
        gen_ids = model.generate(prompt_ids, max_new_tokens=10, temperature=0.2, top_k=5)
    response = tokenizer.decode(gen_ids)

    return full_prompt, response.strip()


def main():
    parser = argparse.ArgumentParser(description="Valutazione & Confronto GPT-2 Small — Progetto 21.0")
    parser.add_argument("--checkpoint", type=str, default=None, help="Percorso checkpoint .pt")
    parser.add_argument("--seq_len", type=int, default=256, help="Lunghezza sequenza di test loss")
    parser.add_argument("--max_new_tokens", type=int, default=45, help="Token da generare per prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperatura di generazione")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt singolo personalizzato")
    parser.add_argument("--compare_gpt2", action="store_true", default=True, help="Abilita confronto con GPT-2 Small")
    parser.add_argument("--output", type=str, default=None, help="Percorso report markdown")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_custom = tiktoken.get_encoding("gpt2")

    report_lines = []
    def rprint(text: str = ""):
        print(text)
        report_lines.append(text)

    rprint("=" * 75)
    rprint("🔬 VALUTAZIONE E BENCHMARK COMPARATIVO — Progetto 21.0 (FAM-LLM)")
    rprint(f"💻 Device: {device} | Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rprint("=" * 75)

    # 1. Carica Modello Locale FAM-LLM
    proj_dir = Path(__file__).parent.parent
    checkpoints_dir = proj_dir / "checkpoints"
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        step_files = list(checkpoints_dir.glob("fam_step_*.pt")) if checkpoints_dir.exists() else []
        if step_files:
            def extract_step(p):
                try:
                    return int(p.stem.split("_")[-1])
                except Exception:
                    return -1
            ckpt_path = max(step_files, key=extract_step)
        else:
            ckpt_path = checkpoints_dir / "fam_latest.pt"

    if ckpt_path and ckpt_path.exists():
        rprint(f"📦 Caricamento FAM-LLM da: {ckpt_path.name}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        config = checkpoint.get("config", {
            "vocab_size": 50257,
            "d_model": 384,
            "n_layers": 6,
            "num_heads": 6,
            "num_kv_heads": 2,
            "chunk_size": 32,
        })
        step = checkpoint.get("step", "Inconosciuto")
        loss_ckpt = checkpoint.get("loss", None)
        rprint(f"   Step salvato: {step} | Loss salvata: {f'{loss_ckpt:.4f}' if loss_ckpt else 'N/A'}")
        model_fam = FourierAssociativeMatrixLLM(**config).to(device)
        model_fam.load_state_dict(checkpoint["model_state_dict"])
    else:
        step = "Iniziale"
        rprint("⚠️ Checkpoint non trovato — inizializzazione modello FAM-LLM base.")
        config = {
            "vocab_size": 50257,
            "d_model": 384,
            "n_layers": 6,
            "num_heads": 6,
            "num_kv_heads": 2,
            "chunk_size": 32,
        }
        model_fam = FourierAssociativeMatrixLLM(**config).to(device)

    model_fam.eval()
    fam_params = sum(p.numel() for p in model_fam.parameters())
    rprint(f"🧠 Parametri FAM-LLM (Progetto 21.0): {fam_params / 1e6:.2f}M")

    # 2. Carica GPT-2 Small da HuggingFace se richiesto
    model_gpt2 = None
    tokenizer_gpt2 = None
    if args.compare_gpt2 and HAS_GPT2_HF:
        rprint("\n📦 Caricamento GPT-2 Small (HuggingFace 124M)...")
        try:
            tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
            model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
            model_gpt2.eval()
            rprint("✅ GPT-2 Small caricato con successo!")
        except Exception as e:
            rprint(f"⚠️ Impossibile caricare GPT-2 HuggingFace ({e}). Solo modello locale.")

    # 3. Metriche Loss & Perplexity
    rprint("\n" + "-" * 75)
    rprint("📊 CALCOLO METRICHE DI VALUTAZIONE (Dataset Reale GPT-2)")
    dataloader, dataset = get_dataloader(seq_len=args.seq_len, batch_size=4, shuffle=False, max_tokens=10000)
    x, y = next(iter(dataloader))
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        _, loss, _ = model_fam(x, targets=y)
        ppl_fam = math.exp(min(loss.item(), 20.0))

    rprint(f"   FAM-LLM Cross-Entropy Loss : {loss.item():.4f}")
    rprint(f"   FAM-LLM Perplexity (PPL)   : {ppl_fam:.2f}")

    if model_gpt2:
        with torch.no_grad():
            outputs = model_gpt2(x, labels=y)
            loss_gpt2 = outputs.loss.item()
            ppl_gpt2 = math.exp(min(loss_gpt2, 20.0))
        rprint(f"   GPT-2 Small Loss           : {loss_gpt2:.4f}")
        rprint(f"   GPT-2 Small Perplexity     : {ppl_gpt2:.2f}")

    # 4. Suite Prompt Comparativi
    rprint("\n" + "=" * 75)
    rprint("💬 SUITE GENERAZIONE PROMPT COMPARATIVI")
    rprint("=" * 75)

    default_prompts = [
        "The earthquake victims were",
        "The scientific breakthrough in Fourier analysis",
        "Artificial intelligence will transform",
        "In modern computer architectures",
        "The history of deep learning shows",
        "Mathematics provides the foundation for",
    ]

    prompts = [args.prompt] if args.prompt else default_prompts
    comparison_results = []

    for i, prompt in enumerate(prompts, 1):
        rprint(f"\n--- [{i}/{len(prompts)}] Prompt: \"{prompt}\" ---")

        # FAM-LLM O(1)
        prompt_ids = torch.tensor(tokenizer_custom.encode(prompt), dtype=torch.long, device=device)
        with torch.no_grad():
            gen_ids = model_fam.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        output_fam = tokenizer_custom.decode(gen_ids).strip()
        print(f"🔮 FAM-LLM (O(1)):  \033[96m{prompt} {output_fam}\033[0m")

        # GPT-2 Small
        output_gpt2 = "N/A"
        if model_gpt2:
            input_hf = tokenizer_gpt2.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen_hf = model_gpt2.generate(
                    input_hf,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    do_sample=True,
                    pad_token_id=tokenizer_gpt2.eos_token_id,
                )
            output_gpt2 = tokenizer_gpt2.decode(gen_hf[0][input_hf.shape[1]:], skip_special_tokens=True).strip()
            print(f"🤖 GPT-2 (HF 124M): \033[93m{prompt} {output_gpt2}\033[0m")

        comparison_results.append({
            "prompt": prompt,
            "fam": output_fam,
            "gpt2": output_gpt2,
        })

    # 5. Test Associative Recall (Needle in a Haystack)
    rprint("\n" + "-" * 75)
    rprint("🔍 TEST MEMORIA ASSOCIATIVA (Needle-in-a-Haystack - Distance 128)")
    rprint("-" * 75)
    needle_prompt, needle_response = test_associative_recall(model_fam, tokenizer_custom, device, distance=128)
    rprint(f"   Target risposta attesa : 'ALPHA_99'")
    rprint(f"   Risposta FAM-LLM O(1)  : '{needle_response}'")

    # 6. Salva Report Markdown
    reports_dir = proj_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else (reports_dir / f"eval_compare_gpt2_step{step}_{ts}.md")

    md_content = f"""# Report Comparativo Progetto 21.0 (FAM-LLM) vs GPT-2 Small

**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Step Checkpoint:** `{step}`  
**Device:** `{device}`  
**Parametri FAM-LLM:** `{fam_params / 1e6:.2f}M` | **GPT-2 Small:** `124M`  

---

## 📊 Metriche e Perplexity

| Modello | Architecture | Cross-Entropy Loss | Perplexity (PPL) | Complexity |
|---|---|---|---|---|
| **FAM-LLM 30M** | Fourier Associative Matrix | `{loss.item():.4f}` | `{ppl_fam:.2f}` | **O(1) Recurrent / O(N) Chunked** |
| **GPT-2 Small 124M** | Standard Transformer Self-Attention | {f"{loss_gpt2:.4f}" if model_gpt2 else "N/A"} | {f"{ppl_gpt2:.2f}" if model_gpt2 else "N/A"} | O(N^2) Self-Attention |

---

## 💬 Prompt Generati (Side-by-Side)

"""
    for res in comparison_results:
        md_content += f"### Prompt: `{res['prompt']}`\n\n"
        md_content += f"**🔮 FAM-LLM (Progetto 21.0 - 30M O(1)):**\n```text\n{res['prompt']} {res['fam']}\n```\n\n"
        if model_gpt2:
            md_content += f"**🤖 GPT-2 Small (HuggingFace - 124M):**\n```text\n{res['prompt']} {res['gpt2']}\n```\n\n"
        md_content += "---\n"

    md_content += f"""
## 🔍 Test Memoria Associativa (Needle-in-a-Haystack)

- **Coppia Chiave-Valore:** `The secret access code is ALPHA_99.`
- **Distanza Contesto Rumore:** `128 token`
- **Output Recuperato da Matrix State:** `{needle_response}`

---
*Report generato automaticamente da `src/evaluate.py`.*
"""

    out_path.write_text(md_content, encoding="utf-8")
    print(f"\n📄 Report salvato in: {out_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
