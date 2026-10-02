"""
Prepare Dataset for Progetto 21.0 (FAM-LLM)
===========================================
Estrae esattamente i primi 100.000 token dal corpus pre-tokenizzato GPT-2 (OpenWebText),
salvandoli sia in formato compatto binario (.bin uint16) sia come testo decodificato (.txt)
con relativi metadati JSON.
"""

import os
import json
from pathlib import Path
import numpy as np
import tiktoken

def prepare_100k_dataset(
    target_tokens: int = 100_000,
    source_bin: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    workspace_root = Path(__file__).parent.parent.parent
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_bin_path = output_dir / f"gpt2_{target_tokens//1000}k_tokens.bin"
    out_txt_path = output_dir / f"gpt2_{target_tokens//1000}k_tokens.txt"
    out_json_path = output_dir / "dataset_info.json"

    if source_bin is None:
        candidate_sources = [
            workspace_root / "progetto20.0" / "data" / "openwebtext_large.bin",
            workspace_root / "data" / "openwebtext_large.bin",
        ]
        for src in candidate_sources:
            if src.exists() and src.stat().st_size >= target_tokens * 2:
                source_bin = src
                break

    tokenizer = tiktoken.get_encoding("gpt2")

    if source_bin and source_bin.exists():
        print(f"📖 Lettura primi {target_tokens:,} token da sorgente GPT-2: {source_bin}")
        raw_data = np.memmap(source_bin, dtype=np.uint16, mode="r")
        tokens = np.array(raw_data[:target_tokens], dtype=np.uint16)
    else:
        # Fallback se non c'è il binario gigante: carica testo e tokenizza
        txt_candidate = workspace_root / "progetto20.0" / "data" / "openwebtext_large.txt"
        if txt_candidate.exists():
            print(f"🔄 Tokenizzazione dei primi token da {txt_candidate} ...")
            with open(txt_candidate, "r", encoding="utf-8", errors="ignore") as f:
                # Leggi ~600KB di testo (più che sufficiente per 100k token)
                text = f.read(1_000_000)
            token_list = tokenizer.encode(text, allowed_special="all")[:target_tokens]
            tokens = np.array(token_list, dtype=np.uint16)
        else:
            raise FileNotFoundError("Impossibile trovare un corpus GPT-2 sorgente valido nel workspace.")

    # Salva il binario
    with open(out_bin_path, "wb") as f:
        f.write(tokens.tobytes())

    # Decodifica il testo e salva
    text_content = tokenizer.decode(tokens.tolist())
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    # Info e metadati
    info = {
        "dataset_name": f"GPT-2 OpenWebText First {target_tokens} Tokens",
        "num_tokens": int(len(tokens)),
        "vocab_size": tokenizer.n_vocab,
        "tokenizer": "tiktoken gpt2",
        "char_count": len(text_content),
        "source": str(source_bin) if source_bin else "openwebtext_large.txt",
        "binary_file": str(out_bin_path.name),
        "text_file": str(out_txt_path.name),
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"✅ Dataset preparato con successo:")
    print(f"   - File binario: {out_bin_path} ({out_bin_path.stat().st_size:,} byte, {len(tokens):,} token)")
    print(f"   - File testo:   {out_txt_path} ({len(text_content):,} caratteri)")
    print(f"   - Info JSON:    {out_json_path}")
    return out_bin_path, out_txt_path

if __name__ == "__main__":
    prepare_100k_dataset()
