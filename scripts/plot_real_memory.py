#!/usr/bin/env python3
"""
Script per generare il grafico ad alta risoluzione del benchmark scientifico e imparziale:
FSTLLM 2.0 (Target Checkpoint) vs Standard Transformer con KV-Cache reale (SDPA).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT_DIR / "benchmark_true_process_memory.json"
OUTPUT_DIR = ROOT_DIR / "grafici"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_PATH = OUTPUT_DIR / "benchmark_vera_memoria_processo_30M.png"

def main():
    if not JSON_PATH.exists():
        print(f"❌ File JSON non trovato: {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    comp = data["comparison"]
    contexts = [c["context_length"] for c in comp]
    fst_total = [c["fstllm_total_rss_mb"] for c in comp]
    tr_total = [c["transformer_total_rss_mb"] for c in comp]
    fst_cache = [c["fstllm_cache_kb"] for c in comp]
    tr_cache = [c["transformer_cache_kb"] for c in comp]
    fst_speed = [c["fstllm_speed_tok_s"] for c in comp]
    tr_speed = [c["transformer_speed_tok_s"] for c in comp]

    # Stile scientifico scuro elegante
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(19, 5.8), dpi=300)
    fig.patch.set_facecolor("#0f172a") # Dark Slate 900

    col_fst = "#38bdf8"       # Sky Blue neon
    col_trans = "#f43f5e"     # Rose/Red neon
    text_color = "#f8fafc"

    # --- PANNELLO 1: Memoria Totale Processo (OS RSS RAM) ---
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_facecolor("#1e293b")
    x = np.arange(len(contexts))
    width = 0.35

    rects1 = ax1.bar(x - width/2, fst_total, width, label="FSTLLM 2.0 (Target Ckpt)", color=col_fst, edgecolor="#0284c7", linewidth=1.2, alpha=0.9)
    rects2 = ax1.bar(x + width/2, tr_total, width, label="Standard Transformer (KV-Cache)", color=col_trans, edgecolor="#e11d48", linewidth=1.2, alpha=0.9)

    ax1.set_title("A. Memoria Totale Processo (OS RSS RAM)", fontsize=12.5, fontweight="bold", color=text_color, pad=12)
    ax1.set_xlabel("Lunghezza Contesto (Token)", fontsize=11, color="#cbd5e1", fontweight="bold")
    ax1.set_ylabel("RAM Totale Processo (MB)", fontsize=11, color="#cbd5e1", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(contexts, color=text_color, fontweight="bold")
    ax1.tick_params(colors="#94a3b8", labelsize=10)
    ax1.set_ylim(0, 650)
    ax1.grid(color="#334155", linestyle="--", linewidth=0.7, alpha=0.7)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}M", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontsize=8.5, color=col_fst, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}M", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontsize=8.5, color=col_trans, fontweight="bold")

    leg1 = ax1.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor=text_color, fontsize=9.5, loc="upper left")
    leg1.get_frame().set_alpha(0.9)

    # --- PANNELLO 2: Memoria Cache / Stato di Inferenza (Scala Log) ---
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_facecolor("#1e293b")

    ax2.plot(contexts, fst_cache, marker="o", linewidth=2.5, markersize=8, color=col_fst, label="FSTLLM: Stato O(1) [67.5 KB]")
    ax2.plot(contexts, tr_cache, marker="s", linewidth=2.5, markersize=8, color=col_trans, label="Transformer: KV-Cache O(S)")

    ax2.set_yscale("log")
    ax2.set_title("B. Dimensione Memoria di Cache (KB - Scala Log)", fontsize=12.5, fontweight="bold", color=text_color, pad=12)
    ax2.set_xlabel("Lunghezza Contesto (Token)", fontsize=11, color="#cbd5e1", fontweight="bold")
    ax2.set_ylabel("Dimensione Cache (KB)", fontsize=11, color="#cbd5e1", fontweight="bold")
    ax2.set_xticks(contexts)
    ax2.set_xticklabels(contexts, color=text_color, fontweight="bold")
    ax2.tick_params(colors="#94a3b8", labelsize=10)
    ax2.grid(color="#334155", linestyle="--", linewidth=0.7, alpha=0.7)

    # Annotazione all'ultimo punto
    ax2.annotate(f"FSTLLM: {fst_cache[-1]} KB", (contexts[-1], fst_cache[-1]),
                 xytext=(-85, -15), textcoords="offset points", color=col_fst, fontweight="bold", fontsize=9)
    ax2.annotate(f"Trans: {tr_cache[-1]/1024:.1f} MB (x1102)", (contexts[-1], tr_cache[-1]),
                 xytext=(-110, 10), textcoords="offset points", color=col_trans, fontweight="bold", fontsize=9)

    leg2 = ax2.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor=text_color, fontsize=9.5, loc="center left")
    leg2.get_frame().set_alpha(0.9)

    # --- PANNELLO 3: Velocità di Decodifica Reale (Tokens/sec) ---
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor("#1e293b")

    ax3.plot(contexts, fst_speed, marker="o", linewidth=2.5, markersize=8, color=col_fst, label="FSTLLM 2.0")
    ax3.plot(contexts, tr_speed, marker="s", linewidth=2.5, markersize=8, color=col_trans, label="Standard Transformer")

    ax3.set_title("C. Throughput di Decodifica Reale (Token/s)", fontsize=12.5, fontweight="bold", color=text_color, pad=12)
    ax3.set_xlabel("Lunghezza Contesto (Token)", fontsize=11, color="#cbd5e1", fontweight="bold")
    ax3.set_ylabel("Velocità di Generazione (tok/s)", fontsize=11, color="#cbd5e1", fontweight="bold")
    ax3.set_xticks(contexts)
    ax3.set_xticklabels(contexts, color=text_color, fontweight="bold")
    ax3.tick_params(colors="#94a3b8", labelsize=10)
    ax3.set_ylim(0, 16)
    ax3.grid(color="#334155", linestyle="--", linewidth=0.7, alpha=0.7)

    for i, txt in enumerate(fst_speed):
        ax3.annotate(f"{txt} t/s", (contexts[i], fst_speed[i] + 0.6), color=col_fst, fontsize=8.5, ha="center", fontweight="bold")
    for i, txt in enumerate(tr_speed):
        ax3.annotate(f"{txt} t/s", (contexts[i], tr_speed[i] + 0.6), color=col_trans, fontsize=8.5, ha="center", fontweight="bold")

    leg3 = ax3.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor=text_color, fontsize=9.5, loc="upper right")
    leg3.get_frame().set_alpha(0.9)

    # Titolo Generale
    fig.suptitle("Confronto Scientifico Imparziale — FSTLLM 2.0 vs Standard Transformer (Processi Isolati, 30M, Intel i5)",
                 fontsize=14.5, fontweight="bold", color="#ffffff", y=1.02)

    plt.tight_layout()
    plt.savefig(IMG_PATH, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"✅ Grafico generato con successo in: {IMG_PATH}")

if __name__ == "__main__":
    main()
