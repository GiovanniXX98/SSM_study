#!/usr/bin/env python3
"""
Monitor Training Real-Time — Progetto 21.0: Fourier Associative Matrix LLM (FAM-LLM)
=====================================================================================
"""

import sys
import os
import time
import math
import re
import argparse
from pathlib import Path

# Imposta backend headless prima di matplotlib
import matplotlib
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

PROJ_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJ_DIR / "training_telemetry.csv"
DEFAULT_GRAPH = PROJ_DIR / "telemetry_graph.png"
DEFAULT_LOG = PROJ_DIR / "training.log"


def detect_max_steps_from_env_or_log(fallback: int = 29297) -> int:
    """Tenta di rilevare i veri max_steps da training.log."""
    if DEFAULT_LOG.exists():
        try:
            with open(DEFAULT_LOG, "r", encoding="utf-8", errors="ignore") as f:
                head = [f.readline() for _ in range(50)]
            for line in head:
                # Cerca pattern tipo: Step Totali: 29297, Max Step: 500 o --max_steps 29297
                m = re.search(r"(?:Step Totali|Max Step|max_steps)[:\s=]+(\d+)", line, re.IGNORECASE)
                if m:
                    val = int(m.group(1))
                    if val > 0:
                        return val
        except Exception:
            pass
    return fallback


def load_telemetry(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    try:
        # Legge il contenuto del file in memoria per evitare collisioni con le scritture attive
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if len(lines) < 2:
            return None
        from io import StringIO
        df = pd.read_csv(StringIO("".join(lines)), on_bad_lines="skip", engine="python")
        if df.empty:
            return None

        df.rename(columns={"learning_rate": "lr"}, inplace=True)

        for col in ["step", "loss", "perplexity", "lr", "tokens_per_sec", "elapsed_sec", "epoch"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["step", "loss"])
        if df.empty:
            return None

        df["step"] = df["step"].astype(int)
        df.sort_values(by="step", inplace=True)
        df.drop_duplicates(subset=["step"], keep="last", inplace=True)
        return df
    except Exception:
        return None


def format_time(seconds: float) -> str:
    if seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
        return "--:--:--"
    sec = int(seconds)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        return f"{d}g {h:02d}h:{m:02d}m"
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def calculate_eta(df: pd.DataFrame, current_step: int, target_steps: int) -> str:
    remaining_steps = max(0, target_steps - current_step)
    if remaining_steps == 0:
        return "Completato"

    sps = 0.0

    # 1. Finestra mobile recente (ultimi 30 step registrati)
    window = df.tail(30)
    if len(window) >= 3:
        s_diff = window["step"].iloc[-1] - window["step"].iloc[0]
        t_diff = window["elapsed_sec"].iloc[-1] - window["elapsed_sec"].iloc[0]
        if s_diff > 0 and t_diff > 0.5:
            sps = s_diff / t_diff

    # 2. Fallback su tempo complessivo
    if sps <= 0.001:
        total_time = df["elapsed_sec"].iloc[-1]
        done_steps = current_step - df["step"].iloc[0]
        if done_steps > 0 and total_time > 1.0:
            sps = done_steps / total_time

    # 3. Fallback sul throughput token/s
    if sps <= 0.001 and "tokens_per_sec" in df.columns:
        tok_s = df["tokens_per_sec"].iloc[-1]
        if tok_s > 100:
            # Batch 8 x Seq 256 = 2048 token per step
            sps = tok_s / 2048.0

    if sps <= 0.001:
        return "--:--:--"

    return format_time(remaining_steps / sps)


def display_dashboard(df: pd.DataFrame, target_steps: int, is_inline: bool = True):
    if df is None or df.empty:
        msg = "⏳ In attesa delle prime metriche in training_telemetry.csv..."
        if sys.stdout.isatty():
            sys.stdout.write(f"\r\033[K{msg}")
            sys.stdout.flush()
        else:
            print(msg)
        return

    latest = df.iloc[-1]
    step = int(latest["step"])
    loss = float(latest["loss"])
    ppl = float(latest.get("perplexity", np.exp(min(loss, 15))))
    lr = float(latest.get("lr", 0.0))
    tok_s = float(latest.get("tokens_per_sec", 0.0))
    elapsed = float(latest.get("elapsed_sec", 0.0))

    # Auto-correzione target se max_steps passato è inferiore allo step corrente
    if target_steps <= step:
        log_steps = detect_max_steps_from_env_or_log(fallback=29297)
        if log_steps > step:
            target_steps = log_steps
        else:
            target_steps = max(29297, int(step * 1.2))

    trend = "➡️"
    if len(df) >= 5:
        prev_loss = df["loss"].iloc[-5]
        if loss < prev_loss - 0.05:
            trend = "📉"
        elif loss > prev_loss + 0.05:
            trend = "📈"

    pct = min(100.0, (step / target_steps) * 100.0) if target_steps > 0 else 0.0
    bar_len = 12
    filled = int(bar_len * (pct / 100.0))
    bar_str = f"[{'█' * filled}{'░' * (bar_len - filled)}]"

    eta_str = calculate_eta(df, step, target_steps)
    ppl_fmt = f"{ppl:,.2f}" if ppl < 1e5 else f"{ppl:.2e}"

    raw_line = (
        f"{bar_str} Step {step:5d}/{target_steps:<5d} ({pct:5.1f}%) | "
        f"Loss: {loss:6.2f} {trend} | PPL: {ppl_fmt:>8} | "
        f"LR: {lr:.2e} | Speed: {tok_s:6.0f} tok/s | "
        f"Tempo: {format_time(elapsed)} | ETA: {eta_str}"
    )

    if is_inline:
        if sys.stdout.isatty():
            sys.stdout.write(f"\r\033[K{raw_line}")
            sys.stdout.flush()
        else:
            print(raw_line)
    else:
        print(raw_line)


def generate_plot(df: pd.DataFrame, out_path: Path = DEFAULT_GRAPH) -> bool:
    if df is None or len(df) < 2:
        return False

    steps = df["step"].values
    loss = df["loss"].values
    ppl = df["perplexity"].values if "perplexity" in df.columns else np.exp(np.clip(loss, 0, 15))
    lr = df["lr"].values if "lr" in df.columns else np.zeros_like(steps)
    tok_s = df["tokens_per_sec"].values if "tokens_per_sec" in df.columns else np.zeros_like(steps)

    fig = plt.figure(figsize=(14, 8), facecolor="#0e1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_ppl = fig.add_subplot(gs[0, 1])
    ax_lr = fig.add_subplot(gs[1, 0])
    ax_tok = fig.add_subplot(gs[1, 1])

    for ax in [ax_loss, ax_ppl, ax_lr, ax_tok]:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    ax_loss.plot(steps, loss, color="#ff4b4b", linewidth=1.5, alpha=0.5, label="Loss")
    if len(loss) >= 5:
        w = min(25, len(loss))
        ax_loss.plot(steps, pd.Series(loss).rolling(w, min_periods=1).mean(), color="#ff2a2a", linewidth=2.0, label="MA")
    ax_loss.set_title("Cross-Entropy Loss", color="#f0f6fc", fontsize=11, fontweight="bold")
    ax_loss.grid(True, alpha=0.15, color="#8b949e")
    ax_loss.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

    clean_ppl = np.nan_to_num(ppl, nan=1e8, posinf=1e8, neginf=1.0)
    clean_ppl = np.clip(clean_ppl, 1.0, 1e12)
    ax_ppl.plot(steps, clean_ppl, color="#ffa600", linewidth=1.8)
    ax_ppl.set_title("Perplexity (Log Scale)", color="#f0f6fc", fontsize=11, fontweight="bold")
    ax_ppl.set_yscale("log")
    ax_ppl.grid(True, alpha=0.15, color="#8b949e")

    ax_lr.plot(steps, lr, color="#58a6ff", linewidth=2.0)
    ax_lr.set_title("Learning Rate", color="#f0f6fc", fontsize=11, fontweight="bold")
    ax_lr.grid(True, alpha=0.15, color="#8b949e")

    ax_tok.plot(steps, tok_s, color="#2ea043", linewidth=1.2, alpha=0.4)
    if len(tok_s) >= 5:
        w = min(20, len(tok_s))
        ax_tok.plot(steps, pd.Series(tok_s).rolling(w, min_periods=1).mean(), color="#3fb950", linewidth=2.0)
    ax_tok.set_title("Throughput (Tokens/s)", color="#f0f6fc", fontsize=11, fontweight="bold")
    ax_tok.grid(True, alpha=0.15, color="#8b949e")

    try:
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.35, wspace=0.25)
        plt.savefig(out_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
        return True
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Monitor Training Real-Time")
    parser.add_argument("--watch", action="store_true", help="Monitoraggio live")
    parser.add_argument("--plot", action="store_true", help="Salva snapshot PNG ed esce")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--max_steps", type=int, default=29297)
    parser.add_argument("--refresh", type=float, default=2.0)
    parser.add_argument("--save_graph_every", type=int, default=30)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    target_steps = args.max_steps

    if args.plot and not args.watch:
        df = load_telemetry(csv_path)
        if df is not None:
            generate_plot(df, out_path=DEFAULT_GRAPH)
            print(f"✅ Grafico salvato in: {DEFAULT_GRAPH}")
            display_dashboard(df, target_steps=target_steps, is_inline=False)
        return

    print("=" * 80)
    print("📡 MONITOR TELEMETRIA — Progetto 21.0 (FAM-LLM 30M)")
    print(f"📄 CSV: {csv_path}")
    print("⌨️  Premi Ctrl+C per uscire.")
    print("=" * 80)

    last_graph_update = 0.0

    try:
        while True:
            df = load_telemetry(csv_path)
            display_dashboard(df, target_steps=target_steps, is_inline=True)

            now = time.time()
            if df is not None and len(df) >= 2 and (now - last_graph_update >= args.save_graph_every):
                try:
                    generate_plot(df, out_path=DEFAULT_GRAPH)
                    last_graph_update = now
                except Exception:
                    pass

            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoraggio interrotto.")


if __name__ == "__main__":
    main()