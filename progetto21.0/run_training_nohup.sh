#!/usr/bin/env bash
# ==============================================================================
# Script di avvio training in NOHUP per Progetto 21.0 (FAM-LLM)
# ==============================================================================

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${PROJ_DIR}/.." && pwd)"
PYTHON_EXEC="${ROOT_DIR}/.venv/bin/python"

if [ ! -f "${PYTHON_EXEC}" ]; then
    PYTHON_EXEC="python3"
fi

cd "${PROJ_DIR}"

# Verifica se il training è già in esecuzione
if [ -f "training.pid" ]; then
    PID=$(cat training.pid)
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "⚠️ Training già in esecuzione con PID: ${PID}"
        echo "   Usa ./stop_training.sh per interromperlo, oppure:"
        echo "   tail -f training.log"
        exit 1
    fi
fi

echo "======================================================================="
echo "🚀 Avvio Training NOHUP — Progetto 21.0 (FAM-LLM)"
echo "📂 Directory: ${PROJ_DIR}"
echo "🐍 Python:    ${PYTHON_EXEC}"
echo "🎯 Dataset:   Primi 100.000 token GPT-2 (OpenWebText)"
echo "⚡ Ottimizz.: AdamW (lr=5e-4, Cosine Annealing, decoupled weight decay)"
echo "======================================================================="

# Parametri personalizzabili passati allo script o default
STEPS=${1:-500}
BATCH_SIZE=${2:-4}
SEQ_LEN=${3:-256}
MAX_TOKENS=${4:-1000000}

echo "🎯 Target Token: ${MAX_TOKENS} token GPT-2 (su 62M disponibili)"

nohup "${PYTHON_EXEC}" -u src/train.py \
    --max_tokens "${MAX_TOKENS}" \
    --max_steps "${STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --seq_len "${SEQ_LEN}" \
    --lr 5e-4 \
    --min_lr 2e-5 \
    --warmup_steps 40 \
    > training.log 2>&1 &

PID=$!
echo "${PID}" > training.pid

echo "✅ Processo avviato con PID: ${PID}"
echo "📝 Log in tempo reale: tail -f training.log"
echo "📡 Monitor console:    ${PYTHON_EXEC} src/monitor_training.py --watch --max_steps ${STEPS}"
echo "📊 Snapshot grafico:   ${PYTHON_EXEC} src/monitor_training.py --plot"
echo "⏹️ Per terminare:      ./stop_training.sh"
echo "======================================================================="
