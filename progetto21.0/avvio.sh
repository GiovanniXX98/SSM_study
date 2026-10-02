#!/usr/bin/env bash
# ==============================================================================
# Script di AVVIO Training Olografico in NOHUP (Progetto 21.0: FAM-LLM 30M)
# Target: Full Run OpenWebText (60 Milioni di Token)
# ==============================================================================

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${PROJ_DIR}/.." && pwd)"
PYTHON_EXEC="${ROOT_DIR}/.venv/bin/python"

if [ ! -f "${PYTHON_EXEC}" ]; then
    PYTHON_EXEC="python3"
fi

cd "${PROJ_DIR}"

# Verifica se il training è già attivo
if [ -f "training.pid" ]; then
    PID=$(cat training.pid)
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "⚠️ Training già in esecuzione con PID: ${PID}"
        echo "   Puoi seguire i log con: tail -f training.log"
        echo "   Oppure fermarlo con:    ./stop_training.sh"
        exit 1
    fi
fi

# Parametri impostati per 180.000.000 token (3 epoche complete di OpenWebText):
# Batch 8 * Seq 256 = 2.048 token/step -> ~87.890 steps totali
MAX_TOKENS=${1:-180000000}
BATCH_SIZE=${2:-8}
SEQ_LEN=${3:-256}

TOKENS_PER_STEP=$((BATCH_SIZE * SEQ_LEN))
DEFAULT_STEPS=$(( (MAX_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP ))
STEPS=${4:-$DEFAULT_STEPS}

echo "======================================================================="
echo "🌌 AVVIO TRAINING FINALE — Progetto 21.0 (FAM-LLM 30M Multi-Epoch)"
echo "⚡ Algoritmo:  AdamW con Matrice degli Stati K^T*V e Decadimento Ritenuto"
echo "🧠 Modello:    30 Milioni di parametri (FAM Associative Matrix)"
echo "🎯 Token:      ${MAX_TOKENS} token (3 Epoche da OpenWebText GPT-2)"
echo "🔢 Step Totali:${STEPS} | Batch: ${BATCH_SIZE} | Sequenza: ${SEQ_LEN}"
echo "📦 Throughput: ${TOKENS_PER_STEP} token/step"
echo "📁 Cartella:   ${PROJ_DIR}"
echo "======================================================================="

# Pulizia vecchi log se necessario
if [ -f "training.log" ]; then
    mv training.log "training_last.log" 2>/dev/null
fi

# Scheduler e intervalli riproporzionati per ~88k steps
nohup "${PYTHON_EXEC}" -u src/train_holographic.py \
    --max_steps "${STEPS}" \
    --max_tokens "${MAX_TOKENS}" \
    --batch_size "${BATCH_SIZE}" \
    --seq_len "${SEQ_LEN}" \
    --d_model 512 \
    --n_layers 8 \
    --num_heads 8 \
    --lr 6e-4 \
    --min_lr 3e-5 \
    --warmup_steps 1000 \
    --save_interval 2000 \
    --sample_interval 1000 \
    > training.log 2>&1 &

PID=$!
disown $PID
echo "${PID}" > training.pid

echo "✅ PROCESSO AVVIATO CON SUCCESSO IN NOHUP (PID: ${PID})"
echo "-----------------------------------------------------------------------"
echo "📝 Log live:          tail -f training.log"
echo "📡 Dashboard con ETA: ${PYTHON_EXEC} src/monitor_training.py --watch --max_steps ${STEPS}"
echo "📊 Snapshot grafico:  ${PYTHON_EXEC} src/monitor_training.py --plot"
echo "🔬 Confronto GPT-2:   ${PYTHON_EXEC} src/compare_gpt2.py"
echo "🛑 Per arrestare:     ./stop_training.sh"
echo "======================================================================="