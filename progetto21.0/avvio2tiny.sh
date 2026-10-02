#!/usr/bin/env bash
# ==============================================================================
# Script di AVVIO Training TinyStories BPE in NOHUP (Progetto 21.0: FAM-LLM 30M)
# Target: Dataset TinyStories (GPT-2 BPE 50.257 Token, MemMap Zero RAM)
# Compatibile al 100% per trasferimento su altro PC (Auto-detect CUDA / CPU)
# ==============================================================================

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${PROJ_DIR}/.." && pwd)"
PYTHON_EXEC="${ROOT_DIR}/.venv/bin/python"

if [ ! -f "${PYTHON_EXEC}" ]; then
    PYTHON_EXEC="python3"
fi

cd "${PROJ_DIR}"

PID_FILE="training_tinystories.pid"
LOG_FILE="training_tinystories.log"

# Verifica se il training TinyStories è già attivo
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️ Training TinyStories già in esecuzione con PID: ${OLD_PID}"
        echo "   Puoi seguire i log con: tail -f ${LOG_FILE}"
        echo "   Oppure fermarlo con:    ./stop_tinystories.sh"
        exit 1
    fi
fi

# Parametri personalizzabili da riga di comando
STEPS=${1:-1000}
BATCH_SIZE=${2:-8}
SEQ_LEN=${3:-128}
LR=${4:-6e-4}

echo "======================================================================="
echo "🌌 AVVIO TRAINING TINYSTORIES BPE — Progetto 21.0 (FAM-LLM 30M)"
echo "⚡ Algoritmo:      Fourier Associative Matrix (K^T * V in C^{HDxHD})"
echo "🧠 Architettura:   29.94 Milioni di parametri (FAM 6 Layers, d_model 384)"
echo "📖 Vocabolario:    GPT-2 BPE Standard (50.257 token)"
echo "🎯 Step Target:    ${STEPS} | Batch: ${BATCH_SIZE} | Sequenza: ${SEQ_LEN}"
echo "📁 Cartella:       ${PROJ_DIR}"
echo "💾 Checkpoints:    ${PROJ_DIR}/checkpoints/tinystories/"
echo "======================================================================="

# Rilevamento Hardware prima dell'avvio
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi > /dev/null 2>&1; then
    echo "🟢 GPU NVIDIA RILEVATA: Il training sfrutterà l'accelerazione CUDA!"
else
    echo "🟡 NESSUNA GPU RILEVATA: Il training userà tutti i core CPU multicore disponibili."
fi

# Archivia vecchio log se presente
if [ -f "${LOG_FILE}" ]; then
    mv "${LOG_FILE}" "training_tinystories_last.log" 2>/dev/null
fi

# Avvio del processo in background con nohup disown
nohup "${PYTHON_EXEC}" -u src/train_tinystories.py \
    --max_steps "${STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --seq_len "${SEQ_LEN}" \
    --lr "${LR}" \
    --eval_interval 50 \
    --device "auto" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
disown $PID
echo "${PID}" > "${PID_FILE}"

echo "✅ PROCESSO AVVIATO CON SUCCESSO IN NOHUP (PID: ${PID})"
echo "-----------------------------------------------------------------------"
echo "📝 Log live:          tail -f ${LOG_FILE}"
echo "📡 Dashboard grafica: ${PYTHON_EXEC} src/monitor_training.py --csv tinystories_bpe_telemetry.csv --max_steps ${STEPS}"
echo "📊 Snapshot grafico:  ${PYTHON_EXEC} src/monitor_training.py --csv tinystories_bpe_telemetry.csv --plot"
echo "🛑 Per arrestare:     ./stop_tinystories.sh"
echo "======================================================================="
