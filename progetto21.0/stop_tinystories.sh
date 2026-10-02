#!/usr/bin/env bash
# ==============================================================================
# Script di ARRESTO Training TinyStories (Progetto 21.0)
# ==============================================================================

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJ_DIR}"

PID_FILE="training_tinystories.pid"

if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "🛑 Arresto del processo TinyStories in corso (PID: ${PID})..."
        kill "$PID" 2>/dev/null
        sleep 2
        if ps -p "$PID" > /dev/null 2>&1; then
            kill -9 "$PID" 2>/dev/null
        fi
        rm -f "${PID_FILE}"
        echo "✅ Training TinyStories arrestato con successo."
    else
        echo "ℹ️ Il processo con PID ${PID} non risulta attivo."
        rm -f "${PID_FILE}"
    fi
else
    # Cerca eventuali processi residui per nome
    PIDS=$(pgrep -f "train_tinystories.py")
    if [ -n "$PIDS" ]; then
        echo "🛑 Arresto dei processi train_tinystories.py residui: ${PIDS}"
        kill -9 $PIDS 2>/dev/null
        echo "✅ Processi terminati."
    else
        echo "ℹ️ Nessun training TinyStories attivo."
    fi
fi
