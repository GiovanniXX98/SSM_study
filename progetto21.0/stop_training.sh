#!/usr/bin/env bash
# ==============================================================================
# Script di arresto training per Progetto 21.0
# ==============================================================================

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJ_DIR}"

if [ -f "training.pid" ]; then
    PID=$(cat training.pid)
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "🛑 Invio segnale SIGTERM al processo training PID: ${PID}..."
        kill -15 "$PID"
        sleep 2
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "⚠️ Processo ancora attivo, forzo terminazione con SIGKILL..."
            kill -9 "$PID"
        fi
        rm -f training.pid
        echo "✅ Training arrestato con successo."
        exit 0
    else
        echo "ℹ️ Il processo PID ${PID} non risulta attivo."
        rm -f training.pid
    fi
fi

# Controllo generico su eventuali script di training di progetto21.0
PIDS=$(pgrep -f "progetto21.0/src/train")
if [ -n "$PIDS" ]; then
    echo "🛑 Rilevati processi orfani train ($PIDS), arresto in corso..."
    kill -15 $PIDS
    sleep 1
    echo "✅ Processi terminati."
else
    echo "ℹ️ Nessun processo di training per Progetto 21.0 trovato attivo."
fi
