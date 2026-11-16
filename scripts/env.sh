#!/usr/bin/env bash
# Готовит окружение и выставляет переменную PY. Подключается из остальных скриптов:
#   source scripts/env.sh
#
# По умолчанию создаёт venv в .venv и ставит туда зависимости. Настройки:
#   PYTHON=python3.12      каким интерпретатором создавать venv
#   USE_SYSTEM_PYTHON=1    не создавать venv, работать в системном python
#                          (в Google Colab включается само: torch с CUDA там уже установлен)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

if [[ "${USE_SYSTEM_PYTHON:-0}" == 1 || -n "${COLAB_RELEASE_TAG:-}" ]]; then
    PY="$(command -v python3)"
else
    VENV="${VENV:-$ROOT/.venv}"
    if [[ ! -x "$VENV/bin/python" ]]; then
        for candidate in "${PYTHON:-}" python3.12 python3.11 python3.13 python3.10 python3; do
            if [[ -n "$candidate" ]] && command -v "$candidate" >/dev/null; then
                break
            fi
        done
        echo ">> создаю venv в $VENV ($candidate)"
        "$candidate" -m venv "$VENV"
    fi
    PY="$VENV/bin/python"
fi

if ! "$PY" -c "import torch, numpy, pandas, pyarrow" 2>/dev/null; then
    echo ">> ставлю зависимости"
    "$PY" -m pip install --quiet --upgrade pip
    if ! "$PY" -c "import torch" 2>/dev/null; then
        # на Linux без видеокарты берём CPU-сборку torch: она на порядок меньше сборки с CUDA
        if [[ "$(uname -s)" == Linux ]] && ! command -v nvidia-smi >/dev/null; then
            "$PY" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
        else
            "$PY" -m pip install --quiet torch
        fi
    fi
    "$PY" -m pip install --quiet -r requirements.txt
fi
