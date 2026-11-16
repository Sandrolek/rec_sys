#!/usr/bin/env bash
# Веса + датасет → вероятности следующего объекта для нескольких строк.
# Скрипт сам готовит окружение, скачивает данные и веса и запускает предсказание.
#
#   bash scripts/predict.sh               # MovieLens-100k, 3 строки, веса weights/twotower_ml100k.pt
#   bash scripts/predict.sh lastfm 2      # Last.fm 1K, 2 строки, опубликованные веса авторов (430 МБ)
#   WEIGHTS=weights/twotower_ml100k_logq.pt bash scripts/predict.sh movielens 5   # другие веса
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

DATASET="${1:-movielens}"
ROWS="${2:-3}"

case "$DATASET" in
    movielens)
        bash scripts/download.sh movielens
        WEIGHTS="${WEIGHTS:-weights/twotower_ml100k.pt}"
        if [[ ! -f "$WEIGHTS" ]]; then
            echo ">> весов $WEIGHTS нет, сначала обучаю модель"
            bash scripts/train_movielens.sh
        fi
        ;;
    lastfm)
        bash scripts/download.sh lastfm
        if [[ -z "${WEIGHTS:-}" ]]; then
            bash scripts/download.sh lastfm-weights
            WEIGHTS=weights/loopback_two_tower_epoch3.pt
        fi
        ;;
    *)
        echo "использование: bash scripts/predict.sh [movielens|lastfm] [число строк]" >&2
        exit 1
        ;;
esac

"$PY" -m twotower.predict --dataset "$DATASET" --weights "$WEIGHTS" --rows "$ROWS"
