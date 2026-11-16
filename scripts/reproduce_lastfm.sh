#!/usr/bin/env bash
# Демонстрационный датасет авторов (Last.fm 1K): воспроизведение заявленных метрик.
#   bash scripts/reproduce_lastfm.sh         # оба шага
#   bash scripts/reproduce_lastfm.sh eval    # 1) оценить опубликованные веса авторов
#   bash scripts/reproduce_lastfm.sh train   # 2) обучить с нуля с гиперпараметрами авторов и оценить
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

STAGE="${1:-all}"
if [[ "$STAGE" != all && "$STAGE" != eval && "$STAGE" != train ]]; then
    echo "использование: bash scripts/reproduce_lastfm.sh [all|eval|train]" >&2
    exit 1
fi

bash scripts/download.sh lastfm

if [[ "$STAGE" != train ]]; then
    echo "=== Last.fm 1K: оценка опубликованных весов авторов ==="
    bash scripts/download.sh lastfm-weights
    "$PY" -m twotower.evaluate --weights weights/loopback_two_tower_epoch3.pt --out results/lastfm_author.json
fi

if [[ "$STAGE" != eval ]]; then
    echo "=== Last.fm 1K: обучение с нуля (3 эпохи, батч 4096, lr 1e-3, weight decay 1e-5) ==="
    "$PY" -m twotower.train --dataset lastfm --epochs 3 --batch-size 4096 --lr 1e-3 --weight-decay 1e-5 \
        --out weights/twotower_lastfm.pt --results results/lastfm_retrained.json
fi
