#!/usr/bin/env bash
# Второй датасет: тот же алгоритм на MovieLens-100k.
# Два разбиения по времени:
#   global — как у авторов: все события на общей временной шкале, 80/10/10
#   user   — те же доли внутри истории каждого пользователя
# Для каждого разбиения:
#   asis — гиперпараметры авторов как есть (3 эпохи, батч 4096)
#   bs*  — подбор размера батча по val, до 300 эпох с ранней остановкой, остальное не меняется
#   logq — лучший из bs* плюс поправка logQ; её нет у авторов, она показывает причину проигрыша
# Размеры батча можно задать так: BATCH_SIZES="512 2048 4096" bash scripts/train_movielens.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

bash scripts/download.sh movielens
rm -f results/ml100k_*.json weights/ml100k_*.pt # результаты прошлых запусков не должны попасть в отчёт

train() { # train ИМЯ_ЗАПУСКА [параметры twotower.train]
    local run="$1"
    shift
    echo "=== $run ==="
    "$PY" -m twotower.train --dataset movielens --out "weights/$run.pt" --results "results/$run.json" "$@"
}

for SPLIT in global user; do
    train "ml100k_${SPLIT}_asis" --split "$SPLIT"
    for BS in ${BATCH_SIZES:-512 2048 4096}; do
        train "ml100k_${SPLIT}_bs${BS}" --split "$SPLIT" --batch-size "$BS" --epochs 300 --patience 20
    done
    BEST="$("$PY" -m twotower.report --best "ml100k_${SPLIT}_bs")"
    train "ml100k_${SPLIT}_logq" --split "$SPLIT" --batch-size "${BEST##*_bs}" --epochs 300 --patience 20 --logq
done

# веса для scripts/predict.sh, оба со сплита user: лучший запуск алгоритма авторов и он же с logQ
BEST="$("$PY" -m twotower.report --best ml100k_user_bs)"
cp "weights/$BEST.pt" weights/twotower_ml100k.pt
cp weights/ml100k_user_logq.pt weights/twotower_ml100k_logq.pt
echo "Лучший запуск на сплите user: $BEST, его веса скопированы в weights/twotower_ml100k.pt"
