#!/usr/bin/env bash
# Всё исследование одной командой: Last.fm 1K → MovieLens-100k → отчёт → пример предсказания.
# На видеокарте уровня T4 занимает около получаса, почти всё время — обучение на Last.fm.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

bash scripts/reproduce_lastfm.sh
bash scripts/train_movielens.sh
"$PY" -m twotower.report
bash scripts/predict.sh movielens 3
