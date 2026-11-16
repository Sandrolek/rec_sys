#!/usr/bin/env bash
# Скачивает датасеты и веса с Hugging Face в data/ и weights/. Уже скачанное пропускает.
#   bash scripts/download.sh movielens        # MovieLens-100k, 1.4 МБ
#   bash scripts/download.sh lastfm           # Last.fm 1K в обработке авторов, 190 МБ
#   bash scripts/download.sh lastfm-weights   # опубликованный чекпойнт авторов, 430 МБ
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

HF=https://huggingface.co

fetch() { # fetch URL ПУТЬ: качает во временный файл, чтобы обрыв не оставил битый
    local url="$1" dest="$2"
    if [[ -s "$dest" ]]; then
        return
    fi
    mkdir -p "$(dirname "$dest")"
    echo ">> скачиваю $dest"
    curl --location --fail --silent --show-error --retry 3 --output "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

if [[ $# -eq 0 ]]; then
    echo "использование: bash scripts/download.sh [movielens] [lastfm] [lastfm-weights]" >&2
    exit 1
fi

for target in "$@"; do
    case "$target" in
        movielens)
            fetch "$HF/datasets/DukeNLPGroup/movielens-100k/resolve/main/data/train-00000-of-00001.parquet" \
                data/ml100k/train.parquet
            ;;
        lastfm)
            for name in train val test vocab track_labels; do
                fetch "$HF/datasets/DanielRegaladoCardoso/lastfm-1k-twotower/resolve/main/$name.parquet" \
                    "data/lastfm/$name.parquet"
            done
            ;;
        lastfm-weights)
            fetch "$HF/DanielRegaladoCardoso/loopback-twotower/resolve/main/two_tower_epoch3.pt" \
                weights/loopback_two_tower_epoch3.pt
            ;;
        *)
            echo "неизвестная цель: $target (нужно movielens, lastfm или lastfm-weights)" >&2
            exit 1
            ;;
    esac
done
