"""Загрузка датасетов в общий формат: взаимодействия (user, item) по времени и группа каждого объекта.

Файлы заранее скачивает scripts/download.sh:
    data/lastfm/{train,val,test,vocab,track_labels}.parquet — DanielRegaladoCardoso/lastfm-1k-twotower
    data/ml100k/train.parquet                               — DukeNLPGroup/movielens-100k
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Interactions:
    name: str
    n_users: int
    n_items: int
    n_groups: int
    train: np.ndarray  # (n, 2) int64: [user, item], строки по возрастанию времени
    val: np.ndarray
    test: np.ndarray
    item_group: np.ndarray  # (n_items,) группа объекта: исполнитель трека / набор жанров фильма
    labels: Callable[[], np.ndarray]  # названия объектов, загружаются по требованию
    _keys: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.train_users = np.unique(self.train[:, 0])

    def keys(self, splits: tuple[str, ...]) -> np.ndarray:
        """Отсортированные уникальные ключи пар user * n_items + item из указанных сплитов."""
        if splits not in self._keys:
            pairs = np.concatenate([getattr(self, s) for s in splits])
            self._keys[splits] = np.unique(pairs[:, 0] * self.n_items + pairs[:, 1])
        return self._keys[splits]


def _require(path: Path, target: str) -> Path:
    if not path.exists():
        raise SystemExit(f"Нет файла {path}. Сначала: bash scripts/download.sh {target}")
    return path


def load_lastfm(data_dir: str = "data") -> Interactions:
    """Last.fm 1K в обработке авторов: готовые индексы и временной сплит 80/10/10."""
    d = Path(data_dir) / "lastfm"
    cols = ["user_idx", "track_idx", "artist_idx", "ts"]
    parts = {
        s: pd.read_parquet(_require(d / f"{s}.parquet", "lastfm"), columns=cols).sort_values("ts", kind="stable")
        for s in ("train", "val", "test")
    }
    vocab = pd.read_parquet(_require(d / "vocab.parquet", "lastfm")).iloc[0]
    n_items = int(vocab["n_tracks"])

    # исполнитель — постоянный признак трека, поэтому берём его из всех сплитов
    item_group = np.zeros(n_items, dtype=np.int64)
    for part in parts.values():
        item_group[part["track_idx"].to_numpy()] = part["artist_idx"].to_numpy()

    def labels() -> np.ndarray:
        path = _require(d / "track_labels.parquet", "lastfm")
        return pd.read_parquet(path).sort_values("track_idx")["label"].to_numpy()

    pairs = {s: np.array(p[["user_idx", "track_idx"]], dtype=np.int64) for s, p in parts.items()}
    return Interactions(
        "lastfm", int(vocab["n_users"]), n_items, int(vocab["n_artists"]), item_group=item_group, labels=labels, **pairs
    )


def train_only_groups(ds: Interactions) -> np.ndarray:
    """Группы как в демо авторов: исполнитель известен только у треков из train, у остальных 0."""
    in_train = np.zeros(ds.n_items, dtype=bool)
    in_train[ds.train[:, 1]] = True
    return np.where(in_train, ds.item_group, 0)


def load_movielens(data_dir: str = "data", split: str = "global", min_rating: float = 0.0) -> Interactions:
    """MovieLens-100k: каждая оценка не ниже min_rating — взаимодействие «посмотрел фильм».

    split="global" — как у авторов: все события по времени, старейшие 80% → train, 10% → val, 10% → test.
    split="user"   — те же доли, но внутри истории каждого пользователя (у всех есть история в train).
    """
    path = _require(Path(data_dir) / "ml100k" / "train.parquet", "movielens")
    cols = ["user_id", "movie_id", "movie_title", "movie_genres", "user_rating", "timestamp"]
    df = pd.read_parquet(path, columns=cols)
    df = df[df["user_rating"] >= min_rating].copy()

    # идентификаторы лежат байтовыми строками: b"138" → 138 → плотный индекс
    user_ids, df["user"] = np.unique(df["user_id"].map(int), return_inverse=True)
    movie_ids, df["item"] = np.unique(df["movie_id"].map(int), return_inverse=True)
    df = df.sort_values(["timestamp", "user", "item"]).reset_index(drop=True)

    movies = df.drop_duplicates("item").sort_values("item")
    # группа фильма (аналог исполнителя) — его набор жанров, например "4-14" = Comedy + Romance
    genre_sets = movies["movie_genres"].map(lambda g: "-".join(str(x) for x in sorted(g)))
    groups, item_group = np.unique(genre_sets.to_numpy(dtype=str), return_inverse=True)
    titles = movies["movie_title"].to_numpy()

    if split == "global":
        frac = np.arange(len(df)) / len(df)
    elif split == "user":
        by_user = df.groupby("user")["item"]
        frac = (by_user.cumcount() / by_user.transform("size")).to_numpy()
    else:
        raise SystemExit(f"Неизвестный сплит {split!r}: нужен global или user")
    pairs = df[["user", "item"]].to_numpy(dtype=np.int64)
    return Interactions(
        "movielens",
        len(user_ids),
        len(movie_ids),
        len(groups),
        train=pairs[frac < 0.8],
        val=pairs[(frac >= 0.8) & (frac < 0.9)],
        test=pairs[frac >= 0.9],
        item_group=item_group.astype(np.int64),
        labels=lambda: titles,
    )


def load_dataset(name: str, data_dir: str = "data", split: str = "global", min_rating: float = 0.0) -> Interactions:
    if name == "lastfm":
        return load_lastfm(data_dir)
    if name == "movielens":
        return load_movielens(data_dir, split, min_rating)
    raise SystemExit(f"Неизвестный датасет {name!r}: нужен lastfm или movielens")
