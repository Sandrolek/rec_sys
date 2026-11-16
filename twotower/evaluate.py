"""Оценка retrieval-качества: top-K по всему каталогу, несколько вариантов протокола, бейзлайны.

Авторы пишут только «Recall@K на 847 отложенных пользователях с фильтрацией прослушанных треков»,
а код оценки недоступен. Поэтому каждая модель оценивается сразу в нескольких конфигурациях
(CONFIGS) и по нескольким определениям метрики (METRICS); какой вариант совпадает с заявленными
числами, показывает twotower.report.

    python -m twotower.evaluate --weights weights/loopback_two_tower_epoch3.pt --out results/lastfm_author.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import Interactions, load_dataset, train_only_groups
from .model import load_checkpoint

KS = (10, 50, 100)
# (оцениваемый сплит, сплиты с уже виденными объектами: они убираются и из выдачи, и из целей)
CONFIGS = [("test", ("train", "val")), ("test", ("train",)), ("test", ()), ("val", ("train",)), ("val", ())]
# recall        — доля целевых объектов пользователя, попавших в top-K (среднее по пользователям)
# hit           — есть ли в top-K хотя бы один целевой объект
# recall_capped — попадания / min(K, число целевых)
# event         — доля событий периода, чей объект попал в top-K пользователя (среднее по событиям)
# first         — попал ли в top-K первый по времени целевой объект («следующий новый»)
METRICS = ("recall", "hit", "recall_capped", "event", "first")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def config_name(split: str, seen: tuple[str, ...]) -> str:
    return f"{split}/seen={'+'.join(seen) or 'none'}"


def member(x: np.ndarray, sorted_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Маска «x есть в sorted_keys» и позиции найденных элементов."""
    pos = np.searchsorted(sorted_keys, x).clip(max=len(sorted_keys) - 1)
    return sorted_keys[pos] == x, pos


@dataclass
class Targets:
    users: np.ndarray  # пользователи, у которых есть хотя бы один целевой объект
    rel_keys: np.ndarray  # отсортированные ключи целевых пар (user, item)
    rel_events: np.ndarray  # сколько событий периода приходится на каждую целевую пару
    n_rel: np.ndarray  # число целевых объектов пользователя
    n_events: np.ndarray  # число целевых событий пользователя
    first: np.ndarray  # первый по времени целевой объект пользователя
    n_seen: np.ndarray  # сколько объектов исключено из выдачи пользователя


def build_targets(ds: Interactions, split: str, seen: tuple[str, ...]) -> Targets:
    events = getattr(ds, split)
    keys = events[:, 0] * ds.n_items + events[:, 1]
    seen_keys = ds.keys(seen) if seen else None
    if seen_keys is not None:
        keys = keys[~member(keys, seen_keys)[0]]  # повторы уже виденного целями не считаются
    rel_keys, first_pos, rel_events = np.unique(keys, return_index=True, return_counts=True)
    users, start, n_rel = np.unique(rel_keys // ds.n_items, return_index=True, return_counts=True)
    if len(users) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return Targets(users, rel_keys, rel_events, n_rel, empty, empty, empty)
    first = keys[np.minimum.reduceat(first_pos, start)] % ds.n_items
    n_seen = np.zeros(len(users), dtype=np.int64)
    if seen_keys is not None:
        lo = np.searchsorted(seen_keys, users * ds.n_items)
        n_seen = np.searchsorted(seen_keys, (users + 1) * ds.n_items) - lo
    return Targets(users, rel_keys, rel_events, n_rel, np.add.reduceat(rel_events, start), first, n_seen)


@torch.no_grad()
def topk_items(score_fn, users, seen_keys, n_items: int, device, k: int = max(KS)) -> np.ndarray:
    """Top-k объектов каждого пользователя по всему каталогу; пары из seen_keys исключаются."""
    step = max(1, int(2e8) // n_items)  # около 0.8 ГБ оценок float32 на батч пользователей
    out = []
    for s in range(0, len(users), step):
        u = users[s : s + step]
        scores = score_fn(torch.as_tensor(u, device=device))  # (B, n_items)
        if seen_keys is not None:
            lo = np.searchsorted(seen_keys, u * n_items)
            hi = np.searchsorted(seen_keys, (u + 1) * n_items)
            rows = np.repeat(np.arange(len(u)), hi - lo)
            cols = np.concatenate([seen_keys[a:b] for a, b in zip(lo, hi)]) % n_items
            scores[torch.as_tensor(rows, device=device), torch.as_tensor(cols, device=device)] = -torch.inf
        out.append(scores.topk(min(k, n_items), dim=1).indices.cpu().numpy())
    return np.concatenate(out)


def model_scorer(model, item_group: np.ndarray, device):
    model.eval()
    item_vecs = model.encode_items(torch.as_tensor(item_group, device=device))
    return lambda u: model.user_tower(u) @ item_vecs.T


def popularity_scorer(ds: Interactions, device):
    """Бейзлайн: популярность объекта = число разных пользователей с ним в train."""
    counts = np.bincount(ds.keys(("train",)) % ds.n_items, minlength=ds.n_items)
    pop = torch.as_tensor(counts, dtype=torch.float32, device=device)
    return lambda u: pop.repeat(len(u), 1)


def scored_numerators(top: np.ndarray, tg: Targets, n_items: int) -> dict:
    """Числители метрик для каждого пользователя по его top-K списку."""
    hit, pos = member(tg.users[:, None] * n_items + top, tg.rel_keys)
    events = np.where(hit, tg.rel_events[pos], 0)
    return {
        k: {
            "hits": hit[:, :k].sum(1),
            "any": hit[:, :k].any(1),
            "events": events[:, :k].sum(1),
            "first": (top[:, :k] == tg.first[:, None]).any(1),
        }
        for k in KS
    }


def random_numerators(tg: Targets, n_items: int) -> dict:
    """Те же числители в среднем для случайной выдачи (точные формулы вместо симуляции)."""

    def lgamma(x):
        return torch.lgamma(torch.as_tensor(x, dtype=torch.float64)).numpy()

    m = (n_items - tg.n_seen).astype(np.float64)  # размер каталога после исключения виденного
    out = {}
    for k in KS:
        p = np.minimum(k / m, 1.0)  # вероятность конкретного объекта попасть в top-K
        free = np.maximum(m - tg.n_rel - k, 0)
        # P(ни одного попадания) = C(m - r, k) / C(m, k)
        log_miss = lgamma(m - tg.n_rel + 1) - lgamma(free + 1) - lgamma(m + 1) + lgamma(np.maximum(m - k, 0) + 1)
        any_hit = np.where(m - tg.n_rel >= k, 1 - np.exp(log_miss), 1.0)
        out[k] = {"hits": p * tg.n_rel, "any": any_hit, "events": p * tg.n_events, "first": p}
    return out


def aggregate(num: dict, tg: Targets, mask: np.ndarray) -> dict:
    n_rel, n_events = tg.n_rel[mask], tg.n_events[mask]
    res: dict = {"n_users": int(mask.sum()), "recall_se": {}, **{name: {} for name in METRICS}}
    for k in KS:
        hits = num[k]["hits"][mask]
        recall = hits / n_rel
        res["recall"][k] = float(recall.mean())
        res["recall_se"][k] = float(recall.std() / np.sqrt(len(recall)))  # стандартная ошибка среднего
        res["hit"][k] = float(num[k]["any"][mask].mean())
        res["recall_capped"][k] = float((hits / np.minimum(k, n_rel)).mean())
        res["event"][k] = float(num[k]["events"][mask].sum() / n_events.sum())
        res["first"][k] = float(num[k]["first"][mask].mean())
    return res


def evaluate(scorer, ds: Interactions, split: str, seen: tuple[str, ...], device) -> dict:
    """Метрики одного скорера в одной конфигурации; отдельно для warm/cold пользователей.

    scorer — функция «пользователи → оценки всех объектов» или строка "random".
    Cold — пользователь без единого взаимодействия в train.
    """
    tg = build_targets(ds, split, seen)
    if len(tg.users) == 0:
        return {}
    if scorer == "random":
        num = random_numerators(tg, ds.n_items)
    else:
        top = topk_items(scorer, tg.users, ds.keys(seen) if seen else None, ds.n_items, device)
        num = scored_numerators(top, tg, ds.n_items)
    warm = member(tg.users, ds.train_users)[0]
    subsets = {"all": np.ones(len(tg.users), dtype=bool), "warm": warm, "cold": ~warm}
    return {name: aggregate(num, tg, mask) for name, mask in subsets.items() if mask.any()}


def full_eval(model, ds: Interactions, device) -> dict:
    """Модель и бейзлайны во всех конфигурациях: {скорер: {конфигурация: {подмножество: метрики}}}."""
    # скореры создаются по одному: каждый держит в памяти векторы всего каталога
    scorers = {
        "model": lambda: model_scorer(model, ds.item_group, device),
        "popularity": lambda: popularity_scorer(ds, device),
        "random": lambda: "random",
    }
    if ds.name == "lastfm":
        scorers["model_artists_train"] = lambda: model_scorer(model, train_only_groups(ds), device)
    results = {}
    for name, make_scorer in scorers.items():
        scorer = make_scorer()
        results[name] = {config_name(split, seen): evaluate(scorer, ds, split, seen, device) for split, seen in CONFIGS}
    return results


def dataset_stats(ds: Interactions) -> dict:
    return {
        "n_users": ds.n_users,
        "n_items": ds.n_items,
        "n_groups": ds.n_groups,
        "n_train": len(ds.train),
        "n_val": len(ds.val),
        "n_test": len(ds.test),
    }


def print_summary(results: dict, config: str = "test/seen=train+val") -> None:
    print(f"\nКонфигурация {config}, все пользователи:")
    print(f"{'':<22}" + "".join(f"{f'recall@{k}':>12}" for k in KS) + "".join(f"{f'hit@{k}':>10}" for k in KS))
    for name, by_config in results.items():
        res = by_config.get(config, {}).get("all")
        if res:
            cells = "".join(f"{res['recall'][k]:>12.4f}" for k in KS) + "".join(f"{res['hit'][k]:>10.4f}" for k in KS)
            print(f"{name:<22}{cells}   (пользователей: {res['n_users']})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Оценка сохранённых весов two-tower модели")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True, help="куда записать JSON с метриками")
    ap.add_argument("--dataset", help="по умолчанию берётся из чекпойнта; у весов авторов это lastfm")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    device = pick_device()
    model, config = load_checkpoint(args.weights, device)
    config = {"dataset": "lastfm", "split": "global", "min_rating": 0.0, **config}
    if args.dataset:
        config["dataset"] = args.dataset
    ds = load_dataset(config["dataset"], args.data_dir, config["split"], config["min_rating"])
    print(f"Веса {args.weights}, датасет {ds.name}, устройство {device}")

    results = full_eval(model, ds, device)
    print_summary(results)
    out = {"run": Path(args.out).stem, "weights": args.weights, "config": config, "stats": dataset_stats(ds), "eval": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"Метрики записаны в {args.out}")


if __name__ == "__main__":
    main()
