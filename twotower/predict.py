"""Веса + несколько строк датасета → вероятности следующего объекта.

Берёт из test несколько пользователей с историей и для каждого считает
P(объект | пользователь) = softmax по всему каталогу от <u, t> · exp(log_temp),
исключая уже виденные объекты. Печатает топ предсказаний и вероятность того объекта,
который пользователь на самом деле выбрал следующим.

    python -m twotower.predict --weights weights/twotower_ml100k.pt --rows 3
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .data import load_dataset
from .evaluate import build_targets, member
from .model import load_checkpoint

HISTORY = ("train", "val")  # всё, что известно о пользователе до тестового периода


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Вероятности следующего объекта для нескольких строк датасета")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--rows", type=int, default=3, help="сколько строк (пользователей) взять из test")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--dataset", help="по умолчанию берётся из чекпойнта; у весов авторов это lastfm")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    model, config = load_checkpoint(args.weights)
    config = {"dataset": "lastfm", "split": "global", "min_rating": 0.0, **config}
    ds = load_dataset(args.dataset or config["dataset"], args.data_dir, config["split"], config["min_rating"])
    labels = ds.labels()
    print(f"Веса: {args.weights}\nДатасет: {ds.name}, {ds.n_users} пользователей, {ds.n_items} объектов\n")

    item_vecs = model.encode_items(torch.as_tensor(ds.item_group))
    seen_keys = ds.keys(HISTORY)
    tg = build_targets(ds, "test", HISTORY)  # tg.first — первый новый объект пользователя в test
    warm = np.flatnonzero(member(tg.users, ds.train_users)[0])
    if len(warm) == 0:
        raise SystemExit("В test нет пользователей с историей в train — предсказывать не для кого")
    picked = warm[np.linspace(0, len(warm) - 1, min(args.rows, len(warm))).astype(int)]

    past = np.concatenate([getattr(ds, s) for s in HISTORY])
    for n, j in enumerate(picked, 1):
        user, target = int(tg.users[j]), int(tg.first[j])
        lo, hi = np.searchsorted(seen_keys, [user * ds.n_items, (user + 1) * ds.n_items])
        logits = (model.user_tower(torch.tensor([user])) @ item_vecs.T)[0] * model.log_temp.exp()
        logits[torch.as_tensor(seen_keys[lo:hi] % ds.n_items)] = -torch.inf
        probs = torch.softmax(logits, dim=0)
        rank = int((probs > probs[target]).sum()) + 1

        recent = past[past[:, 0] == user, 1][-3:]
        print(f"[{n}/{len(picked)}] пользователь {user}, в истории {hi - lo} объектов, последние:")
        for item in recent:
            print(f"      {labels[item]}")
        print(f"  следующий новый объект на самом деле: {labels[target]}")
        print(f"      P = {probs[target].item():.3g}, место {rank} из {ds.n_items - (hi - lo)}")
        print(f"  топ-{args.topk} предсказаний модели:")
        top = probs.topk(args.topk)
        for p, item in zip(top.values.tolist(), top.indices.tolist()):
            mark = "  ← угадано" if item == target else ""
            print(f"      P = {p:.3g}  {labels[item]}{mark}")
        print()


if __name__ == "__main__":
    main()
