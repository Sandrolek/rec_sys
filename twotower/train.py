"""Обучение two-tower модели: симметричный InfoNCE с негативами из батча, AdamW.

Значения по умолчанию — гиперпараметры из карточки модели авторов
(3 эпохи, батч 4096, lr 1e-3, weight decay 1e-5).

    python -m twotower.train --dataset lastfm --out weights/twotower_lastfm.pt --results results/lastfm_retrained.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .data import load_dataset
from .evaluate import dataset_stats, evaluate, full_eval, model_scorer, pick_device, print_summary
from .model import TwoTower, info_nce_loss, save_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description="Обучение two-tower модели")
    ap.add_argument("--dataset", required=True, choices=["lastfm", "movielens"])
    ap.add_argument("--split", default="global", choices=["global", "user"], help="сплит MovieLens")
    ap.add_argument("--min-rating", type=float, default=0.0, help="MovieLens: минимальная оценка для взаимодействия")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=0, help="ранняя остановка по val Recall@10; 0 — выключена")
    ap.add_argument("--logq", action="store_true", help="поправка logQ на популярность негативов (у авторов её нет)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True, help="куда сохранить веса")
    ap.add_argument("--results", required=True, help="куда записать JSON с метриками")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    ds = load_dataset(args.dataset, args.data_dir, args.split, args.min_rating)
    stats = dataset_stats(ds)
    print(f"Датасет {ds.name}: {stats}, устройство {device}")

    model = TwoTower(ds.n_users, ds.n_items, ds.n_groups).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=device.type == "cuda"
    )
    users = torch.as_tensor(ds.train[:, 0], device=device)
    items = torch.as_tensor(ds.train[:, 1], device=device)
    groups = torch.as_tensor(ds.item_group, device=device)
    n, bs = len(users), args.batch_size
    steps = math.ceil(n / bs)
    log_q_user = log_q_item = None
    if args.logq:
        log_q_user = torch.bincount(users, minlength=ds.n_users).clamp(min=1).div(n).log()
        log_q_item = torch.bincount(items, minlength=ds.n_items).clamp(min=1).div(n).log()
    log_every = steps // 10 if steps >= 1000 else 0  # на длинных эпохах печатаем ход обучения

    history, best_recall, best_epoch, best_state = [], -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        perm = torch.randperm(n, device=device)
        losses = []
        for step, s in enumerate(range(0, n, bs), 1):
            idx = perm[s : s + bs]
            user, item = users[idx], items[idx]
            logits = model(user, item, groups[item])
            if args.logq:
                loss = info_nce_loss(logits, log_q_user[user], log_q_item[item])
            else:
                loss = info_nce_loss(logits)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.detach())
            if log_every and step % log_every == 0:
                recent = torch.stack(losses[-log_every:]).mean().item()
                print(f"  шаг {step}/{steps}, loss {recent:.3f}, {step / (time.time() - started):.1f} шаг/с")
        losses = torch.stack(losses).cpu()
        train_seconds = time.time() - started

        val = evaluate(model_scorer(model, ds.item_group, device), ds, "val", ("train",), device)
        val_recall = val["all"]["recall"][10] if val else float("nan")
        history.append(
            {
                "epoch": epoch,
                "loss": losses.mean().item(),
                "loss_last100": losses[-100:].mean().item(),
                "val_recall@10": val_recall,
                "temperature": model.log_temp.exp().item(),
                "seconds": train_seconds,
            }
        )
        print(
            f"эпоха {epoch}/{args.epochs}: loss {history[-1]['loss']:.3f} "
            f"(последние 100 шагов {history[-1]['loss_last100']:.3f}), "
            f"val Recall@10 {val_recall:.4f}, {train_seconds:.0f} с"
        )
        if val_recall > best_recall:
            best_recall, best_epoch = val_recall, epoch
            if args.patience:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif args.patience and epoch - best_epoch >= args.patience:
            print(f"val Recall@10 не растёт {args.patience} эпох — останавливаюсь")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Беру веса лучшей эпохи: {best_epoch}")
    used_epoch = best_epoch if args.patience else history[-1]["epoch"]

    results = full_eval(model, ds, device)
    print_summary(results)

    config = {k: v for k, v in vars(args).items() if k not in ("out", "results", "data_dir")}
    config["used_epoch"] = used_epoch
    for path in (args.out, args.results):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.out, model, config)
    out = {
        "run": Path(args.results).stem,
        "weights": args.out,
        "config": config,
        "device": str(device),
        "stats": stats,
        "history": history,
        "eval": results,
    }
    Path(args.results).write_text(json.dumps(out, indent=1))
    print(f"Веса: {args.out}, метрики: {args.results}")


if __name__ == "__main__":
    main()
