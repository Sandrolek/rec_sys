"""Сводный отчёт по results/*.json: протокол авторов, воспроизведение на Last.fm, MovieLens-100k.

    python -m twotower.report                         # печатает отчёт и пишет results/REPORT.md
    python -m twotower.report --best ml100k_user_bs   # имя лучшего по val запуска с таким префиксом
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import CONFIGS, KS, METRICS, config_name

# карточка модели DanielRegaladoCardoso/loopback-twotower
CLAIM = [0.0708, 0.2172, 0.3140]
CLAIM_RANDOM = [6.7e-6, 3.3e-5, 6.7e-5]
CLAIM_USERS, CLAIM_LOSS = 847, 5.6

# (скорер, конфигурация, метрика, подмножество пользователей), если весов авторов нет и подбирать протокол не по чему
DEFAULT_PROTOCOL = ("model", "test/seen=train+val", "recall", "all")
# как получен вектор пользователя; перебирается при поиске протокола авторов
SCORER_NOTE = {
    "model": "башня пользователя",
    "model_artists_train": "башня пользователя, исполнитель только у треков из train",
    "model_seed_tracks": "среднее векторов треков пользователя, как в демо авторов",
    "model_seed_events": "среднее векторов треков с весом по числу прослушиваний",
}
SUBSET_NOTE = {"all": "все", "warm": "с историей в train"}
HEADER = ["", "@10", "@50", "@100", "польз."]


def metric(run: dict, scorer: str, config: str, name: str, subset: str = "all") -> list[float] | None:
    res = run["eval"].get(scorer, {}).get(config, {}).get(subset)
    return [res[name][str(k)] for k in KS] if res else None


def n_users(run: dict, scorer: str, config: str, subset: str = "all") -> int:
    return run["eval"].get(scorer, {}).get(config, {}).get(subset, {}).get("n_users", 0)


def best_run(runs: dict, prefix: str) -> str | None:
    """Запуск с лучшим val Recall@10 среди тех, чьё имя начинается с prefix."""
    scores = {name: metric(run, "model", "val/seen=train", "recall") for name, run in runs.items() if name.startswith(prefix)}
    scores = {name: vals[0] for name, vals in scores.items() if vals}
    return max(scores, key=scores.get) if scores else None


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows] + [""]


def cells(vals: list[float] | None) -> list[str]:
    return [f"{v:.4f}" for v in vals] if vals else ["—"] * len(KS)


def row(label: str, run: dict, scorer: str, config: str, name: str, subset: str = "all") -> list[str]:
    users = n_users(run, scorer, config, subset) if scorer.startswith("model") else ""
    return [label, *cells(metric(run, scorer, config, name, subset)), str(users)]


def lift(vals: list[float] | None, base: list[float] | None) -> str:
    return f"×{vals[0] / base[0]:.1f}" if vals and base and base[0] > 0 else "—"


def find_protocol(author: dict) -> tuple[tuple[str, str, str, str], list[str]]:
    """Перебирает варианты протокола на весах авторов и выбирает ближайший к заявленным числам."""
    ranked = []
    for scorer in SCORER_NOTE:
        for split, seen in CONFIGS:
            config = config_name(split, seen)
            for subset in SUBSET_NOTE:
                if subset == "warm" and n_users(author, scorer, config, "warm") == n_users(author, scorer, config):
                    continue  # холодных пользователей нет, вариант совпадает с «все»
                for name in METRICS:
                    vals = metric(author, scorer, config, name, subset)
                    if vals:
                        gap = max(abs(v / c - 1) for v, c in zip(vals, CLAIM))
                        ranked.append((gap, scorer, config, name, subset, vals))
    ranked.sort()
    rows = [
        [
            f"`{config}`",
            name,
            SCORER_NOTE[scorer],
            f"{SUBSET_NOTE[subset]}: {n_users(author, scorer, config, subset)}",
            *cells(vals),
            f"{gap:.0%}",
        ]
        for gap, scorer, config, name, subset, vals in ranked[:8]
    ]
    lines = [
        f"Заявлено: Recall@10/50/100 = {' / '.join(f'{c:.4f}' for c in CLAIM)} на {CLAIM_USERS} пользователях. "
        "Код оценки недоступен, поэтому веса авторов оценены во всех вариантах протокола; "
        "ниже восемь ближайших к заявленным числам (отклонение — худшее из трёх K). "
        "Вариантов около двухсот, и близкий может найтись случайно: протокол считаем найденным, "
        f"только если отклонение — единицы процентов, а пользователей около {CLAIM_USERS}.",
        "",
        *table(["конфигурация", "метрика", "вектор пользователя", "пользователи", "@10", "@50", "@100", "отклонение"], rows),
    ]
    return ranked[0][1:5], lines


def lastfm_section(
    author: dict | None, retrained: dict | None, scorer: str, config: str, name: str, subset: str
) -> tuple[list[str], list[list[str]]]:
    base = author or retrained
    random, popular = metric(base, "random", config, name, subset), metric(base, "popularity", config, name, subset)
    rows = [["Заявлено авторами", *cells(CLAIM), str(CLAIM_USERS)]]
    if author:
        rows.append(row("Веса авторов, наша оценка", author, scorer, config, name, subset))
    if retrained:
        rows.append(row("Обучение с нуля, наш запуск", retrained, scorer, config, name, subset))
    rows.append(["Бейзлайн: популярное", *cells(popular), ""])
    rows.append(["Бейзлайн: случайная выдача", *[f"{v:.2e}" for v in random], ""])
    lines = table(HEADER, rows)
    if retrained:
        last = retrained["history"][-1]
        lines += [
            f"Loss в конце обучения: {last['loss_last100']:.2f} (у авторов {CLAIM_LOSS}); "
            f"эпоха на {retrained['device']} — {last['seconds'] / 60:.1f} мин.",
            "",
        ]
    summary = [["Last.fm 1K, заявлено авторами", f"{CLAIM[0]:.4f}", f"×{CLAIM[0] / CLAIM_RANDOM[0]:.0f}", "—"]]
    for label, run in (("Last.fm 1K, веса авторов", author), ("Last.fm 1K, обучение с нуля", retrained)):
        ours = metric(run, scorer, config, name, subset) if run else None
        if ours:  # в результатах, посчитанных старой версией кода, такого варианта может не быть
            summary.append([label, f"{ours[0]:.4f}", lift(ours, random), lift(ours, popular)])
    return lines, summary


def movielens_section(runs: dict, split: str, title: str, config: str, name: str) -> tuple[list[str], list[list[str]]]:
    asis = runs.get(f"ml100k_{split}_asis")
    tuned = runs.get(best_run(runs, f"ml100k_{split}_bs") or "")
    logq = runs.get(f"ml100k_{split}_logq")
    base = tuned or asis
    if not base:
        return [], []
    rows = []
    if asis:
        rows.append(row("Гиперпараметры авторов (3 эпохи, батч 4096)", asis, "model", config, name))
    if tuned:
        cfg = tuned["config"]
        rows.append(row(f"Подбор по val: батч {cfg['batch_size']}, эпоха {cfg['used_epoch']}", tuned, "model", config, name))
        if n_users(tuned, "model", config, "cold"):
            rows.append(row("— из них с историей в train", tuned, "model", config, name, "warm"))
            rows.append(row("— из них без истории (холодный старт)", tuned, "model", config, name, "cold"))
    if logq:
        cfg = logq["config"]
        rows.append(row(f"То же с поправкой logQ, которой нет у авторов: эпоха {cfg['used_epoch']}", logq, "model", config, name))
    rows.append(row("Бейзлайн: популярное", base, "popularity", config, name))
    rows.append(row("Бейзлайн: случайная выдача", base, "random", config, name))
    ours = metric(base, "model", config, name)
    summary = [
        f"MovieLens-100k, сплит {split}",
        f"{ours[0]:.4f}",
        lift(ours, metric(base, "random", config, name)),
        lift(ours, metric(base, "popularity", config, name)),
    ]
    return [f"Сплит `{split}` — {title}:", "", *table(HEADER, rows)], [summary]


def build(runs: dict) -> list[str]:
    sections = []  # (заголовок, строки); номера расставляются в конце, потому что части может не быть
    author, retrained = runs.get("lastfm_author"), runs.get("lastfm_retrained")
    scorer, config, name, subset = DEFAULT_PROTOCOL
    if author:
        (scorer, config, name, subset), body = find_protocol(author)
        sections.append(("Какой протокол оценки у авторов", body))

    summary = []
    if author or retrained:
        body, rows = lastfm_section(author, retrained, scorer, config, name, subset)
        sections.append(("Воспроизведение на Last.fm 1K", body))
        summary += rows
    movielens = []
    splits = (("global", "общая временная шкала, как у авторов"), ("user", "по времени внутри каждого пользователя"))
    for split, title in splits:
        body, rows = movielens_section(runs, split, title, config, name)
        movielens += body
        summary += rows
    if movielens:
        sections.append(("MovieLens-100k", movielens))
    if summary:
        note = (
            "«К случайной выдаче» и «к популярному» — во сколько раз модель лучше бейзлайна при K = 10 "
            "(для MovieLens взят запуск с подбором по val). Каталоги отличаются на три порядка: 1.5 млн треков "
            "против 1.7 тыс. фильмов, поэтому датасеты сравниваются по этим отношениям, а не по самим значениям метрики."
        )
        sections.append(("Сравнение", [*table(["", "@10", "к случайной выдаче", "к популярному"], summary), note, ""]))

    protocol = f"Основной протокол: `{config}`, метрика `{name}`; в таблицах значения при K = 10 / 50 / 100."
    if author or retrained:
        protocol += f" На Last.fm 1K вектор пользователя — {SCORER_NOTE[scorer]}, пользователи — {SUBSET_NOTE[subset]}."
    lines = ["# TwoTower (loopback): Last.fm 1K и MovieLens-100k", "", protocol, ""]
    for number, (title, body) in enumerate(sections, 1):
        lines += [f"## {number}. {title}", "", *body]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Сводный отчёт по результатам запусков")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--best", metavar="PREFIX", help="напечатать имя лучшего по val запуска и выйти")
    args = ap.parse_args()

    runs = {p.stem: json.loads(p.read_text()) for p in sorted(Path(args.results_dir).glob("*.json"))}
    if args.best:
        name = best_run(runs, args.best)
        if not name:
            raise SystemExit(f"Нет запусков с префиксом {args.best!r} в {args.results_dir}")
        print(name)
        return
    text = "\n".join(build(runs))
    Path(args.results_dir, "REPORT.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
