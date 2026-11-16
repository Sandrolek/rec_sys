"""Two-tower модель loopback.

Архитектура перенесена из model.py авторов (HF Space DanielRegaladoCardoso/loopback, Apache-2.0)
без изменений в именах слоёв, поэтому их чекпойнт two_tower_epoch3.pt загружается как есть.

    пользователь: user_id  ─► Embedding(64) ─► MLP(256 → out_dim) ─► L2-норма
    объект:       item_id  ─► Embedding(64) ┐
                  group_id ─► Embedding(64) ┴► MLP(256 → out_dim) ─► L2-норма
    score = <u, t> · exp(log_temp)

У авторов объект — трек, группа — исполнитель. На MovieLens объект — фильм, группа — набор жанров.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class UserTower(nn.Module):
    def __init__(self, n_users: int, embed_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.proj = mlp(embed_dim, 256, out_dim)

    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.user_emb(user_idx)), dim=-1)


class TrackTower(nn.Module):
    def __init__(self, n_tracks: int, n_artists: int, embed_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.track_emb = nn.Embedding(n_tracks, embed_dim)
        self.artist_emb = nn.Embedding(n_artists, embed_dim)
        self.proj = mlp(embed_dim * 2, 256, out_dim)

    def forward(self, track_idx: torch.Tensor, artist_idx: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.track_emb(track_idx), self.artist_emb(artist_idx)], dim=-1)
        return F.normalize(self.proj(x), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_tracks: int, n_artists: int, out_dim: int = 128):
        super().__init__()
        self.user_tower = UserTower(n_users, out_dim=out_dim)
        self.track_tower = TrackTower(n_tracks, n_artists, out_dim=out_dim)
        self.log_temp = nn.Parameter(torch.tensor(0.0))  # обучаемая температура, как в CLIP

    def forward(
        self, user_idx: torch.Tensor, track_idx: torch.Tensor, artist_idx: torch.Tensor
    ) -> torch.Tensor:
        u = self.user_tower(user_idx)  # (B, D)
        t = self.track_tower(track_idx, artist_idx)  # (B, D)
        return (u @ t.T) * self.log_temp.exp()  # (B, B)

    @torch.no_grad()
    def encode_items(self, item_group: torch.Tensor, batch_size: int = 65536) -> torch.Tensor:
        """Векторы всего каталога: строка i — объект i с группой item_group[i]."""
        idx = torch.arange(len(item_group), device=item_group.device)
        chunks = [
            self.track_tower(idx[s : s + batch_size], item_group[s : s + batch_size])
            for s in range(0, len(idx), batch_size)
        ]
        return torch.cat(chunks)


def info_nce_loss(
    logits: torch.Tensor, log_q_user: torch.Tensor | None = None, log_q_item: torch.Tensor | None = None
) -> torch.Tensor:
    """Симметричный InfoNCE: на диагонали положительные пары, остальное — негативы из батча.

    log_q_user, log_q_item — логарифмы частот пользователей и объектов батча в обучающих данных
    для поправки logQ (Yi et al., 2019). У авторов её нет: популярный объект чаще попадает
    в негативы и получает заниженную оценку. Поправка вычитает log-частоту кандидата из его оценки.
    """
    targets = torch.arange(logits.size(0), device=logits.device)
    to_items, to_users = logits, logits.T
    if log_q_item is not None:
        to_items = to_items - log_q_item
    if log_q_user is not None:
        to_users = to_users - log_q_user
    return 0.5 * (F.cross_entropy(to_items, targets) + F.cross_entropy(to_users, targets))


def save_checkpoint(path, model: TwoTower, config: dict) -> None:
    """Формат авторов (model / vocab / embed_dim) плюс config с параметрами запуска."""
    vocab = {
        "n_users": model.user_tower.user_emb.num_embeddings,
        "n_tracks": model.track_tower.track_emb.num_embeddings,
        "n_artists": model.track_tower.artist_emb.num_embeddings,
    }
    embed_dim = model.user_tower.proj[-1].out_features
    torch.save(
        {"model": model.state_dict(), "vocab": vocab, "embed_dim": embed_dim, "config": config},
        path,
    )


def load_checkpoint(path, device: str | torch.device = "cpu") -> tuple[TwoTower, dict]:
    # weights_only=True: из файла читаются только тензоры и простые типы, чужой код не исполняется
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    vocab = ckpt["vocab"]
    model = TwoTower(vocab["n_users"], vocab["n_tracks"], vocab["n_artists"], out_dim=ckpt["embed_dim"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt.get("config", {})
