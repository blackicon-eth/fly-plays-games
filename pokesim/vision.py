"""Encoder A: read where the nearest object is, from pixels alone.

A small CNN that regresses the oracle's screen offset `(dx, dy)` -- pixels,
negative left/up -- from the grayscale screen. That vector is what drives the
fly (through `FeatureDetectors`) and what the action loop steers by. Horizontal
flip with `dx -> -dx` is a free augmentation for this task.

`evaluate` reports the honest number: contiguous temporal folds are held out, so
the score can't be inflated by near-duplicate neighbouring frames, and a shuffled
target should collapse to chance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from flybrain.reservoir import auc


class TargetCNN(nn.Module):
    """Three conv/pool blocks then a small head; two numbers out (dx, dy)."""

    def __init__(self, size: tuple[int, int] = (36, 40)):
        super().__init__()
        h, w = size
        feat = 32 * (h // 4) * (w // 4)   # two pools: localisation needs the grid
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(feat, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class VisionEncoder:
    """Encoder A. `predict` returns (dx, dy) in screen pixels."""

    def __init__(self, size: tuple[int, int] = (36, 40), scale: float = 80.0,
                 seed: int = 0, threads: int = 8):
        self.size, self.scale, self.seed = size, scale, seed
        torch.set_num_threads(threads)
        self.net: TargetCNN | None = None

    def _tensor(self, frames: np.ndarray) -> torch.Tensor:
        return torch.tensor(frames[:, None].astype(np.float32) / 255.0)

    def _augment(self, xb: torch.Tensor, yb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flip = torch.rand(len(xb)) < 0.5
        xb[flip] = torch.flip(xb[flip], dims=[3])
        yb = yb.clone()
        yb[flip, 0] = -yb[flip, 0]
        return xb, yb

    def _fit_net(self, X: torch.Tensor, y: np.ndarray, epochs: int, batch: int, lr: float) -> TargetCNN:
        net = TargetCNN(self.size)
        opt = torch.optim.Adam(net.parameters(), lr)
        loss = nn.MSELoss()
        for ep in range(epochs):
            net.train()
            perm = np.random.default_rng(ep).permutation(len(y))
            for k in range(0, len(y), batch):
                j = perm[k:k + batch]
                xb, yb = self._augment(X[j].clone(), torch.tensor(y[j]))
                opt.zero_grad()
                loss(net(xb), yb).backward()
                opt.step()
        return net

    def _targets(self, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        return np.stack([dx, dy], axis=1) / self.scale

    def fit(self, frames: np.ndarray, dx: np.ndarray, dy: np.ndarray, epochs: int = 20,
            batch: int = 256, lr: float = 1e-3) -> "VisionEncoder":
        torch.manual_seed(self.seed)
        self.net = self._fit_net(self._tensor(frames), self._targets(dx, dy), epochs, batch, lr)
        return self

    def predict(self, frames: np.ndarray) -> np.ndarray:
        """(n, 2) array of (dx, dy) in screen pixels."""
        assert self.net is not None, "call fit() first"
        self.net.eval()
        with torch.no_grad():
            return self.net(self._tensor(frames)).numpy() * self.scale

    def predict_dx(self, frames: np.ndarray) -> np.ndarray:
        return self.predict(frames)[:, 0]

    def evaluate(self, frames: np.ndarray, dx: np.ndarray, dy: np.ndarray, side: np.ndarray,
                 k: int = 5, epochs: int = 20, batch: int = 256, lr: float = 1e-3) -> dict:
        """Contiguous temporal CV: side AUC, sign accuracy, dx/dy correlation."""
        torch.manual_seed(self.seed)
        X = self._tensor(frames)
        y = self._targets(dx, dy)
        pred = np.zeros_like(y)
        for test in np.array_split(np.arange(len(y)), k):
            train = np.setdiff1d(np.arange(len(y)), test)
            net = self._fit_net(X[train], y[train], epochs, batch, lr)
            net.eval()
            with torch.no_grad():
                pred[test] = net(X[test]).numpy()
        pred *= self.scale
        sign_acc = float(((pred[:, 0] > 0) == (side == 1)).mean())
        corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else 0.0
        return {"side_auc": float(auc(side, pred[:, 0])), "sign_acc": sign_acc,
                "dx_corr": corr(pred[:, 0], dx), "dy_corr": corr(pred[:, 1], dy)}

    def save(self, path: str | Path) -> None:
        torch.save({"state": self.net.state_dict(), "size": self.size, "scale": self.scale}, path)

    @classmethod
    def load(cls, path: str | Path) -> "VisionEncoder":
        d = torch.load(path, weights_only=True)
        enc = cls(size=tuple(d["size"]), scale=float(d["scale"]))
        enc.net = TargetCNN(enc.size)
        enc.net.load_state_dict(d["state"])
        enc.net.eval()
        return enc
