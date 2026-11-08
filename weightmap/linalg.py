"""Closed-form linear algebra: linear CKA, ridge regression, R^2. Float64 on CPU for stability."""
from __future__ import annotations

from dataclasses import dataclass

import torch


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X.double() - X.double().mean(0)
    Y = Y.double() - Y.double().mean(0)
    num = torch.linalg.norm(X.T @ Y) ** 2
    den = torch.linalg.norm(X.T @ X) * torch.linalg.norm(Y.T @ Y)
    return float(num / den)


@dataclass
class Ridge:
    W: torch.Tensor      # [d_in, d_out]
    b: torch.Tensor      # [d_out]
    lam: float

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return X.to(self.W.dtype) @ self.W + self.b


def fit_ridge(X: torch.Tensor, Y: torch.Tensor, lam_rel: float) -> Ridge:
    """Ridge with penalty lam_rel * mean feature variance * n (scale-free across models)."""
    X, Y = X.double(), Y.double()
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    lam = lam_rel * float(X.var(0).mean()) * len(X)
    A = Xc.T @ Xc + lam * torch.eye(X.shape[1], dtype=X.dtype)
    W = torch.linalg.solve(A, Xc.T @ Yc)
    return Ridge(W=W, b=my - mx @ W, lam=lam_rel)


def r2(model: Ridge, X: torch.Tensor, Y: torch.Tensor) -> float:
    Y = Y.double()
    resid = ((Y - model(X.double())) ** 2).sum()
    total = ((Y - Y.mean(0)) ** 2).sum()
    return float(1 - resid / total)


def fit_ridge_cv(X: torch.Tensor, Y: torch.Tensor, grid, holdout: float, seed: int) -> tuple[Ridge, float]:
    """Pick lambda on a held-out split; return (model refit on all data, held-out R^2 of best lambda)."""
    n = len(X)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    ntr = int((1 - holdout) * n)
    tr, te = perm[:ntr], perm[ntr:]
    best = None
    for lam in grid:
        m = fit_ridge(X[tr], Y[tr], lam)
        s = r2(m, X[te], Y[te])
        if best is None or s > best[1]:
            best = (lam, s)
    return fit_ridge(X, Y, best[0]), best[1]


def heldout_r2(X: torch.Tensor, Y: torch.Tensor, lam: float, holdout: float, seed: int) -> float:
    n = len(X)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    ntr = int((1 - holdout) * n)
    m = fit_ridge(X[perm[:ntr]], Y[perm[:ntr]], lam)
    return r2(m, X[perm[ntr:]], Y[perm[ntr:]])
