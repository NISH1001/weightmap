"""Closed-form linear algebra: linear CKA, ridge regression, R^2, and train/test splitting.
Float64 on CPU for stability."""
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


def split(n: int, holdout: float, seed: int, groups: torch.Tensor | None = None):
    """Train/test row indices. With `groups` (e.g. the text each token came from), whole groups go to
    one side, so near-duplicate tokens of one text can't sit on both sides of the split."""
    g = torch.Generator().manual_seed(seed)
    if groups is None:
        perm = torch.randperm(n, generator=g)
        ntr = int((1 - holdout) * n)
        return perm[:ntr], perm[ntr:]
    uniq = groups.unique()
    uniq = uniq[torch.randperm(len(uniq), generator=g)]
    test_groups = uniq[: max(1, int(round(holdout * len(uniq))))]
    is_test = torch.isin(groups, test_groups)
    return (~is_test).nonzero().squeeze(-1), is_test.nonzero().squeeze(-1)


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


def fit_ridge_cv(X: torch.Tensor, Y: torch.Tensor, grid, tr: torch.Tensor, te: torch.Tensor) -> tuple[Ridge, float]:
    """Pick lambda on (tr, te); return (model refit on all rows, held-out R^2 of the best lambda)."""
    best = None
    for lam in grid:
        s = r2(fit_ridge(X[tr], Y[tr], lam), X[te], Y[te])
        if best is None or s > best[1]:
            best = (lam, s)
    return fit_ridge(X, Y, best[0]), best[1]


def heldout_r2(X: torch.Tensor, Y: torch.Tensor, lam: float, tr: torch.Tensor, te: torch.Tensor) -> float:
    return r2(fit_ridge(X[tr], Y[tr], lam), X[te], Y[te])


def shuffled_r2(X: torch.Tensor, Y: torch.Tensor, lam: float, tr: torch.Tensor, te: torch.Tensor, seed: int) -> float:
    """Control: fit on training rows with the X<->Y pairing broken, evaluate on true held-out pairs."""
    perm = tr[torch.randperm(len(tr), generator=torch.Generator().manual_seed(seed + 1))]
    return r2(fit_ridge(X[perm], Y[tr], lam), X[te], Y[te])
