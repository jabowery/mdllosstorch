# mdllosstorch.py
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _calc_moments(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    # x: [B, F]
    mean = x.mean(dim=0)
    centered = x - mean
    var = centered.pow(2).mean(dim=0).clamp_min(1e-12)
    std = var.sqrt()
    z = centered / std
    skew = z.pow(3).mean(dim=0)
    kurt = z.pow(4).mean(dim=0) - 3.0
    return {"mean": mean, "var": var, "skew": skew, "kurt": kurt}


def _moment_distance(
    cur: Dict[str, torch.Tensor],
    tgt: Optional[Dict[str, torch.Tensor]] = None,
    w: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    if tgt is None:
        tgt = {
            "mean": torch.zeros_like(cur["mean"]),
            "var": torch.ones_like(cur["var"]),
            "skew": torch.zeros_like(cur["skew"]),
            "kurt": torch.zeros_like(cur["kurt"]),
        }
    if w is None:
        w = {"mean": 1.0, "var": 2.0, "skew": 1.5, "kurt": 1.0}

    loss = torch.tensor(0.0, device=cur["mean"].device)
    # var: compare in log-space to be scale-aware
    loss = loss + w["var"] * (torch.log(cur["var"]) - torch.log(tgt["var"])).pow(2).mean()
    loss = loss + w["mean"] * (cur["mean"] - tgt["mean"]).pow(2).mean()
    loss = loss + w["skew"] * (cur["skew"] - tgt["skew"]).pow(2).mean()
    loss = loss + w["kurt"] * (cur["kurt"] - tgt["kurt"]).pow(2).mean()
    return loss


def yeo_johnson(x: torch.Tensor, lam: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable Yeo–Johnson transform and log|Jacobian| without in-place masked writes.

    x:   [B, F]
    lam: [F] or [B, F]
    returns: (y, log_jacobian_per_sample=[B])
    """
    if lam.dim() == 1:
        lam = lam.unsqueeze(0).expand_as(x)  # [B, F]

    pos = x >= 0
    neg = ~pos

    eps = 1e-8

    # Positive branch
    # y = ((x+1)^lam - 1)/lam ; special-case lam≈0 -> log1p(x)
    ap = x + 1.0
    y_pos_general = ((ap).pow(lam) - 1.0) / lam
    y_pos_l0 = torch.log1p(x)
    y_pos = torch.where(torch.abs(lam) < eps, y_pos_l0, y_pos_general)

    # Negative branch
    # y = -(((1-x)^(2-lam) - 1)/(2-lam)) ; special-case (2-lam)≈0 -> -log1p(-x)
    an = 1.0 - x
    l2 = 2.0 - lam
    y_neg_general = -((an).pow(l2) - 1.0) / l2
    y_neg_l0 = -torch.log1p(-x)
    y_neg = torch.where(torch.abs(l2) < eps, y_neg_l0, y_neg_general)

    # Select by sign of x
    y = torch.where(pos, y_pos, y_neg)

    # log|Jacobian|
    # pos:  (lam-1)*log1p(x)
    # neg:  (1-lam)*log1p(-x)
    lj_pos = (lam - 1.0) * torch.log1p(x.clamp_min(0))  # clamp_min avoids log(<=0) numerics
    lj_neg = (1.0 - lam) * torch.log1p((-x).clamp_min(0))
    lj = torch.where(pos, lj_pos, lj_neg)

    log_jacobian = lj.sum(dim=1)  # [B]
    return y, log_jacobian


class _Normalizer(nn.Module):
    """
    Predicts per-feature lambda in [-2, 2] from residual moments.
    Stateful and trained inside MDLLoss via a proxy moment loss.
    """

    def __init__(self, num_features: int, hidden: int = 128):
        super().__init__()
        self.num_features = num_features
        in_dim = num_features * 4  # mean, std, skew, kurt
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_features),
            nn.Tanh(),
        )

    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        # residuals: [B, F]
        m = _calc_moments(residuals)
        std = m["var"].sqrt().clamp_min(1e-8)
        feats = torch.stack([m["mean"], std, m["skew"], m["kurt"]], dim=0).reshape(-1)  # [4F]
        x = feats.unsqueeze(0)  # [1, 4F]
        lam = self.net(x).squeeze(0) * 2.0  # [-2, 2], shape [F]
        return lam


def _gaussian_bits(centered: torch.Tensor) -> torch.Tensor:
    # centered: [B, F], variance per feature
    var = centered.var(dim=0, unbiased=False).clamp_min(1e-12)
    diff_bits_per_sample = 0.5 * math.log2(2 * math.pi * math.e) + 0.5 * torch.log2(var)
    return centered.size(0) * diff_bits_per_sample.sum()


def _quant_bits(n_entries: int, resolution: float) -> float:
    return n_entries * math.log2(1.0 / resolution)


def _param_bits(model: nn.Module, param_resolution: float) -> torch.Tensor:
    device = (
        next(model.parameters()).device
        if any(p.requires_grad for p in model.parameters())
        else torch.device("cpu")
    )
    total = torch.tensor(0.0, device=device)
    for p in model.parameters():
        if not p.requires_grad:
            continue
        flat = p.reshape(-1)
        if flat.numel() == 0:
            continue
        var = flat.var(unbiased=False).clamp_min(1e-12)
        diff = flat.numel() * (0.5 * math.log2(2 * math.pi * math.e) + 0.5 * torch.log2(var))
        quant = _quant_bits(flat.numel(), param_resolution)
        total = total + diff + quant
    return total


class MDLLoss(nn.Module):
    """
    Callable MDL loss with an internal, stateful normalizer model.

    Usage:
        loss_fn = MDLLoss()
        bits = loss_fn(x, yhat, model)

    Behavior:
      • Computes residual bits using Yeo–Johnson (λ predicted by internal normalizer).
      • Adds parameter bits for `model`.
      • Trains the internal normalizer in-place each call via a lightweight
        moment-matching proxy objective (no gradients flow into `model` from this step).
    """

    def __init__(
        self,
        data_resolution: float = 1e-6,
        param_resolution: float = 1e-6,
        normalizer_hidden: int = 128,
        normalizer_lr: float = 1e-3,
        train_normalizer_steps: int = 1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.data_resolution = data_resolution
        self.param_resolution = param_resolution
        self.normalizer_hidden = normalizer_hidden
        self.normalizer_lr = normalizer_lr
        self.train_normalizer_steps = train_normalizer_steps
        self.device = device
        self._normalizer: Optional[_Normalizer] = None
        self._optim: Optional[torch.optim.Optimizer] = None

    def _ensure_normalizer(self, feat_dim: int, device: torch.device):
        if self._normalizer is None or self._normalizer.num_features != feat_dim:
            self._normalizer = _Normalizer(feat_dim, hidden=self.normalizer_hidden).to(device)
            self._optim = torch.optim.Adam(self._normalizer.parameters(), lr=self.normalizer_lr)

    @torch.no_grad()
    def _predict_lambda(self, residuals: torch.Tensor) -> torch.Tensor:
        self._normalizer.eval()
        return self._normalizer(residuals)

    def _train_normalizer(self, residuals: torch.Tensor):
        self._normalizer.train()
        for _ in range(self.train_normalizer_steps):
            lam = self._normalizer(residuals.detach())
            y, _ = yeo_johnson(residuals.detach(), lam)
            moments = _calc_moments(y)
            loss = _moment_distance(moments)
            self._optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._normalizer.parameters(), 1.0)
            self._optim.step()

    def forward(self, x: torch.Tensor, yhat: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
        Returns total MDL (bits) as a differentiable scalar tensor.
        Gradients flow into `model` (through yhat) but the internal normalizer
        is updated with its own optimizer/state.
        """
        if x.dim() != 2 or yhat.dim() != 2:
            raise ValueError("x and yhat must be [batch, features].")
        if x.shape != yhat.shape:
            raise ValueError("x and yhat must have same shape.")

        device = yhat.device
        B, F = x.shape
        self._ensure_normalizer(F, device)

        # Train/update the normalizer's λ prediction on current residuals
        with torch.no_grad():
            residuals_detached = (x - yhat).detach()
        self._train_normalizer(residuals_detached)

        # Predict λ (no grad for normalizer, to keep model/normalizer decoupled)
        with torch.no_grad():
            lam = self._predict_lambda(residuals_detached)  # [F]

        # Apply transform with fixed λ (treat λ as constant wrt model grads)
        y_trans, logJ = yeo_johnson(x - yhat, lam)  # keep graph through (x - yhat)
        centered = y_trans - y_trans.mean(dim=0, keepdim=True)

        # Residual coding bits (Gaussian) + Jacobian correction + quantization + λ cost
        diff_bits = _gaussian_bits(centered)
        jac_bits = -logJ.sum() / math.log(2.0)
        quant_bits = _quant_bits(B * F, self.data_resolution)
        lambda_bits = F * math.log2(100.0)  # ~6.64 bits/feature (coarse prior)

        residual_bits = diff_bits + jac_bits + quant_bits + lambda_bits

        # Parameter bits for the model
        parameter_bits = _param_bits(model, self.param_resolution)

        total_bits = residual_bits + parameter_bits
        return total_bits
