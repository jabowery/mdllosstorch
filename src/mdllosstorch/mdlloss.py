import torch
from torch import nn
from .residuals import residual_bits_transformed_gradsafe
from .parameters import parameter_bits_model_student_t


class MDLLoss(nn.Module):
    """Total MDL loss in bits:
    L = residual_bits (Yeo-Johnson/Box-Cox + Jacobian + discretization)
      + parameter_bits (Student-t with discretization)
    """
    def __init__(self, method: str = "yeo-johnson",
                 data_resolution: float = 1e-6,
                 param_resolution: float = 1e-6,
                 include_transform_param_bits: bool = True,
                 lam_grid: torch.Tensor = None,
                 coder: str = "legacy"):
        super().__init__()
        self.method = method
        self.data_resolution = float(data_resolution)
        self.param_resolution = float(param_resolution)
        self.coder = coder  # NEW: "legacy" (default) or "gauss_nml"
        self.include_transform_param_bits = include_transform_param_bits
        self._lam_grid = lam_grid
    def forward(self, original: torch.Tensor, reconstructed: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        if self.coder == "gauss_nml":
            # NEW: quantization-aware, variance-floored, absolute residual coder
            residuals = original - reconstructed
            res_bits = gauss_nml_bits(
                residuals,
                data_resolution=self.data_resolution,
                per_feature=True,
                include_quantization=True,
            )
            par_bits = parameter_bits_model_student_t(
                model, include_param_bits=True, param_resolution=self.param_resolution
            )
            return res_bits + par_bits
        else:
            # Existing transform-based residual coder
            lam_grid = self._lam_grid.to(device=original.device, dtype=original.dtype) if self._lam_grid is not None else None
            res_bits = residual_bits_transformed_gradsafe(
                original=original,
                reconstructed=reconstructed,
                lam_grid=lam_grid,
                method=self.method,
                include_param_bits=self.include_transform_param_bits,
                data_resolution=self.data_resolution,
            )
            par_bits = parameter_bits_model_student_t(
                model, include_param_bits=True, param_resolution=self.param_resolution
            )
            return res_bits + par_bits

# === Auto data-resolution + convenience wrappers ===

import numpy as _np
def _estimate_global_resolution_from_tensor(
    x: torch.Tensor,
    sample_per_col: int = 20_000,
    min_positive: float = 1e-12,
    clamp_low: float = 1e-12,
    clamp_high: float = 2.5e-1,
) -> float:
    if x is None:
        return 1e-6
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    with torch.no_grad():
        x_np = x.detach().to("cpu").numpy().astype(_np.float64, copy=False)
    gaps = []
    n_rows, n_cols = x_np.shape
    rng = _np.random.default_rng(42)
    for j in range(n_cols):
        col = x_np[:, j]
        col = col[~_np.isnan(col)]
        if col.size == 0:
            continue
        if sample_per_col > 0 and col.size > sample_per_col:
            col = rng.choice(col, size=sample_per_col, replace=False)
        col = _np.asarray(col, dtype=_np.float64)
        col = _np.unique(_np.sort(col))
        if col.size < 2:
            continue
        diffs = _np.diff(col)
        pos = diffs[diffs > min_positive]
        if pos.size:
            gaps.append(float(_np.min(pos)))
    if not gaps:
        return 1e-6
    res = float(_np.median(gaps))
    return float(_np.clip(res, clamp_low, clamp_high))

def compute_mdl(
    x: torch.Tensor,
    yhat: torch.Tensor,
    model: torch.nn.Module,
    *,
    method: str = "yeo-johnson",
    data_resolution: float | str = "auto",
    param_resolution: float = 1e-6,
) -> torch.Tensor:
    if isinstance(data_resolution, str) and data_resolution.lower() == "auto":
        dr = _estimate_global_resolution_from_tensor(x)
    else:
        dr = float(data_resolution)
    loss = MDLLoss(method=method, data_resolution=dr, param_resolution=param_resolution)
    return loss(x, yhat, model)

def report_mdl(
    x: torch.Tensor,
    yhat: torch.Tensor,
    model: torch.nn.Module,
    *,
    method: str = "yeo-johnson",
    data_resolution: float | str = "auto",
    param_resolution: float = 1e-6,
) -> dict:
    if isinstance(data_resolution, str) and data_resolution.lower() == "auto":
        dr = _estimate_global_resolution_from_tensor(x)
    else:
        dr = float(data_resolution)
    loss = MDLLoss(method=method, data_resolution=dr, param_resolution=param_resolution)
    total = loss(x, yhat, model)
    param_bits = loss.calculate_parameter_bits(model) if hasattr(loss, "calculate_parameter_bits") else None
    residual_bits = loss.calculate_residual_bits(x, yhat) if hasattr(loss, "calculate_residual_bits") else None
    n = max(int(x.numel()), 1)
    return {
        "total_bits": float(total.item()),
        "bits_per_entry": float(total.item()) / n,
        "parameter_bits": (float(param_bits.item()) if isinstance(param_bits, torch.Tensor) else param_bits),
        "residual_bits": (float(residual_bits.item()) if isinstance(residual_bits, torch.Tensor) else residual_bits),
        "data_resolution": float(dr),
        "method": method,
    }
