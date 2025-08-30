import math
import torch

_LOG2 = math.log(2.0)


def _log2(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x) / _LOG2


def _safe_var(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.var(x, unbiased=False).clamp_min(eps)


def _yj_transform_and_logabsdet_jac(r: torch.Tensor, lam: float):
    """Yeo-Johnson T(r; lambda) and sum of log|T'(r)| in natural log."""
    rp = r >= 0
    rn = ~rp
    t = torch.empty_like(r)

    if lam != 0.0:
        t[rp] = ((r[rp] + 1.0) ** lam - 1.0) / lam
    else:
        t[rp] = torch.log1p(r[rp])

    lam2 = 2.0 - lam
    if lam2 != 0.0:
        t[rn] = -(((1.0 - r[rn]) ** lam2) - 1.0) / lam2
    else:
        t[rn] = -torch.log1p(-r[rn])

    logabsdet = torch.zeros((), dtype=r.dtype, device=r.device)
    if rp.any():
        logabsdet = logabsdet + (lam - 1.0) * torch.log1p(r[rp]).sum()
    if rn.any():
        logabsdet = logabsdet + (1.0 - lam) * torch.log1p(-r[rn]).sum()
    return t, logabsdet


def _bc_transform_and_logabsdet_jac(r: torch.Tensor, lam: float, c: float):
    """Box-Cox T(r; lambda) requiring r+c>0 and sum log|T'(r)| (natural log)."""
    z = r + c
    z = torch.clamp(z, min=1e-9)
    if lam != 0.0:
        t = ((z**lam) - 1.0) / lam
    else:
        t = torch.log(z)
    logabsdet = (lam - 1.0) * torch.log(z).sum()
    return t, logabsdet


def residual_bits_transformed_gradsafe(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    lam_grid: torch.Tensor = None,
    method: str = "yeo-johnson",
    offset_c: float = None,
    include_param_bits: bool = True,
    data_resolution: float = 1e-6,
) -> torch.Tensor:
    """MDL residual bits with Yeo-Johnson/Box-Cox, Jacobian, lambda bits, and discretization.
    Select lambda off-graph (no_grad) to keep gradients stable."""

    r_full = (original - reconstructed).flatten()
    mask = torch.isfinite(r_full)
    if not mask.any():
        return torch.tensor(float("inf"), device=original.device, dtype=original.dtype)
    r = r_full[mask]

    if lam_grid is None:
        lam_grid = torch.linspace(-2.0, 2.0, 81, device=r.device, dtype=r.dtype)

    with torch.no_grad():
        rd = r.detach()
        best = None
        for lam in lam_grid.tolist():
            if method == "yeo-johnson":
                t, logabsdet_nat = _yj_transform_and_logabsdet_jac(rd, float(lam))
                c = None
            elif method == "box-cox":
                c = offset_c
                if c is None:
                    c = float(torch.clamp(-(rd.min()) + 1e-6, min=1e-9).item())
                t, logabsdet_nat = _bc_transform_and_logabsdet_jac(rd, float(lam), c)
            else:
                raise ValueError("method must be 'yeo-johnson' or 'box-cox'")

            t = t - t.mean()
            var_t = _safe_var(t)
            n = t.numel()
            bits_gauss = 0.5 * n * _log2(
                torch.tensor(2.0 * math.pi * math.e, device=rd.device, dtype=rd.dtype)
            ) + 0.5 * n * _log2(var_t)
            bits_jac = -(logabsdet_nat / _LOG2)
            bits_param = (0.5 * math.log2(max(2, n))) if include_param_bits else 0.0
            if method == "box-cox" and offset_c is None:
                bits_param += 0.5 * math.log2(max(2, n))
            bits_disc = n * math.log2(1.0 / data_resolution)
            total = bits_gauss + bits_jac + bits_param + bits_disc
            if (best is None) or (total < best[0]):
                best = (total, float(lam), c)
        lam_star, c_star = best[1], best[2]

    if method == "yeo-johnson":
        t, logabsdet_nat = _yj_transform_and_logabsdet_jac(r, lam_star)
    else:
        if c_star is None:
            c_star = float(torch.clamp(-(r.min()) + 1e-6, min=1e-9).item())
        t, logabsdet_nat = _bc_transform_and_logabsdet_jac(r, lam_star, c_star)

    t = t - t.mean()
    var_t = _safe_var(t)
    n = t.numel()
    bits_gauss = 0.5 * n * _log2(
        torch.tensor(2.0 * math.pi * math.e, device=r.device, dtype=r.dtype)
    ) + 0.5 * n * _log2(var_t)
    bits_jac = -(logabsdet_nat / _LOG2)
    bits_param = (0.5 * math.log2(max(2, n))) if include_param_bits else 0.0
    if method == "box-cox" and offset_c is None and method == "box-cox":
        bits_param += 0.5 * math.log2(max(2, n))
    bits_disc = n * math.log2(1.0 / data_resolution)

    return bits_gauss + bits_jac + bits_param + bits_disc


def residual_bits_transformed_softmin(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    lam_grid: torch.Tensor = None,
    tau: float = 1.0,
    method: str = "yeo-johnson",
    offset_c: float = None,
    include_param_bits: bool = True,
    data_resolution: float = 1e-6,
) -> torch.Tensor:
    """Fully differentiable softmin over lambda grid (temperature tau)."""
    r = (original - reconstructed).flatten()
    r = r[torch.isfinite(r)]
    n = r.numel()
    if n == 0:
        return torch.tensor(float("inf"), device=original.device, dtype=original.dtype)
    if lam_grid is None:
        lam_grid = torch.linspace(-2.0, 2.0, 81, device=r.device, dtype=r.dtype)

    B = []
    for lam in lam_grid:
        lamf = float(lam.item())
        if method == "yeo-johnson":
            t, logabsdet_nat = _yj_transform_and_logabsdet_jac(r, lamf)
            c = None
        elif method == "box-cox":
            c = offset_c
            if c is None:
                c = float(torch.clamp(-(r.min()) + 1e-6, min=1e-9).item())
            t, logabsdet_nat = _bc_transform_and_logabsdet_jac(r, lamf, c)
        else:
            raise ValueError("method must be 'yeo-johnson' or 'box-cox'")

        t = t - t.mean()
        var_t = _safe_var(t)
        bits_gauss = 0.5 * n * _log2(
            torch.tensor(2.0 * math.pi * math.e, device=r.device, dtype=r.dtype)
        ) + 0.5 * n * _log2(var_t)
        bits_jac = -(logabsdet_nat / _LOG2)
        bits_param = (0.5 * math.log2(max(2, n))) if include_param_bits else 0.0
        if method == "box-cox" and offset_c is None:
            bits_param += 0.5 * math.log2(max(2, n))
        bits_disc = n * math.log2(1.0 / data_resolution)
        B.append(bits_gauss + bits_jac + bits_param + bits_disc)

    B = torch.stack(B)
    w = torch.softmax(-B / tau, dim=0)
    return (w * B).sum()
def _eps():
    return 1e-12
def gauss_nml_bits(
    residuals: torch.Tensor,
    *,
    data_resolution: float,
    per_feature: bool = True,
    include_quantization: bool = True,
    ) -> torch.Tensor:
    """
    Absolute code length (in bits) for residuals using a Gaussian-with-unknown-variance
    NML-style approximation, with a quantization-aware variance floor.
    """
    x = residuals
    if x.ndim == 1:
        x = x.view(-1, 1)

    n, d = x.shape
    n = max(int(n), 1)
    d = max(int(d), 1)

    delta = float(max(data_resolution, _eps()))
    sigma2_floor = (delta ** 2) / 12.0

    var = x.float().pow(2).mean(dim=0) - x.float().mean(dim=0).pow(2)
    var = torch.clamp(var, min=sigma2_floor)

    diff_per_feat = 0.5 * torch.log2(2.0 * math.pi * math.e * var.clamp_min(_eps()))
    diff_bits = n * diff_per_feat.sum()

    q_const = 0.0
    if include_quantization:
        q_const = n * d * math.log2(max(1.0 / delta, 1.0))

    penalty_sigma = 0.5 * d * math.log2(float(n))

    total_bits = diff_bits + q_const + penalty_sigma
    return total_bits
