import math

import torch
from torch.distributions import StudentT

_LOG2E = 1.0 / math.log(2.0)


def parameter_bits_model_student_t(
    model: torch.nn.Module,
    include_param_bits: bool = True,
    param_resolution: float = 1e-6,
    use_parallel_sa: bool = False,
) -> torch.Tensor:
    """Sum of Student-t parameter bits over all model parameters that require grad."""
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    total = torch.tensor(0.0, device=device)
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total = total + parameter_bits_student_t_gradsafe(
            p,
            include_param_bits=include_param_bits,
            param_resolution=param_resolution,
            use_parallel_sa=use_parallel_sa,
        )
    return total


def parameter_bits_student_t_gradsafe(
    w: torch.Tensor,
    include_param_bits: bool = True,
    param_resolution: float = 1e-6,
    nu_grid=(1.5, 2, 3, 5, 8, 16, 32, 64),
    sigma_scales=(0.25, 0.5, 1.0, 2.0, 4.0),
    use_parallel_sa: bool = False,
) -> torch.Tensor:
    """MDL bits for a parameter tensor under a Student-t(nu, sigma) prior.

    Args:
        use_parallel_sa: If True, use parallel simulated annealing instead of grid search
    """
    x = w.flatten()
    x = x[torch.isfinite(x)]
    n = x.numel()
    if n == 0:
        return torch.tensor(0.0, device=w.device, dtype=w.dtype)

    if use_parallel_sa:
        # Use parallel simulated annealing search
        from .parallel_sa import MDLParallelHyperparameterSearch

        if not hasattr(parameter_bits_student_t_gradsafe, "_sa_search"):
            parameter_bits_student_t_gradsafe._sa_search = MDLParallelHyperparameterSearch()

        with torch.no_grad():
            xd = x.detach()
            nu_star, sigma_scale = (
                parameter_bits_student_t_gradsafe._sa_search.search_student_t_params(
                    xd, param_resolution
                )
            )
            med = torch.median(xd.abs()).item() + 1e-12
            base = max(med / 0.6745, param_resolution)
            sigma_star = max(base * sigma_scale, param_resolution)
    else:
        # Original grid search implementation
        with torch.no_grad():
            xd = x.detach()
            med = torch.median(xd.abs()).item() + 1e-12
            base = max(med / 0.6745, param_resolution)
            sigmas = [base * s for s in sigma_scales]
            best = None
            for nu in nu_grid:
                dist = StudentT(df=float(nu), loc=0.0, scale=1.0)
                for sigma in sigmas:
                    nll_nat = -dist.log_prob(xd / sigma).sum() + xd.numel() * math.log(sigma)
                    bits = nll_nat * _LOG2E
                    if include_param_bits:
                        bits += 0.5 * math.log2(max(2, n)) + 0.5 * math.log2(max(2, n))
                    bits += n * math.log2(1.0 / param_resolution)
                    if (best is None) or (bits < best[0]):
                        best = (bits, float(nu), float(sigma))
            nu_star, sigma_star = best[1], best[2]

    dist = StudentT(df=nu_star, loc=0.0, scale=1.0)
    sigma_star = max(float(sigma_star), param_resolution)
    nll_nat = -dist.log_prob(x / sigma_star).sum() + x.numel() * math.log(sigma_star)
    bits = nll_nat * _LOG2E
    if include_param_bits:
        bits = bits + 0.5 * math.log2(max(2, n)) + 0.5 * math.log2(max(2, n))
    bits = bits + n * math.log2(1.0 / param_resolution)
    return bits
