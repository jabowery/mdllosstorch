import math
import torch
import logging
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
    from .debug_logging import log_tensor_moments, logger
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"=== Parameter Bits Analysis ===")
        log_tensor_moments(w, f"Original parameters (shape {list(w.shape)})")
    
    x = w.flatten()
    x = x[torch.isfinite(x)]
    n = x.numel()
    if n == 0:
        return torch.tensor(0.0, device=w.device, dtype=w.dtype)

    if logger.isEnabledFor(logging.DEBUG):
        if n != w.numel():
            logger.debug(f"  Filtered to {n} finite parameters (removed {w.numel() - n} non-finite)")
        log_tensor_moments(x, "Filtered parameters (finite only)")

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

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"  Optimal Student-t params: nu={nu_star:.4f}, sigma={sigma_star:.6f}")
        # Show standardized parameters under optimal distribution
        standardized = x / sigma_star
        log_tensor_moments(standardized, "Standardized parameters (x/sigma)")

    dist = StudentT(df=nu_star, loc=0.0, scale=1.0)
    sigma_star = max(float(sigma_star), param_resolution)
    nll_nat = -dist.log_prob(x / sigma_star).sum() + x.numel() * math.log(sigma_star)
    bits = nll_nat * _LOG2E
    if include_param_bits:
        bits = bits + 0.5 * math.log2(max(2, n)) + 0.5 * math.log2(max(2, n))
    bits = bits + n * math.log2(1.0 / param_resolution)
    
    if logger.isEnabledFor(logging.DEBUG):
        param_penalty = (0.5 * math.log2(max(2, n)) + 0.5 * math.log2(max(2, n))) if include_param_bits else 0.0
        discretization_bits = n * math.log2(1.0 / param_resolution)
        nll_bits = nll_nat * _LOG2E
        logger.debug(f"  NLL bits: {nll_bits:.2f}")
        logger.debug(f"  Parameter penalty: {param_penalty:.2f}")
        logger.debug(f"  Discretization bits: {discretization_bits:.2f}")
        logger.debug(f"  Total bits: {bits.item():.2f}")
    
    return bits
def parameter_bits_by_layer(
    model: torch.nn.Module,
    param_resolution: float = 1e-6,
    use_parallel_sa: bool = False,
) -> torch.Tensor:
    """MDL bits with per-layer parameter modeling - each layer type gets separate treatment."""
    from .debug_logging import log_tensor_moments, logger
    
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    total_bits = torch.tensor(0.0, device=device)
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"=== Per-Layer Parameter Analysis ===")
    
    # Group parameters by layer type and properties
    weight_groups = []
    bias_groups = []
    norm_groups = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Skip parameters that are essentially constant
        if param.std() < 1e-10:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"  Skipping constant parameter: {name} (std={param.std().item():.2e})")
            continue
            
        # Categorize parameters
        if 'weight' in name.lower() and param.ndim >= 2:
            weight_groups.append((name, param))
        elif 'bias' in name.lower():
            bias_groups.append((name, param))
        elif any(x in name.lower() for x in ['norm', 'scale']):
            norm_groups.append((name, param))
        else:
            # Unknown - treat as weight
            weight_groups.append((name, param))
    
    # Process each group separately
    for group_name, param_list in [("Weight matrices", weight_groups), 
                                   ("Bias vectors", bias_groups), 
                                   ("Norm parameters", norm_groups)]:
        if not param_list:
            continue
            
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"  Processing {len(param_list)} {group_name}")
            
        for name, param in param_list:
            param_bits = parameter_bits_student_t_gradsafe(
                param, 
                param_resolution=param_resolution,
                use_parallel_sa=use_parallel_sa
            )
            total_bits = total_bits + param_bits
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"    {name}: {param_bits.item():.1f} bits")
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"  Total parameter bits: {total_bits.item():.1f}")
        
    return total_bits
