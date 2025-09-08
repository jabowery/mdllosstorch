# --- put near your imports ---
import math
from typing import Dict, Tuple

import torch

try:
    from torch import vmap  # PyTorch ≥ 2.1
except Exception:
    from torch.func import vmap  # PyTorch 2.0 fallback


def exception_bits(mask: torch.Tensor, n_total: torch.Tensor, resolution: torch.Tensor) -> torch.Tensor:
    """
    Calculate MDL cost for exceptions identified by mask.
    
    mask: [N] boolean tensor identifying exception positions
    n_total: scalar tensor, total number of positions
    resolution: scalar tensor, resolution for encoding exception positions
    """
    n_exceptions = mask.sum()
    
    # No exceptions case
    no_ex_case = torch.tensor(0.0, device=mask.device, dtype=torch.get_default_dtype())
    
    # Position encoding bits: each exception position costs log2(1/resolution)
    pos_bits = n_exceptions * torch.log2(1.0 / resolution.clamp_min(1e-12))
    
    # Partition bits: choose which positions are exceptions
    has_exceptions = n_exceptions > 0
    has_partial = (n_exceptions > 0) & (n_exceptions < n_total)
    
    # Safe lgamma computation
    n_safe = n_total.clamp_min(1.0)
    n_ex_safe = n_exceptions.clamp_min(1.0)
    n_normal_safe = (n_total - n_exceptions).clamp_min(1.0)
    
    log_binom = torch.where(
        has_partial,
        torch.lgamma(n_safe + 1) - torch.lgamma(n_ex_safe + 1) - torch.lgamma(n_normal_safe + 1),
        torch.tensor(0.0, device=mask.device, dtype=torch.get_default_dtype())
    )
    partition_bits = torch.where(has_partial, log_binom / math.log(2.0), no_ex_case)
    
    total_exception_bits = torch.where(has_exceptions, pos_bits + partition_bits, no_ex_case)
    
    return total_exception_bits


def nanvar(x: torch.Tensor, unbiased: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NaN-aware variance calculation that returns both variance and exception bits.
    Returns: (variance, exception_bits)
    """
    finite_mask = torch.isfinite(x)
    n_finite = finite_mask.sum()
    n_total = torch.tensor(x.numel(), device=x.device, dtype=x.dtype)

    # Handle case with too few finite values
    too_few_case = torch.tensor(1e-12, device=x.device, dtype=x.dtype)

    # Compute variance without boolean indexing
    x_clean = torch.where(finite_mask, x, torch.tensor(0.0, device=x.device, dtype=x.dtype))
    mean_val = x_clean.sum() / n_finite.clamp_min(1)

    # Compute squared deviations
    sq_dev = torch.where(
        finite_mask, (x - mean_val) ** 2, torch.tensor(0.0, device=x.device, dtype=x.dtype)
    )

    # Variance calculation
    if unbiased:
        var_numerator = sq_dev.sum()
        var_denominator = (n_finite - 1).clamp_min(1)
    else:
        var_numerator = sq_dev.sum()
        var_denominator = n_finite.clamp_min(1)
    
    raw_variance = var_numerator / var_denominator
    
    # Check for degenerate variance (all values identical or near-identical)
    variance_threshold = 1e-12
    degenerate_var_mask = raw_variance < variance_threshold
    
    # For degenerate cases, use minimum variance but count as exception
    corrected_variance = torch.where(degenerate_var_mask, too_few_case, raw_variance)
    
    # Calculate exception bits for degenerate variance
    n_degenerate = torch.where(degenerate_var_mask, torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device))
    degenerate_resolution = torch.tensor(1e-6, device=x.device, dtype=x.dtype)
    
    var_exception_bits = exception_bits(
        degenerate_var_mask.unsqueeze(0), 
        torch.tensor(1.0, device=x.device), 
        degenerate_resolution
    )
    
    # Add bits for non-finite values
    non_finite_mask = ~finite_mask
    non_finite_resolution = torch.tensor(1e-6, device=x.device, dtype=x.dtype)
    non_finite_bits = exception_bits(non_finite_mask, n_total, non_finite_resolution)
    
    total_exception_bits = var_exception_bits + non_finite_bits
    
    final_variance = torch.where(n_finite <= 1, too_few_case, corrected_variance)
    
    return final_variance, total_exception_bits


def nanmax(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NaN-aware maximum that returns both result and exception bits.
    """
    finite_mask = torch.isfinite(x)
    n_finite = finite_mask.sum()
    n_total = torch.tensor(x.numel(), device=x.device, dtype=x.dtype)

    # Handle case with no finite values
    no_finite_case = torch.tensor(0.0, device=x.device, dtype=x.dtype)

    # Replace non-finite values with -inf, then take max
    x_clean = torch.where(
        finite_mask, x, torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
    )
    has_finite_case = x_clean.max()

    result = torch.where(n_finite == 0, no_finite_case, has_finite_case)
    
    # Exception bits for non-finite values
    non_finite_mask = ~finite_mask
    non_finite_resolution = torch.tensor(1e-6, device=x.device, dtype=x.dtype)
    exception_bits_val = exception_bits(non_finite_mask, n_total, non_finite_resolution)
    
    return result, exception_bits_val


def nanmin(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NaN-aware minimum that returns both result and exception bits.
    """
    finite_mask = torch.isfinite(x)
    n_finite = finite_mask.sum()
    n_total = torch.tensor(x.numel(), device=x.device, dtype=x.dtype)

    # Handle case with no finite values
    no_finite_case = torch.tensor(0.0, device=x.device, dtype=x.dtype)

    # Replace non-finite values with +inf, then take min
    x_clean = torch.where(
        finite_mask, x, torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    )
    has_finite_case = x_clean.min()

    result = torch.where(n_finite == 0, no_finite_case, has_finite_case)
    
    # Exception bits for non-finite values
    non_finite_mask = ~finite_mask
    non_finite_resolution = torch.tensor(1e-6, device=x.device, dtype=x.dtype)
    exception_bits_val = exception_bits(non_finite_mask, n_total, non_finite_resolution)
    
    return result, exception_bits_val


def two_param_yeo_johnson(x: torch.Tensor, lam1: torch.Tensor, lam2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Two-parameter Yeo-Johnson that returns (y, log_jac, exception_bits).
    VMAP COMPATIBLE: No if statements, pure tensor operations.
    """
    x = x.to(dtype=torch.get_default_dtype())
    lam1 = torch.as_tensor(lam1, dtype=x.dtype, device=x.device)
    lam2 = torch.as_tensor(lam2, dtype=x.dtype, device=x.device)

    # Track extreme input values that need special handling
    x_near_neg_one = x <= -0.9999  # Problematic for log1p in positive branch
    x_near_pos_one = x >= 0.9999   # Problematic for log(1-x) in negative branch
    extreme_x_mask = x_near_neg_one | x_near_pos_one
    
    # VMAP COMPATIBLE: Always calculate exception bits, let torch.where handle zeros
    x_resolution = torch.tensor(1e-6, device=x.device, dtype=x.dtype)
    x_exception_bits = exception_bits(extreme_x_mask, torch.tensor(x.numel(), device=x.device, dtype=x.dtype), x_resolution)
    
    # Apply corrections for extreme values using torch.where (differentiable)
    x = torch.where(x_near_neg_one, torch.tensor(-0.9999, device=x.device, dtype=x.dtype), x)
    x = torch.where(x_near_pos_one, torch.tensor(0.9999, device=x.device, dtype=x.dtype), x)
    
    # Track extreme lambda values
    lam1_extreme = (torch.abs(lam1) > 2.0)
    lam2_extreme = (lam2 < 0.1) | (lam2 > 3.9)
    
    # VMAP COMPATIBLE: Always calculate exception bits
    lam_resolution = torch.tensor(1e-4, device=x.device, dtype=x.dtype)
    lam1_exception_bits = exception_bits(lam1_extreme.unsqueeze(0), torch.tensor(1.0, device=x.device), lam_resolution)
    lam2_exception_bits = exception_bits(lam2_extreme.unsqueeze(0), torch.tensor(1.0, device=x.device), lam_resolution)
    
    # Apply lambda corrections using clamp (differentiable)
    lam1 = lam1.clamp(-2.0, 2.0)
    lam2 = lam2.clamp(0.1, 3.9)

    pos = x >= 0
    neg = ~pos

    # Positive branch (λ1) - track domain violations
    eps = 1e-6
    lam1_is_zero = torch.abs(lam1) < eps
    
    # Safe operations with domain tracking
    x_pos = torch.where(pos, x, torch.tensor(0.0, device=x.device, dtype=x.dtype))
    
    # Log branch - VMAP COMPATIBLE: always calculate exception bits
    log_arg = 1 + x_pos
    log_domain_violation = log_arg <= 0
    log_resolution = torch.tensor(1e-8, device=x.device, dtype=x.dtype)
    log_exception_bits = exception_bits(log_domain_violation, torch.tensor(x.numel(), device=x.device, dtype=x.dtype), log_resolution)
    log_arg = log_arg.clamp_min(1e-12)  # Always apply correction
    
    y_pos_log = torch.log1p(x_pos)
    lj_pos_log = -torch.log1p(x_pos)

    # Power branch
    base_pos = log_arg  # Already protected above
    y_pos_power = (torch.pow(base_pos, lam1) - 1) / lam1.clamp_min(eps)
    lj_pos_power = (lam1 - 1) * torch.log1p(x_pos)

    y_pos = torch.where(lam1_is_zero, y_pos_log, y_pos_power)
    lj_pos = torch.where(lam1_is_zero, lj_pos_log, lj_pos_power)

    # Negative branch (λ2) - track domain violations
    lam2_is_two = torch.abs(lam2 - 2.0) < eps
    
    x_neg = torch.where(neg, x, torch.tensor(0.0, device=x.device, dtype=x.dtype))
    
    # Log branch - VMAP COMPATIBLE: always calculate exception bits
    neg_log_arg = 1 - x_neg
    neg_log_domain_violation = neg_log_arg <= 0
    neg_log_resolution = torch.tensor(1e-8, device=x.device, dtype=x.dtype)
    neg_log_exception_bits = exception_bits(neg_log_domain_violation, torch.tensor(x.numel(), device=x.device, dtype=x.dtype), neg_log_resolution)
    neg_log_arg = neg_log_arg.clamp_min(1e-12)  # Always apply correction
    
    y_neg_log = -torch.log1p(-x_neg)
    lj_neg_log = -torch.log1p(-x_neg)

    # Power branch
    base_neg = neg_log_arg  # Already protected above
    power_term = torch.pow(base_neg, 2 - lam2)
    y_neg_power = -(power_term - 1) / (2 - lam2).clamp_min(eps)
    lj_neg_power = (1 - lam2) * torch.log(base_neg)

    y_neg = torch.where(lam2_is_two, y_neg_log, y_neg_power)
    lj_neg = torch.where(lam2_is_two, lj_neg_log, lj_neg_power)

    # Combine branches
    y = torch.where(pos, y_pos, y_neg)
    lj = torch.where(pos, lj_pos, lj_neg)
    
    # Check for extreme output values - VMAP COMPATIBLE: always calculate
    extreme_y = (torch.abs(y) > 10.0) | (torch.abs(lj) > 10.0)
    output_resolution = torch.tensor(1e-6, device=x.device, dtype=x.dtype)
    output_exception_bits = exception_bits(extreme_y, torch.tensor(x.numel(), device=x.device, dtype=x.dtype), output_resolution)
    
    # Apply output bounds (always, using clamp which is differentiable)
    y = y.clamp(-10.0, 10.0)
    lj = lj.clamp(-10.0, 10.0)

    # Sum all exception bits
    total_exception_bits = x_exception_bits + lam1_exception_bits + lam2_exception_bits + log_exception_bits + neg_log_exception_bits + output_exception_bits

    return y, lj, total_exception_bits


def feature_complexity(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Heuristic complexity calculation - VMAP COMPATIBLE: no if statements.
    """
    finite = torch.isfinite(x)
    n = finite.sum()

    small_n_case = torch.tensor(1.0, device=x.device, dtype=x.dtype)

    # Compute complexity with exception tracking - always calculate all exception bits
    xf = torch.where(finite, x, torch.nan)
    var, var_exception_bits = nanvar(xf, unbiased=False)
    
    med = torch.nanmedian(xf)
    abs_skew = torch.nanmean(torch.abs(xf - med))
    std = torch.sqrt(var)
    xmax, max_exception_bits = nanmax(xf)
    xmin, min_exception_bits = nanmin(xf)
    
    # Check for division by very small std - always calculate exception bits
    small_std_mask = std < 1e-12
    std_resolution = torch.tensor(1e-8, device=x.device, dtype=x.dtype)
    std_exception_bits = exception_bits(small_std_mask.unsqueeze(0), torch.tensor(1.0, device=x.device), std_resolution)
    std = std.clamp_min(1e-12)  # Always apply correction
    
    range_ratio = (xmax - xmin) / std

    # Protect logarithm domain - always calculate exception bits
    log_arg = 1 + var
    log_domain_violation = log_arg <= 0
    log_resolution = torch.tensor(1e-8, device=x.device, dtype=x.dtype)
    log_exception_bits = exception_bits(log_domain_violation.unsqueeze(0), torch.tensor(1.0, device=x.device), log_resolution)
    log_arg = log_arg.clamp_min(1e-12)  # Always apply correction
    
    score = 1.0 + 0.1 * torch.log1p(var) + 0.05 * abs_skew + 0.02 * range_ratio
    full_complexity = score.clamp(0.1, 10.0)

    final_complexity = torch.where(n < 10, small_n_case, full_complexity)
    
    # Sum all exception bits
    total_exception_bits = var_exception_bits + max_exception_bits + min_exception_bits + std_exception_bits + log_exception_bits
    
    return final_complexity, total_exception_bits


def feature_resolution(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Robust per-feature resolution - VMAP COMPATIBLE: no if statements.
    """
    finite = torch.isfinite(x)
    too_few_case = torch.tensor(1e-6, device=x.device, dtype=x.dtype)

    # Track non-finite values - always calculate exception bits
    non_finite_mask = ~finite
    n_total = torch.tensor(x.numel(), device=x.device, dtype=x.dtype)
    non_finite_resolution = torch.tensor(1e-6, device=x.device, dtype=x.dtype)
    non_finite_bits = exception_bits(non_finite_mask, n_total, non_finite_resolution)

    xs, _ = torch.sort(
        torch.where(finite, x, torch.tensor(float("inf"), device=x.device, dtype=x.dtype))
    )
    diffs = torch.diff(xs)
    pos = torch.where(
        diffs > 1e-12, diffs, torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
    )
    res = torch.nanmedian(pos)

    # Check for invalid resolution - always calculate exception bits
    invalid_res_mask = ~torch.isfinite(res)
    res_resolution = torch.tensor(1e-8, device=x.device, dtype=x.dtype)
    res_exception_bits = exception_bits(invalid_res_mask.unsqueeze(0), torch.tensor(1.0, device=x.device), res_resolution)
    res = torch.where(invalid_res_mask, torch.tensor(1e-6, device=x.device, dtype=x.dtype), res)  # Always apply correction
    
    final_res = res.clamp(1e-12, 1e-1)
    result = torch.where(finite.sum() < 2, too_few_case, final_res)
    
    # Sum exception bits
    total_exception_bits = non_finite_bits + res_exception_bits
    
    return result, total_exception_bits


def feature_bits(y: torch.Tensor, log_jac: torch.Tensor, resolution: torch.Tensor, accumulated_exception_bits: torch.Tensor) -> torch.Tensor:
    """
    Bits calculation - VMAP COMPATIBLE: no if statements.
    """
    finite = torch.isfinite(y)
    n_f = finite.sum()
    n = torch.tensor(y.numel(), device=y.device, dtype=y.dtype)
    n_ex = n - n_f

    few_points_case = torch.tensor(1.0, device=y.device, dtype=y.dtype)

    # Compute variance with exception tracking - always calculate exception bits
    y_clean = torch.where(finite, y, torch.tensor(0.0, device=y.device, dtype=y.dtype))
    mean_val = y_clean.sum() / n_f.clamp_min(1)
    sq_dev = torch.where(
        finite, (y - mean_val) ** 2, torch.tensor(0.0, device=y.device, dtype=y.dtype)
    )
    
    raw_var = sq_dev.sum() / n_f.clamp_min(1)
    
    # Track variance degeneracy - always calculate exception bits
    var_degenerate_mask = raw_var < 1e-12
    var_resolution = torch.tensor(1e-8, device=y.device, dtype=y.dtype)
    var_exception_bits = exception_bits(var_degenerate_mask.unsqueeze(0), torch.tensor(1.0, device=y.device), var_resolution)
    
    var = raw_var.clamp_min(1e-12)  # Always apply correction

    # Differential entropy
    many_points_case = n_f * (0.5 * math.log2(2 * math.pi * math.e) + 0.5 * torch.log2(var))
    diff_bits = torch.where(n_f > 1, many_points_case, few_points_case)

    # Jacobian contribution
    no_finite_case = torch.tensor(0.0, device=y.device, dtype=y.dtype)
    lj_clean = torch.where(finite, log_jac, torch.tensor(0.0, device=y.device, dtype=y.dtype))
    has_finite_case = lj_clean.sum() / math.log(2.0)
    lj_bits = torch.where(n_f > 0, has_finite_case, no_finite_case)

    # Exception position bits
    log2_inv_res = torch.log2(1.0 / resolution.clamp_min(1e-12))
    ex_bits = torch.where(
        n_ex > 0, n_ex * log2_inv_res, torch.tensor(0.0, device=y.device, dtype=y.dtype)
    )

    # Partition bits for exceptions
    no_ex_case = torch.tensor(0.0, device=y.device, dtype=y.dtype)
    has_partial_ex = (n_ex > 0) & (n_ex < n)

    log_binom = torch.where(
        has_partial_ex,
        torch.lgamma(n + 1) - torch.lgamma(n_ex + 1) - torch.lgamma(n - n_ex + 1),
        torch.tensor(0.0, device=y.device, dtype=y.dtype),
    )
    part_bits = torch.where(has_partial_ex, log_binom / math.log(2.0), no_ex_case)

    # Lambda prior
    lambda_bits = torch.tensor(2 * math.log2(100.0), device=y.device, dtype=y.dtype)

    # Total bits including all accumulated exceptions
    total_bits = diff_bits + lj_bits + ex_bits + part_bits + lambda_bits + accumulated_exception_bits + var_exception_bits
    
    return total_bits.clamp_min(0.1)


def process_parameter_tensor(
    param_tensor: torch.Tensor, fixed_resolution: torch.Tensor
) -> torch.Tensor:
    """
    Process parameter tensor - VMAP COMPATIBLE: no if statements.
    """
    # Track extreme parameter values - always calculate exception bits
    extreme_param_mask = torch.abs(param_tensor) > 10.0
    param_resolution = torch.tensor(1e-8, device=param_tensor.device, dtype=param_tensor.dtype)
    param_exception_bits = exception_bits(
        extreme_param_mask, 
        torch.tensor(param_tensor.numel(), device=param_tensor.device, dtype=param_tensor.dtype), 
        param_resolution
    )
    
    # Apply correction for extreme values (always)
    y = param_tensor.clamp(-10.0, 10.0)
    log_jac = torch.zeros_like(param_tensor)

    bits = feature_bits(y, log_jac, fixed_resolution, param_exception_bits)
    return bits





def evaluate_yj_score(x: torch.Tensor, lam1: torch.Tensor, lam2: torch.Tensor) -> torch.Tensor:
    """
    Evaluate YJ transformation including all exception costs.
    """
    y, log_jac, yj_exception_bits = two_param_yeo_johnson(x, lam1, lam2)
    resolution, res_exception_bits = feature_resolution(x)
    total_exception_bits = yj_exception_bits + res_exception_bits
    return feature_bits(y, log_jac, resolution, total_exception_bits)


def process_single_feature_sa_stateless(
    x_col: torch.Tensor, lam1: torch.Tensor, lam2: torch.Tensor, random_proposals: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process one feature column with full exception tracking.
    All vmap compatibility preserved, but now with proper MDL accounting.
    """
    # Current score (includes all exception costs)
    current_score = evaluate_yj_score(x_col, lam1, lam2)

    # Generate proposals with reduced variance for stability
    temperature = 1.0
    std_dev = 0.05 * temperature

    # Create 5 proposal pairs (preserving vmap serial computation pattern)
    prop1_lam1 = lam1 + std_dev * random_proposals[0]
    prop1_lam2 = lam2 + std_dev * random_proposals[1]
    prop1_score = evaluate_yj_score(x_col, prop1_lam1, prop1_lam2)

    prop2_lam1 = lam1 + std_dev * random_proposals[2]
    prop2_lam2 = lam2 + std_dev * random_proposals[3]
    prop2_score = evaluate_yj_score(x_col, prop2_lam1, prop2_lam2)

    prop3_lam1 = lam1 + std_dev * random_proposals[4]
    prop3_lam2 = lam2 + std_dev * random_proposals[5]
    prop3_score = evaluate_yj_score(x_col, prop3_lam1, prop3_lam2)

    prop4_lam1 = lam1 + std_dev * random_proposals[6]
    prop4_lam2 = lam2 + std_dev * random_proposals[7]
    prop4_score = evaluate_yj_score(x_col, prop4_lam1, prop4_lam2)

    prop5_lam1 = lam1 + std_dev * random_proposals[8]
    prop5_lam2 = lam2 + std_dev * random_proposals[9]
    prop5_score = evaluate_yj_score(x_col, prop5_lam1, prop5_lam2)

    # Stack all candidates (preserving vmap tensor operations)
    all_lam1 = torch.stack([lam1, prop1_lam1, prop2_lam1, prop3_lam1, prop4_lam1, prop5_lam1])
    all_lam2 = torch.stack([lam2, prop1_lam2, prop2_lam2, prop3_lam2, prop4_lam2, prop5_lam2])
    all_scores = torch.stack(
        [current_score, prop1_score, prop2_score, prop3_score, prop4_score, prop5_score]
    )

    # Find best using one-hot encoding (preserving vmap no-indexing pattern)
    min_score = torch.min(all_scores)
    is_best = (all_scores == min_score).float()
    
    total_weight = is_best.sum().clamp_min(1e-8)
    best_lam1 = (all_lam1 * is_best).sum() / total_weight
    best_lam2 = (all_lam2 * is_best).sum() / total_weight

    # Calculate final bits with best lambdas (includes all exception costs)
    y_col, lj_col, final_exception_bits = two_param_yeo_johnson(x_col, best_lam1, best_lam2)
    resolution, res_exception_bits = feature_resolution(x_col)
    total_exception_bits = final_exception_bits + res_exception_bits
    bits = feature_bits(y_col, lj_col, resolution, total_exception_bits)

    return bits, best_lam1, best_lam2




def optimize_feature_lambdas(x: torch.Tensor, complexity_hint: torch.Tensor):
    """
    Fallback lambda optimization (preserved for compatibility).
    """
    finite = torch.isfinite(x)
    too_few_case = (
        torch.tensor(0.0, device=x.device, dtype=x.dtype),
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
    )

    # Simple heuristic lambdas based on complexity
    lam1 = complexity_hint * 0.1
    lam2 = complexity_hint * 0.1 + 1.0
    enough_data_case = (lam1, lam2)

    use_simple = finite.sum() < 10
    final_lam1 = torch.where(use_simple, too_few_case[0], enough_data_case[0])
    final_lam2 = torch.where(use_simple, too_few_case[1], enough_data_case[1])

    return final_lam1, final_lam2



# Add this diagnostic version to debug the issue
class MDLLoss(torch.nn.Module):
    def __init__(self, param_resolution_scale: float = 1e-8, weight_model_bits: float = 1.0):
        super().__init__()
        self.param_resolution_scale = param_resolution_scale
        self.weight_model_bits = weight_model_bits
        self.feature_lambdas = {}
        self.feature_temperatures = {}
        self.feature_initialized = set()

    def get_current_lambdas_tensor(self, num_features: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        lam1_list = []
        lam2_list = []
        for i in range(num_features):
            if i in self.feature_lambdas:
                l1, l2 = self.feature_lambdas[i]
            else:
                l1, l2 = torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)
            lam1_list.append(l1)
            lam2_list.append(l2)
        return torch.stack(lam1_list), torch.stack(lam2_list)

    def update_all_feature_states(self, new_lam1_tensor: torch.Tensor, new_lam2_tensor: torch.Tensor):
        for i in range(new_lam1_tensor.shape[0]):
            self.feature_lambdas[i] = (new_lam1_tensor[i].detach(), new_lam2_tensor[i].detach())
            current_temp = self.feature_temperatures.get(i, 1.0)
            self.feature_temperatures[i] = current_temp * 0.95

    def forward(self, residuals: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        print(f"\n=== MDL DEBUG: Residuals shape {residuals.shape} ===")
        print(f"Residuals range: [{residuals.min().item():.6f}, {residuals.max().item():.6f}]")
        print(f"Residuals std: {residuals.std().item():.6f}")
        
        num_features = residuals.shape[1]
        device = residuals.device

        current_lam1, current_lam2 = self.get_current_lambdas_tensor(num_features, device)
        print(f"Lambda ranges: λ1 [{current_lam1.min().item():.3f}, {current_lam1.max().item():.3f}], λ2 [{current_lam2.min().item():.3f}, {current_lam2.max().item():.3f}]")

        random_proposals = 0.5 * torch.randn(num_features, 10, device=device)

        # Process each feature individually to track exception bits
        total_data_bits = torch.tensor(0.0, device=device)
        total_exception_bits = torch.tensor(0.0, device=device)
        
        for i in range(num_features):
            x_col = residuals[:, i]
            lam1 = current_lam1[i]
            lam2 = current_lam2[i]
            rand_props = random_proposals[i]
            
            # Manual processing to track exception bits
            y, log_jac, yj_exception_bits = two_param_yeo_johnson(x_col, lam1, lam2)
            resolution, res_exception_bits = feature_resolution(x_col)
            feature_exception_bits = yj_exception_bits + res_exception_bits
            feature_bits_val = feature_bits(y, log_jac, resolution, feature_exception_bits)
            
            total_data_bits += feature_bits_val
            total_exception_bits += feature_exception_bits
            
            if i < 3:  # Log first 3 features
                print(f"Feature {i}: bits={feature_bits_val.item():.1f}, exceptions={feature_exception_bits.item():.3f}")

        print(f"Total data bits: {total_data_bits.item():.1f}")
        print(f"Total exception bits: {total_exception_bits.item():.3f}")
        
        # Model complexity
        model_bits = self.compute_model_bits(model)
        print(f"Model bits: {model_bits.item():.1f}")
        
        total_loss = total_data_bits + self.weight_model_bits * model_bits
        print(f"Final loss: {total_loss.item():.1f}")
        
        return total_loss.clamp_min(0.1)

    def compute_model_bits(self, model: torch.nn.Module) -> torch.Tensor:
        resolutions = self.estimate_parameter_resolutions(model)
        total_bits = torch.tensor(0.1, device=next(model.parameters()).device)
        total_param_exceptions = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, param in model.named_parameters():
            if not param.requires_grad or name not in resolutions:
                continue
            flat_param = param.reshape(-1)
            if flat_param.numel() == 0:
                continue
            
            # Track parameter exceptions
            extreme_param_mask = torch.abs(flat_param) > 10.0
            n_extreme = extreme_param_mask.sum().item()
            if n_extreme > 0:
                print(f"Parameter {name}: {n_extreme}/{flat_param.numel()} extreme values")
            
            param_bits = process_parameter_tensor(flat_param, resolutions[name])
            if total_bits.device != param_bits.device:
                total_bits = total_bits.to(param_bits.device)
            total_bits = total_bits + param_bits

        print(f"Parameter exception count: {total_param_exceptions.item()}")
        return total_bits.clamp_min(0.1)

    def estimate_parameter_resolutions(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        resolutions = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            flat = param.data.reshape(-1)
            if flat.numel() == 0:
                continue
            resolution = torch.tensor(
                self.param_resolution_scale, device=param.device, dtype=param.dtype
            ).clamp_min(1e-12)
            resolutions[name] = resolution
        return resolutions

    def is_initialized(self, feature_idx: int) -> bool:
        """Check if a feature has been initialized (unchanged)."""
        return feature_idx in self.feature_initialized

    def initialize_feature(self, feature_idx: int, initial_temp: float):
        """Initialize a feature's SA state (unchanged)."""
        self.feature_lambdas[feature_idx] = (torch.tensor(0.0), torch.tensor(1.0))
        self.feature_temperatures[feature_idx] = initial_temp
        self.feature_initialized.add(feature_idx)

    def get_lambdas(self, feature_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current lambda pair for a feature (unchanged)."""
        if feature_idx not in self.feature_lambdas:
            return torch.tensor(0.0), torch.tensor(1.0)
        return self.feature_lambdas[feature_idx]

    def get_temperature(self, feature_idx: int) -> float:
        """Get current temperature for a feature (unchanged)."""
        return self.feature_temperatures.get(feature_idx, 1.0)

    def update_feature_state(
        self, feature_idx: int, new_lam1: torch.Tensor, new_lam2: torch.Tensor, new_temp: float
    ):
        """Update a feature's SA state (unchanged)."""
        self.feature_lambdas[feature_idx] = (new_lam1.detach(), new_lam2.detach())
        self.feature_temperatures[feature_idx] = new_temp

