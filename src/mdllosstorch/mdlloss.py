# --- put near your imports ---
import math
from typing import Dict, Tuple

import torch

try:
    from torch import vmap  # PyTorch ≥ 2.1
except Exception:
    from torch.func import vmap  # PyTorch 2.0 fallback


def nanvar(x: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """
    NaN-aware variance calculation.
    """
    finite_mask = torch.isfinite(x)
    n_finite = finite_mask.sum()

    # Handle case with too few finite values
    too_few_case = torch.tensor(0.0, device=x.device, dtype=x.dtype)

    # Compute variance without boolean indexing
    # Replace non-finite values with 0 for mean calculation
    x_clean = torch.where(finite_mask, x, torch.tensor(0.0, device=x.device, dtype=x.dtype))
    mean_val = x_clean.sum() / n_finite.clamp_min(1)  # Avoid division by zero

    # Compute squared deviations, masking out non-finite values
    sq_dev = torch.where(
        finite_mask, (x - mean_val) ** 2, torch.tensor(0.0, device=x.device, dtype=x.dtype)
    )

    # Compute variance
    if unbiased:
        enough_data_case = sq_dev.sum() / (n_finite - 1).clamp_min(1)
    else:
        enough_data_case = sq_dev.sum() / n_finite.clamp_min(1)

    # Use torch.where instead of if-statement
    return torch.where(n_finite <= 1, too_few_case, enough_data_case)


def nanmax(x: torch.Tensor) -> torch.Tensor:
    """
    NaN-aware maximum.
    """
    finite_mask = torch.isfinite(x)
    n_finite = finite_mask.sum()

    # Handle case with no finite values
    no_finite_case = torch.tensor(float("nan"), device=x.device, dtype=x.dtype)

    # Replace non-finite values with -inf, then take max
    x_clean = torch.where(
        finite_mask, x, torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
    )
    has_finite_case = x_clean.max()

    # Use torch.where instead of if-statement
    return torch.where(n_finite == 0, no_finite_case, has_finite_case)


def nanmin(x: torch.Tensor) -> torch.Tensor:
    """
    NaN-aware minimum.
    """
    finite_mask = torch.isfinite(x)
    n_finite = finite_mask.sum()

    # Handle case with no finite values
    no_finite_case = torch.tensor(float("nan"), device=x.device, dtype=x.dtype)

    # Replace non-finite values with +inf, then take min
    x_clean = torch.where(
        finite_mask, x, torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    )
    has_finite_case = x_clean.min()

    # Use torch.where instead of if-statement
    return torch.where(n_finite == 0, no_finite_case, has_finite_case)


def two_param_yeo_johnson(x: torch.Tensor, lam1: torch.Tensor, lam2: torch.Tensor):
    """
    Two-parameter Yeo-Johnson for a feature.
    x: [N] (1-D), lam1, lam2: scalars (0-D tensors or floats)
    returns: y:[N], log_jac:[N]
    """
    x = x.to(dtype=torch.get_default_dtype())
    lam1 = torch.as_tensor(lam1, dtype=x.dtype, device=x.device)
    lam2 = torch.as_tensor(lam2, dtype=x.dtype, device=x.device)

    pos = x >= 0
    neg = ~pos
    # positive branch (λ1) - vectorized
    # Case 1: lam1 ≈ 0 (use log1p)
    eps = torch.finfo(lam1.dtype).eps * 5
    #    lam1_is_zero = torch.isclose(lam1, torch.tensor(0.0, device=x.device, dtype=x.dtype))
    lam1_is_zero = torch.abs(lam1) < eps
    y_pos_log = torch.log1p(x)
    lj_pos_log = -torch.log1p(x)

    # Case 2: lam1 ≠ 0 (use power transform)
    y_pos_power = ((1 + x) ** lam1 - 1) / lam1
    lj_pos_power = (lam1 - 1) * torch.log1p(x)

    # Select based on lambda value
    y_pos = torch.where(lam1_is_zero, y_pos_log, y_pos_power)
    lj_pos = torch.where(lam1_is_zero, lj_pos_log, lj_pos_power)

    # negative branch (λ2) - vectorized
    # Case 1: lam2 ≈ 2 (use log1p)
    # lam2_is_two = torch.isclose(lam2, torch.tensor(2.0, device=x.device, dtype=x.dtype))
    lam2_is_two = torch.abs(lam2 - 2.0) < eps
    y_neg_log = -torch.log1p(-x)
    lj_neg_log = -torch.log1p(-x)

    # Case 2: lam2 ≠ 2 (use power transform)
    base = (1 - x).clamp_min(1e-18)
    y_neg_power = -((base ** (2 - lam2) - 1) / (2 - lam2))
    lj_neg_power = (1 - lam2) * torch.log(base)

    # Select based on lambda value
    y_neg = torch.where(lam2_is_two, y_neg_log, y_neg_power)
    lj_neg = torch.where(lam2_is_two, lj_neg_log, lj_neg_power)

    # Combine positive and negative branches using torch.where
    y = torch.where(pos, y_pos, y_neg)
    lj = torch.where(pos, lj_pos, lj_neg)

    return y, lj


def feature_complexity(x: torch.Tensor) -> torch.Tensor:
    """
    Heuristic complexity for a feature (scalar).
    x: [N]
    """
    finite = torch.isfinite(x)
    n = finite.sum()

    # Replace if-statement with torch.where for vmap compatibility
    small_n_case = torch.tensor(1.0, device=x.device, dtype=x.dtype)

    # Compute the full complexity calculation
    xf = torch.where(finite, x, torch.nan)
    var = nanvar(xf, unbiased=False).clamp_min(0.0)
    med = torch.nanmedian(xf)
    abs_skew = torch.nanmean(torch.abs(xf - med))
    std = torch.sqrt(nanvar(xf, unbiased=False).clamp_min(1e-12))
    xmax = nanmax(xf)
    xmin = nanmin(xf)
    range_ratio = (xmax - xmin) / (std + 1e-12)

    score = 1.0 + 0.1 * torch.log1p(var) + 0.05 * abs_skew + 0.02 * range_ratio
    full_complexity = score.clamp(0.1, 10.0)

    # Use torch.where instead of if-statement
    return torch.where(n < 10, small_n_case, full_complexity)


def feature_resolution(x: torch.Tensor) -> torch.Tensor:
    """
    Robust per-feature resolution (scalar): median positive gap in sorted finite values.
    x: [N]
    """
    finite = torch.isfinite(x)

    # Handle case with too few finite values
    too_few_case = torch.tensor(1e-6, device=x.device, dtype=x.dtype)

    # Compute full resolution calculation
    xs, _ = torch.sort(
        torch.where(finite, x, torch.tensor(float("inf"), device=x.device, dtype=x.dtype))
    )
    diffs = torch.diff(xs)
    pos = torch.where(
        diffs > 1e-12, diffs, torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
    )
    res = torch.nanmedian(pos)

    # Handle case where no valid differences found
    no_valid_case = torch.tensor(1e-6, device=x.device, dtype=x.dtype)
    valid_res = torch.where(torch.isfinite(res), res, no_valid_case)
    final_res = valid_res.clamp(1e-12, 1e-1)

    # Use torch.where instead of if-statement
    return torch.where(finite.sum() < 2, too_few_case, final_res)


def feature_bits(y: torch.Tensor, log_jac: torch.Tensor, resolution: torch.Tensor) -> torch.Tensor:
    """
    Bits for a feature (scalar).
    y: [N] transformed residuals; log_jac: [N]; resolution: scalar
    """
    finite = torch.isfinite(y)
    n_f = finite.sum()
    n = torch.tensor(y.numel(), device=y.device, dtype=y.dtype)
    n_ex = n - n_f

    # Handle variance calculation without boolean indexing
    few_points_case = torch.tensor(0.0, device=y.device, dtype=y.dtype)

    # Compute variance manually without boolean indexing
    y_clean = torch.where(finite, y, torch.tensor(0.0, device=y.device, dtype=y.dtype))
    mean_val = y_clean.sum() / n_f.clamp_min(1)
    sq_dev = torch.where(
        finite, (y - mean_val) ** 2, torch.tensor(0.0, device=y.device, dtype=y.dtype)
    )
    var = (sq_dev.sum() / n_f.clamp_min(1)).clamp_min(1e-12)

    many_points_case = n_f * (0.5 * math.log2(2 * math.pi * math.e) + 0.5 * torch.log2(var))
    diff_bits = torch.where(n_f > 1, many_points_case, few_points_case)

    # log jacobian contribution - fix this too
    no_finite_case = torch.tensor(0.0, device=y.device, dtype=y.dtype)

    # Replace log_jac[finite].sum() with masked sum
    lj_clean = torch.where(finite, log_jac, torch.tensor(0.0, device=y.device, dtype=y.dtype))
    has_finite_case = lj_clean.sum() / math.log(2.0)
    lj_bits = torch.where(n_f > 0, has_finite_case, no_finite_case)

    # exception positions encoded at given resolution - keep in tensor form
    log2_inv_res = torch.log2(1.0 / resolution)  # Keep as tensor
    ex_bits = torch.where(
        n_ex > 0, n_ex * log2_inv_res, torch.tensor(0.0, device=y.device, dtype=y.dtype)
    )

    # partition bits for exceptions - fix the lgamma calls too
    no_ex_case = torch.tensor(0.0, device=y.device, dtype=y.dtype)

    # Calculate log_binom when we have some but not all exceptions
    has_partial_ex = (n_ex > 0) & (n_ex < n)

    # Use torch.lgamma instead of math.lgamma to keep tensor operations
    log_binom = torch.where(
        has_partial_ex,
        torch.lgamma(n + 1) - torch.lgamma(n_ex + 1) - torch.lgamma(n - n_ex + 1),
        torch.tensor(0.0, device=y.device, dtype=y.dtype),
    )
    part_bits = torch.where(has_partial_ex, log_binom / math.log(2.0), no_ex_case)

    # simple λ prior (constant per feature)
    lambda_bits = torch.tensor(2 * math.log2(100.0), device=y.device, dtype=y.dtype)

    return diff_bits + lj_bits + ex_bits + part_bits + lambda_bits


def evaluate_yj_score(x: torch.Tensor, lam1: torch.Tensor, lam2: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the YJ transformation and return bits score for given lambdas.
    x: [N] - feature values
    lam1, lam2: scalar tensors - lambda parameters
    Returns: scalar bits score
    """
    y, log_jac = two_param_yeo_johnson(x, lam1, lam2)
    resolution = feature_resolution(x)
    return feature_bits(y, log_jac, resolution)


def optimize_feature_lambdas_sa(
    x: torch.Tensor, feature_idx: torch.Tensor, mdl_loss_ref
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulated annealing optimization for (λ1, λ2) for a single feature.
    x: [N] - feature values
    feature_idx: scalar tensor - index of this feature
    mdl_loss_ref: reference to MDLLoss instance for state access
    Returns: (lam1, lam2) as scalar tensors
    """
    idx = int(feature_idx.item())

    # Get current state
    current_lam1, current_lam2 = mdl_loss_ref.get_lambdas(idx)
    temperature = mdl_loss_ref.get_temperature(idx)

    # Current score
    current_score = evaluate_yj_score(x, current_lam1, current_lam2)

    # Generate 5 proposals around current lambdas
    std_dev = 0.1 * temperature  # Scale proposals by temperature

    # Create 5 proposal pairs (explicitly serial computation)
    prop1_lam1 = current_lam1 + std_dev * torch.randn(1, device=x.device, dtype=x.dtype)
    prop1_lam2 = current_lam2 + std_dev * torch.randn(1, device=x.device, dtype=x.dtype)
    prop1_score = evaluate_yj_score(x, prop1_lam1, prop1_lam2)

    prop2_lam1 = current_lam1 + std_dev * torch.randn(1, device=x.device, dtype=x.dtype)
    prop2_lam2 = current_lam2 + std_dev * torch.randn(1, device=x.device, dtype=x.dtype)
    prop2_score = evaluate_yj_score(x, prop2_lam1, prop2_lam2)

    prop3_lam1 = current_lam1 + std_dev * torch.randn(1, device=x.device, dtype=x.dtype)
    prop3_lam2 = current_lam2 + std_dev * torch.randn(1, device=x.device, dtype=x.dtype)
    prop3_score = evaluate_yj_score(x, prop3_lam1, prop3_lam2)

    prop4_lam1 = current_lam1 + std_dev * torch.randn(1, device=x.device, dtype=x.dtype)
    prop4_lam2 = current_lam2 + std_dev * torch.randn(1, device=x.device, dtype=x.dtype)
    prop4_score = evaluate_yj_score(x, prop4_lam1, prop4_lam2)

    prop5_lam1 = current_lam1 + std_dev * torch.randn(1, device=x.device, dtype=x.dtype)
    prop5_lam2 = current_lam2 + std_dev * torch.randn(1, device=x.device, dtype=x.dtype)
    prop5_score = evaluate_yj_score(x, prop5_lam1, prop5_lam2)

    # Stack all candidates (current + 5 proposals)
    all_lam1 = torch.stack(
        [
            current_lam1,
            prop1_lam1.squeeze(),
            prop2_lam1.squeeze(),
            prop3_lam1.squeeze(),
            prop4_lam1.squeeze(),
            prop5_lam1.squeeze(),
        ]
    )
    all_lam2 = torch.stack(
        [
            current_lam2,
            prop1_lam2.squeeze(),
            prop2_lam2.squeeze(),
            prop3_lam2.squeeze(),
            prop4_lam2.squeeze(),
            prop5_lam2.squeeze(),
        ]
    )
    all_scores = torch.stack(
        [current_score, prop1_score, prop2_score, prop3_score, prop4_score, prop5_score]
    )

    # Find best
    best_idx = torch.argmin(all_scores)
    best_lam1 = all_lam1[best_idx]
    best_lam2 = all_lam2[best_idx]

    # Update state (this will be called from the main thread after vmap)
    mdl_loss_ref.update_feature_state(idx, best_lam1, best_lam2, temperature * 0.95)

    return best_lam1, best_lam2


def optimize_feature_lambdas(x: torch.Tensor, complexity_hint: torch.Tensor):
    """
    Fallback lambda optimization (simplified version).
    This is kept for compatibility but won't be used in SA version.
    """
    finite = torch.isfinite(x)
    too_few_case = (
        torch.tensor(0.0, device=x.device, dtype=x.dtype),
        torch.tensor(0.0, device=x.device, dtype=x.dtype),
    )

    # Simple heuristic lambdas based on complexity
    lam1 = complexity_hint * 0.1
    lam2 = complexity_hint * 0.1 + 1.0
    enough_data_case = (lam1, lam2)

    # Use torch.where pattern (though this returns tuples, need to handle carefully)
    use_simple = finite.sum() < 10
    final_lam1 = torch.where(use_simple, too_few_case[0], enough_data_case[0])
    final_lam2 = torch.where(use_simple, too_few_case[1], enough_data_case[1])

    return final_lam1, final_lam2


def process_single_feature_sa_stateless(
    x_col: torch.Tensor, lam1: torch.Tensor, lam2: torch.Tensor, random_proposals: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process one feature column through SA pipeline (stateless version).

    x_col: [N] - one feature column
    lam1, lam2: scalar tensors - current lambda values
    random_proposals: [10] tensor of random numbers for proposals
    Returns: (bits, best_lam1, best_lam2) for this feature
    """
    # 1) Complexity (for temperature scaling)
    complexity = feature_complexity(x_col)

    # 2) Current score
    current_score = evaluate_yj_score(x_col, lam1, lam2)

    # 3) Generate 5 proposals using pre-generated random numbers
    temperature = 1.0  # We'll make this adaptive later
    std_dev = 0.1 * temperature

    # Create 5 proposal pairs using the random numbers
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

    # Stack all candidates (current + 5 proposals)
    all_lam1 = torch.stack([lam1, prop1_lam1, prop2_lam1, prop3_lam1, prop4_lam1, prop5_lam1])
    all_lam2 = torch.stack([lam2, prop1_lam2, prop2_lam2, prop3_lam2, prop4_lam2, prop5_lam2])
    all_scores = torch.stack(
        [current_score, prop1_score, prop2_score, prop3_score, prop4_score, prop5_score]
    )

    # Find best using one-hot encoding instead of indexing
    min_score = torch.min(all_scores)
    is_best = (all_scores == min_score).float()  # One-hot vector for best score(s)

    # If there are ties, this will average them (could also take first by using argmax trick)
    # To take first in case of tie, use: is_best = torch.zeros_like(all_scores); is_best[torch.argmin(all_scores)] = 1.0
    # But let's use the averaging approach to avoid indexing:
    total_weight = is_best.sum().clamp_min(1e-8)  # Avoid division by zero
    best_lam1 = (all_lam1 * is_best).sum() / total_weight
    best_lam2 = (all_lam2 * is_best).sum() / total_weight

    # Calculate final bits with best lambdas
    y_col, lj_col = two_param_yeo_johnson(x_col, best_lam1, best_lam2)
    resolution = feature_resolution(x_col)
    bits = feature_bits(y_col, lj_col, resolution)

    return bits, best_lam1, best_lam2


def process_parameter_tensor(
    param_tensor: torch.Tensor, fixed_resolution: torch.Tensor
) -> torch.Tensor:
    """
    Process a parameter tensor as a "feature" with fixed resolution.

    param_tensor: [N] - flattened parameter values
    fixed_resolution: scalar tensor - predetermined resolution for this parameter type
    Returns: scalar bits for this parameter tensor
    """
    # Skip complexity and lambda optimization for parameters - use identity transform
    # This treats parameters as already "normalized"
    y = param_tensor
    log_jac = torch.zeros_like(param_tensor)  # Identity transform has zero jacobian

    # Calculate bits using the fixed resolution
    bits = feature_bits(y, log_jac, fixed_resolution)

    return bits


class MDLLoss(torch.nn.Module):
    def __init__(self, param_resolution_scale: float = 1e-12, weight_model_bits: float = 1.0):
        """
        param_resolution_scale: Base resolution for parameter quantization
        weight_model_bits: Relative weight for model complexity vs data complexity
        """
        super().__init__()
        self.param_resolution_scale = param_resolution_scale
        self.weight_model_bits = weight_model_bits

        # SA state management
        self.feature_lambdas = {}  # feature_idx -> (lam1, lam2)
        self.feature_temperatures = {}  # feature_idx -> temperature
        self.feature_initialized = set()  # track which features have been initialized

    def get_current_lambdas_tensor(
        self, num_features: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current lambda tensors for all features."""
        lam1_list = []
        lam2_list = []

        for i in range(num_features):
            if i in self.feature_lambdas:
                l1, l2 = self.feature_lambdas[i]
            else:
                l1, l2 = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            lam1_list.append(l1)
            lam2_list.append(l2)

        return torch.stack(lam1_list), torch.stack(lam2_list)

    def update_all_feature_states(
        self, new_lam1_tensor: torch.Tensor, new_lam2_tensor: torch.Tensor
    ):
        """Update all feature states from tensors."""
        for i in range(new_lam1_tensor.shape[0]):
            self.feature_lambdas[i] = (new_lam1_tensor[i].detach(), new_lam2_tensor[i].detach())
            # Simple temperature decay
            current_temp = self.feature_temperatures.get(i, 1.0)
            self.feature_temperatures[i] = current_temp * 0.95

    # ... (keep other methods the same)

    def forward(self, residuals: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """
        residuals: [N, num_features] - data residuals to encode
        model: nn.Module - model whose parameters contribute to complexity
        """
        num_features = residuals.shape[1]
        device = residuals.device

        # Get current lambda states as tensors
        current_lam1, current_lam2 = self.get_current_lambdas_tensor(num_features, device)

        # Generate random proposals outside of vmap (10 random numbers per feature)
        random_proposals = torch.randn(num_features, 10, device=device)

        # Data complexity: process residual features with SA (vmap compatible)
        def process_stateless(x_col, lam1, lam2, rand_props):
            return process_single_feature_sa_stateless(x_col, lam1, lam2, rand_props)

        # vmap over features
        results = vmap(process_stateless, in_dims=(1, 0, 0, 0), out_dims=(0, 0, 0))(
            residuals, current_lam1, current_lam2, random_proposals
        )
        bits_features, new_lam1, new_lam2 = results

        # Update states (outside vmap)
        self.update_all_feature_states(new_lam1, new_lam2)

        data_bits = bits_features.sum()

        # Model complexity: process parameter tensors as features
        model_bits = self.compute_model_bits(model)

        # Weighted combination
        return data_bits + self.weight_model_bits * model_bits

    def is_initialized(self, feature_idx: int) -> bool:
        """Check if a feature has been initialized."""
        return feature_idx in self.feature_initialized

    def initialize_feature(self, feature_idx: int, initial_temp: float):
        """Initialize a feature's SA state."""
        self.feature_lambdas[feature_idx] = (torch.tensor(0.0), torch.tensor(0.0))
        self.feature_temperatures[feature_idx] = initial_temp
        self.feature_initialized.add(feature_idx)

    def get_lambdas(self, feature_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current lambda pair for a feature."""
        if feature_idx not in self.feature_lambdas:
            return torch.tensor(0.0), torch.tensor(0.0)
        return self.feature_lambdas[feature_idx]

    def get_temperature(self, feature_idx: int) -> float:
        """Get current temperature for a feature."""
        return self.feature_temperatures.get(feature_idx, 1.0)

    def update_feature_state(
        self, feature_idx: int, new_lam1: torch.Tensor, new_lam2: torch.Tensor, new_temp: float
    ):
        """Update a feature's SA state."""
        self.feature_lambdas[feature_idx] = (new_lam1.detach(), new_lam2.detach())
        self.feature_temperatures[feature_idx] = new_temp

    def estimate_parameter_resolutions(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        Estimate resolution for each parameter based on its empirical distribution.
        Returns tensors for differentiability.
        """
        resolutions = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            flat = param.data.reshape(-1)
            if flat.numel() == 0:
                continue

            # Use the base resolution scale
            resolution = torch.tensor(
                self.param_resolution_scale, device=param.device, dtype=param.dtype
            )
            resolutions[name] = resolution

        return resolutions

    def compute_model_bits(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Compute model complexity bits by treating each parameter tensor as a "feature".
        """
        resolutions = self.estimate_parameter_resolutions(model)
        total_bits = torch.tensor(0.0)

        for name, param in model.named_parameters():
            if not param.requires_grad or name not in resolutions:
                continue

            # Flatten parameter tensor
            flat_param = param.reshape(-1)
            if flat_param.numel() == 0:
                continue

            # Process as feature with fixed resolution
            param_bits = process_parameter_tensor(flat_param, resolutions[name])

            # Move to same device if needed
            if total_bits.device != param_bits.device:
                total_bits = total_bits.to(param_bits.device)

            total_bits = total_bits + param_bits

        return total_bits
