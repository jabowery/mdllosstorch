# mdlloss.py - Clean API with private implementation details
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class MDLNumericalError(Exception):
    """Exception for truly unrecoverable numerical issues.""" 
    pass


def _estimate_feature_complexity(residuals: torch.Tensor) -> torch.Tensor:
    """
    Estimate optimization complexity for each feature.
    Higher complexity means higher initial temperature needed.
    """
    B, F = residuals.shape
    complexity = torch.zeros(F, device=residuals.device)
    
    for j in range(F):
        col = residuals[:, j]
        finite_col = col[torch.isfinite(col)]
        
        if finite_col.numel() < 10:
            complexity[j] = 1.0  # Default
            continue
        
        # Indicators of optimization difficulty
        var = finite_col.var()
        abs_skew = torch.abs(finite_col - finite_col.median()).mean()
        range_ratio = (finite_col.max() - finite_col.min()) / (finite_col.std() + 1e-8)
        
        # Combine into complexity score
        complexity[j] = 1.0 + 0.1 * torch.log1p(var) + 0.05 * abs_skew + 0.02 * range_ratio
        
    return complexity.clamp(0.1, 10.0)


def _evaluate_two_param_score(col_residuals: torch.Tensor, lam1: float, lam2: float) -> float:
    """Evaluate quality of lambda parameters for a single feature."""
    try:
        # Apply transform
        lam1_tensor = torch.tensor(lam1, device=col_residuals.device)
        lam2_tensor = torch.tensor(lam2, device=col_residuals.device)
        
        y, _ = _two_param_yeo_johnson(
            col_residuals.unsqueeze(1), 
            lam1_tensor.unsqueeze(0), 
            lam2_tensor.unsqueeze(0)
        )
        
        finite_mask = torch.isfinite(y[:, 0])
        if finite_mask.sum() < 5:
            return 1e6
        
        y_finite = y[finite_mask, 0]
        
        # Score based on normality (minimize departure from standard normal)
        mean = y_finite.mean()
        var = y_finite.var(unbiased=False)
        centered = y_finite - mean
        std = torch.sqrt(var + 1e-12)
        z = centered / std
        
        skew = z.pow(3).mean()
        kurt = z.pow(4).mean() - 3.0
        
        # Penalize extreme departures from normality
        score = skew.pow(2) + kurt.pow(2) + (var - 1.0).pow(2) + mean.pow(2)
        return score.item()
        
    except Exception:
        return 1e6


def _optimize_feature_lambdas_sa(col_residuals: torch.Tensor, 
                               complexity: float,
                               max_steps: int = 2000) -> Tuple[float, float]:
    """
    Optimize lambda1, lambda2 for a single feature using simulated annealing.
    Temperature schedule adapts based on feature complexity.
    """
    # Initialize
    current_lam1 = torch.randn(1).item() * 0.5
    current_lam2 = torch.randn(1).item() * 0.5
    current_score = _evaluate_two_param_score(col_residuals, current_lam1, current_lam2)
    
    best_lam1, best_lam2, best_score = current_lam1, current_lam2, current_score
    
    # Temperature schedule based on complexity
    initial_temp = complexity * 2.0
    
    accept_count = 0
    total_proposals = 0
    
    for step in range(max_steps):
        # Temperature schedule
        progress = step / max_steps
        temp = initial_temp * (0.995 ** step)
        
        # Early termination if temperature too low
        if temp < 0.001:
            break
        
        # Adaptive compute: fewer proposals as temperature decreases
        temp_ratio = temp / initial_temp
        n_proposals = max(1, int(10 * temp_ratio))
        
        for _ in range(n_proposals):
            # Propose new parameters
            step_size = temp * 0.1
            proposed_lam1 = current_lam1 + torch.randn(1).item() * step_size
            proposed_lam2 = current_lam2 + torch.randn(1).item() * step_size
            
            # Clamp to safe bounds
            proposed_lam1 = max(-1.9, min(1.9, proposed_lam1))
            proposed_lam2 = max(-1.9, min(1.9, proposed_lam2))
            
            # Evaluate
            proposed_score = _evaluate_two_param_score(col_residuals, proposed_lam1, proposed_lam2)
            
            # Accept/reject
            total_proposals += 1
            if proposed_score < current_score or torch.rand(1).item() < math.exp(-(proposed_score - current_score) / temp):
                current_lam1, current_lam2, current_score = proposed_lam1, proposed_lam2, proposed_score
                accept_count += 1
                
                # Track best
                if current_score < best_score:
                    best_lam1, best_lam2, best_score = current_lam1, current_lam2, current_score
    
    return best_lam1, best_lam2


def _optimize_all_features_parallel(residuals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimize lambda parameters for all features using parallel simulated annealing.
    """
    B, F = residuals.shape
    
    # Estimate per-feature complexity
    complexities = _estimate_feature_complexity(residuals)
    
    lambda1_opt = torch.zeros(F, device=residuals.device)
    lambda2_opt = torch.zeros(F, device=residuals.device)
    
    # Optimize each feature independently (could be parallelized)
    for j in range(F):
        col_residuals = residuals[:, j]
        finite_mask = torch.isfinite(col_residuals)
        
        if finite_mask.sum() < 10:
            # Not enough data, use default
            lambda1_opt[j] = 0.0
            lambda2_opt[j] = 0.0
            continue
        
        col_finite = col_residuals[finite_mask]
        lam1, lam2 = _optimize_feature_lambdas_sa(col_finite, complexities[j].item())
        
        lambda1_opt[j] = lam1
        lambda2_opt[j] = lam2
    
    return lambda1_opt, lambda2_opt


def _estimate_per_parameter_resolutions(model: nn.Module) -> Dict[str, float]:
    """
    Estimate resolution for each parameter based on its empirical distribution.
    """
    resolutions = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        flat = param.data.reshape(-1)
        if flat.numel() == 0:
            continue
        
        # Estimate resolution from parameter value distribution
        sorted_vals, _ = torch.sort(flat)
        if sorted_vals.numel() > 1:
            diffs = torch.diff(sorted_vals)
            pos_diffs = diffs[diffs > 1e-12]
            if pos_diffs.numel() > 0:
                resolution = pos_diffs.median().item()
            else:
                resolution = 1e-6
        else:
            resolution = 1e-6
            
        # Conservative clipping
        resolution = max(1e-12, min(1e-3, resolution))
        resolutions[name] = resolution
    
    return resolutions


def _estimate_per_feature_resolutions(residuals: torch.Tensor) -> torch.Tensor:
    """
    Estimate resolution for each feature based on its residual distribution.
    """
    B, F = residuals.shape
    resolutions = torch.zeros(F, device=residuals.device)
    
    for j in range(F):
        col = residuals[:, j]
        finite_col = col[torch.isfinite(col)]
        
        if finite_col.numel() < 2:
            resolutions[j] = 1e-6
            continue
        
        # Estimate from unique value gaps
        sorted_vals, _ = torch.sort(finite_col)
        unique_vals = torch.unique(sorted_vals)
        
        if unique_vals.numel() > 1:
            diffs = torch.diff(unique_vals)
            pos_diffs = diffs[diffs > 1e-12]
            if pos_diffs.numel() > 0:
                resolution = pos_diffs.median()
            else:
                resolution = torch.tensor(1e-6, device=residuals.device)
        else:
            resolution = torch.tensor(1e-6, device=residuals.device)
        
        resolutions[j] = resolution.clamp(1e-12, 1e-1)
    
    return resolutions


def _compute_parameter_bits(model: nn.Module, param_resolutions: Dict[str, float]) -> torch.Tensor:
    """Compute parameter encoding cost using per-parameter resolutions."""
    try:
        device = next(iter(model.parameters())).device
    except StopIteration:
        return torch.tensor(0.0)
    
    total = torch.tensor(0.0, device=device)
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        flat = param.reshape(-1)
        if flat.numel() == 0:
            continue
        
        resolution = param_resolutions.get(name, 1e-6)
        
        # Differential entropy
        var = flat.var(unbiased=False).clamp_min(1e-12)
        diff_bits = flat.numel() * (0.5 * math.log2(2 * math.pi * math.e) + 0.5 * torch.log2(var))
        
        # Quantization
        quant_bits = flat.numel() * math.log2(1.0 / resolution)
        
        total = total + diff_bits + quant_bits
    
    return total


class MDLLoss(nn.Module):
    """
    Sophisticated per-feature MDL loss with adaptive optimization.
    
    Estimates all resolutions automatically - no global parameters needed.
    Uses two-parameter Yeo-Johnson transform with simulated annealing optimization.
    """

    def __init__(
        self,
        normalizer_lr: float = 1e-3,
        sa_max_steps: int = 2000,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.normalizer_lr = normalizer_lr
        self.sa_max_steps = sa_max_steps
        self.device = device

    def forward(self, x: torch.Tensor, yhat: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
        Compute MDL using sophisticated per-feature optimization.
        
        Args:
            x: Original data [batch, features]
            yhat: Model reconstruction [batch, features]  
            model: PyTorch model used for reconstruction
            
        Returns:
            total_bits: Total description length in bits
        """
        if x.dim() != 2 or yhat.dim() != 2:
            raise ValueError("x and yhat must be [batch, features].")
        if x.shape != yhat.shape:
            raise ValueError("x and yhat must have same shape.")

        device = yhat.device
        B, F = x.shape
        
        if not torch.isfinite(x).all() or not torch.isfinite(yhat).all():
            raise MDLNumericalError("Input contains non-finite values")
        
        # Compute residuals
        residuals = x - yhat
        if not torch.isfinite(residuals).all():
            raise MDLNumericalError("Residuals contain non-finite values")

        # Optimize lambda parameters per feature
        lambda1_opt, lambda2_opt = _optimize_all_features_parallel(residuals)
        
        # Estimate per-feature resolutions
        feature_resolutions = _estimate_per_feature_resolutions(residuals)
        
        # Compute residual encoding cost
        residual_bits = _compute_per_feature_bits(residuals, lambda1_opt, lambda2_opt, feature_resolutions)
        
        # Estimate per-parameter resolutions and compute parameter cost
        param_resolutions = _estimate_per_parameter_resolutions(model)
        parameter_bits = _compute_parameter_bits(model, param_resolutions)
        
        # Total MDL
        total_bits = residual_bits + parameter_bits
        
        if not torch.isfinite(total_bits):
            raise MDLNumericalError("Non-finite MDL computed")
        
        return total_bits
def _two_param_yeo_johnson(x: torch.Tensor, lambda1: torch.Tensor, lambda2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two-parameter Yeo-Johnson transform with separate lambdas for positive and negative branches.
    
    Args:
        x: Input values [B, F] 
        lambda1: Parameters for positive branch [F]
        lambda2: Parameters for negative branch [F]
        
    Returns:
        y: Transformed values [B, F]
        log_jacobian_terms: Log Jacobian terms per feature [B, F] - NOT collapsed to [B]
    """
    if x.numel() == 0:
        raise MDLNumericalError("Empty input tensor")
    
    B, F = x.shape
    if lambda1.shape != (F,) or lambda2.shape != (F,):
        raise ValueError("Lambda parameters must have shape [F]")
    
    # Expand lambdas to match data shape
    lam1 = lambda1.unsqueeze(0).expand(B, F)  # [B, F]
    lam2 = lambda2.unsqueeze(0).expand(B, F)  # [B, F]
    
    y = torch.zeros_like(x)
    log_jac_terms = torch.zeros_like(x)  # [B, F] - per feature, per sample
    
    # Positive branch: x >= 0
    pos_mask = x >= 0
    if pos_mask.any():
        x_pos = x[pos_mask]
        lam1_pos = lam1[pos_mask]
        
        eps = 1e-8
        lam1_abs = torch.abs(lam1_pos)
        
        # Safe bounds check
        safe_pos = (x_pos.abs() <= 1e3) & (lam1_abs <= 1.9)
        
        if safe_pos.any():
            x_safe = x_pos[safe_pos]
            lam_safe = lam1_pos[safe_pos]
            
            # Transform: ((x+1)^lambda - 1)/lambda or log(x+1) 
            y_general = ((x_safe + 1.0).pow(lam_safe) - 1.0) / lam_safe
            y_l0 = torch.log1p(x_safe)
            y_transformed = torch.where(torch.abs(lam_safe) < eps, y_l0, y_general)
            
            # Log Jacobian: (lambda-1) * log(x+1)
            lj_safe = (lam_safe - 1.0) * torch.log1p(x_safe)
            
            # Store results maintaining [B, F] structure
            y_pos_safe = torch.full_like(x_pos, float('nan'))
            lj_pos_safe = torch.zeros_like(x_pos)
            
            y_pos_safe[safe_pos] = y_transformed
            lj_pos_safe[safe_pos] = lj_safe
            
            y[pos_mask] = y_pos_safe
            log_jac_terms[pos_mask] = lj_pos_safe
    
    # Negative branch: x < 0
    neg_mask = x < 0
    if neg_mask.any():
        x_neg = x[neg_mask]
        lam2_neg = lam2[neg_mask]
        
        eps = 1e-8
        
        # Safe bounds: need x > -0.99 and reasonable lambda2
        safe_neg = (x_neg > -0.99) & (torch.abs(lam2_neg) <= 1.9)
        
        if safe_neg.any():
            x_safe = x_neg[safe_neg]
            lam_safe = lam2_neg[safe_neg]
            l2 = 2.0 - lam_safe
            
            # Transform: -(((1-x)^(2-lambda) - 1)/(2-lambda)) or -log(1-x)
            y_general = -((1.0 - x_safe).pow(l2) - 1.0) / l2
            y_l0 = -torch.log1p(-x_safe)
            y_transformed = torch.where(torch.abs(l2) < eps, y_l0, y_general)
            
            # Log Jacobian: (1-lambda) * log(1-x)  
            lj_safe = (1.0 - lam_safe) * torch.log1p(-x_safe)
            
            # Store results maintaining [B, F] structure
            y_neg_safe = torch.full_like(x_neg, float('nan'))
            lj_neg_safe = torch.zeros_like(x_neg)
            
            y_neg_safe[safe_neg] = y_transformed
            lj_neg_safe[safe_neg] = lj_safe
            
            y[neg_mask] = y_neg_safe
            log_jac_terms[neg_mask] = lj_neg_safe
    
    # Return per-feature log Jacobian terms [B, F], NOT collapsed to [B]
    return y, log_jac_terms
def _compute_per_feature_bits(residuals: torch.Tensor, 
                           lambda1: torch.Tensor, 
                           lambda2: torch.Tensor,
                           resolutions: torch.Tensor) -> torch.Tensor:
    """
    Compute bits for each feature independently using its transform and resolution.
    Properly respects per-feature architecture throughout.
    """
    B, F = residuals.shape
    total_bits = torch.tensor(0.0, device=residuals.device)
    
    # Apply per-feature transforms - now returns per-feature Jacobian terms
    y_trans, log_jac_terms = _two_param_yeo_johnson(residuals, lambda1, lambda2)
    
    # Process each feature independently - including its Jacobian contribution
    for j in range(F):
        col_transformed = y_trans[:, j]
        col_jac_terms = log_jac_terms[:, j]  # Per-feature Jacobian terms
        finite_mask = torch.isfinite(col_transformed)
        n_finite = finite_mask.sum().item()
        n_exceptions = B - n_finite
        
        if n_finite > 0:
            # Differential entropy for transformed finite values
            y_finite = col_transformed[finite_mask]
            if y_finite.numel() > 1:
                var = y_finite.var(unbiased=False).clamp_min(1e-12)
                diff_bits = n_finite * (0.5 * math.log2(2 * math.pi * math.e) + 0.5 * torch.log2(var))
            else:
                diff_bits = torch.tensor(0.0, device=residuals.device)
        else:
            diff_bits = torch.tensor(0.0, device=residuals.device)
        
        # Per-feature Jacobian contribution
        finite_jac = col_jac_terms[finite_mask]
        if finite_jac.numel() > 0:
            jacobian_bits_j = finite_jac.sum() / math.log(2.0)
        else:
            jacobian_bits_j = torch.tensor(0.0, device=residuals.device)
        
        # Direct encoding for exceptions
        exception_bits = n_exceptions * math.log2(1.0 / resolutions[j].item()) if n_exceptions > 0 else 0.0
        
        # Partition overhead
        if n_exceptions > 0 and n_exceptions < B:
            log_binom = (math.lgamma(B + 1) - 
                        math.lgamma(n_exceptions + 1) - 
                        math.lgamma(B - n_exceptions + 1))
            partition_bits = log_binom / math.log(2.0)
        else:
            partition_bits = 0.0
        
        # Lambda encoding (2 parameters per feature)
        lambda_bits = 2 * math.log2(100.0)
        
        # Sum all per-feature contributions
        feature_bits = diff_bits + jacobian_bits_j + exception_bits + partition_bits + lambda_bits
        total_bits = total_bits + feature_bits
    
    return total_bits
