"""
MDL Loss with Proper Separation of Concerns

This approach maintains clean separation between:
1. MDL calculation (stateless)
2. Normalizer management (stateful)
3. Training coordination (user-controlled)

The MDLLoss class becomes a pure calculator, while a separate NormalizerTrainer
handles the normalizer optimization. This allows users to control both optimizers
explicitly while still providing a clean interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import math
from typing import Tuple, Dict, Optional, Union
import warnings

class MDLLoss(nn.Module):
    """
    Pure MDL calculator that works with any normalization function.
    Does not maintain internal state or optimizers.
    """
    
    def __init__(self, 
                 data_resolution: float = 1e-6,
                 param_resolution: float = 1e-6):
        """
        Initialize MDL Loss calculator.
        
        Args:
            data_resolution: Quantization resolution for data
            param_resolution: Quantization resolution for parameters
        """
        super().__init__()
        self.data_resolution = data_resolution
        self.param_resolution = param_resolution
    
    def forward(self, 
                x: torch.Tensor, 
                yhat: torch.Tensor, 
                model: nn.Module,
                normalize_fn: Optional[callable] = None) -> torch.Tensor:
        """
        Calculate MDL loss in bits.
        
        Args:
            x: True values [batch_size, num_features]
            yhat: Predicted values [batch_size, num_features]  
            model: The model being evaluated
            normalize_fn: Optional function that takes residuals and returns (normalized, log_jacobian)
            
        Returns:
            Total MDL loss in bits (scalar tensor)
        """
        # Ensure inputs are properly shaped
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        if yhat.dim() > 2:
            yhat = yhat.view(yhat.shape[0], -1)
            
        # Calculate residuals
        residuals = x - yhat
        
        # Apply normalization if provided
        if normalize_fn is not None:
            normalized_residuals, log_jacobian = normalize_fn(residuals)
        else:
            # No normalization - assume residuals are already well-distributed
            normalized_residuals = residuals
            log_jacobian = torch.zeros(residuals.shape[0], device=residuals.device)
        
        # Calculate MDL components
        residual_bits = self._calculate_residual_bits(
            residuals, normalized_residuals, log_jacobian
        )
        parameter_bits = self._calculate_parameter_bits(model)
        
        return residual_bits + parameter_bits
    
    def _calculate_residual_bits(self, residuals: torch.Tensor, 
                               normalized_residuals: torch.Tensor,
                               log_jacobian: torch.Tensor) -> torch.Tensor:
        """Calculate bits needed to encode residuals."""
        batch_size, num_features = normalized_residuals.shape
        
        # Center the normalized residuals
        centered = normalized_residuals - normalized_residuals.mean(dim=0, keepdim=True)
        
        # Calculate Gaussian encoding bits per feature
        variance = centered.var(dim=0, unbiased=False).clamp_min(1e-12)
        
        # Differential entropy in bits: 0.5 * log2(2πe * σ²)
        LOG2_2PIE = 0.5 * math.log2(2 * math.pi * math.e)
        diff_bits_per_sample = LOG2_2PIE + 0.5 * torch.log2(variance)
        diff_bits = batch_size * diff_bits_per_sample.sum()
        
        # Jacobian correction (convert from natural log to bits)
        jacobian_bits = -log_jacobian.sum() / math.log(2.0)
        
        # Quantization bits
        quant_bits = batch_size * num_features * math.log2(1.0 / self.data_resolution)
        
        return diff_bits + jacobian_bits + quant_bits
    
    def _calculate_parameter_bits(self, model: nn.Module) -> torch.Tensor:
        """Calculate bits needed to encode model parameters."""
        total_bits = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for param in model.parameters():
            if param.requires_grad:
                flat_param = param.flatten()
                n = flat_param.numel()
                
                if n > 0:
                    # Simple Gaussian encoding
                    variance = flat_param.var(unbiased=False).clamp_min(1e-12)
                    
                    # Differential entropy + quantization
                    diff_bits = n * (0.5 * math.log2(2 * math.pi * math.e) + 
                                   0.5 * torch.log2(variance))
                    quant_bits = n * math.log2(1.0 / self.param_resolution)
                    
                    total_bits = total_bits + diff_bits + quant_bits
        
        return total_bits


class NormalizerTrainer:
    """
    Manages training of normalization models separately from MDL calculation.
    This class maintains the normalizer state and provides the normalize function.
    """
    def __init__(self,
                num_features: int,
                normalize_method: str = "yeo-johnson",
                lr: float = 1e-4,
                moment_weights: Optional[Dict[str, float]] = None):
        """
   Initialize normalizer trainer with direct lambda parameters.
   
   Args:
       num_features: Number of input features
       normalize_method: "yeo-johnson" or "box-cox"
       lr: Learning rate for lambda parameters
       moment_weights: Weights for different moments in loss
   """
        self.num_features = num_features
        self.normalize_method = normalize_method
        
        # Direct lambda parameters - one per feature
        self.lambdas = nn.Parameter(torch.zeros(num_features))
        self.optimizer = Adam([self.lambdas], lr=lr)
        
        # Moment loss configuration
        self.moment_weights = moment_weights or {
            'mean': 1.0, 'variance': 2.0, 'skewness': 1.5, 'kurtosis': 1.0
        }
    def get_normalize_function(self):
        """
   Return a normalization function using current lambda parameters.
   """
        def normalize_fn(residuals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Apply normalization transform to residuals."""
            # Use current lambda parameters directly
            current_lambdas = self.lambdas.detach()
            
            # Apply transform
            if self.normalize_method == "yeo-johnson":
                return yeo_johnson_normalize(residuals, current_lambdas)
            elif self.normalize_method == "box-cox":
                return box_cox_normalize(residuals, current_lambdas)
            else:
                raise ValueError(f"Unknown normalize method: {self.normalize_method}")
        
        return normalize_fn
    def get_stats(self, residuals: torch.Tensor) -> Dict[str, float]:
        """Get normalization effectiveness statistics."""
        with torch.no_grad():
            # Get normalization using current lambdas
            normalize_fn = self.get_normalize_function()
            normalized_residuals, _ = normalize_fn(residuals)
            
            # Calculate moments
            orig_moments = calculate_statistical_moments(residuals)
            norm_moments = calculate_statistical_moments(normalized_residuals)
            
            return {
                'lambda_mean': self.lambdas.mean().item(),
                'lambda_std': self.lambdas.std().item(),
                'lambda_min': self.lambdas.min().item(),
                'lambda_max': self.lambdas.max().item(),
                'orig_mean_abs': orig_moments['mean'].abs().mean().item(),
                'norm_mean_abs': norm_moments['mean'].abs().mean().item(),
                'orig_var_mean': orig_moments['variance'].mean().item(),
                'norm_var_mean': norm_moments['variance'].mean().item(),
                'orig_skew_abs': orig_moments['skewness'].abs().mean().item(),
                'norm_skew_abs': norm_moments['skewness'].abs().mean().item(),
            }
    def update(self, residuals: torch.Tensor, regularization_weight: float = 1e-4) -> float:
        """
   Update lambda parameters using moment loss + L2 regularization.
   
   Args:
       residuals: Current residuals to normalize
       regularization_weight: L2 penalty weight for lambda parameters
       
   Returns:
       Combined loss value (moment + L2)
   """
        # Apply normalization with current lambdas (gradients enabled)
        if self.normalize_method == "yeo-johnson":
            normalized_residuals, _ = yeo_johnson_normalize(residuals.detach(), self.lambdas)
        else:
            normalized_residuals, _ = box_cox_normalize(residuals.detach(), self.lambdas)
        
        # Calculate moment loss in 4D space
        current_moments = calculate_statistical_moments(normalized_residuals)
        moment_loss = moment_distance_loss(current_moments, weights=self.moment_weights)
        
        # Add L2 regularization to prevent lambda parameters from growing too large
        l2_penalty = regularization_weight * torch.norm(self.lambdas)**2
        total_loss = moment_loss + l2_penalty
        
        # Update lambda parameters
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([self.lambdas], max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item()


def yeo_johnson_normalize(x: torch.Tensor, lam: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Yeo-Johnson normalization with Jacobian determinant."""
    if lam.dim() == 1 and x.dim() == 2:
        lam = lam.unsqueeze(0).expand(x.shape[0], -1)
    
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    
    normalized = torch.zeros_like(x)
    
    # Positive values
    if pos_mask.any():
        x_pos = x[pos_mask]
        lam_pos = lam[pos_mask]
        
        near_zero = torch.abs(lam_pos) < 1e-8
        if near_zero.any():
            normalized[pos_mask & (torch.abs(lam) < 1e-8)] = torch.log1p(x_pos[near_zero])
        
        not_zero = ~near_zero
        if not_zero.any():
            x_nz = x_pos[not_zero]
            lam_nz = lam_pos[not_zero]
            normalized[pos_mask & (torch.abs(lam) >= 1e-8)] = ((x_nz + 1.0) ** lam_nz - 1.0) / lam_nz
    
    # Negative values
    if neg_mask.any():
        x_neg = x[neg_mask]
        lam_neg = lam[neg_mask]
        lam2 = 2.0 - lam_neg
        
        near_zero = torch.abs(lam2) < 1e-8
        if near_zero.any():
            normalized[neg_mask & (torch.abs(2.0 - lam) < 1e-8)] = -torch.log1p(-x_neg[near_zero])
        
        not_zero = ~near_zero
        if not_zero.any():
            x_nz = x_neg[not_zero]
            lam2_nz = lam2[not_zero]
            normalized[neg_mask & (torch.abs(2.0 - lam) >= 1e-8)] = -(((1.0 - x_nz) ** lam2_nz - 1.0) / lam2_nz)
    
    # Jacobian calculation
    log_jacobian_terms = torch.zeros_like(x)
    
    if pos_mask.any():
        log_jacobian_terms[pos_mask] = (lam[pos_mask] - 1.0) * torch.log1p(x[pos_mask])
    
    if neg_mask.any():
        log_jacobian_terms[neg_mask] = (1.0 - lam[neg_mask]) * torch.log1p(-x[neg_mask])
    
    log_jacobian = log_jacobian_terms.sum(dim=1)
    
    return normalized, log_jacobian


def box_cox_normalize(x: torch.Tensor, lam: torch.Tensor, offset: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Box-Cox normalization with Jacobian determinant."""
    if lam.dim() == 1 and x.dim() == 2:
        lam = lam.unsqueeze(0).expand(x.shape[0], -1)
    
    min_val = x.min(dim=0, keepdim=True)[0]
    c = torch.clamp(-min_val + offset, min=offset)
    z = torch.clamp(x + c, min=1e-12)
    
    near_zero = torch.abs(lam) < 1e-8
    normalized = torch.zeros_like(z)
    
    if near_zero.any():
        normalized[near_zero] = torch.log(z[near_zero])
    
    not_zero = ~near_zero
    if not_zero.any():
        z_nz = z[not_zero]
        lam_nz = lam[not_zero]
        normalized[not_zero] = (z_nz ** lam_nz - 1.0) / lam_nz
    
    log_jacobian = ((lam - 1.0) * torch.log(z)).sum(dim=1)
    
    return normalized, log_jacobian


def calculate_statistical_moments(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Calculate first four statistical moments for each feature."""
    mean = x.mean(dim=0)
    centered = x - mean
    
    variance = centered.pow(2).mean(dim=0)
    std = torch.sqrt(variance.clamp_min(1e-12))
    
    standardized = centered / std.clamp_min(1e-12)
    
    skewness = standardized.pow(3).mean(dim=0)
    kurtosis = standardized.pow(4).mean(dim=0) - 3.0
    
    return {
        'mean': mean,
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurtosis
    }


def moment_distance_loss(current_moments: Dict[str, torch.Tensor], 
                        target_moments: Optional[Dict[str, torch.Tensor]] = None,
                        weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
    """Calculate weighted distance from target moments."""
    if target_moments is None:
        target_moments = {
            'mean': torch.zeros_like(current_moments['mean']),
            'variance': torch.ones_like(current_moments['variance']),
            'skewness': torch.zeros_like(current_moments['skewness']),
            'kurtosis': torch.zeros_like(current_moments['kurtosis'])
        }
    
    if weights is None:
        weights = {'mean': 1.0, 'variance': 2.0, 'skewness': 1.5, 'kurtosis': 1.0}
    
    total_loss = torch.tensor(0.0, device=current_moments['mean'].device)
    
    for moment_name, weight in weights.items():
        if moment_name in current_moments and moment_name in target_moments:
            if moment_name == 'variance':
                current_log = torch.log(current_moments[moment_name].clamp_min(1e-12))
                target_log = torch.log(target_moments[moment_name].clamp_min(1e-12))
                error = (current_log - target_log).pow(2).mean()
            else:
                error = (current_moments[moment_name] - target_moments[moment_name]).pow(2).mean()
            
            total_loss = total_loss + weight * error
    
    return total_loss


# Usage example demonstrating the clean separation
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create data
    torch.manual_seed(42)
    num_samples, num_features = 1000, 784
    
    # Create diverse data distributions
    dummy_data = torch.cat([
        torch.randn(num_samples, num_features // 4) * 0.5,
        torch.distributions.Exponential(2.0).sample((num_samples, num_features // 4)),
        torch.rand(num_samples, num_features // 4) * 10,
        torch.randn(num_samples, num_features // 4).abs()
    ], dim=1)
    
    dataset = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dummy_data),
        batch_size=64, shuffle=True
    )
    
    # Create model and both loss components
    model = torch.nn.Sequential(
        torch.nn.Linear(num_features, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, num_features)
    )
    
    # Initialize both components
    mdl_loss = MDLLoss()
    normalizer_trainer = NormalizerTrainer(num_features)
    
    # Create separate optimizers - clean separation!
    model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # normalizer has its own optimizer internal to NormalizerTrainer
    
    print("Training with Clean Separation of Concerns")
    print("=" * 50)
    
    for epoch in range(5):
        epoch_mdl_loss = 0.0
        epoch_moment_loss = 0.0
        num_batches = 0
        
        for batch_idx, (batch_data,) in enumerate(dataset):
            if batch_data.dim() > 2:
                batch_data = batch_data.view(batch_data.shape[0], -1)
            
            # Forward pass
            yhat = model(batch_data)
            residuals = batch_data - yhat
            
            # Get current normalization function
            normalize_fn = normalizer_trainer.get_normalize_function()
            
            # Calculate MDL loss with normalization
            mdl_loss_value = mdl_loss(batch_data, yhat, model, normalize_fn)
            
            # Update main model based on MDL
            model_optimizer.zero_grad()
            mdl_loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model_optimizer.step()
            
            # Update normalizer based on moment distance (separate objective!)
            moment_loss_value = normalizer_trainer.update(residuals.detach())
            
            # Logging
            epoch_mdl_loss += mdl_loss_value.item()
            epoch_moment_loss += moment_loss_value
            num_batches += 1
            
            if batch_idx % 100 == 0:
                stats = normalizer_trainer.get_stats(residuals.detach())
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: "
                          f"MDL = {mdl_loss_value.item():.0f}, "
                          f"Moment = {moment_loss_value:.3f}, "
                          f"Lambda μ = {stats['lambda_mean']:.3f}")
        
        avg_mdl = epoch_mdl_loss / num_batches
        avg_moment = epoch_moment_loss / num_batches
        print(f"Epoch {epoch}: MDL = {avg_mdl:.0f}, Moment = {avg_moment:.3f}")
    
    print("\nFinal Analysis:")
    with torch.no_grad():
        test_batch = next(iter(dataset))[0]
        if test_batch.dim() > 2:
            test_batch = test_batch.view(test_batch.shape[0], -1)
        
        yhat_final = model(test_batch)
        residuals = test_batch - yhat_final
        
        normalize_fn = normalizer_trainer.get_normalize_function()
        final_mdl = mdl_loss(test_batch, yhat_final, model, normalize_fn)
        
        stats = normalizer_trainer.get_stats(residuals)
        
        print(f"Final MDL: {final_mdl.item():.0f} bits")
        print(f"Normalization: {stats['orig_var_mean']:.3f} → {stats['norm_var_mean']:.3f} (variance)")
        print(f"Lambda range: μ={stats['lambda_mean']:.3f}, σ={stats['lambda_std']:.3f}")
