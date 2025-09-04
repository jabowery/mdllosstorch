"""
MDL Loss for PyTorch - Simple Interface

This module provides a clean interface for calculating Minimum Description Length (MDL) 
loss in PyTorch models. The MDLLoss class maintains an internal normalization model
that learns to optimize residual distributions for better compression.

Usage:
    from mdllosstorch import MDLLoss
    
    model = torch.nn.Linear(10, 10)
    x = torch.randn(32, 10)
    yhat = model(x)
    
    loss_fn = MDLLoss()
    bits = loss_fn(x, yhat, model)
    print("Total MDL (bits):", bits.item())
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import math
from typing import Tuple, Dict, Optional
import warnings

class MDLLoss(nn.Module):
    """
    MDL Loss function that maintains an internal normalization model.
    
    The loss function calculates the total bits needed to encode both the model
    parameters and the residuals (prediction errors), using a learned normalization
    to improve residual compression.
    """
    
    def __init__(self, 
                 normalize_method: str = "yeo-johnson",
                 data_resolution: float = 1e-6,
                 param_resolution: float = 1e-6,
                 normalizer_lr: float = 1e-4,
                 hidden_dim: int = 128,
                 update_normalizer: bool = True):
        """
        Initialize MDL Loss function.
        
        Args:
            normalize_method: "yeo-johnson" or "box-cox"
            data_resolution: Quantization resolution for data
            param_resolution: Quantization resolution for parameters
            normalizer_lr: Learning rate for internal normalizer model
            hidden_dim: Hidden dimension for normalizer network
            update_normalizer: Whether to update normalizer during training
        """
        super().__init__()
        
        self.normalize_method = normalize_method
        self.data_resolution = data_resolution
        self.param_resolution = param_resolution
        self.normalizer_lr = normalizer_lr
        self.hidden_dim = hidden_dim
        self.update_normalizer = update_normalizer
        
        # Internal state
        self.normalizer_model = None
        self.normalizer_optimizer = None
        self.num_features = None
        self._initialized = False
        
        # Moment loss weights
        self.moment_weights = {
            'mean': 1.0, 
            'variance': 2.0, 
            'skewness': 1.5, 
            'kurtosis': 1.0
        }
    
    def _initialize_normalizer(self, num_features: int):
        """Initialize the normalizer model for the given feature dimension."""
        if self._initialized and self.num_features == num_features:
            return
            
        self.num_features = num_features
        self.normalizer_model = NormalizerModel(num_features, self.hidden_dim)
        
        if self.update_normalizer:
            self.normalizer_optimizer = Adam(
                self.normalizer_model.parameters(), 
                lr=self.normalizer_lr
            )
        
        self._initialized = True
    
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
        
        # Lambda parameter encoding cost
        param_bits = num_features * math.log2(100)  # ~6.6 bits per lambda
        
        total_bits = diff_bits + jacobian_bits + quant_bits + param_bits
        
        return total_bits
    
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
    def forward(self, x: torch.Tensor, yhat: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """
   Calculate MDL loss in bits.
   
   Args:
       x: True values [batch_size, num_features]
       yhat: Predicted values [batch_size, num_features]  
       model: The model being evaluated
       
   Returns:
       Total MDL loss in bits (scalar tensor)
   """
        # Ensure inputs are properly shaped
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        if yhat.dim() > 2:
            yhat = yhat.view(yhat.shape[0], -1)
            
        batch_size, num_features = x.shape
        
        # Initialize normalizer if needed
        self._initialize_normalizer(num_features)
        
        # Calculate residuals
        residuals = x - yhat
        
        # Get normalization parameters (always detach to avoid graph issues)
        with torch.no_grad():
            predicted_lambdas = self.normalizer_model(residuals)
        
        # Apply normalization transform
        if self.normalize_method == "yeo-johnson":
            normalized_residuals, log_jacobian = yeo_johnson_normalize(
                residuals, predicted_lambdas
            )
        elif self.normalize_method == "box-cox":
            normalized_residuals, log_jacobian = box_cox_normalize(
                residuals, predicted_lambdas
            )
        else:
            raise ValueError(f"Unknown normalize method: {self.normalize_method}")
        
        # Calculate MDL components
        residual_bits = self._calculate_residual_bits(
            residuals, normalized_residuals, log_jacobian
        )
        parameter_bits = self._calculate_parameter_bits(model)
        
        total_bits = residual_bits + parameter_bits
        
        # Update normalizer model if in training mode (in separate forward pass)
        if self.update_normalizer and self.training and self.normalizer_optimizer is not None:
            self._update_normalizer_separate(residuals.detach())
        
        return total_bits
    def _update_normalizer_separate(self, residuals: torch.Tensor):
        """Update the normalizer model in a separate forward pass to avoid graph conflicts."""
        # Fresh forward pass for normalizer with gradients enabled
        predicted_lambdas = self.normalizer_model(residuals)
        
        # Recompute normalization with gradients
        if self.normalize_method == "yeo-johnson":
            normalized_residuals, _ = yeo_johnson_normalize(residuals, predicted_lambdas)
        else:
            normalized_residuals, _ = box_cox_normalize(residuals, predicted_lambdas)
        
        # Calculate moment loss
        current_moments = calculate_statistical_moments(normalized_residuals)
        moment_loss = moment_distance_loss(current_moments, weights=self.moment_weights)
        
        # Update normalizer
        self.normalizer_optimizer.zero_grad()
        moment_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.normalizer_model.parameters(), max_norm=1.0)
        self.normalizer_optimizer.step()


class NormalizerModel(nn.Module):
    """Model to predict optimal lambda parameters for each feature."""
    
    def __init__(self, num_features: int, hidden_dim: int = 128):
        super().__init__()
        self.num_features = num_features
        
        # Input: 4 statistics per feature (mean, std, skew, kurt)
        stats_input_dim = num_features * 4
        
        self.feature_analyzer = nn.Sequential(
            nn.Linear(stats_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_features),
            nn.Tanh()  # Output in [-1, 1], will be scaled to [-2, 2]
        )
        
    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        """
        Predict lambda parameters based on residual statistics.
        
        Args:
            residuals: [batch_size, num_features]
        
        Returns:
            lambdas: [num_features] in range [-2, 2]
        """
        batch_size, num_features = residuals.shape
        
        # Calculate per-feature statistics
        stats = []
        for i in range(num_features):
            feat_data = residuals[:, i]
            mean_val = feat_data.mean()
            std_val = feat_data.std().clamp_min(1e-8)
            
            standardized = (feat_data - mean_val) / std_val
            skew = standardized.pow(3).mean()
            kurt = standardized.pow(4).mean() - 3.0
            
            stats.extend([mean_val, std_val, skew, kurt])
        
        stats_tensor = torch.stack(stats).unsqueeze(0)  # [1, num_features * 4]
        
        # Predict lambdas
        raw_lambdas = self.feature_analyzer(stats_tensor).squeeze(0)  # [num_features]
        
        # Scale from [-1, 1] to [-2, 2]
        lambdas = raw_lambdas * 2.0
        
        return lambdas


def yeo_johnson_normalize(x: torch.Tensor, lam: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Yeo-Johnson normalization with Jacobian determinant."""
    if lam.dim() == 1 and x.dim() == 2:
        lam = lam.unsqueeze(0).expand(x.shape[0], -1)
    
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    
    normalized = torch.zeros_like(x)
    
    # Positive values: ((x+1)^λ - 1) / λ if λ≠0, else log(x+1)
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
    
    # Negative values: -(((1-x)^(2-λ) - 1) / (2-λ)) if 2-λ≠0, else -log(1-x)
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
    
    # Ensure positivity
    min_val = x.min(dim=0, keepdim=True)[0]
    c = torch.clamp(-min_val + offset, min=offset)
    z = torch.clamp(x + c, min=1e-12)
    
    # Box-Cox transform: (z^λ - 1) / λ if λ≠0, else log(z)
    near_zero = torch.abs(lam) < 1e-8
    normalized = torch.zeros_like(z)
    
    if near_zero.any():
        normalized[near_zero] = torch.log(z[near_zero])
    
    not_zero = ~near_zero
    if not_zero.any():
        z_nz = z[not_zero]
        lam_nz = lam[not_zero]
        normalized[not_zero] = (z_nz ** lam_nz - 1.0) / lam_nz
    
    # Jacobian: (λ-1) * log(z), summed over features
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
                # Use log-space for variance to handle scale differences
                current_log = torch.log(current_moments[moment_name].clamp_min(1e-12))
                target_log = torch.log(target_moments[moment_name].clamp_min(1e-12))
                error = (current_log - target_log).pow(2).mean()
            else:
                error = (current_moments[moment_name] - target_moments[moment_name]).pow(2).mean()
            
            total_loss = total_loss + weight * error
    
    return total_loss


# Example usage
if __name__ == "__main__":
    # Example with the exact interface you requested
    import torch
    
    model = torch.nn.Linear(10, 10)
    x = torch.randn(32, 10)
    yhat = model(x)
    
    loss_fn = MDLLoss()
    bits = loss_fn(x, yhat, model)
    print("Total MDL (bits):", bits.item())
    
    # Example with training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(10):
        yhat = model(x)
        mdl_loss = loss_fn(x, yhat, model)
        
        optimizer.zero_grad()
        mdl_loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}: MDL = {mdl_loss.item():.2f} bits")
