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
    def __init__(self, 
                normalize_method: str = "yeo-johnson",
                data_resolution: float = 1e-6,
                param_resolution: float = 1e-6,
                normalizer_lr: float = 1e-4,
                hidden_dim: int = 128,
                update_normalizer: bool = True,
                debug_logging: bool = False,
                log_frequency: int = 50):
        """
   Initialize MDL Loss function.
   
   Args:
       normalize_method: "yeo-johnson" or "box-cox"
       data_resolution: Quantization resolution for data
       param_resolution: Quantization resolution for parameters
       normalizer_lr: Learning rate for internal normalizer model
       hidden_dim: Hidden dimension for normalizer network
       update_normalizer: Whether to update normalizer during training
       debug_logging: Enable detailed convergence logging
       log_frequency: Log debug info every N calls
   """
        super().__init__()
        
        self.normalize_method = normalize_method
        self.data_resolution = data_resolution
        self.param_resolution = param_resolution
        self.normalizer_lr = normalizer_lr
        self.hidden_dim = hidden_dim
        self.update_normalizer = update_normalizer
        self.debug_logging = debug_logging
        self.log_frequency = log_frequency
        
        # Internal state
        self.normalizer_model = None
        self.normalizer_optimizer = None
        self.num_features = None
        self._initialized = False
        self._call_count = 0
        
        # Moment loss weights
        self.moment_weights = {
            'mean': 1.0, 
            'variance': 2.0, 
            'skewness': 1.5, 
            'kurtosis': 1.0
        }
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
        self._call_count += 1
        
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
        
        # Debug logging
        if self.debug_logging and (self._call_count % self.log_frequency == 0):
            self._log_convergence_metrics(residuals, normalized_residuals, predicted_lambdas, log_jacobian)
        
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
    def _log_convergence_metrics(self, residuals: torch.Tensor, normalized_residuals: torch.Tensor, 
                              predicted_lambdas: torch.Tensor, log_jacobian: torch.Tensor):
        """Log detailed metrics about normalization convergence."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Calculate moments for original and normalized residuals
        orig_moments = calculate_statistical_moments(residuals)
        norm_moments = calculate_statistical_moments(normalized_residuals)
        
        # Lambda statistics
        lambda_mean = predicted_lambdas.mean().item()
        lambda_std = predicted_lambdas.std().item()
        lambda_min = predicted_lambdas.min().item()
        lambda_max = predicted_lambdas.max().item()
        
        # Normalization effectiveness metrics
        orig_mean_abs = orig_moments['mean'].abs().mean().item()
        norm_mean_abs = norm_moments['mean'].abs().mean().item()
        
        orig_var_mean = orig_moments['variance'].mean().item()
        norm_var_mean = norm_moments['variance'].mean().item()
        
        orig_skew_abs = orig_moments['skewness'].abs().mean().item()
        norm_skew_abs = norm_moments['skewness'].abs().mean().item()
        
        orig_kurt_abs = orig_moments['kurtosis'].abs().mean().item()
        norm_kurt_abs = norm_moments['kurtosis'].abs().mean().item()
        
        # Jacobian statistics
        jacobian_mean = log_jacobian.mean().item()
        jacobian_std = log_jacobian.std().item()
        
        # Compression effectiveness
        orig_entropy = self._estimate_entropy(residuals)
        norm_entropy = self._estimate_entropy(normalized_residuals)
        entropy_reduction = orig_entropy - norm_entropy
        
        logger.info(f"\n=== MDL Normalization Debug (Call #{self._call_count}) ===")
        logger.info(f"Lambda Parameters:")
        logger.info(f"  Mean: {lambda_mean:.4f}, Std: {lambda_std:.4f}")
        logger.info(f"  Range: [{lambda_min:.4f}, {lambda_max:.4f}]")
        
        logger.info(f"Moment Convergence:")
        logger.info(f"  Mean |deviation|:  {orig_mean_abs:.6f} → {norm_mean_abs:.6f} ({((norm_mean_abs/orig_mean_abs-1)*100):+.2f}%)")
        logger.info(f"  Variance:          {orig_var_mean:.6f} → {norm_var_mean:.6f} ({((norm_var_mean-1)*100):+.2f}% from target=1)")
        logger.info(f"  |Skewness|:        {orig_skew_abs:.6f} → {norm_skew_abs:.6f} ({((norm_skew_abs/orig_skew_abs-1)*100):+.2f}%)")
        logger.info(f"  |Kurtosis|:        {orig_kurt_abs:.6f} → {norm_kurt_abs:.6f} ({((norm_kurt_abs/orig_kurt_abs-1)*100):+.2f}%)")
        
        logger.info(f"Transform Quality:")
        logger.info(f"  Jacobian: μ={jacobian_mean:.4f}, σ={jacobian_std:.4f}")
        logger.info(f"  Entropy reduction: {entropy_reduction:.4f} bits/sample")
        
        # Gaussianity test (simple normality check)
        gaussian_score = self._gaussianity_score(normalized_residuals)
        logger.info(f"  Gaussianity score: {gaussian_score:.4f} (higher = more Gaussian)")
        
        # Bit calculation breakdown
        centered = normalized_residuals - normalized_residuals.mean(dim=0, keepdim=True)
        variance = centered.var(dim=0, unbiased=False).clamp_min(1e-12)
        theoretical_bits = 0.5 * torch.log2(2 * 3.14159 * 2.71828 * variance).sum().item()
        logger.info(f"  Theoretical encoding bits/sample: {theoretical_bits:.2f}")
        
        logger.info("=" * 50)
    def _estimate_entropy(self, x: torch.Tensor, bins: int = 50) -> float:
        """Estimate differential entropy using histogram method."""
        x_flat = x.flatten()
        
        # Create histogram
        hist, bin_edges = torch.histogram(x_flat, bins=bins, density=True)
        bin_width = (bin_edges[1] - bin_edges[0]).item()
        
        # Convert to probabilities and calculate entropy
        probs = hist * bin_width
        probs = probs[probs > 1e-12]  # Remove zeros
        
        entropy = -(probs * torch.log2(probs)).sum().item()
        return entropy
    def _gaussianity_score(self, x: torch.Tensor) -> float:
        """Simple Gaussianity score based on moment deviations from standard normal."""
        moments = calculate_statistical_moments(x)
        
        # Score based on how close moments are to standard normal (0,1,0,0)
        mean_score = 1.0 / (1.0 + moments['mean'].abs().mean().item())
        var_score = 1.0 / (1.0 + (moments['variance'] - 1.0).abs().mean().item())  
        skew_score = 1.0 / (1.0 + moments['skewness'].abs().mean().item())
        kurt_score = 1.0 / (1.0 + moments['kurtosis'].abs().mean().item())
        
        # Weighted average (variance most important for encoding efficiency)
        total_score = (mean_score + 2*var_score + skew_score + kurt_score) / 5.0
        return total_score
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
        
        # Log normalizer training progress
        if self.debug_logging and (self._call_count % self.log_frequency == 0):
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Normalizer Update - Moment Loss: {moment_loss.item():.6f}")


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
if __name__ == "__main__":
    # Set up logging for debug output
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create realistic dataset matching original example
    torch.manual_seed(42)
    
    # Generate more complex, realistic data (like MNIST flattened)
    num_samples = 1000
    num_features = 784
    
    # Create data with various distributions to make normalization meaningful
    dummy_data = torch.cat([
        torch.randn(num_samples // 4, num_features // 4) * 0.5,  # Normal
        torch.exponential(torch.ones(num_samples // 4, num_features // 4) * 2.0),  # Exponential (skewed)
        torch.uniform(torch.zeros(num_samples // 4, num_features // 4), 
                     torch.ones(num_samples // 4, num_features // 4) * 10),  # Uniform
        torch.randn(num_samples // 4, num_features // 4).pow(2)  # Chi-squared-like
    ], dim=1)
    
    # Add some structure/correlation to make reconstruction meaningful
    noise = torch.randn(num_samples, num_features) * 0.1
    dummy_data = dummy_data + noise
    
    print(f"Dataset shape: {dummy_data.shape}")
    print(f"Data range: [{dummy_data.min().item():.3f}, {dummy_data.max().item():.3f}]")
    print(f"Data mean: {dummy_data.mean().item():.3f}, std: {dummy_data.std().item():.3f}")
    
    # Create DataLoader for batch training
    dataset = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dummy_data), 
        batch_size=64, 
        shuffle=True
    )
    
    # Create autoencoder model (matching original architecture)
    input_dim = num_features
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        # Decoder
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, input_dim)
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize MDL loss with debug logging
    loss_fn = MDLLoss(
        debug_logging=True, 
        log_frequency=100,  # Log every 100 batches
        normalize_method="yeo-johnson"
    )
    
    # Optimizer for main model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\n" + "="*60)
    print("TRAINING WITH DUAL-MODEL CONVERGENCE")
    print("="*60)
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        epoch_mdl_loss = 0.0
        epoch_moment_loss = 0.0
        num_batches = 0
        
        model.train()
        loss_fn.train()
        
        for batch_idx, (batch_data,) in enumerate(dataset):
            # Ensure proper shape
            if batch_data.dim() > 2:
                batch_data = batch_data.view(batch_data.shape[0], -1)
            
            # Forward pass
            yhat = model(batch_data)
            mdl_loss = loss_fn(batch_data, yhat, model)
            
            # Get moment loss for logging (separate computation)
            with torch.no_grad():
                residuals = batch_data - yhat
                if hasattr(loss_fn, 'normalizer_model') and loss_fn.normalizer_model is not None:
                    pred_lambdas = loss_fn.normalizer_model(residuals)
                    if loss_fn.normalize_method == "yeo-johnson":
                        norm_residuals, _ = yeo_johnson_normalize(residuals, pred_lambdas)
                    else:
                        norm_residuals, _ = box_cox_normalize(residuals, pred_lambdas)
                    current_moments = calculate_statistical_moments(norm_residuals)
                    moment_loss = moment_distance_loss(current_moments, weights=loss_fn.moment_weights)
                    epoch_moment_loss += moment_loss.item()
                else:
                    epoch_moment_loss += 0.0
            
            # Backward pass for main model
            optimizer.zero_grad()
            mdl_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_mdl_loss += mdl_loss.item()
            num_batches += 1
            
            # Log batch progress
            if batch_idx % 100 == 0:
                moment_val = moment_loss.item() if 'moment_loss' in locals() else 0.0
                logger.info(f"Epoch {epoch}, Batch {batch_idx}: "
                          f"MDL Loss = {mdl_loss.item():.2f}, "
                          f"Moment Loss = {moment_val:.4f}")
        
        # Epoch summary
        avg_mdl = epoch_mdl_loss / num_batches
        avg_moment = epoch_moment_loss / num_batches
        logger.info(f"Epoch {epoch}: Avg MDL Loss = {avg_mdl:.2f}, "
                   f"Avg Moment Loss = {avg_moment:.4f}")
        
        print(f"Epoch {epoch}: MDL = {avg_mdl:.2f} bits, Moment = {avg_moment:.4f}")
    
    logger.info("Training completed successfully!")
    
    print("\n" + "="*60)
    print("FINAL CONVERGENCE ANALYSIS")
    print("="*60)
    
    # Final analysis
    model.eval()
    loss_fn.eval()
    
    with torch.no_grad():
        # Test on first batch
        test_batch = next(iter(dataset))[0]
        if test_batch.dim() > 2:
            test_batch = test_batch.view(test_batch.shape[0], -1)
        
        yhat_final = model(test_batch)
        final_loss = loss_fn(test_batch, yhat_final, model)
        
        print(f"Final MDL Loss: {final_loss.item():.2f} bits")
        
        # Show normalization effectiveness
        residuals = test_batch - yhat_final
        original_moments = calculate_statistical_moments(residuals)
        
        if hasattr(loss_fn, 'normalizer_model') and loss_fn.normalizer_model is not None:
            pred_lambdas = loss_fn.normalizer_model(residuals)
            norm_residuals, _ = yeo_johnson_normalize(residuals, pred_lambdas)
            normalized_moments = calculate_statistical_moments(norm_residuals)
            
            print(f"\nNormalization Effectiveness:")
            print(f"Original  - Mean: {original_moments['mean'].abs().mean():.4f}, "
                  f"Var: {original_moments['variance'].mean():.4f}, "
                  f"Skew: {original_moments['skewness'].abs().mean():.4f}")
            print(f"Normalized- Mean: {normalized_moments['mean'].abs().mean():.4f}, "
                  f"Var: {normalized_moments['variance'].mean():.4f}, "
                  f"Skew: {normalized_moments['skewness'].abs().mean():.4f}")
if __name__ == "__main__":
    # Set up logging for debug output
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    # Example with the exact interface you requested
    import torch
    
    model = torch.nn.Linear(10, 10)
    x = torch.randn(32, 10)
    yhat = model(x)
    
    # Enable debug logging
    loss_fn = MDLLoss(debug_logging=True, log_frequency=1)
    bits = loss_fn(x, yhat, model)
    print("Total MDL (bits):", bits.item())
    
    # Example with training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("\n" + "="*60)
    print("TRAINING WITH DEBUG LOGGING")
    print("="*60)
    
    for epoch in range(10):
        yhat = model(x)
        mdl_loss = loss_fn(x, yhat, model)
        
        optimizer.zero_grad()
        mdl_loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}: MDL = {mdl_loss.item():.2f} bits")


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
