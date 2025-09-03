import logging
import torch
import math

logger = logging.getLogger(__name__)

def log_tensor_moments(tensor: torch.Tensor, name: str, prefix: str = ""):
   """Log statistical moments of a tensor for debugging."""
   if not logger.isEnabledFor(logging.DEBUG):
       return
   
   if tensor.numel() == 0:
       logger.debug(f"{prefix}{name}: empty tensor")
       return
   
   with torch.no_grad():
       flat = tensor.flatten()
       finite_mask = torch.isfinite(flat)
       if not finite_mask.any():
           logger.debug(f"{prefix}{name}: no finite values")
           return
       
       finite_vals = flat[finite_mask]
       n = finite_vals.numel()
       
       mean_val = finite_vals.mean().item()
       var_val = finite_vals.var(unbiased=False).item()
       std_val = math.sqrt(var_val)
       
       min_val = finite_vals.min().item()
       max_val = finite_vals.max().item()
       
       # Percentiles
       sorted_vals, _ = torch.sort(finite_vals)
       p25_idx = max(0, int(0.25 * n) - 1)
       p50_idx = max(0, int(0.50 * n) - 1)
       p75_idx = max(0, int(0.75 * n) - 1)
       
       p25 = sorted_vals[p25_idx].item()
       p50 = sorted_vals[p50_idx].item() 
       p75 = sorted_vals[p75_idx].item()
       
       # Skewness and kurtosis approximations
       centered = finite_vals - mean_val
       m3 = (centered ** 3).mean().item()
       m4 = (centered ** 4).mean().item()
       skewness = m3 / (std_val ** 3) if std_val > 1e-12 else 0.0
       kurtosis = m4 / (var_val ** 2) - 3.0 if var_val > 1e-12 else 0.0
       
       logger.debug(
           f"{prefix}{name}: n={n}, mean={mean_val:.6f}, std={std_val:.6f}, "
           f"min={min_val:.6f}, max={max_val:.6f}, "
           f"p25={p25:.6f}, p50={p50:.6f}, p75={p75:.6f}, "
           f"skew={skewness:.6f}, kurt={kurtosis:.6f}"
       )

def log_transform_comparison(original: torch.Tensor, transformed: torch.Tensor, name: str):
   """Log before/after comparison for transforms."""
   if not logger.isEnabledFor(logging.DEBUG):
       return
   
   logger.debug(f"=== Transform Analysis: {name} ===")
   log_tensor_moments(original, "Before", "  ")
   log_tensor_moments(transformed, "After", "  ")