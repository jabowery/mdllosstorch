import math
import torch
import logging

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
    from .debug_logging import log_tensor_moments, logger
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"=== Gaussian NML Analysis ===")
        log_tensor_moments(residuals, "Input residuals (original shape)")
    
    x = residuals
    if x.ndim == 1:
        x = x.view(-1, 1)

    n, d = x.shape
    n = max(int(n), 1)
    d = max(int(d), 1)

    if logger.isEnabledFor(logging.DEBUG):
        if d > 1:
            logger.debug(f"  Reshaped to: {n} samples x {d} features")
            log_tensor_moments(x, "Reshaped residuals")
        else:
            logger.debug(f"  Processing as: {n} samples x {d} feature (no reshape needed)")

    delta = float(max(data_resolution, _eps()))
    sigma2_floor = (delta ** 2) / 12.0

    var = x.float().pow(2).mean(dim=0) - x.float().mean(dim=0).pow(2)
    var = torch.clamp(var, min=sigma2_floor)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"  Data resolution: {delta:.2e}")
        logger.debug(f"  Variance floor: {sigma2_floor:.2e}")
        if d == 1:
            logger.debug(f"  Raw variance: {var.item():.6e}")
            logger.debug(f"  Clamped variance: {var.item():.6e}")
        else:
            logger.debug(f"  Raw variance range: [{var.min().item():.2e}, {var.max().item():.2e}]")
            logger.debug(f"  Mean variance: {var.mean().item():.2e}")

    diff_per_feat = 0.5 * torch.log2(2.0 * math.pi * math.e * var.clamp_min(_eps()))
    diff_bits = n * diff_per_feat.sum()

    q_const = 0.0
    if include_quantization:
        q_const = n * d * math.log2(max(1.0 / delta, 1.0))

    penalty_sigma = 0.5 * d * math.log2(float(n))

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"  Differential bits: {diff_bits.item():.2f}")
        logger.debug(f"  Quantization bits: {q_const:.2f}")
        logger.debug(f"  Parameter penalty: {penalty_sigma:.2f}")
        logger.debug(f"  Total bits: {(diff_bits + q_const + penalty_sigma).item():.2f}")

    total_bits = diff_bits + q_const + penalty_sigma
    return total_bits
def residual_bits_transformed_gradsafe(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    lam_grid: torch.Tensor = None,
    method: str = "yeo-johnson",
    offset_c: float = None,
    include_param_bits: bool = True,
    data_resolution: float = 1e-6,
    use_parallel_sa: bool = False,
) -> torch.Tensor:
    """MDL residual bits with Yeo-Johnson/Box-Cox, Jacobian, lambda bits, and discretization.
    
    Args:
        use_parallel_sa: If True, use parallel simulated annealing instead of grid search
    """
    from .debug_logging import log_tensor_moments, log_transform_comparison, logger
    
    r_full = (original - reconstructed).flatten()
    mask = torch.isfinite(r_full)
    if not mask.any():
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("  TRANSFORM SEARCH FAILED: No finite residuals found")
            logger.debug("  FALLBACK: Using identity transform (no transform applied)")
        return torch.tensor(float("inf"), device=original.device, dtype=original.dtype)
    r = r_full[mask]

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"=== Residual Transform Analysis ({method}) ===")
        log_tensor_moments(original, "Original data")
        log_tensor_moments(reconstructed, "Reconstructed data") 
        log_tensor_moments(r_full, "Raw residuals")
        log_tensor_moments(r, "Filtered residuals (finite only)")

    lam_star = None
    c_star = None
    search_failed = False

    if use_parallel_sa:
        # Use parallel simulated annealing search
        try:
            from .parallel_sa import MDLParallelHyperparameterSearch
            
            if not hasattr(residual_bits_transformed_gradsafe, '_sa_search'):
                residual_bits_transformed_gradsafe._sa_search = MDLParallelHyperparameterSearch()
            
            with torch.no_grad():
                rd = r.detach()
                lam_star = residual_bits_transformed_gradsafe._sa_search.search_lambda_params(rd, method, data_resolution)
                if method == "box-cox" and offset_c is None:
                    c_star = float(torch.clamp(-(rd.min()) + 1e-6, min=1e-9).item())
                else:
                    c_star = offset_c
                    
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"  PARALLEL SA SEARCH: Found lambda={lam_star:.4f}")
                if method == "box-cox":
                    logger.debug(f"  PARALLEL SA SEARCH: Found c={c_star:.6f}")
                    
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"  PARALLEL SA SEARCH FAILED: {str(e)}")
            search_failed = True
    else:
        # Original grid search implementation
        try:
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
                        
                if best is None:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"  GRID SEARCH FAILED: No valid lambda found in grid search")
                    search_failed = True
                else:
                    lam_star, c_star = best[1], best[2]
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"  GRID SEARCH: Found lambda={lam_star:.4f} with {len(lam_grid)} candidates")
                        if method == "box-cox":
                            logger.debug(f"  GRID SEARCH: Found c={c_star:.6f}")
                            
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"  GRID SEARCH FAILED: {str(e)}")
            search_failed = True

    # Handle search failure with identity fallback
    if search_failed or lam_star is None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("  TRANSFORM SEARCH FAILED: Using identity transform (lambda=0)")
            logger.debug("  FALLBACK: No transform applied to residuals")
        lam_star = 0.0
        c_star = None if method == "yeo-johnson" else float(torch.clamp(-(r.min()) + 1e-6, min=1e-9).item())

    # Apply final transform
    if method == "yeo-johnson":
        t, logabsdet_nat = _yj_transform_and_logabsdet_jac(r, lam_star)
    else:
        if c_star is None:
            c_star = float(torch.clamp(-(r.min()) + 1e-6, min=1e-9).item())
        t, logabsdet_nat = _bc_transform_and_logabsdet_jac(r, lam_star, c_star)

    t = t - t.mean()
    
    if logger.isEnabledFor(logging.DEBUG):
        if abs(lam_star) < 1e-6:
            logger.debug("  FINAL TRANSFORM: Identity (lambda ≈ 0)")
        else:
            logger.debug(f"  FINAL TRANSFORM: {method} with lambda={lam_star:.4f}")
        log_transform_comparison(r, t, f"{method} transform")
        var_t = _safe_var(t)
        logger.debug(f"  Final variance after mean-centering: {var_t.item():.6f}")

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
def residual_bits_per_feature(
    residuals: torch.Tensor,
    method: str = "yeo-johnson", 
    data_resolution: float = 1e-6,
    use_parallel_sa: bool = False,
) -> torch.Tensor:
    """MDL bits with per-feature transforms - each feature gets its own lambda."""
    from .debug_logging import log_tensor_moments, log_transform_comparison, logger
    
    if residuals.ndim == 1:
        residuals = residuals.view(-1, 1)
    
    n_samples, n_features = residuals.shape
    total_bits = torch.tensor(0.0, device=residuals.device, dtype=residuals.dtype)
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"=== Per-Feature Residual Analysis ===")
        logger.debug(f"  Processing {n_features} features with {n_samples} samples each")
        log_tensor_moments(residuals, "All residuals")
    
    # Track statistics for reporting
    transform_improvements = []
    
    for feat_idx in range(n_features):
        r = residuals[:, feat_idx]
        r = r[torch.isfinite(r)]
        
        if r.numel() == 0:
            continue
            
        # Log original feature statistics
        r_original_var = torch.var(r, unbiased=False).item()
        r_original_skew = None
        r_original_kurt = None
        
        if logger.isEnabledFor(logging.DEBUG) and feat_idx % max(1, n_features // 20) == 0:
            # Calculate skewness and kurtosis for detailed features
            with torch.no_grad():
                r_mean = r.mean()
                r_std = r.std(unbiased=False)
                if r_std > 1e-12:
                    r_centered = r - r_mean
                    r_original_skew = (r_centered ** 3).mean().item() / (r_std ** 3).item()
                    r_original_kurt = (r_centered ** 4).mean().item() / (r_std ** 4).item() - 3.0
            
            logger.debug(f"  === Feature {feat_idx} Transform Analysis ===")
            log_tensor_moments(r, f"Feature {feat_idx} before transform")
            
        # Find optimal transform for this feature
        if use_parallel_sa:
            from .parallel_sa import MDLParallelHyperparameterSearch
            sa_search = MDLParallelHyperparameterSearch(memory_limit_mb=500)
            try:
                lam_star = sa_search.search_lambda_params(r, method, data_resolution)
            except:
                lam_star = 0.0
        else:
            # Grid search for this feature
            lam_grid = torch.linspace(-2.0, 2.0, 21, device=r.device, dtype=r.dtype)
            best_var = float('inf')
            lam_star = 0.0
            
            for lam in lam_grid:
                lam_val = float(lam.item())
                try:
                    if method == "yeo-johnson":
                        t, _ = _yj_transform_and_logabsdet_jac(r, lam_val)
                    elif method == "box-cox":
                        c = float(torch.clamp(-(r.min()) + 1e-6, min=1e-9).item())
                        t, _ = _bc_transform_and_logabsdet_jac(r, lam_val, c)
                    else:
                        continue
                        
                    t = t - t.mean()
                    var_t = torch.var(t, unbiased=False)
                    if var_t < best_var:
                        best_var = var_t
                        lam_star = lam_val
                except:
                    continue
        
        # Apply transform and compute bits for this feature
        try:
            if method == "yeo-johnson":
                t, logabsdet_nat = _yj_transform_and_logabsdet_jac(r, lam_star)
            elif method == "box-cox":
                c = float(torch.clamp(-(r.min()) + 1e-6, min=1e-9).item())
                t, logabsdet_nat = _bc_transform_and_logabsdet_jac(r, lam_star, c)
            else:
                t = r
                logabsdet_nat = 0.0
                
            t = t - t.mean()
            var_t = _safe_var(t)
            n = t.numel()
            
            # Track improvement metrics
            var_improvement = r_original_var - var_t.item()
            var_reduction_pct = (var_improvement / r_original_var) * 100 if r_original_var > 0 else 0
            transform_improvements.append({
                'feature': feat_idx,
                'lambda': lam_star,
                'var_before': r_original_var,
                'var_after': var_t.item(),
                'var_reduction_pct': var_reduction_pct
            })
            
            # Detailed logging for selected features
            if logger.isEnabledFor(logging.DEBUG) and feat_idx % max(1, n_features // 20) == 0:
                log_tensor_moments(t, f"Feature {feat_idx} after transform")
                logger.debug(f"    Lambda: {lam_star:.4f}")
                logger.debug(f"    Variance: {r_original_var:.6f} → {var_t.item():.6f} ({var_reduction_pct:+.1f}%)")
                if r_original_skew is not None:
                    # Calculate post-transform skewness/kurtosis
                    t_std = t.std(unbiased=False)
                    if t_std > 1e-12:
                        t_skew = (t ** 3).mean().item() / (t_std ** 3).item()
                        t_kurt = (t ** 4).mean().item() / (t_std ** 4).item() - 3.0
                        logger.debug(f"    Skewness: {r_original_skew:.3f} → {t_skew:.3f}")
                        logger.debug(f"    Kurtosis: {r_original_kurt:.3f} → {t_kurt:.3f}")
                
                # Show extreme values to check for numerical issues
                logger.debug(f"    Range: [{t.min().item():.3f}, {t.max().item():.3f}]")
                if t.max().item() > 100 or t.min().item() < -100:
                    logger.debug(f"    WARNING: Extreme values detected in transformed feature {feat_idx}")
            
            # Bits calculation
            bits_gauss = 0.5 * n * _log2(torch.tensor(2.0 * math.pi * math.e, device=r.device)) + 0.5 * n * _log2(var_t)
            bits_jac = -(logabsdet_nat / _LOG2)
            bits_param = 0.5 * math.log2(max(2, n))
            bits_disc = n * math.log2(1.0 / data_resolution)
            
            feature_bits = bits_gauss + bits_jac + bits_param + bits_disc
            total_bits = total_bits + feature_bits
            
            # Regular progress logging
            if logger.isEnabledFor(logging.DEBUG) and feat_idx % max(1, n_features // 10) == 0:
                logger.debug(f"    Feature {feat_idx}: lambda={lam_star:.3f}, var_reduction={var_reduction_pct:+.1f}%, bits={feature_bits:.1f}")
                
        except Exception as e:
            # Fallback to identity for problematic features
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"    Feature {feat_idx}: Transform failed ({e}), using identity")
            n = r.numel()
            var_r = _safe_var(r)
            bits_gauss = 0.5 * n * _log2(torch.tensor(2.0 * math.pi * math.e, device=r.device)) + 0.5 * n * _log2(var_r)
            bits_disc = n * math.log2(1.0 / data_resolution)
            feature_bits = bits_gauss + bits_disc
            total_bits = total_bits + feature_bits
            
            transform_improvements.append({
                'feature': feat_idx,
                'lambda': 0.0,
                'var_before': r_original_var,
                'var_after': r_original_var,
                'var_reduction_pct': 0.0
            })
    
    # Summary statistics
    if logger.isEnabledFor(logging.DEBUG) and transform_improvements:
        successful_transforms = [t for t in transform_improvements if abs(t['lambda']) > 1e-6]
        var_reductions = [t['var_reduction_pct'] for t in transform_improvements]
        
        logger.debug(f"  === Transform Summary ===")
        logger.debug(f"    Features processed: {len(transform_improvements)}")
        logger.debug(f"    Non-identity transforms: {len(successful_transforms)} ({100*len(successful_transforms)/len(transform_improvements):.1f}%)")
        if var_reductions:
            import statistics
            logger.debug(f"    Variance reduction: mean={statistics.mean(var_reductions):.1f}%, median={statistics.median(var_reductions):.1f}%")
            logger.debug(f"    Best improvement: {max(var_reductions):.1f}%, worst: {min(var_reductions):.1f}%")
        
        logger.debug(f"  Total per-feature bits: {total_bits.item():.1f}")
        
    return total_bits
