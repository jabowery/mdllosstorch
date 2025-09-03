import math

import torch


class ParallelAdaptiveAnnealingSampler:
    """Parallel adaptive simulated annealing for hyperparameter search in MDL loss"""

    def __init__(self, n_parallel=32, initial_temp=1.0, cooling_rate=0.95, param_bounds=None):
        self.n_parallel = n_parallel
        self.temperature = initial_temp
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.current_best = None
        self.current_best_score = float("inf")
        self.param_bounds = param_bounds or self._default_param_bounds()
        self.iteration_count = 0

    def _default_param_bounds(self):
        return {"nu": (1.5, 64.0), "sigma_scale": (0.25, 4.0), "lambda": (-2.0, 2.0)}
    def sample_candidates(self, current_params=None):
        """Generate n_parallel candidate parameter sets using adaptive sampling"""
        candidates = []
        for _i in range(self.n_parallel):
            try:
                if self.current_best is None or torch.rand(1) < 0.3:
                    # Random exploration (30% chance)
                    candidate = self._random_sample()
                else:
                    # Temperature-based perturbation around best
                    candidate = self._perturb_around_best(self.current_best, self.temperature)
                candidates.append(candidate)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to generate candidate {_i}: {e}")
                logger.error(f"param_bounds: {self.param_bounds}")
                logger.error(f"current_best: {self.current_best}")
                # Add a fallback candidate
                candidates.append(self._get_fallback_candidate())
        return candidates
    def _get_fallback_candidate(self):
        """Generate a safe fallback candidate with all required keys"""
        candidate = {}
        if "nu" in self.param_bounds:
            candidate["nu"] = 2.0  # Safe default
        if "sigma_scale" in self.param_bounds:
            candidate["sigma_scale"] = 1.0  # Safe default
        if "lambda" in self.param_bounds:
            candidate["lambda"] = 0.0  # Safe default
        return candidate
    def update_and_accept(self, candidates, scores):
        """Update best solution using Metropolis acceptance criterion"""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            best_idx = torch.argmin(scores)
            best_candidate = candidates[best_idx]
            best_score = float(scores[best_idx])
            
            # Validate best_candidate has required keys
            required_keys = list(self.param_bounds.keys())
            missing_keys = [k for k in required_keys if k not in best_candidate]
            if missing_keys:
                logger.error(f"Best candidate missing keys: {missing_keys}")
                logger.error(f"Best candidate: {best_candidate}")
                logger.error(f"Required keys: {required_keys}")
                # Use fallback
                best_candidate = self._get_fallback_candidate()

            # Always accept improvements
            if best_score < self.current_best_score:
                self.current_best = best_candidate
                self.current_best_score = best_score
                accepted = True
            else:
                # Probabilistic acceptance for worse solutions
                delta = best_score - self.current_best_score
                # Ensure we work with tensors for torch.exp()
                delta_tensor = torch.tensor(delta, dtype=scores.dtype, device=scores.device)
                temp_tensor = torch.tensor(
                    max(self.temperature, 1e-10), dtype=scores.dtype, device=scores.device
                )
                accept_prob = torch.exp(-delta_tensor / temp_tensor)
                if torch.rand(1, device=scores.device) < accept_prob:
                    self.current_best = best_candidate
                    self.current_best_score = best_score
                    accepted = True
                else:
                    accepted = False

            # Cool down temperature
            self.temperature *= self.cooling_rate
            self.iteration_count += 1

            # Reheat if temperature gets too low (adaptive restart)
            if self.temperature < 0.01 and self.iteration_count % 50 == 0:
                self.temperature = self.initial_temp * 0.5

            # Final validation
            if self.current_best is None:
                logger.error("current_best is None after update_and_accept")
                self.current_best = self._get_fallback_candidate()
                
            required_keys = list(self.param_bounds.keys())
            missing_keys = [k for k in required_keys if k not in self.current_best]
            if missing_keys:
                logger.error(f"Final current_best missing keys: {missing_keys}")
                self.current_best = self._get_fallback_candidate()

            return self.current_best, accepted
            
        except Exception as e:
            logger.error(f"Exception in update_and_accept: {e}")
            logger.error(f"candidates: {candidates}")
            logger.error(f"scores: {scores}")
            # Return safe defaults
            fallback = self._get_fallback_candidate()
            self.current_best = fallback
            self.current_best_score = float('inf')
            return fallback, False
    def _random_sample(self):
        """Sample random parameters within bounds"""
        result = {}
        
        if "nu" in self.param_bounds:
            nu = (
                torch.rand(1) * (self.param_bounds["nu"][1] - self.param_bounds["nu"][0])
                + self.param_bounds["nu"][0]
            )
            result["nu"] = float(nu)
            
        if "sigma_scale" in self.param_bounds:
            sigma_scale = (
                torch.rand(1)
                * (self.param_bounds["sigma_scale"][1] - self.param_bounds["sigma_scale"][0])
                + self.param_bounds["sigma_scale"][0]
            )
            result["sigma_scale"] = float(sigma_scale)
            
        if "lambda" in self.param_bounds:
            lambda_val = (
                torch.rand(1) * (self.param_bounds["lambda"][1] - self.param_bounds["lambda"][0])
                + self.param_bounds["lambda"][0]
            )
            result["lambda"] = float(lambda_val)
            
        return result
    def _perturb_around_best(self, best_params, temperature):
        """Smart perturbation based on temperature"""
        result = {}
        
        if "nu" in best_params and "nu" in self.param_bounds:
            nu_std = temperature * 2.0
            nu = torch.clamp(
                torch.tensor(best_params["nu"]) + torch.randn(1) * nu_std,
                self.param_bounds["nu"][0],
                self.param_bounds["nu"][1],
            )
            result["nu"] = float(nu)
            
        if "sigma_scale" in best_params and "sigma_scale" in self.param_bounds:
            sigma_std = temperature * 0.2
            sigma_scale = torch.clamp(
                torch.tensor(best_params["sigma_scale"]) * torch.exp(torch.randn(1) * sigma_std),
                self.param_bounds["sigma_scale"][0],
                self.param_bounds["sigma_scale"][1],
            )
            result["sigma_scale"] = float(sigma_scale)
            
        if "lambda" in best_params and "lambda" in self.param_bounds:
            lambda_std = temperature * 0.3
            lambda_val = torch.clamp(
                torch.tensor(best_params["lambda"]) + torch.randn(1) * lambda_std,
                self.param_bounds["lambda"][0],
                self.param_bounds["lambda"][1],
            )
            result["lambda"] = float(lambda_val)

        return result


class MDLParallelHyperparameterSearch:
    """Efficient hyperparameter search for MDL loss using parallel SA"""

    def estimate_memory_usage(self, n_residuals):
        """Estimate memory usage for parallel evaluation"""
        bytes_per_float = 4
        arrays_per_candidate = 3  # residuals, transformed, intermediate
        return (
            n_residuals * self.n_parallel * arrays_per_candidate * bytes_per_float / (1024 * 1024)
        )

    def adaptive_batch_size(self, n_residuals):
        """Adjust batch size based on memory constraints"""
        estimated_mb = self.estimate_memory_usage(n_residuals)
        if estimated_mb > self.memory_limit_mb:
            scale_factor = self.memory_limit_mb / estimated_mb
            return max(2, int(self.n_parallel * scale_factor))
        return self.n_parallel

    def _yj_transform_and_logabsdet_jac(self, r, lam):
        """Yeo-Johnson transform with Jacobian determinant"""
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

    def _bc_transform_and_logabsdet_jac(self, r, lam, c):
        """Box-Cox transform with Jacobian determinant"""
        z = torch.clamp(r + c, min=1e-9)

        if lam != 0.0:
            t = ((z**lam) - 1.0) / lam
        else:
            t = torch.log(z)

        logabsdet = (lam - 1.0) * torch.log(z).sum()
        return t, logabsdet

    def _evaluate_student_t_candidates(self, parameters, candidates, param_resolution):
        """Parallel evaluation of Student-t hyperparameters for neural network parameters"""
        n_candidates = len(candidates)
        n_parameters = len(parameters)

        # Extract hyperparameters into tensors for vectorized operations
        nus = torch.tensor(
            [params["nu"] for params in candidates],
            device=parameters.device,
            dtype=parameters.dtype,
        )
        sigma_scales = torch.tensor(
            [params["sigma_scale"] for params in candidates],
            device=parameters.device,
            dtype=parameters.dtype,
        )

        # Compute median-based sigma estimate (shared across all candidates)
        med = torch.median(parameters.abs()).item() + 1e-12
        base_sigma = max(med / 0.6745, 1e-9)

        # Broadcast sigmas: shape [n_candidates]
        sigmas = base_sigma * sigma_scales

        # Broadcast parameters for all candidates: shape [n_candidates, n_parameters]
        parameters_expanded = parameters.unsqueeze(0).expand(n_candidates, -1)
        sigmas_expanded = sigmas.unsqueeze(1).expand(
            -1, n_parameters
        )  # shape [n_candidates, n_parameters]
        nus_expanded = nus.unsqueeze(1).expand(
            -1, n_parameters
        )  # shape [n_candidates, n_parameters]

        # Vectorized Student-t log probability computation
        # StudentT PDF: p(w) = Γ((ν+1)/2) / (√(νπ)·Γ(ν/2)·σ) · (1 + w²/(νσ²))^(-(ν+1)/2)

        # Standardized parameters: shape [n_candidates, n_parameters]
        z = parameters_expanded / sigmas_expanded

        # Vectorized log probability computation
        # log p(w)
        # = log_gamma((ν+1)/2) - 0.5*log(νπ) - log_gamma(ν/2) - log(σ) - ((ν+1)/2)*log(1 + w²/ν)
        log_gamma_half_nu_plus_1 = torch.lgamma((nus_expanded + 1) / 2)
        log_gamma_half_nu = torch.lgamma(nus_expanded / 2)
        log_sqrt_nu_pi = 0.5 * (torch.log(nus_expanded) + math.log(math.pi))
        log_sigma = torch.log(sigmas_expanded)
        log_one_plus_z_sq_over_nu = torch.log1p((z * z) / nus_expanded)

        # Shape: [n_candidates, n_parameters]
        log_probs = (
            log_gamma_half_nu_plus_1
            - log_sqrt_nu_pi
            - log_gamma_half_nu
            - log_sigma
            - ((nus_expanded + 1) / 2) * log_one_plus_z_sq_over_nu
        )

        # Sum log probabilities across parameters for each candidate: shape [n_candidates]
        nll_bits = -log_probs.sum(dim=1) / math.log(2.0)  # Convert to bits

        # Fixed hyperparameter encoding cost (independent of parameter count)
        # Cost to encode ν and σ hyperparameters with reasonable precision
        hyperparameter_bits_constant = 2.0 * math.log2(
            100
        )  # ~13 bits total for both hyperparameters

        # Discretization bits scale with parameter count using user-specified resolution
        discretization_bits_constant = n_parameters * math.log2(1.0 / param_resolution)

        # Total bits for each candidate
        scores = nll_bits + hyperparameter_bits_constant + discretization_bits_constant

        return scores

    def _evaluate_lambda_candidates(self, residuals, candidates, method, data_resolution):
        """Fully parallel evaluation of lambda hyperparameters using vectorized transformations"""
        n_candidates = len(candidates)
        n_residuals = len(residuals)

        # Extract lambdas into tensor: shape [n_candidates]
        lambdas = torch.tensor(
            [params["lambda"] for params in candidates],
            device=residuals.device,
            dtype=residuals.dtype,
        )

        # Broadcast residuals and lambdas: shape [n_candidates, n_residuals]
        residuals_expanded = residuals.unsqueeze(0).expand(n_candidates, -1)
        lambdas_expanded = lambdas.unsqueeze(1).expand(-1, n_residuals)

        if method == "yeo-johnson":
            # Vectorized Yeo-Johnson transformation
            transformed_stack, logabsdet_stack = self._vectorized_yj_transform(
                residuals_expanded, lambdas_expanded
            )
        elif method == "box-cox":
            # Vectorized Box-Cox transformation
            transformed_stack, logabsdet_stack = self._vectorized_bc_transform(
                residuals_expanded, lambdas_expanded
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Mean-center each transformed candidate: shape [n_candidates, n_residuals]
        transformed_centered = transformed_stack - transformed_stack.mean(dim=1, keepdim=True)

        # Vectorized variance computation: shape [n_candidates]
        var_t = torch.var(transformed_centered, dim=1, unbiased=False).clamp_min(1e-12)

        # Vectorized bits computation
        bits_gauss = 0.5 * n_residuals * math.log2(
            2.0 * math.pi * math.e
        ) + 0.5 * n_residuals * torch.log2(var_t)
        bits_jac = -(logabsdet_stack / math.log(2.0))

        # Fixed hyperparameter encoding cost (independent of residual count)
        hyperparameter_bits_constant = math.log2(100)  # ~6.6 bits for λ hyperparameter
        if method == "box-cox":
            hyperparameter_bits_constant += math.log2(100)  # Another ~6.6 bits for c hyperparameter

        # Discretization bits using user-specified resolution
        discretization_bits_constant = n_residuals * math.log2(1.0 / data_resolution)

        total_bits = (
            bits_gauss + bits_jac + hyperparameter_bits_constant + discretization_bits_constant
        )

        return total_bits

    def _vectorized_yj_transform(self, residuals_expanded, lambdas_expanded):
        """Vectorized Yeo-Johnson transformation for all candidates simultaneously"""
        # residuals_expanded: [n_candidates, n_residuals]
        # lambdas_expanded: [n_candidates, n_residuals]

        # Split into positive and negative masks
        positive_mask = residuals_expanded >= 0
        negative_mask = ~positive_mask

        # Initialize output tensors
        transformed = torch.zeros_like(residuals_expanded)
        logabsdet_total = torch.zeros(
            residuals_expanded.shape[0],
            device=residuals_expanded.device,
            dtype=residuals_expanded.dtype,
        )

        # Handle positive values: T(r) = ((r+1)^λ - 1) / λ if λ≠0, else log(r+1)
        if positive_mask.any():
            r_pos = residuals_expanded[positive_mask]
            lam_pos = lambdas_expanded[positive_mask]

            # Separate λ=0 and λ≠0 cases
            lam_zero_mask = torch.abs(lam_pos) < 1e-10
            lam_nonzero_mask = ~lam_zero_mask

            if lam_zero_mask.any():
                transformed[positive_mask][lam_zero_mask] = torch.log1p(r_pos[lam_zero_mask])

            if lam_nonzero_mask.any():
                r_nz = r_pos[lam_nonzero_mask]
                lam_nz = lam_pos[lam_nonzero_mask]
                transformed[positive_mask][lam_nonzero_mask] = (
                    (r_nz + 1.0) ** lam_nz - 1.0
                ) / lam_nz

            # Jacobian for positive values: (λ-1) * log(r+1)
            logabsdet_pos = (lambdas_expanded[positive_mask] - 1.0) * torch.log1p(r_pos)
            # Sum across residuals for each candidate
            for i in range(residuals_expanded.shape[0]):
                mask_i = positive_mask[i]
                if mask_i.any():
                    logabsdet_total[i] += logabsdet_pos[positive_mask[i]].sum()

        # Handle negative values: T(r) = -(((1-r)^(2-λ) - 1) / (2-λ)) if 2-λ≠0, else -log(1-r)
        if negative_mask.any():
            r_neg = residuals_expanded[negative_mask]
            lam_neg = lambdas_expanded[negative_mask]
            lam2 = 2.0 - lam_neg

            # Separate (2-λ)=0 and (2-λ)≠0 cases
            lam2_zero_mask = torch.abs(lam2) < 1e-10
            lam2_nonzero_mask = ~lam2_zero_mask

            if lam2_zero_mask.any():
                transformed[negative_mask][lam2_zero_mask] = -torch.log1p(-r_neg[lam2_zero_mask])

            if lam2_nonzero_mask.any():
                r_nz = r_neg[lam2_nonzero_mask]
                lam2_nz = lam2[lam2_nonzero_mask]
                transformed[negative_mask][lam2_nonzero_mask] = -(
                    ((1.0 - r_nz) ** lam2_nz - 1.0) / lam2_nz
                )

            # Jacobian for negative values: (1-λ) * log(1-r)
            logabsdet_neg = (1.0 - lambdas_expanded[negative_mask]) * torch.log1p(-r_neg)
            # Sum across residuals for each candidate
            for i in range(residuals_expanded.shape[0]):
                mask_i = negative_mask[i]
                if mask_i.any():
                    logabsdet_total[i] += logabsdet_neg[negative_mask[i]].sum()

        return transformed, logabsdet_total

    def _vectorized_bc_transform(self, residuals_expanded, lambdas_expanded):
        """Vectorized Box-Cox transformation for all candidates simultaneously"""
        # residuals_expanded: [n_candidates, n_residuals]
        # lambdas_expanded: [n_candidates, n_residuals]

        # Compute offset for each candidate (to ensure positivity)
        # c = max(0, -min(residuals) + ε) for each candidate
        min_per_candidate = residuals_expanded.min(dim=1, keepdim=True)[0]  # [n_candidates, 1]
        c_per_candidate = torch.clamp(-min_per_candidate + 1e-6, min=1e-9)  # [n_candidates, 1]

        # Apply offset: z = r + c
        z = residuals_expanded + c_per_candidate
        z = torch.clamp(z, min=1e-9)  # Ensure positivity

        # Box-Cox transformation: T(z) = (z^λ - 1) / λ if λ≠0, else log(z)
        lam_zero_mask = torch.abs(lambdas_expanded) < 1e-10
        lam_nonzero_mask = ~lam_zero_mask

        transformed = torch.zeros_like(z)

        # λ = 0 case
        if lam_zero_mask.any():
            transformed[lam_zero_mask] = torch.log(z[lam_zero_mask])

        # λ ≠ 0 case
        if lam_nonzero_mask.any():
            z_nz = z[lam_nonzero_mask]
            lam_nz = lambdas_expanded[lam_nonzero_mask]
            transformed[lam_nonzero_mask] = (z_nz**lam_nz - 1.0) / lam_nz

        # Jacobian: (λ-1) * log(z), summed over residuals for each candidate
        logabsdet_per_element = (lambdas_expanded - 1.0) * torch.log(z)
        logabsdet_total = logabsdet_per_element.sum(dim=1)  # [n_candidates]

        return transformed, logabsdet_total
    def __init__(self, n_parallel=32, memory_limit_mb=3000):
        self.n_parallel = n_parallel
        self.memory_limit_mb = memory_limit_mb
        self.student_t_sampler = None
        self.lambda_sampler = None
    def search_student_t_params(self, parameters, param_resolution=1e-6):
        """Search for optimal Student-t parameters using parallel SA"""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            if self.student_t_sampler is None:
                self.student_t_sampler = ParallelAdaptiveAnnealingSampler(
                    n_parallel=self.adaptive_batch_size(len(parameters)),
                    param_bounds={"nu": (1.5, 64.0), "sigma_scale": (0.25, 4.0)},
                )

            logger.debug(f"Student-t search with param_bounds: {self.student_t_sampler.param_bounds}")
            candidates = self.student_t_sampler.sample_candidates()
            
            if not candidates:
                raise ValueError("No candidates generated by sampler")
                
            logger.debug(f"Generated {len(candidates)} candidates: {candidates}")
            
            scores = self._evaluate_student_t_candidates(parameters, candidates, param_resolution)
            logger.debug(f"Scores: {scores}")
            
            best_params, accepted = self.student_t_sampler.update_and_accept(candidates, scores)
            logger.debug(f"Best params after update_and_accept: {best_params}")
            
            if best_params is None:
                raise ValueError("best_params is None")
                
            if 'nu' not in best_params:
                raise ValueError(f"'nu' not in best_params: {best_params}")
                
            if 'sigma_scale' not in best_params:
                raise ValueError(f"'sigma_scale' not in best_params: {best_params}")

            return best_params["nu"], best_params["sigma_scale"]
            
        except Exception as e:
            logger.error(f"Parallel SA search failed: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to simple grid search
            with torch.no_grad():
                xd = parameters.detach()
                med = torch.median(xd.abs()).item() + 1e-12
                base = max(med / 0.6745, param_resolution)
                
                # Simple grid search over common values
                nu_grid = [1.5, 2, 3, 5, 8, 16, 32, 64]
                sigma_scales = [0.25, 0.5, 1.0, 2.0, 4.0]
                sigmas = [base * s for s in sigma_scales]
                
                best = None
                from torch.distributions import StudentT
                for nu in nu_grid:
                    dist = StudentT(df=float(nu), loc=0.0, scale=1.0)
                    for sigma in sigmas:
                        nll_nat = -dist.log_prob(xd / sigma).sum() + xd.numel() * math.log(sigma)
                        bits = nll_nat / math.log(2.0)
                        if (best is None) or (bits < best[0]):
                            best = (bits, float(nu), float(sigma))
                
                if best is None:
                    logger.error("Grid search also failed, using ultimate fallback")
                    return 2.0, 1.0
                    
                nu_star = best[1]
                sigma_star = best[2]
                sigma_scale = sigma_star / base
                
                logger.debug(f"Grid search fallback result: nu={nu_star}, sigma_scale={sigma_scale}")
                return nu_star, sigma_scale
    def search_lambda_params(self, residuals, method="yeo-johnson", data_resolution=1e-6):
        """Search for optimal transformation lambda using parallel SA"""
        if self.lambda_sampler is None:
            self.lambda_sampler = ParallelAdaptiveAnnealingSampler(
                n_parallel=self.adaptive_batch_size(len(residuals)),
                param_bounds={"lambda": (-2.0, 2.0)},
            )

        candidates = self.lambda_sampler.sample_candidates()
        scores = self._evaluate_lambda_candidates(residuals, candidates, method, data_resolution)
        best_params, accepted = self.lambda_sampler.update_and_accept(candidates, scores)

        return best_params["lambda"]
