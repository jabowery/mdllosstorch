import math
import torch
from torch.distributions import StudentT


class ParallelAdaptiveAnnealingSampler:
   """Parallel adaptive simulated annealing for hyperparameter search in MDL loss"""
   
   def __init__(self, n_parallel=8, initial_temp=1.0, cooling_rate=0.95, param_bounds=None):
       self.n_parallel = n_parallel
       self.temperature = initial_temp
       self.initial_temp = initial_temp
       self.cooling_rate = cooling_rate
       self.current_best = None
       self.current_best_score = float('inf')
       self.param_bounds = param_bounds or self._default_param_bounds()
       self.iteration_count = 0
       
   def _default_param_bounds(self):
       return {
           'nu': (1.5, 64.0),
           'sigma_scale': (0.25, 4.0),
           'lambda': (-2.0, 2.0)
       }

   def sample_candidates(self, current_params=None):
       """Generate n_parallel candidate parameter sets using adaptive sampling"""
       candidates = []
       for i in range(self.n_parallel):
           if self.current_best is None or torch.rand(1) < 0.3:
               # Random exploration (30% chance)
               candidate = self._random_sample()
           else:
               # Temperature-based perturbation around best
               candidate = self._perturb_around_best(self.current_best, self.temperature)
           candidates.append(candidate)
       return candidates

   def _random_sample(self):
       """Sample random parameters within bounds"""
       nu = torch.rand(1) * (self.param_bounds['nu'][1] - self.param_bounds['nu'][0]) + self.param_bounds['nu'][0]
       sigma_scale = torch.rand(1) * (self.param_bounds['sigma_scale'][1] - self.param_bounds['sigma_scale'][0]) + self.param_bounds['sigma_scale'][0]
       lambda_val = torch.rand(1) * (self.param_bounds['lambda'][1] - self.param_bounds['lambda'][0]) + self.param_bounds['lambda'][0]
       return {'nu': float(nu), 'sigma_scale': float(sigma_scale), 'lambda': float(lambda_val)}

   def _perturb_around_best(self, best_params, temperature):
       """Smart perturbation based on temperature"""
       nu_std = temperature * 2.0
       sigma_std = temperature * 0.2
       lambda_std = temperature * 0.3
       
       nu = torch.clamp(
           torch.tensor(best_params['nu']) + torch.randn(1) * nu_std,
           self.param_bounds['nu'][0], self.param_bounds['nu'][1]
       )
       sigma_scale = torch.clamp(
           torch.tensor(best_params['sigma_scale']) * torch.exp(torch.randn(1) * sigma_std),
           self.param_bounds['sigma_scale'][0], self.param_bounds['sigma_scale'][1]
       )
       lambda_val = torch.clamp(
           torch.tensor(best_params['lambda']) + torch.randn(1) * lambda_std,
           self.param_bounds['lambda'][0], self.param_bounds['lambda'][1]
       )
       
       return {'nu': float(nu), 'sigma_scale': float(sigma_scale), 'lambda': float(lambda_val)}
   def update_and_accept(self, candidates, scores):
      """Update best solution using Metropolis acceptance criterion"""
      best_idx = torch.argmin(scores)
      best_candidate = candidates[best_idx]
      best_score = float(scores[best_idx])
      
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
         temp_tensor = torch.tensor(max(self.temperature, 1e-10), dtype=scores.dtype, device=scores.device)
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
      
      return self.current_best, accepted


class MDLParallelHyperparameterSearch:
   """Efficient hyperparameter search for MDL loss using parallel SA"""
   
   def __init__(self, n_parallel=8, memory_limit_mb=100):
       self.n_parallel = n_parallel
       self.memory_limit_mb = memory_limit_mb
       self.param_sampler = None
       self.lambda_sampler = None
       
   def estimate_memory_usage(self, n_residuals):
       """Estimate memory usage for parallel evaluation"""
       bytes_per_float = 4
       arrays_per_candidate = 3  # residuals, transformed, intermediate
       return n_residuals * self.n_parallel * arrays_per_candidate * bytes_per_float / (1024 * 1024)
   
   def adaptive_batch_size(self, n_residuals):
       """Adjust batch size based on memory constraints"""
       estimated_mb = self.estimate_memory_usage(n_residuals)
       if estimated_mb > self.memory_limit_mb:
           scale_factor = self.memory_limit_mb / estimated_mb
           return max(2, int(self.n_parallel * scale_factor))
       return self.n_parallel
   
   def search_student_t_params(self, residuals):
       """Search for optimal Student-t parameters using parallel SA"""
       if self.param_sampler is None:
           self.param_sampler = ParallelAdaptiveAnnealingSampler(
               n_parallel=self.adaptive_batch_size(len(residuals)),
               param_bounds={
                   'nu': (1.5, 64.0),
                   'sigma_scale': (0.25, 4.0),
                   'lambda': (-2.0, 2.0)
               }
           )
       
       candidates = self.param_sampler.sample_candidates()
       scores = self._evaluate_student_t_candidates(residuals, candidates)
       best_params, accepted = self.param_sampler.update_and_accept(candidates, scores)
       
       return best_params['nu'], best_params['sigma_scale']
   
   def search_lambda_params(self, residuals, method='yeo-johnson'):
       """Search for optimal transformation lambda using parallel SA"""
       if self.lambda_sampler is None:
           self.lambda_sampler = ParallelAdaptiveAnnealingSampler(
               n_parallel=self.adaptive_batch_size(len(residuals)),
               param_bounds={'lambda': (-2.0, 2.0)}
           )
       
       candidates = self.lambda_sampler.sample_candidates()
       scores = self._evaluate_lambda_candidates(residuals, candidates, method)
       best_params, accepted = self.lambda_sampler.update_and_accept(candidates, scores)
       
       return best_params['lambda']
   
   def _evaluate_student_t_candidates(self, residuals, candidates):
       """Parallel evaluation of Student-t parameter candidates"""
       scores = torch.zeros(len(candidates), device=residuals.device)
       
       for i, params in enumerate(candidates):
           nu, sigma_scale = params['nu'], params['sigma_scale']
           
           # Compute median-based sigma estimate
           med = torch.median(residuals.abs()).item() + 1e-12
           sigma = max(med / 0.6745, 1e-9) * sigma_scale
           
           # Evaluate Student-t NLL + parameter bits
           dist = StudentT(df=float(nu), loc=0.0, scale=1.0)
           nll_nat = -dist.log_prob(residuals / sigma).sum() + len(residuals) * math.log(sigma)
           bits = nll_nat / math.log(2.0)
           
           # Add parameter encoding bits
           n = len(residuals)
           param_bits = 0.5 * math.log2(max(2, n)) + 0.5 * math.log2(max(2, n))
           discretization_bits = n * math.log2(1000.0)  # 1e-6 resolution
           
           scores[i] = bits + param_bits + discretization_bits
       
       return scores
   
   def _evaluate_lambda_candidates(self, residuals, candidates, method):
       """Parallel evaluation of lambda parameter candidates"""
       scores = torch.zeros(len(candidates), device=residuals.device)
       
       for i, params in enumerate(candidates):
           lam = params['lambda']
           
           try:
               if method == 'yeo-johnson':
                   t, logabsdet_nat = self._yj_transform_and_logabsdet_jac(residuals, lam)
               else:  # box-cox
                   c = float(torch.clamp(-(residuals.min()) + 1e-6, min=1e-9).item())
                   t, logabsdet_nat = self._bc_transform_and_logabsdet_jac(residuals, lam, c)
               
               # Mean-center and compute variance
               t = t - t.mean()
               var_t = torch.var(t, unbiased=False).clamp_min(1e-12)
               
               # Compute total bits
               n = len(t)
               bits_gauss = 0.5 * n * math.log2(2.0 * math.pi * math.e) + 0.5 * n * math.log2(var_t)
               bits_jac = -(logabsdet_nat / math.log(2.0))
               bits_param = 0.5 * math.log2(max(2, n))
               bits_disc = n * math.log2(1000.0)  # 1e-6 resolution
               
               scores[i] = bits_gauss + bits_jac + bits_param + bits_disc
               
           except Exception:
               scores[i] = float('inf')  # Invalid parameter
       
       return scores
   
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
           t = ((z ** lam) - 1.0) / lam
       else:
           t = torch.log(z)
       
       logabsdet = (lam - 1.0) * torch.log(z).sum()
       return t, logabsdet
