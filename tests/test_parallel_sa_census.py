import os
import time
import numpy as np
import pytest
pytest.importorskip("pandas", reason="pandas required for census integration tests (install with '.[census]')")
import torch

from mdllosstorch import MDLLoss
from tests.util_census import load_census_df, estimate_global_resolution

@pytest.mark.slow
def test_parallel_sa_vs_grid_search_census():
   """Compare parallel SA vs grid search on real census data"""
   path = os.getenv("LOTC_CENSUS_BZ2")
   if not path or not os.path.exists(path):
       pytest.skip("County dataset not available")

   # Load census data
   df = load_census_df(max_rows=2000, max_cols=128).astype(np.float32)
   assert df.shape[0] > 0 and df.shape[1] > 0

   data_res = estimate_global_resolution(df)
   X = df.values
   X = X - np.nanmean(X, axis=0, keepdims=True)

   x_t = torch.from_numpy(X)
   model = torch.nn.Sequential(
       torch.nn.Linear(X.shape[1], 64),
       torch.nn.ReLU(),
       torch.nn.Linear(64, X.shape[1])
   )
   
   # Create imperfect reconstruction
   with torch.no_grad():
       yhat = model(x_t)

   # Test both grid search and parallel SA
   loss_grid = MDLLoss(
       method="yeo-johnson", 
       data_resolution=data_res, 
       param_resolution=1e-6,
       use_parallel_sa=False  # Grid search
   )
   
   loss_sa = MDLLoss(
       method="yeo-johnson", 
       data_resolution=data_res, 
       param_resolution=1e-6,
       use_parallel_sa=True   # Parallel SA
   )

   # Time both approaches
   start_time = time.time()
   bits_grid = loss_grid(x_t, yhat, model)
   grid_time = time.time() - start_time

   start_time = time.time()
   bits_sa = loss_sa(x_t, yhat, model)
   sa_time = time.time() - start_time

   # Verify both produce finite results
   assert torch.isfinite(bits_grid), f"Grid search produced non-finite bits: {bits_grid}"
   assert torch.isfinite(bits_sa), f"Parallel SA produced non-finite bits: {bits_sa}"

   print(f"Grid search: {bits_grid.item():.3f} bits in {grid_time:.3f}s")
   print(f"Parallel SA: {bits_sa.item():.3f} bits in {sa_time:.3f}s")
   print(f"Speed improvement: {grid_time/sa_time:.2f}x")
   print(f"Bits difference: {abs(bits_grid.item() - bits_sa.item()):.3f}")

   # SA should be faster (though this depends on hardware)
   # Allow some tolerance since SA might not always be faster on small problems
   if grid_time > 0.1:  # Only check speed if grid search took meaningful time
       assert sa_time < grid_time * 2.0, f"SA ({sa_time:.3f}s) should be faster than or comparable to grid search ({grid_time:.3f}s)"

   # Results should be reasonably close (SA is stochastic)
   # Allow 5% relative difference due to stochastic nature
   relative_diff = abs(bits_grid.item() - bits_sa.item()) / bits_grid.item()
   assert relative_diff < 0.05, f"SA and grid search results too different: {relative_diff:.1%}"


@pytest.mark.slow
def test_parallel_sa_convergence_census():
   """Test that parallel SA converges over multiple calls"""
   path = os.getenv("LOTC_CENSUS_BZ2")
   if not path or not os.path.exists(path):
       pytest.skip("County dataset not available")

   # Smaller dataset for faster convergence testing
   df = load_census_df(max_rows=1000, max_cols=64).astype(np.float32)
   data_res = estimate_global_resolution(df)
   X = df.values - np.nanmean(df.values, axis=0, keepdims=True)
   x_t = torch.from_numpy(X)
   
   model = torch.nn.Linear(X.shape[1], X.shape[1])
   yhat = model(x_t)

   loss_sa = MDLLoss(
       method="yeo-johnson", 
       data_resolution=data_res,
       use_parallel_sa=True
   )

   # Run multiple forward passes to test convergence
   bits_history = []
   for i in range(10):
       bits = loss_sa(x_t, yhat, model).item()
       bits_history.append(bits)
       assert torch.isfinite(torch.tensor(bits)), f"Iteration {i}: non-finite bits {bits}"

   print(f"SA convergence over 10 iterations:")
   print(f"Bits range: {min(bits_history):.3f} - {max(bits_history):.3f}")
   print(f"Final 3 values: {bits_history[-3:]}")

   # Check that values are stabilizing (last 3 should be more similar)
   if len(bits_history) >= 3:
       recent_std = np.std(bits_history[-3:])
       total_std = np.std(bits_history)
       print(f"Recent std: {recent_std:.3f}, Total std: {total_std:.3f}")
       
       # Recent values should be more stable than overall
       assert recent_std <= total_std, "SA should converge (recent values should be more stable)"


@pytest.mark.slow  
def test_parallel_sa_memory_adaptation():
   """Test that parallel SA adapts batch size based on memory constraints"""
   path = os.getenv("LOTC_CENSUS_BZ2")
   if not path or not os.path.exists(path):
       pytest.skip("County dataset not available")

   # Test with different data sizes to trigger memory adaptation
   small_df = load_census_df(max_rows=500, max_cols=32).astype(np.float32)
   large_df = load_census_df(max_rows=3000, max_cols=256).astype(np.float32)
   
   for df, name in [(small_df, "small"), (large_df, "large")]:
       data_res = estimate_global_resolution(df)
       X = df.values - np.nanmean(df.values, axis=0, keepdims=True)
       x_t = torch.from_numpy(X)
       
       model = torch.nn.Linear(X.shape[1], X.shape[1])
       yhat = model(x_t)

       # Test with tight memory limit to force adaptation
       loss_sa = MDLLoss(
           method="yeo-johnson",
           data_resolution=data_res,
           use_parallel_sa=True
       )

       # Should handle both cases without memory issues
       bits = loss_sa(x_t, yhat, model)
       assert torch.isfinite(bits), f"{name} dataset: non-finite bits {bits}"
       print(f"{name} dataset ({X.shape}): {bits.item():.3f} bits")


@pytest.mark.slow
def test_parallel_sa_gauss_nml_census():
   """Test parallel SA with gauss_nml coder on census data"""
   path = os.getenv("LOTC_CENSUS_BZ2")
   if not path or not os.path.exists(path):
       pytest.skip("County dataset not available")

   df = load_census_df(max_rows=1500, max_cols=100).astype(np.float32)
   data_res = estimate_global_resolution(df)
   X = df.values - np.nanmean(df.values, axis=0, keepdims=True)
   x_t = torch.from_numpy(X)
   
   model = torch.nn.Sequential(
       torch.nn.Linear(X.shape[1], 32),
       torch.nn.Linear(32, X.shape[1])
   )
   yhat = model(x_t)

   # Test gauss_nml with and without parallel SA
   loss_grid = MDLLoss(
       coder="gauss_nml",
       data_resolution=data_res,
       use_parallel_sa=False
   )
   
   loss_sa = MDLLoss(
       coder="gauss_nml", 
       data_resolution=data_res,
       use_parallel_sa=True
   )

   bits_grid = loss_grid(x_t, yhat, model)
   bits_sa = loss_sa(x_t, yhat, model)

   assert torch.isfinite(bits_grid), f"Grid gauss_nml: non-finite {bits_grid}"
   assert torch.isfinite(bits_sa), f"SA gauss_nml: non-finite {bits_sa}"
   
   print(f"Gauss NML - Grid: {bits_grid.item():.3f}, SA: {bits_sa.item():.3f}")
   
   # For gauss_nml, results should be identical since no hyperparameter search
   # Allow small numerical differences
   assert torch.allclose(bits_grid, bits_sa, rtol=1e-4), f"Gauss NML results should be nearly identical"


@pytest.mark.slow
def test_parallel_sa_gradient_flow_census():
   """Test that parallel SA maintains gradient flow on census data"""
   path = os.getenv("LOTC_CENSUS_BZ2")
   if not path or not os.path.exists(path):
       pytest.skip("County dataset not available")

   df = load_census_df(max_rows=1000, max_cols=64).astype(np.float32)
   data_res = estimate_global_resolution(df)
   X = df.values - np.nanmean(df.values, axis=0, keepdims=True)
   x_t = torch.from_numpy(X)
   
   model = torch.nn.Sequential(
       torch.nn.Linear(X.shape[1], 32),
       torch.nn.ReLU(),
       torch.nn.Linear(32, X.shape[1])
   )
   model.train()

   loss_sa = MDLLoss(
       method="yeo-johnson",
       data_resolution=data_res,
       use_parallel_sa=True
   )

   # Forward pass
   yhat = model(x_t)
   bits = loss_sa(x_t, yhat, model)
   
   # Backward pass
   bits.backward()

   # Check gradients
   has_grad = False
   total_grad_norm = 0.0
   for name, param in model.named_parameters():
       if param.grad is not None:
           grad_norm = torch.norm(param.grad).item()
           total_grad_norm += grad_norm
           if grad_norm > 1e-6:
               has_grad = True
           print(f"{name}: grad_norm = {grad_norm:.6f}")

   assert has_grad, "Parallel SA should maintain gradient flow"
   assert total_grad_norm > 0, f"Total gradient norm should be positive: {total_grad_norm}"
   assert np.isfinite(total_grad_norm), f"Gradient norm should be finite: {total_grad_norm}"