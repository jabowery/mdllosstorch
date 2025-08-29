
# mdllosstorch

MDL-friendly loss components for PyTorch:

- Transformed residual bits: Yeo-Johnson (or Box-Cox) + Jacobian + lambda parameter bits + discretization.
- Parameter bits: Student-t prior (continuous nu,sigma), differentiable w.r.t. weights.
- Discretization correction converts differential code lengths to discrete bits (positive).

## Install

```bash
pip install .  # from project root
```

## Minimal Example

```python
import torch
from mdllosstorch import MDLLoss

model = torch.nn.Linear(10, 10)
x = torch.randn(64, 10)
yhat = model(x)

loss_fn = MDLLoss(method="yeo-johnson")
mdl_bits = loss_fn(x, yhat, model)
mdl_bits.backward()
```

## Tests

```bash
pytest -q
```
