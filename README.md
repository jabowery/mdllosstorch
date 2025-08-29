
# mdllosstorch

Minimum Description Length (MDL) loss components for PyTorch.

This repository provides implementations of:

- Transformed residual coding (Yeo-Johnson / Box-Cox + Jacobian + discretization)
- Student-t parameter coding with discretization
- Unified `MDLLoss` module for PyTorch

---

## Documentation by Role

Different audiences should follow different guides:

- [README.users.md](README.users.md) — For **users** installing and using `mdllosstorch` in their own projects.
- [README.maintainers.md](README.maintainers.md) — For **maintainers** releasing new versions to PyPI.
- [README.developers.md](README.developers.md) — For **developers** contributing to the project.

---

## Quickstart (User)
```bash
pip install mdllosstorch
```

```python
import torch
from mdllosstorch import MDLLoss

model = torch.nn.Linear(10, 10)
x = torch.randn(32, 10)
yhat = model(x)

loss_fn = MDLLoss()
bits = loss_fn(x, yhat, model)
print("Total MDL (bits):", bits.item())
```
