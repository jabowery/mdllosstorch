
# mdllosstorch – Instructions for Users

You are an **end-user** of the library. You want to *install and use* it in your own project.

## Install
```bash
pip install mdllosstorch
```

## Requirements
- Python ≥3.9
- PyTorch ≥1.12

## Example
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

## Extended API
```python
import torch
from mdllosstorch.mdlloss import compute_mdl, report_mdl

x = torch.randn(1000, 64)
yhat = x.clone()  # perfect recon
model = torch.nn.Identity()

bits = compute_mdl(x, yhat, model, data_resolution="auto")
print("total bits:", bits.item())

print(report_mdl(x, yhat, model, data_resolution="auto"))
```

[API Documentation](mdllosstorch_API.pdf)
