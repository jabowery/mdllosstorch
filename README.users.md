
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
