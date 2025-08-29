import torch
from mdllosstorch import (
    MDLLoss,
    residual_bits_transformed_gradsafe,
    parameter_bits_model_student_t,
)


def test_mdl_positivity():
    torch.manual_seed(0)
    model = torch.nn.Linear(10, 10)
    x = torch.randn(100, 10)
    yhat = torch.randn(100, 10)
    loss = MDLLoss()
    bits = loss(x, yhat, model)
    assert bits.item() > 0


def test_perfect_reconstruction():
    torch.manual_seed(0)
    model = torch.nn.Linear(10, 10)
    x = torch.randn(100, 10)
    yhat = x.clone()
    loss = MDLLoss(data_resolution=1e-6, param_resolution=1e-6)
    bits = loss(x, yhat, model)
    assert bits.item() > 0


def test_worse_reconstruction_higher_mdl():
    torch.manual_seed(0)
    model = torch.nn.Linear(10, 10)
    x = torch.randn(100, 10)
    good = x + 0.1 * torch.randn_like(x)
    bad = x + 1.0 * torch.randn_like(x)
    loss = MDLLoss()
    mdl_good = loss(x, good, model).item()
    mdl_bad = loss(x, bad, model).item()
    assert mdl_bad > mdl_good


def test_larger_model_higher_mdl():
    torch.manual_seed(0)
    x = torch.randn(50, 10)
    yhat = torch.randn(50, 10)

    small = torch.nn.Linear(10, 5)
    big = torch.nn.Sequential(
        torch.nn.Linear(10, 100),
        torch.nn.Linear(100, 100),
        torch.nn.Linear(100, 10),
    )
    loss = MDLLoss()
    mdl_small = loss(x, yhat, small).item()
    mdl_big = loss(x, yhat, big).item()
    assert mdl_big > mdl_small


def test_component_additivity():
    torch.manual_seed(0)
    model = torch.nn.Linear(10, 10)
    x = torch.randn(64, 10)
    yhat = torch.randn(64, 10)

    loss = MDLLoss()
    total = loss(x, yhat, model)

    lam_grid = torch.linspace(-2.0, 2.0, 81, device=x.device, dtype=x.dtype)
    res = residual_bits_transformed_gradsafe(
        x, yhat, lam_grid=lam_grid, method="yeo-johnson", data_resolution=1e-6
    )
    par = parameter_bits_model_student_t(model, param_resolution=1e-6)
    assert torch.allclose(total, res + par, atol=1e-4, rtol=1e-5)


def test_gradient_flow():
    torch.manual_seed(0)
    model = torch.nn.Linear(10, 10)
    model.train()
    x = torch.randn(32, 10)
    yhat = model(x)

    loss = MDLLoss()
    bits = loss(x, yhat, model)
    bits.backward()

    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).any() for p in model.parameters()
    )
    assert has_grad
