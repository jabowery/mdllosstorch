import torch
from torch import nn
from .residuals import residual_bits_transformed_gradsafe
from .parameters import parameter_bits_model_student_t


class MDLLoss(nn.Module):
    """Total MDL loss in bits:
    L = residual_bits (Yeo-Johnson/Box-Cox + Jacobian + discretization)
      + parameter_bits (Student-t with discretization)
    """

    def __init__(
        self,
        method: str = "yeo-johnson",
        include_transform_param_bits: bool = True,
        data_resolution: float = 1e-6,
        param_resolution: float = 1e-6,
        lam_grid: torch.Tensor = None,
    ):
        super().__init__()
        self.method = method
        self.include_transform_param_bits = include_transform_param_bits
        self.data_resolution = float(data_resolution)
        self.param_resolution = float(param_resolution)
        self._lam_grid = lam_grid

    def forward(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        model: torch.nn.Module,
    ) -> torch.Tensor:
        if self._lam_grid is None:
            lam_grid = torch.linspace(
                -2.0, 2.0, 81, device=original.device, dtype=original.dtype
            )
        else:
            lam_grid = self._lam_grid.to(device=original.device, dtype=original.dtype)

        res_bits = residual_bits_transformed_gradsafe(
            original=original,
            reconstructed=reconstructed,
            lam_grid=lam_grid,
            method=self.method,
            include_param_bits=self.include_transform_param_bits,
            data_resolution=self.data_resolution,
        )
        par_bits = parameter_bits_model_student_t(
            model, include_param_bits=True, param_resolution=self.param_resolution
        )
        return res_bits + par_bits
