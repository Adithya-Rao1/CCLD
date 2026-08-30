from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def _deriv_kernels(dx: float, dy: float, device, dtype):
    kx = torch.tensor([[-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 1, 3) / (2 * dx)
    ky = torch.tensor([[-1], [0], [1]], dtype=dtype, device=device).view(1, 1, 3, 1) / (2 * dy)
    return kx, ky


def _ddx(field: torch.Tensor, kx: torch.Tensor) -> torch.Tensor:
    return F.conv2d(field.unsqueeze(1), kx, padding=(0, 1)).squeeze(1)


def _ddy(field: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    return F.conv2d(field.unsqueeze(1), ky, padding=(1, 0)).squeeze(1)


def pde_residual_metric(residual: torch.Tensor) -> float:
    return residual.abs().pow(2).mean().item()


E_FLOW_GRID = {"dx": 1.28e-3 / 128, "dy": 1.28e-3 / 128}


def e_flow_residual(fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    kappa, ec_V, u_flow, v_flow = fields["kappa"], fields["ec_V"], fields["u_flow"], fields["v_flow"]
    kx, ky = _deriv_kernels(E_FLOW_GRID["dx"], E_FLOW_GRID["dy"], ec_V.device, ec_V.dtype)

    flow_continuity = _ddx(u_flow, kx) + _ddy(v_flow, ky)

    grad_V_x = _ddx(ec_V, kx)
    grad_V_y = _ddy(ec_V, ky)
    current_continuity = _ddx(kappa * grad_V_x, kx) + _ddy(kappa * grad_V_y, ky)

    return {"flow_continuity": flow_continuity, "current_continuity": current_continuity}


VA_GRID = {"dx": (40 / 128) * 1e-3, "dy": (40 / 128) * 1e-3}
VA_OMEGA = math.pi * 1e5
VA_C_AC = 1.48144e3


def va_residual(fields: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    rho_water = fields["rho_water"]
    p_re, p_im = fields["p_t_re"], fields["p_t_im"]
    Sxx_re, Sxx_im = fields["Sxx_re"], fields["Sxx_im"]
    Sxy_re, Sxy_im = fields["Sxy_re"], fields["Sxy_im"]
    Syy_re, Syy_im = fields["Syy_re"], fields["Syy_im"]
    x_u_re, x_u_im = fields["x_u_re"], fields["x_u_im"]
    x_v_re, x_v_im = fields["x_v_re"], fields["x_v_im"]

    kx, ky = _deriv_kernels(VA_GRID["dx"], VA_GRID["dy"], p_re.device, p_re.dtype)

    def acoustic_residual(p: torch.Tensor) -> torch.Tensor:
        grad_x = _ddx(p, kx)
        grad_y = _ddy(p, ky)
        laplace = _ddx(grad_x / rho_water, kx) + _ddy(grad_y / rho_water, ky)
        return laplace + VA_OMEGA ** 2 * p / (rho_water * VA_C_AC ** 2)

    return {
        "acoustic_real": acoustic_residual(p_re),
        "acoustic_imag": acoustic_residual(p_im),
        "structure_x_real": _ddx(Sxx_re, kx) + _ddy(Sxy_re, ky) + x_u_re,
        "structure_x_imag": _ddx(Sxx_im, kx) + _ddy(Sxy_im, ky) + x_u_im,
        "structure_y_real": _ddx(Sxy_re, kx) + _ddy(Syy_re, ky) + x_v_re,
        "structure_y_imag": _ddx(Sxy_im, kx) + _ddy(Syy_im, ky) + x_v_im,
    }


TE_HEAT_GRID = {"dx": 1e-3, "dy": 1e-3}
TE_HEAT_F = 4e9
TE_HEAT_K0 = 2 * math.pi * TE_HEAT_F / 3e8
TE_HEAT_OMEGA = 2 * math.pi * TE_HEAT_F
TE_HEAT_Q = 1.602
TE_HEAT_MU_R = 1.0
TE_HEAT_EPS_0 = 8.854e-12
TE_HEAT_KB = 8.6173e-5
TE_HEAT_EG = 1.12


def _te_heat_mater_iden(elliptic_params: torch.Tensor, H: int, W: int, device, dtype) -> torch.Tensor:
    coords_x = (torch.arange(H, device=device, dtype=dtype) - (H - 1) / 2) * TE_HEAT_GRID["dx"]
    coords_y = (torch.arange(W, device=device, dtype=dtype) - (W - 1) / 2) * TE_HEAT_GRID["dy"]
    xx, yy = torch.meshgrid(coords_x, coords_y, indexing="ij")
    cx = elliptic_params[:, 0].view(-1, 1, 1)
    cy = elliptic_params[:, 1].view(-1, 1, 1)
    r = elliptic_params[:, 2].view(-1, 1, 1)
    distance_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    return torch.where(distance_sq <= r ** 2, torch.ones_like(distance_sq), -torch.ones_like(distance_sq))


def te_heat_residual(fields: Dict[str, torch.Tensor], elliptic_params: torch.Tensor) -> Dict[str, torch.Tensor]:
    mater = fields["mater"]
    T = fields["T"].clamp(min=1.0)
    Ez = torch.complex(fields["Ez_re"], fields["Ez_im"])

    device, dtype = mater.device, mater.dtype
    _, H, W = mater.shape
    elliptic_params = elliptic_params.to(device=device, dtype=dtype)
    mater_iden = _te_heat_mater_iden(elliptic_params, H, W, device, dtype)

    sigma_coef_map = torch.where(mater_iden > 1e-5, mater, torch.zeros_like(mater))
    sigma_map = TE_HEAT_Q * sigma_coef_map * torch.exp(-TE_HEAT_EG / (TE_HEAT_KB * T))
    sigma_map = torch.where(mater_iden > 1e-5, sigma_map, torch.full_like(sigma_map, 1e-7))
    rho_map = torch.where(mater_iden > 1e-5, torch.full_like(mater, 70.0), mater)
    eps_r = torch.where(mater_iden > 1e-5, torch.full_like(mater, 11.7), torch.ones_like(mater))
    K_E = TE_HEAT_MU_R * TE_HEAT_K0 ** 2 * (eps_r - 1j * sigma_map / (TE_HEAT_OMEGA * TE_HEAT_EPS_0))

    kx_r, ky_r = _deriv_kernels(TE_HEAT_GRID["dx"], TE_HEAT_GRID["dy"], device, dtype)
    c_dtype = torch.complex64 if Ez.dtype == torch.complex64 else torch.complex128
    kx_c, ky_c = kx_r.to(c_dtype), ky_r.to(c_dtype)

    grad_x_E = F.conv2d(Ez.unsqueeze(1), kx_c, padding=(0, 1))
    grad_y_E = F.conv2d(Ez.unsqueeze(1), ky_c, padding=(1, 0))
    laplace_E = (F.conv2d(grad_x_E, kx_c, padding=(0, 1)) + F.conv2d(grad_y_E, ky_c, padding=(1, 0))).squeeze(1)
    result_E = laplace_E + K_E * Ez

    laplace_T = _ddx(_ddx(T, kx_r), kx_r) + _ddy(_ddy(T, ky_r), ky_r)
    result_T = (rho_map * laplace_T + 0.5 * sigma_map * (Ez * torch.conj(Ez))).real

    return {"e_field": result_E, "heat": result_T}
