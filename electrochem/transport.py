"""electrochem.transport

一维 Fick 扩散传质模型工具。
可用于固/液电解质中的锂离子扩散，采用有限差分 + Method-of-Lines，
得到 ODE 系统便于与核心 ``run_ivp`` 求解器耦合。
"""
from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np


def build_uniform_grid(length: float, n: int) -> Tuple[np.ndarray, float]:
    """Generates a uniform 1D grid."""
    x = np.linspace(0.0, length, n)
    dx = x[1] - x[0]
    return x, dx


def build_fick_rhs(params: Dict[str, Any]):
    """Builds the RHS function for 1D Fickian diffusion in Cartesian coordinates."""
    L = params["L"]
    N = int(params.get("N", 50))
    D_eff = params["D_eff"]
    bc = params.get("bc", "neumann")

    x, dx = build_uniform_grid(L, N)

    def rhs(t: float, c: np.ndarray, p: Dict[str, Any]):
        dcdt = np.zeros_like(c)
        dcdt[1:-1] = D_eff * (c[2:] - 2 * c[1:-1] + c[:-2]) / dx**2

        if bc == "neumann":
            dcdt[0] = D_eff * (2 * (c[1] - c[0])) / dx**2
            dcdt[-1] = D_eff * (2 * (c[-2] - c[-1])) / dx**2
        else:
            raise NotImplementedError(bc)
        return dcdt

    return rhs, x


def build_spherical_fick_rhs(params: Dict[str, Any]):
    """Builds the RHS function for Fickian diffusion in spherical coordinates."""
    R = params["R_particle"]
    N = int(params.get("N_particle", 50))
    D_eff = params["D_eff"]
    dr = params.get("dr", 1e-6)
    r = np.linspace(dr, R, N)
    r_safe = np.concatenate([[0], r])

    def rhs(t: float, c: np.ndarray, p: Dict[str, Any]):
        dcdt = np.zeros_like(c)
        j = -D_eff * np.gradient(c, dr)

        dcdt[1:-1] = -(1 / r_safe[1:-1] ** 2) * np.gradient(
            r_safe[1:-1] ** 2 * j[1:-1], dr
        )

        dcdt[0] = 6 * D_eff * (c[1] - c[0]) / dr**2
        I_app = p.get("I_app", 0.0)
        A_particle = 4 * np.pi * R**2
        j_surface = I_app / A_particle / p["F"]
        dcdt[-1] = -3 * j_surface / R
        return dcdt

    return rhs, r 