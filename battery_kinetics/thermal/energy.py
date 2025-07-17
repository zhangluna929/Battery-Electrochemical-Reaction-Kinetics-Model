"""thermal.energy

一维热传导方程 + 内部均匀热源构造工具。

方程： ρ c_p ∂T/∂t = k ∂²T/∂x² + Q_gen
可选边界：绝热 (Neumann 0) 或 对流 (h (T - T_env))
"""
from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np

__all__ = ["build_heat_transport_rhs"]


def build_heat_transport_rhs(params: Dict[str, Any]):
    """Builds the RHS function for 1D heat transport."""
    L = params["L"]
    N = int(params.get("N", 50))
    k = params["k"]
    rho = params["rho"]
    cp = params["cp"]
    Q_gen_param = params.get("Q_gen", 0.0)
    bc = params.get("bc", "neumann")
    h = params.get("h", 0.0)
    T_env = params.get("T_env", 298.15)

    dx = L / (N - 1)
    alpha_th = k / (rho * cp)

    def Q_gen(t):
        return Q_gen_param(t) if callable(Q_gen_param) else Q_gen_param

    def rhs(t: float, T: np.ndarray, p):
        dTdt = np.zeros_like(T)
        q_gen_term = Q_gen(t) / (rho * cp)

        dTdt[1:-1] = alpha_th * (T[2:] - 2 * T[1:-1] + T[:-2]) / dx**2 + q_gen_term
        if bc == "neumann":
            dTdt[0] = alpha_th * 2 * (T[1] - T[0]) / dx**2 + q_gen_term
            dTdt[-1] = alpha_th * 2 * (T[-2] - T[-1]) / dx**2 + q_gen_term
        elif bc == "convective":
            # Assuming one end is insulated and the other has convective cooling.
            dTdt[0] = alpha_th * 2 * (T[1] - T[0]) / dx**2 + q_gen_term
            dTdt[-1] = (
                alpha_th * 2 * (T[-2] - T[-1]) / dx**2
                - (2 * h * (T[-1] - T_env)) / (rho * cp * dx)
                + q_gen_term
            )
        else:
            raise NotImplementedError(bc)
        return dTdt

    return rhs, dx 