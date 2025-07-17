"""thermal.energy

一维热传导方程 + 内部均匀热源构造工具。

方程： ρ c_p ∂T/∂t = k ∂²T/∂x² + Q_gen
可选边界：绝热 (Neumann 0) 或 对流 (h (T - T_env))
"""
from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np

__all__ = ["build_heat_rhs"]


def build_heat_rhs(params: Dict[str, Any]):
    """构造温度场 RHS (Method-of-Lines)。

    必需参数
    ----------
    L : 厚度 (m)
    N : 网格数目
    k : 导热系数 (W m^-1 K^-1)
    rho : 密度 (kg m^-3)
    cp : 比热 (J kg^-1 K^-1)
    Q_gen : 均匀体积热源 (W m^-3) 或 callable(t)

    可选参数
    ----------
    bc : "neumann" | "convective"
    h : 对流换热系数 (W m^-2 K^-1)
    T_env : 环境温度 (K)
    """
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
    alpha = k / (rho * cp)

    def Q_gen(t):
        return Q_gen_param(t) if callable(Q_gen_param) else Q_gen_param

    def rhs(t: float, T: np.ndarray, p):  # noqa: D401
        dTdt = np.zeros_like(T)
        # 内部节点
        dTdt[1:-1] = alpha * (T[2:] - 2 * T[1:-1] + T[:-2]) / dx**2 + Q_gen(t) / (rho * cp)
        if bc == "neumann":
            # 绝热
            dTdt[0] = alpha * 2 * (T[1] - T[0]) / dx**2 + Q_gen(t) / (rho * cp)
            dTdt[-1] = alpha * 2 * (T[-2] - T[-1]) / dx**2 + Q_gen(t) / (rho * cp)
        elif bc == "convective":
            dTdt[0] = alpha * 2 * (T[1] - T[0]) / dx**2 + Q_gen(t) / (rho * cp)
            dTdt[-1] = (
                alpha * 2 * (T[-2] - T[-1]) / dx**2
                - 2 * h * (T[-1] - T_env) / (rho * cp)
                + Q_gen(t) / (rho * cp)
            )
        else:
            raise NotImplementedError(bc)
        return dTdt

    return rhs, dx 