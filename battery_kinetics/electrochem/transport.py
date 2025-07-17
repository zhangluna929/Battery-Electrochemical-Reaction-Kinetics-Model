"""electrochem.transport

一维 Fick 扩散传质模型工具。
可用于固/液电解质中的锂离子扩散，采用有限差分 + Method-of-Lines，
得到 ODE 系统便于与核心 ``run_ivp`` 求解器耦合。
"""
from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np

__all__ = ["build_fick_rhs", "build_uniform_grid"]


def build_uniform_grid(length: float, n: int) -> Tuple[np.ndarray, float]:
    """生成均匀网格坐标与步长。"""
    x = np.linspace(0.0, length, n)
    dx = x[1] - x[0]
    return x, dx


def build_fick_rhs(params: Dict[str, Any]):
    """根据参数字典构建 1D Fick 扩散方程 RHS。

    Parameters
    ----------
    params : Dict[str, Any]
        必要字段：
        - L : 电极/固体厚度 (m)
        - N : 网格节点数 (int)
        - D_eff : 有效扩散系数 (m^2 s^-1)
    可选字段：
        - bc : 边界条件类型，目前支持 "neumann" (default)。
    Returns
    -------
    rhs : Callable(t, y, p) -> ndarray
        适用于 ``core.ode.run_ivp`` 的 RHS。
    x : ndarray
        网格坐标 (m)。
    """
    L = params["L"]
    N = int(params.get("N", 50))
    D_eff = params["D_eff"]
    bc = params.get("bc", "neumann")  # 只实现 Neumann 零通量

    x, dx = build_uniform_grid(L, N)

    # 组装稀疏矩阵?  为简化直接在 RHS 中手动差分
    def rhs(t: float, c: np.ndarray, p):  # noqa: D401
        dcdt = np.zeros_like(c)
        # 内部节点中央差分
        dcdt[1:-1] = D_eff * (c[2:] - 2 * c[1:-1] + c[:-2]) / dx**2

        if bc == "neumann":
            # 左端 ∂c/∂x = 0 ⇒ 使用二阶精度虚拟点：c_-1 = c_1
            dcdt[0] = D_eff * (2 * (c[1] - c[0])) / dx**2
            # 右端
            dcdt[-1] = D_eff * (2 * (c[-2] - c[-1])) / dx**2
        else:
            raise NotImplementedError(bc)
        return dcdt

    return rhs, x


def build_spherical_fick_rhs(params: Dict[str, Any]):
    """构建球坐标系下 Fick 扩散方程的 RHS。"""
    R = params["R_particle"]
    N = int(params.get("N_particle", 50))
    D_eff = params["D_eff"]

    r, dr = build_uniform_grid(R, N)
    r_safe = np.maximum(r, 1e-12)

    def rhs(t: float, c: np.ndarray, p: Dict[str, Any]):
        dcdt = np.zeros_like(c)
        j = -D_eff * np.gradient(c, dr)

        # 内部节点
        dcdt[1:-1] = -(1 / r_safe[1:-1] ** 2) * np.gradient(
            r_safe[1:-1] ** 2 * j[1:-1], dr
        )

        # 中心点 r=0: dC/dr = 0
        dcdt[0] = 6 * D_eff * (c[1] - c[0]) / dr**2

        # 表面 r=R: 通量由外部电流决定
        I_app = p.get("I_app", 0.0)  # A
        A_particle = 4 * np.pi * R**2
        j_surface = I_app / A_particle / p["F"]
        dcdt[-1] = -3 * j_surface / R

        return dcdt

    return rhs, r 