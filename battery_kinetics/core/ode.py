"""core.ode
通用 ODE 求解器封装，便于快速替换计算后端。
"""
from __future__ import annotations

from typing import Callable, Dict, Any, Optional

import numpy as np
from scipy.integrate import solve_ivp

try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore  # noqa: F401

    _JAX_AVAILABLE = True
except ModuleNotFoundError:
    _JAX_AVAILABLE = False


def run_ivp(
    rhs: Callable[[float, np.ndarray, Dict[str, Any]], np.ndarray],
    y0: np.ndarray,
    t_span: tuple[float, float],
    params: Dict[str, Any] | None = None,
    *,
    use_ad: bool = False,
    jac_sparsity: Optional[np.ndarray] = None,
    **kwargs,
):
    """高阶包装 SciPy ``solve_ivp``，支持稀疏 Jacobian、并行化等拓展。

    Parameters
    ----------
    rhs
        右端函数 ``f(t, y, params)``，返回 dydt。
    y0
        初始状态变量。
    t_span
        (t0, tf) 时间区间。
    params
        传递给 ``rhs`` 的其他模型参数。
    kwargs
        透传给 ``solve_ivp`` 的额外关键字参数，例如 ``method``、``rtol`` 等。
    """
    if params is None:
        params = {}

    def _wrap_rhs(t: float, y: np.ndarray):
        return rhs(t, y, params)

    default_opts = {
        "method": "BDF",  # 刚性方程友好
        "rtol": 1e-6,
        "atol": 1e-9,
    }
    # Jacobian via autodiff if requested
    if use_ad:
        if not _JAX_AVAILABLE:
            raise ImportError("use_ad=True 需要安装 jax (`pip install jax jaxlib`) ")

        def jax_rhs(t, y):
            return rhs(t, y, params)

        jac_func = jax.jacfwd(jax_rhs, argnums=1)

        default_opts["jac"] = lambda t, y: np.asarray(jac_func(t, y))

        if jac_sparsity is not None:
            default_opts["jac_sparsity"] = jac_sparsity

    default_opts.update(kwargs)

    sol = solve_ivp(_wrap_rhs, t_span, y0, **default_opts)
    return sol 