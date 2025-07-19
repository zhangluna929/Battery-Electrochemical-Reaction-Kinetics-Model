"""core.ode
通用 ODE 求解器封装，便于快速替换计算后端。
"""
from __future__ import annotations

from typing import Callable, Dict, Any, Optional
import numpy as np
from scipy.integrate import solve_ivp
import jax
from jax import jacfwd

_JAX_AVAILABLE = True
try:
    import jaxlib
except ImportError:
    _JAX_AVAILABLE = False


def ode_solver(
    rhs: Callable,
    y0: np.ndarray,
    t_span: tuple[float, float],
    params: Dict[str, Any] | None = None,
    *,
    use_ad: bool = False,
    **kwargs,
):
    """A wrapper for scipy.integrate.solve_ivp with optional JAX-based AD."""
    if params is None:
        params = {}

    def _wrap_rhs(t: float, y: np.ndarray):
        return rhs(t, y, params)

    default_opts = {
        "method": "BDF",
        "rtol": 1e-6,
        "atol": 1e-9,
    }
    if use_ad:
        if not _JAX_AVAILABLE:
            raise ImportError("JAX is required for automatic differentiation. Please install with `pip install jax jaxlib`.")

        def jax_rhs(t, y):
            return rhs(t, y, params)
        
        jac_func = jax.jacfwd(jax_rhs, argnums=1)
        default_opts["jac"] = lambda t, y: np.asarray(jac_func(t,y))

    default_opts.update(kwargs)

    sol = solve_ivp(_wrap_rhs, t_span, y0, **default_opts)
    return sol
 