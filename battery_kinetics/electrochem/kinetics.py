"""electrochem.kinetics
基础电化学动力学模型。
"""
from __future__ import annotations

import math
from typing import Union, Dict, Any

import numpy as np

F = 96485.3329  # C mol^-1
R = 8.314462618  # J mol^-1 K^-1


def butler_volmer(
    i0: Union[float, np.ndarray],
    alpha_a: float,
    alpha_c: float,
    eta: Union[float, np.ndarray],
    T: float = 298.15,
) -> Union[float, np.ndarray]:
    """Butler-Volmer 方程。

    Parameters
    ----------
    i0
        交换电流密度 (A m^-2)
    alpha_a
        阳极电荷迁移系数。
    alpha_c
        阴极电荷迁移系数。
    eta
        过电位 (V)。
    T
        绝对温度 (K)。
    """
    term1 = math.exp((alpha_a * F * eta) / (R * T))
    term2 = math.exp(-(alpha_c * F * eta) / (R * T))
    return i0 * (term1 - term2)


def butler_volmer_stress_coupled(
    i0: float,
    alpha_a: float,
    alpha_c: float,
    eta_act: float,
    sigma_n: float,
    params: Dict[str, Any],
    T: float = 298.15,
) -> float:
    """应力耦合的 Butler-Volmer 方程。

    总过电位 η_total = η_act + η_mech
    力学过电位 η_mech = - (1-2ν)σ_nΩ / F
    """
    omega = params["partial_molar_volume"]
    nu = params["poisson_ratio"]

    eta_mech = -((1 - 2 * nu) * sigma_n * omega) / F
    eta_total = eta_act + eta_mech
    return butler_volmer(i0, alpha_a, alpha_c, eta_total, T)


# -----------------------------------------------------------------------------
# Marcus-Hush-Chidsey (MHC) 电子转移动力学
# -----------------------------------------------------------------------------


def mhc(i0: Union[float, np.ndarray], lam: float, eta: Union[float, np.ndarray], T: float = 298.15):
    """Marcus–Hush–Chidsey 方程 (简化指数积分近似)。

    Parameters
    ----------
    i0 : float | ndarray
        速率常数对应的交换电流密度 (A m^-2)
    lam : float
        Reorganization energy λ (J mol^-1)
    eta : float | ndarray
        过电位 (V)
    T : float, 默认 298.15 K
    """
    # 转成 J mol^-1
    e = F * eta
    term = e + lam
    kf = np.exp(-((lam - e) ** 2) / (4 * lam * R * T))
    kb = np.exp(-((lam + e) ** 2) / (4 * lam * R * T))
    return i0 * (kf - kb)


def exchange_current(c: Union[float, np.ndarray], c_max: float, k0: float) -> Union[float, np.ndarray]:
    """浓度依赖交换电流：i0 = k0 * c^{1/2} (c_max - c)^{1/2}"""
    return k0 * np.sqrt(c) * np.sqrt(np.maximum(c_max - c, 0))


def open_circuit_voltage(
    c: Union[float, np.ndarray], params: Dict[str, Any]
) -> Union[float, np.ndarray]:
    """计算开路电压 (OCV)。

    使用一个简化的 Nernstian-like 关系：
    OCV = U0_ref + (RT/nF) * ln((c_max - c_surf) / c_surf)

    Parameters
    ----------
    c : 锂离子浓度 (mol m^-3)
    params : 参数字典，需要 "U0_ref", "c_max"
    """
    U0_ref = params.get("U0_ref", 3.4)  # 参考电位, V
    c_max = params["c_max"]
    T = params.get("T", 298.15)
    # 避免除零和 log(0)
    c_safe = np.clip(c, 1e-9, c_max - 1e-9)
    return U0_ref + (R * T / F) * np.log((c_max - c_safe) / c_safe) 