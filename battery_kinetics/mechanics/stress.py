"""mechanics.stress

固态电极中的力学应力计算。
主要关注由于锂离子浓度不均引起的化学-机械应力。
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np

__all__ = ["calc_spherical_stress"]


def calc_spherical_stress(
    c: np.ndarray, r: np.ndarray, params: Dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """计算球形颗粒内部的径向和切向应力。

    该模型基于 Timoshenko 的弹性理论，假设材料各向同性、线弹性。
    应力由浓度梯度引起。

    Parameters
    ----------
    c : np.ndarray
        从 r=0 到 r=R 的锂离子浓度分布。
    r : np.ndarray
        从 r=0 到 r=R 的径向坐标。
    params : Dict[str, Any]
        - E: 杨氏模量 (Pa)
        - nu: 泊松比 (-)
        - omega: 偏摩尔体积 (m^3 mol^-1)
        - c_max: 最大锂离子浓度 (mol m^-3)
        - R_particle: 颗粒半径 (m)

    Returns
    -------
    sigma_r : np.ndarray
        径向应力分布。
    sigma_t : np.ndarray
        切向应力分布。
    """
    E = params["E_modulus"]
    nu = params["poisson_ratio"]
    omega = params["partial_molar_volume"]
    c_avg_initial = params.get("c_avg_initial", np.mean(c))

    # 避免 r=0 处的除零
    r_safe = np.maximum(r, 1e-12)
    dr = r[1] - r[0] if len(r) > 1 else r[0]

    # 平均浓度
    c_avg = (3 / r_safe[-1] ** 3) * np.trapz(c * r_safe**2, r_safe, dx=dr)

    # 积分项
    integral_term = np.zeros_like(r_safe)
    for i in range(1, len(r_safe)):
        integral_term[i] = np.trapz(c[: i + 1] * r_safe[: i + 1] ** 2, r_safe[: i + 1], dx=dr)

    # 应力计算
    term_common = (2 * E * omega) / (3 * (1 - nu))
    sigma_r = term_common * (c_avg - (1 / r_safe**3) * integral_term)
    sigma_t = term_common * (
        (c_avg / 2) + (1 / (2 * r_safe**3)) * integral_term - (c - c_avg_initial)
    )

    # 在球心 r=0 处，径向应力与切向应力相等
    sigma_r[0] = sigma_t[0]

    return sigma_r, sigma_t 