"""mechanics.stress

固态电极中的力学应力计算。
主要关注由于锂离子浓度不均引起的化学-机械应力。
"""
from __future__ import annotations

from typing import Dict, Any
import numpy as np


def calc_spherical_stress(
    c: np.ndarray, r: np.ndarray, params: Dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates radial and tangential stress in a spherical particle."""
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