"""solid_state.interface

固-固界面物理模型。
包括界面电阻、接触损失等。
"""
from __future__ import annotations
from typing import Dict, Any

import numpy as np

__all__ = ["contact_resistance"]


def contact_resistance(sigma_n: float, params: Dict[str, Any]) -> float:
    """计算界面接触电阻。

    一个简化的经验模型，假设电阻随法向压应力的增加而指数下降。
    R_contact = R0 * exp(-beta * sigma_n)

    Parameters
    ----------
    sigma_n : float
        界面上的法向压应力 (Pa)。压应力为正。
    params : Dict[str, Any]
        - R0_contact: 零应力下的本征接触电阻 (Ohm m^2)
        - beta_contact: 电阻-应力耦合系数 (Pa^-1)

    Returns
    -------
    float
        当前应力下的接触电阻 (Ohm m^2)。
    """
    R0 = params.get("R0_contact", 1e-4)
    beta = params.get("beta_contact", 1e-8)

    # 假设只有压应力能改善接触，拉应力（负值）效果忽略或设为上限
    sigma_n_compressive = np.maximum(0, sigma_n)

    return R0 * np.exp(-beta * sigma_n_compressive) 