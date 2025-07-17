"""solid_state.interface

固-固界面物理模型。
包括界面电阻、接触损失等。
"""
from __future__ import annotations
from typing import Dict, Any

import numpy as np

__all__ = ["contact_resistance"]


def contact_resistance(sigma_n: float, params: Dict[str, Any]) -> float:
    """
    Calculates contact resistance based on normal stress at the interface.
    Uses a simplified model: R_contact = R0 * exp(-beta * sigma_n)
    """
    R0 = params.get("R0_contact", 1e-4)
    beta = params.get("beta_contact", 1e-8)

    # 假设只有压应力能改善接触，拉应力（负值）效果忽略或设为上限
    sigma_n_compressive = np.maximum(0, sigma_n)

    return R0 * np.exp(-beta * sigma_n_compressive) 