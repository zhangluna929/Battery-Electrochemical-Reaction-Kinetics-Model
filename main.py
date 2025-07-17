"""主执行脚本
示例：读取参数文件，运行简单的 0D Butler-Volmer 单颗粒放电模型。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from battery_kinetics.core.ode import run_ivp
from battery_kinetics.electrochem.kinetics import butler_volmer
from battery_kinetics.io.loader import load_parameters


def rhs(t: float, y: np.ndarray, p):
    """简单 0D 模型：假设过电位 eta 由固定放电电流 I_control 反算得到。"""
    c, = y  # 单一态变量：锂离子浓度 (占位)
    I = p["I_control"]  # A m^-2
    i0 = p["i0"]  # A m^-2
    alpha_a = p.get("alpha_a", 0.5)
    alpha_c = p.get("alpha_c", 0.5)
    # 假设电极表面积=1 m^2，则 I=i
    # 由 BV 求过电位
    # 对非线性方程可牛顿求解，简化：小电流近似
    eta = (I / (i0 * (alpha_a + alpha_c))) * (p["R"] * p["T"] / p["F"])
    # 此处 c 动力学随便写个占位：dc/dt = -I/F/V
    V_particle = p["V_particle"]
    dcdt = -I / (p["F"] * V_particle)
    return np.array([dcdt])


def main():
    parser = argparse.ArgumentParser(description="0D Butler-Volmer 放电示例")
    parser.add_argument("param_file", type=str, help="参数 YAML/JSON 文件路径")
    args = parser.parse_args()

    params = load_parameters(args.param_file)
    # 常数补充
    params.setdefault("F", 96485.3329)
    params.setdefault("R", 8.314462618)

    y0 = np.array([params["c0"]])
    t_span = (0.0, params["t_end"])
    sol = run_ivp(rhs, y0, t_span, params, max_step=1.0)

    # 可视化
    plt.plot(sol.t, sol.y[0], label="c")
    plt.xlabel("Time / s")
    plt.ylabel("Concentration (arb.)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main() 