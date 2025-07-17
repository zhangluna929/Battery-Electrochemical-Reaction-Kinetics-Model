"""battery_kinetics.cli
命令行工具，安装后可通过 ``python -m battery_kinetics.cli`` 或 ``berkm`` 调用。

目前支持子命令：
- run   运行单个参数文件
- sweep 批量参数扫描（笛卡尔积）
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from joblib import Parallel, delayed

from battery_kinetics.core.ode import run_ivp
from battery_kinetics.io.loader import load_parameters
from battery_kinetics.electrochem.kinetics import butler_volmer

# --------------------------------------------------------------------------------------
# 模型函数集合（后续可注册制）
# --------------------------------------------------------------------------------------

def model_0d_bv(params: Dict[str, Any]):
    """示例 0D Butler-Volmer 模型（沿用 main.py 逻辑）。"""

    def rhs(t: float, y: np.ndarray, p):
        c, = y
        I = p["I_control"]
        i0 = p["i0"]
        eta = (I / (i0 * (p.get("alpha_a", 0.5) + p.get("alpha_c", 0.5)))) * (
            p["R"] * p["T"] / p["F"]
        )
        V_particle = p["V_particle"]
        dcdt = -I / (p["F"] * V_particle)
        return np.array([dcdt])

    y0 = np.array([params["c0"]])
    sol = run_ivp(rhs, y0, (0.0, params["t_end"]), params, max_step=1.0)
    return sol.t, sol.y[0]


MODELS = {"0d_bv": model_0d_bv}

# ------------------------------------------------------------------
# 1D Fick 扩散模型
# ------------------------------------------------------------------

from battery_kinetics.electrochem.transport import build_fick_rhs


def model_1d_fick(params: Dict[str, Any]):
    """1D Fick 扩散示例。

    参数要求：L, N, D_eff, c0, t_end
    """

    rhs, x = build_fick_rhs(params)
    y0 = np.full((len(x),), params["c0"])
    sol = run_ivp(rhs, y0, (0.0, params["t_end"]), params, max_step=params.get("max_step", 0.1))
    return sol.t, sol.y  # 返回 2D 数组 shape (N_nodes, n_times)


MODELS["1d_fick"] = model_1d_fick

# ------------------------------------------------------------------
# 1D Fick + Heat 耦合模型
# ------------------------------------------------------------------

from battery_kinetics.thermal.energy import build_heat_rhs


def model_1d_fick_thermal(params: Dict[str, Any]):
    """耦合浓度扩散 + 温度传导 (简单相互独立热源)。"""

    # 构造两个 RHS，拼接向量
    rhs_c, x = build_fick_rhs(params)
    rhs_T, _ = build_heat_rhs(params)

    N = len(x)

    def rhs(t: float, y: np.ndarray, p):
        c = y[:N]
        T = y[N:]
        dc_dt = rhs_c(t, c, p)
        dT_dt = rhs_T(t, T, p)
        return np.concatenate([dc_dt, dT_dt])

    y0_c = np.full((N,), params["c0"])
    y0_T = np.full((N,), params.get("T0", 298.15))
    y0 = np.concatenate([y0_c, y0_T])

    sol = run_ivp(
        rhs,
        y0,
        (0.0, params["t_end"]),
        params,
        max_step=params.get("max_step", 0.1),
        use_ad=params.get("use_ad", False),
    )
    return sol.t, sol.y.reshape(2, N, -1)  # shape (2, N, n_times)


MODELS["1d_fick_thermal"] = model_1d_fick_thermal

# ------------------------------------------------------------------
# 单颗粒固态物理全耦合模型
# ------------------------------------------------------------------

from battery_kinetics.mechanics.stress import calc_spherical_stress
from battery_kinetics.solid_state.interface import contact_resistance
from battery_kinetics.electrochem.kinetics import butler_volmer_stress_coupled, open_circuit_voltage, exchange_current
from battery_kinetics.electrochem.transport import build_spherical_fick_rhs
from scipy.optimize import root_scalar


def model_particle_solid_state(params: Dict[str, Any]):
    """单颗粒尺度下的电-化-力耦合模型。

    求解变量：球形颗粒内的锂离子浓度分布 c(r, t)。
    模型输出：浓度场、应力场、界面电阻、过电位等随时间的变化。
    """
    rhs_c, r = build_spherical_fick_rhs(params)
    y0 = np.full((len(r),), params["c0"])

    # --- 存储额外的输出结果 ---
    times = []
    voltages = []
    sigma_r_surf = []
    sigma_t_surf = []
    r_contact = []

    def full_rhs(t: float, c: np.ndarray, p: Dict[str, Any]):
        # 1. 计算应力
        sigma_r, sigma_t = calc_spherical_stress(c, r, p)
        sigma_n_surf = sigma_r[-1]

        # 2. 计算浓度/温度依赖的参数
        i0 = exchange_current(c[-1], p["c_max"], p["k0"])
        ocv_surf = open_circuit_voltage(c[-1], p)
        R_contact = contact_resistance(sigma_n_surf, p)

        # 3. 求解 eta_act 以匹配外部电流
        i_app = p["I_app"] / (4 * np.pi * p["R_particle"] ** 2)

        def err_func(eta_act_guess):
            i_calc = butler_volmer_stress_coupled(
                i0, p["alpha_a"], p["alpha_c"], eta_act_guess, sigma_n_surf, p
            )
            return i_calc - i_app

        sol_eta = root_scalar(err_func, bracket=[-2, 2], method="brentq")
        eta_act = sol_eta.root

        # 4. 计算总电压
        V_cell = ocv_surf - eta_act - i_app * R_contact

        # 5. 计算浓度场演化 (RHS)
        p_dynamic = p.copy()
        p_dynamic["I_app"] = p["I_app"] # 确保总电流传递
        dcdt = rhs_c(t, c, p_dynamic)

        # 存储中间结果
        times.append(t)
        sigma_r_surf.append(sigma_r[-1])
        sigma_t_surf.append(sigma_t[-1])
        r_contact.append(R_contact)
        voltages.append(V_cell)

        return dcdt

    sol = run_ivp(full_rhs, y0, (0.0, params["t_end"]), params, max_step=params.get("max_step", 1.0))

    # 将额外结果打包进 sol 对象
    sol.extra_results = {
        "time_extra": np.array(times),
        "sigma_r_surface": np.array(sigma_r_surf),
        "sigma_t_surface": np.array(sigma_t_surf),
        "contact_resistance": np.array(r_contact),
        "voltage": np.array(voltages),
    }
    return sol.t, sol.y


MODELS["particle_solid_state"] = model_particle_solid_state

# --------------------------------------------------------------------------------------
# CLI 实现
# --------------------------------------------------------------------------------------

def _run(args: argparse.Namespace):
    params = load_parameters(args.param_file)
    params.setdefault("F", 96485.3329)
    params.setdefault("R", 8.314462618)

    model_func = MODELS[args.model]
    t, y = model_func(params)

    # 保存或打印
    if args.save:
        out = np.vstack([t, y]).T
        np.savetxt(args.save, out, delimiter=",", header="time,value", comments="")
        print(f"[INFO] 保存结果到 {args.save}")
    else:
        for ti, yi in zip(t, y):
            print(ti, yi)


def _sweep(args: argparse.Namespace):
    base_params = load_parameters(args.param_file)
    sweep_cfg = load_parameters(args.sweep_file)

    keys = list(sweep_cfg.keys())
    values = [sweep_cfg[k] for k in keys]

    combos = list(itertools.product(*values))

    def _run_one(combo):
        params = base_params.copy()
        params.update(dict(zip(keys, combo)))
        combo_tag = "_".join(f"{k}{v}" for k, v in zip(keys, combo))
        print(f"[RUN] {combo_tag}")
        t, y = MODELS[args.model](params)
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            out_path = Path(args.output_dir) / f"{combo_tag}.csv"
            np.savetxt(out_path, np.vstack([t, y]).T, delimiter=",", header="time,value", comments="")
        return combo_tag

    Parallel(n_jobs=args.n_jobs, backend="loky")(_run_one(c) for c in combos)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="berkm", description="Battery Kinetics CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = subparsers.add_parser("run", help="运行单个仿真")
    p_run.add_argument("param_file", type=str, help="参数文件或材料名")
    p_run.add_argument("--model", default="0d_bv", choices=MODELS.keys())
    p_run.add_argument("--save", type=str, help="结果保存到 CSV")
    p_run.set_defaults(func=_run)

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="批量参数扫描")
    p_sweep.add_argument("param_file", type=str, help="基础参数文件")
    p_sweep.add_argument("sweep_file", type=str, help="定义扫描参数的 YAML/JSON")
    p_sweep.add_argument("--model", default="0d_bv", choices=MODELS.keys())
    p_sweep.add_argument("--output_dir", type=str, default="results", help="输出目录")
    p_sweep.add_argument("--n_jobs", type=int, default=1, help="并行进程数 (-1 表示使用全部处理器)")
    p_sweep.set_defaults(func=_sweep)

    return parser


def main(argv: List[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:]) 