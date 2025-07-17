"""battery_kinetics.cli
命令行工具，安装后可通过 ``python -m battery_kinetics.cli`` 或 ``berkm`` 调用。

目前支持子命令：
- run   运行单个参数文件
- sweep 批量参数扫描（笛卡尔积）
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any, Dict, List
from joblib import Parallel, delayed

import numpy as np
from scipy.optimize import root_scalar

from battery_kinetics.models import (
    build_fick_rhs,
    build_heat_rhs,
    build_spherical_fick_rhs,
    calc_spherical_stress,
    exchange_current,
    contact_resistance,
    run_ivp,
    load_parameters,
)


# --------------------------------------------------------------------------------------
# 模型函数集合（后续可注册制）
# --------------------------------------------------------------------------------------

def _model_0d_bv(params: Dict[str, Any]):
    """A simple 0D Butler-Volmer model."""

    def rhs(t: float, y: np.ndarray, p):
        eta_act = y[0]
        i_app = p["I_app"] / (4 * np.pi * p["R_particle"] ** 2)
        return np.array([i_app - exchange_current(eta_act, p["c_max"], p["k0"])])

    y0 = np.array([open_circuit_voltage(params["c0"], params)])
    sol = run_ivp(rhs, y0, (0.0, params["t_end"]), params, max_step=params.get("max_step", 0.1))
    return sol.t, sol.y[0]


# ------------------------------------------------------------------
# 1D Fick 扩散模型
# ------------------------------------------------------------------


def _model_1d_fick(params: Dict[str, Any]):
    """1D Fickian diffusion model."""

    rhs, x = build_fick_rhs(params)
    y0 = np.full((len(x),), params["c0"])
    sol = run_ivp(rhs, y0, (0.0, params["t_end"]), params, max_step=params.get("max_step", 0.1))
    return sol.t, sol.y  # Returns 2D array shape (N_nodes, n_times)


# ------------------------------------------------------------------
# 1D Fick + Heat 耦合模型
# ------------------------------------------------------------------


def _model_1d_fick_thermal(params: Dict[str, Any]):
    """Coupled 1D Fickian diffusion and heat transport."""
    # 构造两个 RHS，拼接向量
    rhs_c, x = build_fick_rhs(params)
    rhs_T, _ = build_heat_rhs(params)
    N = len(x)

    def rhs(t: float, y: np.ndarray, p):
        c, T = y[:N], y[N:]
        # This is a simple one-way coupling for demonstration.
        # A full implementation would have T-dependent D_eff and c-dependent Q_gen.
        dc_dt = rhs_c(t, c, p)
        dT_dt = rhs_T(t, T, p)
        return np.concatenate([dc_dt, dT_dt])

    y0_c = np.full((N,), params["c0"])
    y0_T = np.full((N,), params["T0"])
    sol = run_ivp(
        rhs,
        np.concatenate([y0_c, y0_T]),
        (0.0, params["t_end"]),
        params,
        max_step=params.get("max_step", 0.1),
        use_ad=params.get("use_ad", False),
    )
    return sol.t, sol.y.reshape(2, N, -1)  # shape (2, N, n_times)


# ------------------------------------------------------------------
# 单颗粒固态物理全耦合模型
# ------------------------------------------------------------------


def _model_particle_solid_state(params: Dict[str, Any]):
    """Fully coupled chemo-mechanical model for a single spherical particle."""
    # 求解变量：球形颗粒内的锂离子浓度分布 c(r, t)。
    # 模型输出：浓度场、应力场、界面电阻、过电位等随时间的变化。
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
            p_dynamic = p.copy()
            p_dynamic["I_app"] = p["I_app"]
            eta_act = eta_act_guess
            V_cell = ocv_surf - eta_act - i_app * R_contact
            return i_app - exchange_current(eta_act, p_dynamic["c_max"], p_dynamic["k0"])

        sol_eta = root_scalar(err_func, bracket=[-2, 2], method="brentq")
        eta_act = sol_eta.root

        # 4. 计算总电压
        V_cell = ocv_surf - eta_act - i_app * R_contact

        # 5. 计算浓度场演化 (RHS)
        # 需要把电流作为参数传入 rhs_c
        p_dynamic = p.copy()
        p_dynamic["I_app"] = p["I_app"]
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


MODELS = {
    "0d_bv": _model_0d_bv,
    "1d_fick": _model_1d_fick,
    "1d_fick_thermal": _model_1d_fick_thermal,
    "particle_solid_state": _model_particle_solid_state,
}

# --------------------------------------------------------------------------------------
# CLI 实现
# --------------------------------------------------------------------------------------


def _run(args: argparse.Namespace):
    params = load_parameters(args.param_file)
    model_func = MODELS[args.model]
    t, y = model_func(params)

    # 保存或打印
    if args.save:
        out = np.vstack([t, y]).T
        np.savetxt(args.save, out, delimiter=",", header="time,value", comments="")
        print(f"Results saved to {args.save}")
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
        print(f"Running sweep: {combo_tag}")
        t, y = MODELS[args.model](params)
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            out_path = Path(args.output_dir) / f"{combo_tag}.csv"
            np.savetxt(out_path, np.vstack([t, y]).T, delimiter=",", header="time,value", comments="")
        return combo_tag

    Parallel(n_jobs=args.n_jobs, backend="loky")(_run_one(c) for c in combos)


def _plot(args: argparse.Namespace):
    """Plots data from a saved .npz file."""
    data = np.load(args.npz_file)
    field_data = data[args.field]
    # Assuming plot_heatmap is defined elsewhere or will be added.
    # For now, we'll just print the data.
    print(f"Plotting field: {args.field}")
    print(field_data)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="berkm", description="Battery Kinetics CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_run = subparsers.add_parser("run", help="Run a single simulation.")
    p_run.add_argument("param_file", type=str, help="Path to the parameter file or a material name.")
    p_run.add_argument(
        "--model",
        default="0d_bv",
        choices=MODELS.keys(),
        help="The simulation model to use.",
    )
    p_run.add_argument("--save", type=str, help="Save output to a CSV file.")
    p_run.set_defaults(func=_run)

    p_sweep = subparsers.add_parser("sweep", help="Run a parameter sweep.")
    p_sweep.add_argument("param_file", type=str, help="Base parameter file.")
    p_sweep.add_argument("sweep_file", type=str, help="File defining sweep variables.")
    p_sweep.add_argument(
        "--model",
        default="0d_bv",
        choices=MODELS.keys(),
        help="The simulation model to use.",
    )
    p_sweep.add_argument("--output_dir", type=str, default="results", help="Output directory for results.")
    p_sweep.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs (-1 for all cores).")
    p_sweep.set_defaults(func=_sweep)

    p_plot = subparsers.add_parser("plot", help="Plot results from a saved file.")
    p_plot.add_argument("npz_file", type=str, help="Path to the .npz result file.")
    p_plot.add_argument("--field", type=str, required=True, help="Field to plot (e.g., 'c', 'T', 'voltage').")
    p_plot.add_argument("--title", type=str, default="", help="Title for the plot.")
    p_plot.set_defaults(func=_plot)

    return parser


def main(argv: List[str] | None = None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:]) 