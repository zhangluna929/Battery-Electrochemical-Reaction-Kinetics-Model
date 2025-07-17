# Battery Electrochemical Reaction Kinetics Model (BERKM)

This project provides a comprehensive, multi-physics simulation platform for solid-state lithium batteries. It goes beyond simple electrochemical models to incorporate tightly coupled chemo-mechanical and thermal effects, making it a powerful tool for academic research and industrial R&D in the field of solid-state batteries. The model is designed with a modular architecture, allowing for easy extension and integration of new physics.

本项目为一个针对固态锂电池的、完整的多物理场耦合仿真平台。它超越了简单的电化学模型，深度集成了化学-力学和热学效应的紧密耦合，使其成为固态电池领域学术研究和工业研发的强大工具。该模型采用模块化架构设计，便于未来扩展和集成新的物理模型。

The platform is capable of simulating the time-dependent evolution of various internal state variables within a single active material particle under electrochemical cycling. This includes the lithium concentration profile, stress distribution (radial and tangential), temperature field, and dynamic changes in solid-solid interface properties.

该平台能够模拟在电化学循环过程中，单个活性材料颗粒内部各种状态变量随时间的演化。这包括锂离子浓度分布、应力场（径向与切向）、温度场，以及固-固界面属性的动态变化。

## Core Features

-   **Multi-Physics Coupling Framework**: A fully coupled framework that solves for electrochemical reactions, solid-state diffusion, mechanical stress, and thermal effects simultaneously. The core of the model captures the interplay where lithiation-induced stress affects reaction kinetics, and current flow generates heat that in turn influences transport properties.
-   **Detailed Solid-State Mechanics**: Implements a rigorous model for calculating chemo-mechanical stress within a spherical particle based on the theory of elasticity. It simulates how non-uniform lithium concentration leads to internal stress, which is a critical factor for battery degradation and performance.
-   **Dynamic Interfacial Physics**: The model includes a dynamic solid-solid interface, where contact resistance is not a fixed parameter but a variable that depends on the mechanical pressure at the interface, providing a more realistic simulation of the interfacial behavior in solid-state batteries.
-   **Advanced Electrochemical Kinetics**: Supports multiple kinetic models, including the standard Butler-Volmer equation and its stress-coupled variant, as well as the Marcus-Hush-Chidsey (MHC) model. It also accounts for concentration-dependent exchange current density.
-   **Numerical Engine**: Built upon a robust numerical backend using `scipy.integrate.solve_ivp` with support for stiff ODE systems (BDF method). It incorporates a non-linear algebraic solver (`scipy.optimize.root_scalar`) to self-consistently determine the activation overpotential required to match a given applied current, thus enabling realistic voltage predictions. Optional support for JAX-based automatic differentiation is included to accelerate Jacobian matrix computation for large-scale systems.
-   **Modular and Extensible Architecture**: The code is organized into distinct modules for electrochemistry, mechanics, thermal physics, data I/O, and numerical core, facilitating future development and customization.

-   **多物理场耦合框架**：一个完全耦合的框架，可同时求解电化学反应、固态扩散、力学应力及热效应。模型的核心在于捕捉锂化应力对反应动力学的影响，以及电流产生的热量反过来影响传输特性的相互作用。
-   **精细的固态力学模型**：基于弹性理论，实现了严格的球形颗粒内化学-力学应力计算模型。它模拟了非均匀锂浓度如何导致内部应力，而这正是电池衰退与性能的关键影响因素。
-   **动态界面物理**：模型包含一个动态的固-固界面，其接触电阻不再是固定参数，而是依赖于界面处机械压力的变量，从而为固态电池的界面行为提供了更真实的模拟。
-   **高级电化学动力学**：支持多种动力学模型，包括标准的 Butler-Volmer 方程及其应力耦合变体，以及 Marcus-Hush-Chidsey (MHC) 模型。同时，模型也考虑了浓度依赖的交换电流密度。
-   **数值引擎**：构建于一个稳健的数值后端之上，使用 `scipy.integrate.solve_ivp` 并支持刚性常微分方程组（BDF方法）。平台集成了一个非线性代数求解器 (`scipy.optimize.root_scalar`)，用以自洽地求解在给定外加电流下所需的活化过电位，从而实现真实的电压预测。同时，可选支持基于 JAX 的自动微分，以加速大规模系统的雅可比矩阵计算。
-   **模块化与可扩展架构**：代码被组织为清晰的模块，分别对应电化学、力学、热物理、数据IO和数值核心，极大地便利了未来的二次开发与定制。

## Project Structure

The repository is organized as follows:

```
Battery-Electrochemical-Reaction-Kinetics-Model-main/
├── battery_kinetics/
│   ├── analysis/         # Post-processing: metrics, plotting
│   ├── core/             # Core numerical solvers (ODE, sparsity)
│   ├── data/             # Material properties database (YAML files)
│   │   ├── electrodes/
│   │   ├── electrolytes/
│   │   └── materials/
│   ├── electrochem/      # Electrochemistry models (kinetics, transport)
│   ├── io/               # Data loaders (parameters, experimental data)
│   ├── mechanics/        # Solid mechanics models (stress)
│   ├── solid_state/      # Solid-state specific physics (interfaces)
│   ├── thermal/          # Thermal models (heat transport)
│   ├── __init__.py
│   └── cli.py            # Command-line interface
├── main.py               # Deprecated, use CLI
└── README.md
```

## Getting Started

### Prerequisites

The model relies on several scientific computing libraries in Python. It is recommended to set up a dedicated virtual environment.

```bash
pip install numpy scipy matplotlib pyyaml joblib
# For .mpr file support
pip install pympr
# For optional performance boost via automatic differentiation
pip install jax jaxlib
```

### Running a Simulation

All simulations are controlled via the command-line interface (`cli.py`) and parameter files (`.yml`).

1.  **Prepare a Parameter File**: Create a `params.yml` file to define all physical, material, and operational parameters for the simulation. An example for the fully coupled solid-state particle model is shown below:

    ```yaml
    # params_solid.yml

    # Physical Constants
    F: 96485.3329
    R: 8.314462618
    T: 298.15

    # Electrochemical Parameters
    U0_ref: 3.4      # V
    I_app: 1.0e-5    # A, Applied total current
    k0: 1.0e-9       # Intrinsic reaction rate constant
    c_max: 22800     # mol m^-3
    c0: 1000         # Initial concentration
    alpha_a: 0.5
    alpha_c: 0.5

    # Particle Geometry & Diffusion
    R_particle: 1.0e-6 # m
    N_particle: 20
    D_eff: 1.0e-15

    # Mechanical Properties
    E_modulus: 15.0e9        # Pa, Young's Modulus
    poisson_ratio: 0.3
    partial_molar_volume: 3.4e-6 # m^3 mol^-1

    # Interfacial Properties
    R0_contact: 1.0e-3
    beta_contact: 1.0e-8

    # Simulation Control
    t_end: 3600
    max_step: 10.0
    ```

2.  **Execute the Simulation**: Run the desired model from the command line. For the detailed solid-state particle model, use:

    ```bash
    python -m battery_kinetics.cli run params_solid.yml --model particle_solid_state --save results.npz
    ```

    This command will run the simulation and save all results (concentration profiles, stress, voltage, etc.) into a single compressed NumPy file `results.npz`.

### Available Models

The CLI supports several models via the `--model` flag:

-   `0d_bv`: A simple zero-dimensional Butler-Volmer model.
-   `1d_fick`: 1D Fickian diffusion in Cartesian coordinates.
-   `1d_fick_thermal`: 1D diffusion coupled with heat transport.
-   `particle_solid_state`: The flagship fully coupled chemo-thermo-mechanical model for a single spherical particle.

## Author

-   **lunazhang** 