"""
A multi-physics simulation platform for solid-state lithium batteries.
"""

__version__ = "1.0.0"

# Core numerical tools
from .core.ode import run_ivp

# Data I/O
from .io.loader import load_parameters, load_material
from .io.experiment import read_arbin_csv, read_biologic_mpr

# Physics Models
from .electrochem.kinetics import butler_volmer, mhc, exchange_current, open_circuit_voltage
from .electrochem.transport import build_fick_rhs, build_spherical_fick_rhs
from .thermal.energy import build_heat_transport_rhs
from .mechanics.stress import calc_spherical_stress
from .solid_state.interface import contact_resistance

# Analysis
from .analysis.metrics import calc_metrics
from .analysis.plotting import plot_heatmap 