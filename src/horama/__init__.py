"""Feature-visualization utilities used by TEXTER."""

from .feature_accentuation import accentuation
from .fourier_fv import fourier
from .losses import dot_cossim
from .maco_fv import maco
from .plots import plot_maco

__version__ = "0.3.0"

__all__ = [
    "accentuation",
    "dot_cossim",
    "fourier",
    "maco",
    "plot_maco",
]
