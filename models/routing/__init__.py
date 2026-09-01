from .kinematic_wave import EdgeParameterNetwork, PureDirectedGNN
from .kinematic_wave_optimized import KinematicWaveGNN
from .muskingum import MuskingumGraphRouter

__all__ = [
    "MuskingumGraphRouter",
    "PureDirectedGNN",
    # Retained only for direct import compatibility with pre-rebuild tests.
    "KinematicWaveGNN",
    "EdgeParameterNetwork",
]
