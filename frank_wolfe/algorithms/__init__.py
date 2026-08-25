# __init__.py for the algorithms subpackage

from .base import FrankWolfe
from .away import AwayFrankWolfe
from .boosted import BoostedFrankWolfe
from .mismatch import MismatchFrankWolfe
from .frames import (
    AdaptiveFrames,
    AdaptiveFramesFrankWolfe,
    Frames,
    FramesFrankWolfe,
    StochasticFrames,
    StochasticFramesFrankWolfe,
)
from .sliding import CondGradSliding

__all__ = [
    "FrankWolfe",
    "AwayFrankWolfe",
    "BoostedFrankWolfe",
    "MismatchFrankWolfe",
    "AdaptiveFrames",
    "AdaptiveFramesFrankWolfe",
    "Frames",
    "FramesFrankWolfe",
    "StochasticFrames",
    "StochasticFramesFrankWolfe",
    "CondGradSliding",
]
