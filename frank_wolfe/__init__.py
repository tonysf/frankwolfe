# This is the main __init__.py file for the frankwolfe package

from .core.lmo import create_lmo
from .core.objective import ObjectiveFunction
from .algorithms.away import AwayFrankWolfe
from .algorithms.base import FrankWolfe

__version__ = "0.1.0"

__all__ = ['create_lmo', 'ObjectiveFunction', 'BoostingFrankWolfe', 'MismatchFrankWolfe', 'NoNoFrankWolfe', 'CondGradSliding', 'AwayFrankWolfe', 'FrankWolfe']