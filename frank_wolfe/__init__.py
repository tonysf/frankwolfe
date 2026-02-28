# This is the main __init__.py file for the frankwolfe package

from .core.lmo import create_lmo
from .core.objective import ObjectiveFunction
from .algorithms.base import FrankWolfe
from .algorithms.away import AwayFrankWolfe
from .algorithms.boosted import BoostedFrankWolfe
from .algorithms.mismatch import MismatchFrankWolfe
from .algorithms.nono import NoNoFrankWolfe
from .algorithms.sliding import CondGradSliding

__version__ = "0.1.0"

__all__ = ['create_lmo', 'ObjectiveFunction', 'FrankWolfe', 'AwayFrankWolfe', 'BoostedFrankWolfe', 'MismatchFrankWolfe', 'NoNoFrankWolfe', 'CondGradSliding']