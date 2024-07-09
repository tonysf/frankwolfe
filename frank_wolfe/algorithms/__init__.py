# __init__.py for the algorithms subpackage

from .base import FrankWolfe
from .away import AwayFrankWolfe
from .boosted import BoostedFrankWolfe
from .mismatch import MismatchFrankWolfe
from .nono import NoNoFrankWolfe
from .sliding import CondGradSliding

__all__ = ['FrankWolfe', 'AwayFrankWolfe', 'BoostedFrankWolfe', 'MismatchFrankWolfe', 'NoNoFrankWolfe', 'CondGradSliding']