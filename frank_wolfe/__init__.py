# This is the main __init__.py file for the frankwolfe package

# Import key classes and functions to make them easily accessible
from .core.lmo import create_lmo
from .core.objective import ObjectiveFunction
from .algorithms.away_step import AwayFrankWolfe
from .algorithms.base import FrankWolfe

# You can also define package-level variables here if needed
__version__ = "0.1.0"

# If you want to control what gets imported with "from frank_wolfe import *"
__all__ = ['create_lmo', 'ObjectiveFunction', 'AwayFrankWolfe', 'FrankWolfe']