# read version from installed package
from importlib.metadata import version

__version__ = version("deepmr")

# from . import bloch
from . import io

# from . import optim
# from . import prox

from .testdata import testdata
