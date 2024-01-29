"""
"""
# read version from installed package
from importlib.metadata import version

__version__ = version("deepmr")

# from . import bloch
from . import io

# from . import optim
# from . import prox
from . import signal as _signal
from .testdata import testdata

from .signal import * # noqa

__all__ = [testdata]
__all__.extend(_signal.__all__)
