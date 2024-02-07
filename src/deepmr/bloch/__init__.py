"""Sub-package containing Bloch simulation routines.

Include MR simulation routines for common sequences based on Extended Phase Graphs [1, 2]. 
Currently provided models include MPRAGE, Multiecho MPRAGE (ME-MPRAGE), Fast Spin Echo (FSE)
and T1-T2 Shuffling and balanced / unbalanced MR Fingerprinting.

References
----------
[1] Malik, S.J., Teixeira, R.P.A.G. and Hajnal, J.V. (2018), 
Extended phase graph formalism for systems with magnetization transfer and exchange. 
Magn. Reson. Med., 80: 767-779. https://doi.org/10.1002/mrm.27040
    
"""

from . import model as _model
from . import ops as _ops
from . import blocks as _blocks

from .model import *  # noqa
from .ops import *  # noqa
from .blocks import *  # noqa

__all__ = []
__all__.extend(_model.__all__)
__all__.extend(_ops.__all__)
__all__.extend(_blocks.__all__)
