"""Collection of trajectories and tools used for non-Cartesian MRI."""
"""
Gradient design utils
=====================

The subpackage utils contains a collection of tools 
used for gradient waveform design. The full list is:
    
Config (utils.config)
    config_cartesian_2D     parse arguments for 2D phase encoding planning.
    config_cartesian_3D     parse arguments for 3D phase encoding planning.
    config_stack            parse arguments for 3D stack-of-trajectory creation.
    config_proj             parse arguments for 3D trajectory projection creation.

Defaults (utils.defaults)
    get_defaults            defaults options for waveform generation.
    get_cartesian_defaults  defaults options for cartesian readout generation.

Density compensation (utils.dcf)
    voronoi                 Voronoi-based sampling density estimation.
    
Designer (utils.designer)
    make_trapezoid          trapezoidal gradient design.
    make_cartesian_gradient Cartesian readout gradient design.
    make_arbitrary_gradient general gradient design.
    compose_gradients       pad and stack different waveforms.
    derivative              finite-difference derivative.
    integration             trapezoid-method integration.
    
Encoding Plans (utils.encodeplans)
    prep_2d_phase_plan      initialize 2D phase encoding plan.
    prep_3d_phase_plan      initialize 3D phase encoding plan.

Mask (utils.mask)
    partial_fourier         initialize Partial Fourier undersampling mask.
    parallel_imaging        initialize Parallel Imaging undersampling mask.
    poisson_disk            initialize Compressed Sensing undersampling mask.
    
Misc (utils.misc)
    traj_complex_to_array   convert complex nx + 1j*ny trajectory to (..., 2).
    traj_array_to_complex   convert (..., 2) trajectory to complex nx + 1j*ny.
    scale_traj              scale trajectory to -0.5 0.5.
    pad                     append zeros to gradient waveform.
    flatten_echoes          flatten multiple echo in the readout axis (for gradient and adc)-
    extract_acs             calculate trajectory indexes, dcf and matrix size for ACS samples.
    
Nyquist (utils.nyquist)
    squared_fov             Nyquist criterion for Cartesian 2D sampling.
    cubic_fov               Nyquist criterion for Cartesian 3D sampling.
    radial_fov              Nyquist criterion for Non-Cartesian 2D sampling.
    cylindrical_fov         Nyquist criterion for stack of 2D Non-Cartesian sampling.
    spherical_fov           Nyquist criterion for Non-Cartesian 3D sampling.
    
Tilt (utils.tilt)
    make_tilt               tilt angles list creation
    broadcast_tilt          expand tilt angles list for 3D Stack and 3D Projection trajectories.
    projection              generate 2D/3D Non-Cartesian from base readout + tilt angles.
    
Timing (utils.timing)
    calculate_timing        calculate Echo Times and readout time map for a given interleave.
    
"""

from . import config as _config
from . import dcf as _dcf
from . import defaults as _defaults
from . import designer as _designer
from . import encodeplans as _encodeplans
from . import expansions as _expansions
from . import mask as _mask
from . import misc as _misc
from . import nyquist as _nyquist
from . import ordering as _ordering
from . import tilt as _tilt
from . import timing as _timing

from .config import *  # noqa
from .dcf import *  # noqa
from .defaults import * # noqa
from .designer import *  # noqa
from .encodeplans import * # noqa
from .expansions import * # noqa
from .mask import *  # noqa
from .misc import *  # noqa
from .nyquist import *  # noqa
from .ordering import *  # noqa
from .tilt import *  # noqa
from .timing import *  # noqa

__all__ = []
__all__.extend(_config.__all__)
__all__.extend(_dcf.__all__)
__all__.extend(_defaults.__all__)
__all__.extend(_designer.__all__)
__all__.extend(_encodeplans.__all__)
__all__.extend(_expansions.__all__)
__all__.extend(_mask.__all__)
__all__.extend(_misc.__all__)
__all__.extend(_nyquist.__all__)
__all__.extend(_ordering.__all__)
__all__.extend(_tilt.__all__)
__all__.extend(_timing.__all__)
