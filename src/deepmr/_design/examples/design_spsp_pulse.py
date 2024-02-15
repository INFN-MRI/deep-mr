import numpy as np

from deepmr import design
from deepmr.design.pulses._subroutines._spsp.ss_globals import ss_globals
from deepmr.design.pulses._subroutines._spsp.ss_design import ss_design

# Water/fat chemical shifts
#
df = 0.5e-6 # Conservative shim requirement
water = 4.7e-6			
fat2 = 1.3e-6
fat1 = 0.9e-6

# Convert to frequency
#
B0 = 70000 # G
gamma_h = 4258
fspec = B0 * (np.asarray([(fat1-df), (fat2+df), (water-df), (water+df)]) - water) * gamma_h

water_ctr = (fspec[2] + fspec[3]) / 2
fat_ctr = (fspec[0] + fspec[1]) / 2

# Set up pulse parameters
#
ang = np.pi / 6 # flip angle [Rad]
z_thk = 18 # cm
z_tb = 4 

# Set up spectral/spatial specifications
#
a = np.asarray([0.0, 1.0])
d = np.asarray([0.02, 0.005])
ptype = 'ex'
z_ftype='ls' # Use this to get rid of "Conolly Wings" 
z_d1 = 0.01
z_d2 = 0.01

s_ftype = 'min'	# min-phase spectral 
ss_type='Flyback Half'

# options
sg = ss_globals()
sg.SS_MAX_DURATION = 16e-3
sg.SS_NUM_LOBE_ITERS = 5
sg.SS_VERSE_FRAC = 0.9

#%%
print(z_thk)
print(z_tb)
print(np.asarray([z_d1, z_d2]))
print(fspec)
print(a * ang)
print(d)
print(ptype)
print(z_ftype)
print(s_ftype)
print(ss_type)

g, rf, fs = ss_design(z_thk, z_tb, np.asarray([z_d1, z_d2]), fspec, a * ang, d, ptype, z_ftype, s_ftype, ss_type, sg=sg, verbose=False)

