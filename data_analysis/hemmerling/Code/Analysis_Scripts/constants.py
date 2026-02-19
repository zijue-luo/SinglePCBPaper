# some physics constants
import numpy as np

c_light  = 299792458
kB       = 1.38064852e-23
h_planck = 6.62607004e-34
hbar     = h_planck/(2*np.pi)

amu = 1.66053906660e-27


# AlCl specific

mass_e = 9.1093837015e-31 # in kg
massAl = 26.98153841 # in amu
massCl_35 = 34.96885269 # in amu
massCl_37 = 36.96590258 # in amu

# reduced masses
mu_AlCl35 = (massAl * massCl_35)/(massAl + massCl_35) # in amu
mu_AlCl37 = (massAl * massCl_37)/(massAl + massCl_37) # in amu


# plot colors

STD_BLUE   = '#1f77b4'
STD_RED    = '#d62728'
STD_ORANGE = '#ff7f0e'



