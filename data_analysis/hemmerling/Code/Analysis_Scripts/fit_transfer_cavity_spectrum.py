import numpy as np
import matplotlib.pyplot as plt

from constants import *



def get_laser_freq(n, nc, vc, dv):

    return dv + float(n/nc) * vc


def get_FSR(L):

    return c_light/(2*L)

def get_L(FSR):

    return c_light/(2*FSR)

def get_n(nu, FSR):

    return int(np.floor(nu/FSR))


def measure_laser_freq(vc, dv, v_wavemeter, FSR, dFSR):

    L  = get_L(FSR)
    Lp = get_L(FSR + dFSR)
    Lm = get_L(FSR - dFSR)

    nc  = get_n(vc, FSR)
    
    ncp = get_n(vc, FSR + dFSR)
    
    ncm = get_n(vc, FSR - dFSR)

    v_est = FSR * nc


    # estimate range of n for unknown laser

    n_wm = get_n(v_wavemeter, FSR)

    n_delta = 10

    n_arr = np.arange(n_wm - n_delta, n_wm + n_delta)

    v_laser_arr = np.array([get_laser_freq(x, nc, vc, dv) for x in n_arr])

    err_arr = v_laser_arr - v_wavemeter

    v_laser = v_laser_arr[np.where(abs(err_arr) == min(abs(err_arr)))][0]

    plt.plot(n_arr,abs(err_arr))



    print('''
Frequency: {0:.6f} THz
n = {1} +/- ({2}/{3})
L = {4:.2f} mm +/- ({5:.2f}/{6:.2f}) um

Frequency: n * FSR = {7:.6f} THz

Wavemeter laser frequency: {8:.6f} THz
Measured laser frequency:  {9:.6f} THz
          '''.format(
        vc/1e12,
        nc,
        nc - ncp,
        ncm - nc,
        L / 1e-3,
        (L-Lp)/1e-6,
        (Lm-L)/1e-6,
        v_est/1e12,
        v_wavemeter/1e12,
        v_laser/1e12
        ))




    return




vc = c_light/1046e-9

v_real = c_light/780e-9

v_real = 384.348440e12

v_wavemeter = v_real - 23e6


dnu = -40e6

FSR = 1.49e9


dFSR = 20e3


dn = 10


FSR_arr = np.linspace(FSR - dFSR, FSR + dFSR, 5000)

# vc/FSR needs to be integer

nc_est = np.floor(vc/FSR)

dnu_arr = np.linspace(-100e6, 100e6, 100)


nwm_est = np.floor(v_real/FSR)


#plt.plot( dnu_arr, dnu_arr + n/nc_est * vc )
#
print(-40e6 + nwm_est * FSR)
#
#plt.show()



# the FSR for the other wavelength is different but we can get close to the n, using the same
nwm_est = np.floor(v_wavemeter/FSR)


n = np.arange(int(nwm_est - dn), int(nwm_est + dn))

err = np.abs( dnu + n * FSR - v_wavemeter )

errp = np.abs( dnu + n * (FSR + dFSR) - v_wavemeter )
errm = np.abs( dnu + n * (FSR - dFSR) - v_wavemeter )


errf = lambda n, dFSR: np.abs( dnu + n * (FSR - dFSR) - v_wavemeter )

ind = np.where(err == np.min(err))[0][0]


n_true = n[ind]


v_meas = dnu + n_true * FSR


plt.plot(n, err)
plt.plot(n, errp)
plt.plot(n, errm)

for k in np.linspace(-12e3, 12e3, 5000):

    n0 = np.arange(int(nwm_est - dn), int(nwm_est + dn))

    plt.plot(n0, errf(n0, k))

#plt.ylim(0, 50e6)

plt.ylim(1e6, 1e9)

plt.yscale('log')

plt.show()





print('''
Wavemeter Frequency: {0:.6f} THz
True Frequency: {1:.6f} THz
Mode number: {2}

Measured Frequency: {3:.6f} THz
          '''.format(
        v_wavemeter/1e12,
        v_real/1e12,
        n_true,
        v_meas/1e12
        ))





