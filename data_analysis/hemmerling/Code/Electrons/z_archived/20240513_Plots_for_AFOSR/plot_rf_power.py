import numpy as np
import lmfit
from lmfit import Minimizer, Parameters, report_fit

import matplotlib.pyplot as plt

import matplotlib
import pylab


fontsize = 20


font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : fontsize}

#matplotlib.rc('font', **font)


params = {
            'text.usetex'       : True,     # for latex fonts
            'legend.fontsize'   : fontsize,
            'figure.figsize'    : (12, 5),
            'axes.labelsize'    : fontsize,
            'axes.titlesize'    : fontsize,
            'xtick.labelsize'   : fontsize,
            'ytick.labelsize'   : fontsize,
            'font.size'         : fontsize
            }

pylab.rcParams.update(params)



####################################
# Fit Simple Gaussian
####################################

def fcn2min_func(params, x, data, func = None, plot_fit = False, no_of_points = 500):

    p0 = params['p0']
    p1 = params['p1']
    p2 = params['p2']
    p3 = params['p3']

    if plot_fit == False:
        x_fit = x
    else:
        #x_fit = np.linspace(np.min(x), np.max(x), no_of_points)
        x_fit = np.linspace(0, np.max(x), no_of_points)

    model = p0 * x_fit**0.5
    
    if plot_fit == False:
        return model - data
    else:
        return (x_fit, model)



####################################################################

def my_fit(x, y):

    y_min = np.min(y)
    y_max = np.max(y)
    x_mean = np.mean(x)
    x_min = np.min(x)
    x_max = np.max(x)

    params = Parameters()

    params.add('p0', value=1, min=0.0, max=1e3, vary = True)
    params.add('p1', value=1, min=-3.0, max=3.0, vary = True)
    params.add('p2', value=1, min=x_mean-10e9, max=x_mean+10e9, vary = True)
    params.add('p3', value=1, min=100e6, max=(x_max-x_min), vary = True)

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min_func, params, fcn_args=(x, y, None))
    result = minner.minimize()
    
    (xf, yf) = fcn2min_func(result.params, x, y, plot_fit = True)
    
    (xf_res, yf_res) = fcn2min_func(result.params, x, y, plot_fit = True, no_of_points = len(x))
    
    residuals = yf_res - y

    return (result, xf, yf, residuals)


####################################################################

def load_data(filename):

    d = np.genfromtxt(filename, delimiter = ',', skip_header = 1)

    return d


####################################################################

d = load_data('d.csv')

x = d[:, 1]
y = d[:, 2]
yerr = d[:, 3]
xerr = d[:, 4]

x = 10**(x/10.0)

xerr = 0**(0.2/10.0) + 0*xerr

(result, xf, yf, residuals) = my_fit(x, y)


print(result.params)

plt.figure(figsize = (8, 6))

#plt.plot(x, y, 'o')
#plt.errorbar(x, y + 3, yerr = yerr, xerr = xerr, fmt = '.', ls = '', capsize = 6)
plt.errorbar(x, y + 3, yerr = yerr, fmt = '.', ls = '', capsize = 6, markersize = 10)

plt.plot(xf, yf)

plt.xlim(0, 1.1*max(x))
plt.ylim(0, 1.1*max(y))

plt.xlabel('Transmitted RF power (attenuated) (mW)')
plt.ylabel('Radial Trap Frequency (MHz)')

plt.tight_layout()

plt.show()



