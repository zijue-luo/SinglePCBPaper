import numpy as np
import lmfit
from lmfit import Minimizer, Parameters, report_fit


###############################################
# Fit Multiple Gaussians for the Q Transitions
###############################################

def fcn2min_multi(params, x, data, func = None, plot_fit = False, no_of_points = 500):

    y_offset = params['p0']
    x0 = params['p2']
    w = params['p3']

    if plot_fit == False:
        x_fit = x
    else:
        x_fit = np.linspace(np.min(x), np.max(x), no_of_points)

    model = y_offset

    for k in range(2):
        model += params['a35'] * params['p_a' + str(k)] * np.exp( -(x_fit - x0 - k*params['shift_' + str(k)])**2/(2.0*w**2) )
        model += params['a37'] * params['p_a' + str(k)] * np.exp( -(x_fit - x0 - params['isotope_shift'] - k*params['shift_' + str(k)])**2/(2.0*w**2) )
    
    if plot_fit == False:
        return model - data
    else:
        return (x_fit, model)


def fit_multi_gauss(x, y):

    y_min  = np.min(y)
    y_max  = np.max(y)
    x_mean = np.mean(x)
    x_min  = np.min(x)
    x_max  = np.max(x)

    params = Parameters()

    params.add('p0', value=y_min, min=-1.0, max=1.0, vary = True)
    params.add('p2', value=x_min+1.75e9, min=x_mean-10e9, max=x_mean+10e9, vary = True)
    params.add('p3', value=(x_max-x_min)/50.0, min=10e6, max=(x_max-x_min), vary = True)
    
    params.add('a35', value=0.76, min = 0.0, max = 1.0, vary = True)
    params.add('a37', value=0.24, min = 0.0, max = 1.0, vary = True)
    
    params.add('isotope_shift', value=(7.2e9 - 1.0e9), min=100e6, max=(x_max-x_min), vary = True)
    
    for k in range(2):
        params.add('p_a' + str(k), value=0.006, min=0.0, max=0.1, vary = True)
        params.add('shift_' + str(k), value=750e6, min=0e6, max=(x_max-x_min), vary = True)

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min_multi, params, fcn_args=(x, y, None))
    result = minner.minimize()
    
    (xf, yf) = fcn2min_multi(result.params, x, y, plot_fit = True)
    
    (xf_res, yf_res) = fcn2min_multi(result.params, x, y, plot_fit = True, no_of_points = len(x))
    
    residuals = yf_res - y

    return (result, xf, yf, residuals)


####################################
# Fit Simple Gaussian
####################################

def fcn2min_gauss(params, x, data, func = None, plot_fit = False, no_of_points = 500):

    y_offset = params['p0']
    a = params['p1']
    x0 = params['p2']
    w = params['p3']

    if plot_fit == False:
        x_fit = x
    else:
        x_fit = np.linspace(np.min(x), np.max(x), no_of_points)

    model = y_offset + a * np.exp( -(x_fit - x0)**2/(2.0*w**2) )
    
    if plot_fit == False:
        return model - data
    else:
        return (x_fit, model)


def fit_gauss(x, y):

    y_min = np.min(y)
    y_max = np.max(y)
    x_mean = np.mean(x)
    x_min = np.min(x)
    x_max = np.max(x)

    params = Parameters()

    params.add('p0', value=y_min, min=-1.0, max=1.0, vary = True)
    params.add('p1', value=(y_max - y_min), min=0.0, max=3.0, vary = True)
    params.add('p2', value=x_mean, min=x_mean-10e9, max=x_mean+10e9, vary = True)
    params.add('p3', value=(x_max-x_min)/10.0, min=100e6, max=(x_max-x_min), vary = True)

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min_gauss, params, fcn_args=(x, y, None))
    result = minner.minimize()
    
    (xf, yf) = fcn2min_gauss(result.params, x, y, plot_fit = True)
    
    (xf_res, yf_res) = fcn2min_gauss(result.params, x, y, plot_fit = True, no_of_points = len(x))
    
    residuals = yf_res - y

    return (result, xf, yf, residuals)


####################################
# Fit Simple Gaussian with Slope
####################################

def fcn2min_gauss_with_slope(params, x, data, func = None, plot_fit = False, no_of_points = 500):

    y_offset = params['p0']
    a        = params['p1']
    x0       = params['p2']
    w        = params['p3']
    b        = params['p4']

    if plot_fit == False:
        x_fit = x
    else:
        x_fit = np.linspace(np.min(x), np.max(x), no_of_points)

    #model = y_offset + b * x_fit + a * np.exp( -(x_fit - x0)**2/(2.0*w**2) )
    model = y_offset + b * x_fit + a * w**2/((x_fit - x0)**2 + w**2)
    
    if plot_fit == False:
        return model - data
    else:
        return (x_fit, model)



def fit_gauss_with_slope(x, y):

    y_min  = np.min(y)
    y_max  = np.max(y)
    x_mean = np.mean(x)
    x_min  = np.min(x)
    x_max  = np.max(x)

    params = Parameters()

    params.add('p0', value=y_min, min=y_min - 10, max=y_min + 10, vary = True)
    params.add('p1', value=(y_max - y_min), min=3.0, max=15.0, vary = True)
    params.add('p2', value=x_mean, min=x_mean-30, max=x_mean+30, vary = True)
    params.add('p3', value=(x_max-x_min)/10.0, min=1, max=(x_max-x_min), vary = True)
    params.add('p4', value=-0.001, min=-10, max=10, vary = True)

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min_gauss_with_slope, params, fcn_args=(x, y, None))
    result = minner.minimize()
    
    (xf, yf) = fcn2min_gauss_with_slope(result.params, x, y, plot_fit = True)
    
    (xf_res, yf_res) = fcn2min_gauss_with_slope(result.params, x, y, plot_fit = True, no_of_points = len(x))
    
    residuals = yf_res - y

    return (result, xf, yf, residuals)




####################################
# Fit 2D Line
####################################

def fcn2min_2D_line(params, x, data, func = None, plot_fit = False, no_of_points = 500):

    y0        = params['p0']
    a         = params['p1']
    x0        = params['p2']

    if plot_fit == False:
        x_fit = x
    else:
        x_fit = np.linspace(np.min(x), np.max(x), no_of_points)

    model = y0 + a * (x_fit - x0)
    
    if plot_fit == False:
        return model - data
    else:
        return (x_fit, model)



def fit_2D_line(x, y, vary = False):

    y_min  = np.min(y)
    y_max  = np.max(y)
    x_mean = np.mean(x)
    x_min  = np.min(x)
    x_max  = np.max(x)

    params = Parameters()

    #params.add('p0', value=y_min,           min=y_min - 10, max=y_min + 10, vary = vary)
    #params.add('p1', value=(y_max - y_min), min=0.1, max=150.0, vary = vary)

    params.add('p0', value=139,   min=y_min, max=y_max, vary = vary)
    params.add('p1', value=1.075,   min=0.1, max=3.0, vary = vary)
    params.add('p2', value=x_min, min=x_min, max=x_max, vary = False)


    # do fit, here with leastsq model
    minner = Minimizer(fcn2min_2D_line, params, fcn_args=(x, y, None))
    result = minner.minimize()
    
    (xf, yf) = fcn2min_2D_line(result.params, x, y, plot_fit = True)
    
    (xf_res, yf_res) = fcn2min_2D_line(result.params, x, y, plot_fit = True, no_of_points = len(x))
    
    residuals = yf_res - y

    return (result, xf, yf, residuals)


