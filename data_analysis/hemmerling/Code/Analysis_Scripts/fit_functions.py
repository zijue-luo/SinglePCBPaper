import numpy as np
import lmfit
import copy

from lmfit import Minimizer, Parameters, report_fit




##########################################################################
# Fit Scan Functions
##########################################################################

def get_fit_parameters(d):

    # extracts fit parameters and their stderr
   
    fit_pars = {}

    for k in d.keys():

        fit_pars[k] = [d[k].value, d[k].stderr]

    return fit_pars


###############################################

def run_fit(x, y, fcn2min_type, params, param_counter, func_opt):
    
    # note that the param_counter is an index, but the function takes the number of params

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min_type, params, fcn_args=(x, y, param_counter + 1, func_opt))
    result = minner.minimize()
    
    (xf, yf)         = fcn2min_type(result.params, x, y, param_counter + 1, func_opt, plot_fit = True)
    
    (xf_res, yf_res) = fcn2min_type(result.params, x, y, param_counter + 1, func_opt, plot_fit = True, no_of_points = len(x))
    
    residuals = yf_res - y

    return (result, xf, yf, residuals)



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



####################################################
# Fit Rb-87 Saturated Absorption Spectroscopy
####################################################

def fcn2min_Rb87_sas(params, x, data, no_of_peaks, func_opt = 'lorentzian', plot_fit = False, no_of_points = 500):

    offset = params['offset']

    if plot_fit == False:
        x_fit = x
    else:
        x_fit = np.linspace(np.min(x), np.max(x), no_of_points)

    model = offset

    A   = params['ampl_d']
    w   = params['width_d']
    
    cnt0 = params['cnt_d_0']
    cnt1 = params['cnt_d_1']
        
    # Doppler peak

    if func_opt == 'lorentzian':
        model += A * ( w**2/((x_fit - cnt0)**2 + w**2) +  w**2/((x_fit - cnt1)**2 + w**2) )
    elif func_opt == 'gaussian':
        model += A * ( np.exp( -(x_fit - cnt0)**2/(2.0 * w**2) ) + np.exp( -(x_fit - cnt1)**2/(2.0 * w**2) ) )    
    
    # add Lamb dips
   
    A_l_0 = params['ampl_l_0']
    w_l_0 = params['width_l_0']

    w_l_1 = 2.0 * w_l_0
    w_l_2 = 2.0 * w_l_0
    
    cnt_l_0 = params['cnt_l_0']
    
    # main peak
    model += A_l_0 * w_l_0**2/((x_fit - cnt_l_0)**2 + w_l_0**2)
    
    # right peak
    model += 0.2 * A_l_0 * w_l_1**2/((x_fit - (cnt_l_0 + 133.0))**2 + w_l_1**2)

    # left peak
    model += 0.5 * A_l_0 * w_l_2**2/((x_fit - (cnt_l_0 - 79.0))**2 + w_l_2**2)


    

    if plot_fit == False:
        return model - data
    else:
        return (x_fit, model)


def fit_Rb87_sas(x, y, cnt_guesses = [[0], [0]], width_guess = 1, func_opt = 'gaussian', vary = True):

    fcn2min_type = fcn2min_Rb87_sas

    y_min  = np.min(y)
    y_max  = np.max(y)
    y_mean = np.mean(y)

    x_mean = np.mean(x)
    x_min  = np.min(x)
    x_max  = np.max(x)

    # guess offset and amplitude of Doppler peak, assuming Gaussian

    x_center = np.mean(cnt_guesses[0])

    # take first point of data set
    x1 = x[0]
    y1 = y[0]
    w = width_guess

    hlp = np.exp(+(x1 - x_center)**2/(2*w**2)) - 1

    offset_guess = y1 + (y1 - y_min)/hlp

    amplitude_guess = 1.025 * (y_min - offset_guess) / len(cnt_guesses[0])


    # Lamb dip guess

    delta = 30

    ind = np.where( (x > -600 - delta) & (x < -600 + delta) )

    ind_max = np.where( y[ind] == np.max(y[ind]) )

    lamb_dip_center = x[ind][ind_max][0]


    # fit with these starting parameters

    params = Parameters()
    
    # offset of large Doppler peak
    params.add('offset', value = offset_guess, min = 0, max= 4 * y_max, vary = vary)
 
    params.add('ampl_d', value= amplitude_guess,  min=-3*y_max, max=0,   vary = vary)

    params.add('width_d', value = width_guess, min=0.90*width_guess, max=1.1*width_guess, vary = vary)
  
    params.add('cnt_d_0', value = cnt_guesses[0][0], min=x_min, max=x_max,   vary = vary)
    
    params.add('cnt_d_1', value = cnt_guesses[0][1], min=x_min, max=x_max,   vary = vary)

    # Lamb dips

    params.add('ampl_l_0', value = 0.1, min = 0, max = 1.0,   vary = vary)

    params.add('width_l_0', value = width_guess / 35.0, min = width_guess / 50.0,   max=width_guess / 20.0, vary = vary)
        
    # cros-over center peak
    params.add('cnt_l_0', value = lamb_dip_center, min=x_min, max=x_max,   vary = vary)
    
    ## peak right of cross-over
    #params.add('cnt_l_1', value = lamb_dip_center + 133.0, min=x_min, max=x_max,   vary = vary)
    #
    ## cross-over peak left of cross-over
    #params.add('cnt_l_2', value = lamb_dip_center - 79, min=x_min, max=x_max,   vary = vary)


    return run_fit(x, y, fcn2min_type, params, 0, func_opt)



####################################################
# Fit Saturated Absorption Spectroscopy
####################################################

def fcn2min_sas(params, x, data, no_of_peaks, func_opt = 'lorentzian', plot_fit = False, no_of_points = 500):

    offset = params['offset']

    if plot_fit == False:
        x_fit = x
    else:
        x_fit = np.linspace(np.min(x), np.max(x), no_of_points)

    model = offset

    for k in range(no_of_peaks):
        A   = params['ampl_' + str(k)]
        cnt = params['cnt_' + str(k)]
        w   = params['width_' + str(k)]
        
        if func_opt == 'lorentzian':
            model += A * w**2/((x_fit - cnt)**2 + w**2)
        elif func_opt == 'gaussian':
            model += A * np.exp( -(x_fit - cnt)**2/(2.0 * w**2) )
    
    if plot_fit == False:
        return model - data
    else:
        return (x_fit, model)


def fit_sas(x, y, cnt_guesses = [[0], [0]], width_guess = 1, inverse = 1, func_opt = 'gaussian', vary = True):

    fcn2min_type = fcn2min_sas

    y_min  = np.min(y)
    y_max  = np.max(y)
    y_mean = np.mean(y)

    x_mean = np.mean(x)
    x_min  = np.min(x)
    x_max  = np.max(x)

    params = Parameters()
    
    params.add('offset', value= 1.1 * 0.8 + 0*1.3*np.max(y), min=y_min, max=3.2*y_max, vary = vary)
  
    param_counter = -1

    no_of_doppler = len(cnt_guesses[0])
    
    for k in range(no_of_doppler):
    
        param_counter += 1

        if inverse == 1:
            params.add('ampl_'  + str(param_counter), value=y_max * 0.9,   min=0, max=y_max,   vary = vary)
        else:
            params.add('ampl_'  + str(param_counter), value=-y_max * 0.9,  min=-y_max, max=0,   vary = vary)

        params.add('width_' + str(param_counter), value = width_guess,    min=0.90*width_guess,   max=1.1*width_guess, vary = vary)
        
        # broad Doppler background
        params.add('cnt_'   + str(param_counter), value=cnt_guesses[0][k], min=x_min, max=x_max,   vary = vary)

    no_of_peaks = len(cnt_guesses[1])
    
    for k in range(no_of_peaks):

        param_counter += 1

        if inverse == 1:
            params.add('ampl_'  + str(param_counter), value=(-1.0) * y_mean / 2.0,   min=(-y_max), max = 0,   vary = vary)
        else:
            params.add('ampl_'  + str(param_counter), value = 0*y_mean / 20.0, min = 0, max = 1e-12 + 0*y_max,   vary = vary)

        params.add('width_' + str(param_counter), value = width_guess / 15.0,    min = width_guess / 30.0,   max=width_guess / 5.0, vary = vary)
        
        # Lamp dips
        params.add('cnt_'   + str(param_counter), value=cnt_guesses[1][k], min=x_min, max=x_max,   vary = vary)


    return run_fit(x, y, fcn2min_type, params, param_counter, func_opt)


####################################################
# Fit Multiple Gaussians or Lorentzian Transitions
####################################################

def fcn2min_multi_peaks(params, x, data, no_of_peaks, func_opt = 'lorentzian', plot_fit = False, no_of_points = 500):

    offset = params['offset']

    if plot_fit == False:
        x_fit = x
    else:
        x_fit = np.linspace(np.min(x), np.max(x), no_of_points)

    model = offset

    for k in range(no_of_peaks):
        
        A   = params['ampl_' + str(k)]
        cnt = params['cnt_' + str(k)]
        w   = params['width_' + str(k)]
        
        if func_opt == 'lorentzian':
            model += A * w**2/((x_fit - cnt)**2 + w**2)
        elif func_opt == 'gaussian':
            model += A * np.exp( -(x_fit - cnt)**2/(2.0 * w**2) )
    
    if plot_fit == False:
        return model - data
    else:
        return (x_fit, model)


def fit_multi_peaks(x, y, cnt_guesses = [0], width_guess = 1, func_opt = 'gaussian', vary = True):

    fcn2min_type = fcn2min_multi_peaks
    
    param_counter = -1
    
    y_min  = np.min(y)
    y_max  = np.max(y)
    y_mean = np.mean(y)

    x_mean = np.mean(x)
    x_min  = np.min(x)
    x_max  = np.max(x)

    no_of_peaks = len(cnt_guesses)

    params = Parameters()

    params.add('offset', value=np.mean(y[0:5]), min=y_min, max=y_max, vary = vary)
    
    for k in range(no_of_peaks):
        params.add('ampl_'  + str(k), value=y_mean / 2.0,   min=0, max=(y_max - y_min),   vary = vary)
        params.add('cnt_'   + str(k), value=cnt_guesses[k], min=x_min, max=x_max,   vary = vary)
        params.add('width_' + str(k), value=width_guess,    min=0.10*width_guess,   max=(x_max-x_min)/2.0, vary = vary)

        param_counter += 1

    return run_fit(x, y, fcn2min_type, params, param_counter, func_opt)

####################################
# Fit Simple Gaussian
####################################

def fcn2min_gauss(params, x, data, dummy, func_opt = 'gaussian', plot_fit = False, no_of_points = 1000):

    y_offset = params['p0']
    a        = params['p1']
    x0       = params['p2']
    w        = params['p3']

    if plot_fit == False:
        x_fit = x
    else:
        x_fit = np.linspace(np.min(x), np.max(x), no_of_points)

    if func_opt == 'lorentzian':
       model = y_offset + a * w**2/( (x_fit - x0)**2 + w**2 ) 
    elif func_opt == 'gaussian':
       model = y_offset + a * np.exp( -(x_fit - x0)**2/(2.0*w**2) )
 
    if plot_fit == False:
        return model - data
    else:
        return (x_fit, model)


def fit_gauss(x, y, cnt_guess = 0.0, func_opt = 'gaussian'):

    fcn2min_type = fcn2min_gauss

    y_min = np.min(y)
    y_max = np.max(y)
    x_mean = np.mean(x)
    x_min = np.min(x)
    x_max = np.max(x)

    if y_min == y_max:
        y_max = 1.0

    params = Parameters()

    params.add('p0', value=y_min, min=y_min, max=y_max, vary = True)
    params.add('p1', value=(y_max - y_min), min=0.0, max=100.0, vary = True)
    params.add('p2', value=cnt_guess, min=min(x), max=max(x), vary = True)
    params.add('p3', value=(x_max-x_min)/10.0, min=(x[1] - x[1]), max=(x_max-x_min), vary = True)

    param_counter = 4

    return run_fit(x, y, fcn2min_type, params, param_counter, func_opt)

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

    model = y_offset + b * x_fit + a * np.exp( -(x_fit - x0)**2/(2.0*w**2) )
    
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

    params.add('p0', value=y_min, min=-1.0, max=1.0, vary = True)
    params.add('p1', value=(y_max - y_min), min=0.0, max=3.0, vary = True)
    params.add('p2', value=x_mean, min=x_mean-10e9, max=x_mean+10e9, vary = True)
    params.add('p3', value=(x_max-x_min)/10.0, min=100e6, max=(x_max-x_min), vary = True)
    params.add('p4', value=-1.0, min=-10, max=10, vary = True)

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min_gauss_with_slope, params, fcn_args=(x, y, None))
    result = minner.minimize()
    
    (xf, yf) = fcn2min_gauss_with_slope(result.params, x, y, plot_fit = True)
    
    (xf_res, yf_res) = fcn2min_gauss_with_slope(result.params, x, y, plot_fit = True, no_of_points = len(x))
    
    residuals = yf_res - y

    return (result, xf, yf, residuals)


###############################################
# Fit Linear Slope
###############################################

def fcn2min_linear(params, x, data, plot_fit = False, no_of_points = 500):

    o = params['offset']
    s = params['slope']

    if plot_fit == False:
        x_fit = x
    else:
        x_fit = np.linspace(np.nanmin(x), np.nanmax(x), no_of_points)

    model = s * x_fit + o

    if plot_fit == False:
        return model - data
    else:
        return (x_fit, model)


def fit_linear(x, y, cnt_guesses = [0], width_guess = 1, func = 'linear', vary = True):

    y_min  = np.nanmin(y)
    y_max  = np.nanmax(y)
    y_mean = np.nanmean(y)

    x_mean = np.nanmean(x)
    x_min  = np.nanmin(x)
    x_max  = np.nanmax(x)
    
    my_slope = (np.nanmax(y) - np.nanmin(y))/(np.nanmax(x) - np.nanmin(x))

    params = Parameters()

    params.add('offset', value= np.nanmean(y),      min = y_min, max = y_max, vary = vary)
    params.add('slope',  value= my_slope,   min = -10.0 * my_slope, max = +10.0 * my_slope,     vary = vary)

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min_linear, params, fcn_args=(x, y), nan_policy = 'omit')
    result = minner.minimize()
    
    (xf, yf) = fcn2min_linear(result.params, x, y, plot_fit = True)
    
    (xf_res, yf_res) = fcn2min_linear(result.params, x, y, plot_fit = True, no_of_points = len(x))
    
    residuals = yf_res - y

    return (result, xf, yf, residuals)


###############################################
# Fit Atomic Stark Shift
###############################################

def fcn2min_polarizibility(params, x, data, prefac, plot_fit = False, no_of_points = 500):

    if plot_fit == False:
        x_fit = x
    else:
        x_fit = np.linspace(np.min(x), np.max(x), no_of_points)

    model = -1/2 * prefac * (params['scale'] * x_fit)**2
   
    if plot_fit == False:
        return model - data
    else:
        return (x_fit, model)


def fit_atomic_stark_shift(x, y, prefac = 1.0, vary = True):

    y_min  = np.min(y)
    y_max  = np.max(y)
    y_mean = np.mean(y)

    x_mean = np.mean(x)
    x_min  = np.min(x)
    x_max  = np.max(x)

    params = Parameters()

    params.add('scale', value = 1.0, min = 0.0, max = 1.0e5, vary = vary)
   
    # do fit, here with leastsq model
    minner = Minimizer(fcn2min, params, fcn_args=(x, y, prefac))
    result = minner.minimize()
    
    (xf, yf) = fcn2min(result.params, x, y, prefac, plot_fit = True)
    


    # get error ranges
    hlp_params = copy.deepcopy(result.params)
    hlp_params['scale'].value = result.params['scale'].value + result.params['scale'].stderr

    (xf, yf_low) = fcn2min_polarizibility(hlp_params, x, y, prefac, plot_fit = True)
    
    hlp_params = copy.deepcopy(result.params)
    hlp_params['scale'].value = result.params['scale'].value - result.params['scale'].stderr

    (xf, yf_high) = fcn2min_polarizibility(hlp_params, x, y, prefac, plot_fit = True)


    (xf_res, yf_res) = fcn2min_polarizibility(result.params, x, y, prefac, plot_fit = True, no_of_points = len(x))
    
    residuals = yf_res - y

    #ci = lmfit.conf_interval(minner, result)

    #report_fit(result)

    conf_result = {
            'xf'      : xf,
            'yf_low'  : yf_low,
            'yf'      : yf,
            'yf_high' : yf_high,
            }

    return (result, xf, yf, residuals, conf_result)


