import numpy as np
import lmfit
from lmfit import Minimizer, Parameters, report_fit



# define objective function: returns the array to be minimized
# function to minimize
def fcn2min(params, x, data, my_lines, plot_fit = False):
    
    y_offset    = params['y_offset']
    freq_offset = params['freq_offset']

    a  = []
    w  = []
    xc = []
    for k in range(len(my_lines)):
        a.append(params['a' + str(k)])
        w.append(params['w' + str(k)])

        xc.append(my_lines[k])
                      
    if plot_fit == False:
        x_fit = x
    else:
        no_points = 1000
        x_plot = np.linspace(np.min(x), np.max(x), no_points)
        x_fit = x_plot

    model = y_offset
    for k in range(len(my_lines)):            
        model += a[k] * np.exp( -(x_fit - xc[k] - freq_offset)**2/(2.0*w[k]**2) )

    if plot_fit == False:
        return model - data
    else:
        return (x_plot, model)



def fit_multi_gaussian(x, y, my_lines, freq_offset = 0.0, vary = True):
    
    # fits a sum of gaussians to a data set
    # my_lines is a list of frequency offsets

    params = Parameters()
    
    params.add('y_offset', value = np.min(y), min = 0*np.min(y), max = np.max(y), vary = vary)
    params.add('freq_offset', value = freq_offset, min = np.min(x), max = np.max(x), vary = vary)
    
    for k in range(len(my_lines)):
        params.add('a' + str(k), value = (np.max(y) - np.min(y))/2, min = 0.0, max = np.max(y), vary = vary)
        params.add('w' + str(k), value = (np.max(x) - np.min(x))/40.0, min = x[1] - x[0], max = (np.max(x) - np.min(x))/2.0, vary = vary)
    
    # do fit, here with leastsq model
    minner = Minimizer(fcn2min, params, fcn_args=(x, y, my_lines))
    result = minner.minimize()
    
    # Store the Confidence data from the fit
    con_report = lmfit.fit_report(result.params)
    
    (x_plot, model) = fcn2min(result.params, x, y, my_lines, plot_fit = True)
    
    # get residuals
    (residuals) = fcn2min(result.params, x, y, my_lines)
    
    #:print(result.params)
    
    return (x_plot, model, result, residuals)

