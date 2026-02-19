import numpy as np

import matplotlib.pyplot as plt

from configparser import ConfigParser

import matplotlib
import pylab

import lmfit
from lmfit import Minimizer, Parameters, report_fit



fontsize = 20


font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : fontsize}

#matplotlib.rc('font', **font)


params = {
            'text.usetex'       : True,     # for latex fonts
            'legend.fontsize'   : fontsize,
            'figure.figsize'    : (8, 5),
            'axes.labelsize'    : fontsize,
            'axes.titlesize'    : fontsize,
            'xtick.labelsize'   : fontsize,
            'ytick.labelsize'   : fontsize,
            'font.size'         : fontsize
            }

pylab.rcParams.update(params)




####################################
# Fit Exp Growth
####################################

def fcn2min_func_growth(params, x, data, func = None, plot_fit = False, no_of_points = 500):

    p0 = params['p0']
    p1 = params['p1']
    p2 = params['p2']

    if plot_fit == False:
        x_fit = x
    else:
        #x_fit = np.linspace(np.min(x), np.max(x), no_of_points)
        x_fit = np.linspace(0, np.max(x), no_of_points)

    model = p0 * (1 - np.exp(-(x_fit-p1)/p2))
    
    if plot_fit == False:
        return model - data
    else:
        return (x_fit, model)


####################################################################

def my_fit_growth(x, y):

    y_min = np.min(y)
    y_max = np.max(y)
    x_mean = np.mean(x)
    x_min = np.min(x)
    x_max = np.max(x)

    params = Parameters()

    params.add('p0', value=6.5e3, min=0.0, max=1e4, vary = True)
    params.add('p1', value=0.1, min=-10, max=10, vary = True)
    params.add('p2', value=0.5, min=0, max=30, vary = True)

    # do fit, here with leastsq model
    minner = Minimizer(fcn2min_func_growth, params, fcn_args=(x, y, None))
    result = minner.minimize()
    
    (xf, yf) = fcn2min_func_growth(result.params, x, y, plot_fit = True)
    
    (xf_res, yf_res) = fcn2min_func_growth(result.params, x, y, plot_fit = True, no_of_points = len(x))
    
    residuals = yf_res - y

    return (result, xf, yf, residuals)




####################################
# Fit Exp Decay
####################################

def fcn2min_func(params, x, data, func = None, plot_fit = False, no_of_points = 500):

    p0 = params['p0']
    p1 = params['p1']
    p2 = params['p2']

    if plot_fit == False:
        x_fit = x
    else:
        #x_fit = np.linspace(np.min(x), np.max(x), no_of_points)
        x_fit = np.linspace(0, np.max(x), no_of_points)

    model = p0 * np.exp(-(x_fit-p1)/p2)
    
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

    params.add('p0', value=1, min=0.0, max=1e4, vary = True)
    params.add('p1', value=1, min=-5, max=5, vary = True)
    params.add('p2', value=1, min=0, max=10, vary = True)

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




####################################
# returns moving average
####################################

def moving_average(a, n = 3) :
    
    if not n == 0:
        ret = np.cumsum(a, dtype = float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    else:
        return a

####################################
# reads in config file
####################################

def read_in_config(f):
    
    config = ConfigParser()
    config.read(f)

    sensor_ids = config.sections()
    # make dictionary out of config

    sensors = {}

    for s in sensor_ids:
        opts = config.options(s)
        
        sensors[s] = {}
        for o in opts:
            sensors[s][o] = config.get(s, o)

    return sensors


def extract_timestamps(date, time):

    path = '/Users/boerge/Software/offline_electrons_data/'

    f = open(path + date + '/' + date + '_' + time + '_arr_of_timestamps')

    lines = f.readlines()

    print(lines)
    result = []
    for k in range(len(lines)):

        result.append( [ float(x) for x in lines[k][1:-2].split(',') ] )

    f.close()


    print(result)

    asd

    (h, edges) = np.histogram(tstamps[k], bins = np.linspace(0, 10000, 500))

    sig.append(h)

    trapped_no.append(max(h[10:]))


def get_data(date, time, filename, timestamps = False):

    path = '/Users/boerge/Software/offline_electrons_data/'
    #path = '/home/electrons/software/data/'

    f = open(path + date + '/' + date + '_' + time + '_' + filename)

    lines = f.readlines()

    result = []
    for k in range(len(lines)):

        if not timestamps:
            
            result.append( float(lines[k].strip()) )

        else:

            result.append( [ float(x) for x in lines[k][1:-2].split(',') ] )

    f.close()

    return result


def get_full_data(date, time):

    path = '/Users/boerge/Software/offline_electrons_data/'

    data = {}

    try:
        data['x'] = np.array(get_data(date, time,  'arr_of_setpoints'))
    except:
        data['x'] = []
        print('no x data')

    try:
        data['yt'] = np.array(get_data(date, time,  'trapped_signal'))
        data['yl'] = np.array(get_data(date, time,  'loading_signal'))

    except:

        print('getting spectrum')

        #(yt, yl) = extract_timestamps(date, time)

        data['yt'] = np.array(get_data(date, time,  'spectrum'))
        data['yl'] = 0*data['yt']


    data['config'] = read_in_config(path +  date + '/' + date + '_' + time + '_conf')

    if not 'scanning_parameter' in data['config'].keys():

        data['config']['scanning_parameter'] = {'val' : 'N/A'}

    return data


def get_arr_data(date, time_arr):

    d = []
    for t in time_arr:
        d.append(get_full_data(date, t))

    return d


def plot_arr_data(d, plot_loaded = False):

    for k in range(len(d)):

        plt.figure()

        plt.plot(d[k]['x'], d[k]['yt'], label = 'trapped')
        if plot_loaded:
            plt.plot(d[k]['x'], d[k]['yl'], label = 'loaded')
    
        plt.legend()

        par = d[k]['config']['scanning_parameter']['val']

        if not par == 'N/A':
        
            plt.xlabel("Scanning parameter: {0} ({1})".format(par, d[k]['config'][par]['unit']))

        else:
            
            plt.xlabel("Tickle frequency (MHz)")



        plt.ylabel('MCP counts')

        plt.title('Scan {0}'.format(d[k]['config']['Scan']['filename'][-15:]))

    return



date = '20231113'

time = ['164343'] # ['150932', '151201', '152532', '152655']


ts = get_data(date, time[0], 'arr_of_extraction_times', timestamps = False)
data = get_data(date, time[0], 'arr_of_timestamps', timestamps = True)

ts = np.array(ts)


sig = []
trapped = []

for k in range(len(data)):
    (h, edges) = np.histogram(data[k], bins = np.linspace(0, 10000, 500))

    sig.append(h)

    trapped.append( max(h[3:]) )

sig = np.array(sig)


(result, xf, yf, residuals) = my_fit(ts/1e3, trapped)

print(result.params)

plt.figure()

plt.plot(ts/1e3, trapped, 'o')

plt.plot(xf, yf)

plt.ylim(0, 270)

plt.xlim(-0.25, 10.25)

plt.xlabel('Time (ms)')
plt.ylabel('Trapped Signal (cts)')

plt.tight_layout()








date = '20231207'

time = ['140554'] # ['150932', '151201', '152532', '152655']


ts = get_data(date, time[0], 'scan_x', timestamps = False)
data = get_data(date, time[0], 'arr_of_timestamps', timestamps = True)

ts = np.array(ts)

trapped = []

sig = []

for k in range(len(data)):
    (h, edges) = np.histogram(data[k], bins = np.linspace(0, 1000, 500))

    sig.append(h)

    trapped.append( max(h[105:]) )

plt.figure()



plt.plot(ts, trapped, 'o-')

#plt.ylim(0, 270)

plt.xlim(-0.75, 0.75)

plt.xlabel('Quadrupole U2 (Volt)')
plt.ylabel('Trapped Signal (cts)')

plt.axvline(0.0, color = 'k', ls = '--')

plt.tight_layout()










date = '20240110'

time = ['135741'] # ['150932', '151201', '152532', '152655']


ts = get_data(date, time[0], 'arr_of_setpoints', timestamps = False)
data = get_data(date, time[0], 'arr_of_timestamps', timestamps = True)

ts = np.array(ts)

trapped = []
loaded = []
sig = []

for k in range(len(data)):
    
    (h, edges) = np.histogram(data[k], bins = np.linspace(0, 5000, 2000))

    sig.append(h)

    ind = np.where( abs(edges - ts[k]) < 3.0 )[0][0]

    trapped.append( max(h[ind:]) )

    loaded.append( h )





(result, xf, yf, residuals) = my_fit_growth(ts/1e3, trapped)


print(result.params)

plt.figure()

plt.plot(ts/1e3, trapped, 'o')
plt.plot(xf, yf, '-')

plt.ylim(0, 6500)

#plt.xlim(-0.75, 0.75)

plt.xlabel('Loading Time (ms)')
plt.ylabel('Trapped Signal (cts)')

plt.axvline(0.0, color = 'k', ls = '--')

plt.tight_layout()













plt.show()

