import numpy as np

import matplotlib.pyplot as plt

from configparser import ConfigParser

import matplotlib


font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)


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



date = '20231212'

time = ['133214', '133341'] # ['150932', '151201', '152532', '152655']

data = get_arr_data(date, time)


plot_arr_data(data, plot_loaded = False)



date = '20231211'

time = ['163433', '163854', '163614'] # ['150932', '151201', '152532', '152655']

data = get_arr_data(date, time)


plot_arr_data(data, plot_loaded = False)




plt.show()


