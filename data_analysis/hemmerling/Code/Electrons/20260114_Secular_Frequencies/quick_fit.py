# Preamble file with imports

from preamble import *

import numpy as np
import pickle

from fit_functions import fit_multi_peaks

from read_in_data import read_in_config

from pdf_functions import save_all_plots

from plot_functions import my_scatter_plot

def get_scan_list():

    f = open('scan_list.csv', 'r')

    lines = f.readlines()

    scans = []

    for l in lines[1:]:

        (timestamp, b1, u2) = l.split(',')

        date = timestamp.split('_')[0]
        time = timestamp.split('_')[0]

        scans.append({
            'date': date,
            'time': time,
            'timestamp' : timestamp,
            'u2' : u2,
            'b1' : b1,
            })


    f.close()
    
    return scans

def read_file(base_filename, cat):

    f = open('{0}_{1}'.format(base_filename, cat), 'r')
    d = np.genfromtxt(f)
    f.close()
 
    return d


def get_compensation(conf):

    hlp = {}

    for k in ['U1', 'U2', 'U3', 'U4', 'U5', 'Ex', 'Ey', 'Ez']:

        hlp[k] = conf[k]['val']

    return hlp


def read_data(scans):

    base_folder = '/Users/boerge/Software/offline_electrons_data'

    data = []

    keys = ['ratio_lost', 'arr_of_setpoints']

    for s in scans:

        hlp = {}

        base_filename = '{0}/{1}/{2}'.format(base_folder, s['date'], s['timestamp'])

        hlp['timestamp'] = s['timestamp']

        for k in keys:
            hlp[k] = read_file(base_filename, k)
       

        # read in config

        conf = read_in_config('{0}_conf'.format(base_filename))

        hlp['conf'] = conf

        hlp['rf_power'] = conf['RF_amplitude']['val']
        hlp['tickle_level'] = conf['tickle_level']['val']
        hlp['tickle_pulse_length'] = conf['tickle_pulse_length']['val']

        hlp['compensation'] = get_compensation(conf)


        data.append(hlp)

    return data


#######################################################

def get_scan(scans, timestamp):

    for k in range(len(scans)):

        if scans[k]['timestamp'] == timestamp:
            
            return scans[k]

    return 0

#######################################################

def plot_data(scans):

    plt.figure()

    for k in range(len(scans)):

        s = scans[k]

        plt.plot(s['arr_of_setpoints'], 10 * s['ratio_lost']/max(s['ratio_lost']) + k )

    return

#######################################################

def plot_scan(s):


    plt.plot(s['arr_of_setpoints'], s['ratio_lost'])

    return


def get_xy(s):

    return (s['arr_of_setpoints'], s['ratio_lost'])


#######################################################


scans = get_scan_list()

data = read_data(scans)

print('Found {0} scans'.format(len(data)))


############################
# Fit data
############################

init_c = {
        '20260107_174740' : [43.5, 44],
        '20260107_175216' : [58, 63],
        '20260107_180137' : [84, 90],
        '20260107_181203' : [98, 100],
        '20260107_181704' : [100],
        '20260107_182743' : [104, 108],
        '20251224_144819' : [42],
        '20251224_145254' : [61, 62],
        '20251224_150102' : [81, 83],
        '20251224_150934' : [102, 103.5],
        '20251224_175711' : [52],
        '20251224_180919' : [84, 86],
        '20251224_181339' : [102],
        '20251223_142531' : [34, 38],
        '20251223_143329' : [56, 61, 68],
        '20251223_144635' : [81, 83],
        '20251223_145237' : [94, 97, 100],
        '20251223_150033' : [110],
        '20251223_150438' : [120, 123],
        '20251223_192919' : [37],
        '20251223_193638' : [56, 61],
        '20251223_194955' : [82],
        '20251223_195354' : [99, 101],
        '20251223_200839' : [142, 144, 146],
        '20251222_142532' : [22, 24],
        '20251222_143229' : [63, 66, 69],
        '20251222_145548' : [89, 90, 92],
        '20251222_150242' : [107],
        '20251222_151117' : [116],
        '20251222_185258' : [29, 32],
        '20251222_190012' : [62, 64],
        '20251222_191539' : [83, 85],
        '20251222_192232' : [95, 97],
        '20251222_193035' : [129],
        '20251219_152324' : [9.5, 13, 14],
        '20251219_152848' : [26, 28],
        '20251219_153648' : [59, 66, 70, 74, 79, 81.5, 88, 94, 99, 101, 112],
        '20251219_174546' : [14, 15],
        '20251219_175228' : [29, 30],
        '20251219_175909' : [46],
        '20251219_180433' : [62, 64, 74, 77, 81, 91, 93, 105, 108, 120, 124],
        '20251218_145950' : [8, 9],
        '20251218_150250' : [16],
        '20251218_150947' : [64, 69, 73, 78, 79, 82, 86.5, 88.5, 93, 96, 100],
        '20251218_174138' : [12],
        '20251218_174818' : [24, 25],
        '20251218_175457' : [55, 67, 74, 77, 79, 87, 91, 98.5, 101, 110, 113],
        '20251217_165722' : [9, 16, 18],
        '20251217_171011' : [60, 69, 77, 85, 90, 94, 102],
        }



results = []


for n in range(len(data)):
    
    s = data[n]

    timestamp = s['timestamp']

    print('\n{0} Scan {1}'.format(n, timestamp))

    (x, y) = get_xy(s)

    func = 'gaussian'

    width_guess = 1
    cnt_guesses = init_c[timestamp]
    
    (fit_result, xf, yf, residuals) = fit_multi_peaks(x, y, cnt_guesses = cnt_guesses, width_guess = width_guess, func_opt = func, vary = True)
    
    (fig, ax) = plt.subplots(1, 1)

    my_scatter_plot(ax, x, y)
    ax.plot(xf, yf, 'r--')
   
    
    # get fitting results
    pars = fit_result.params

    my_str = []

    for n in range(len(init_c[timestamp])):

        my_str.append('center: {0:8.2f}, width: {1:8.3f}, amplitude: {2:8.3f}'.format(
            pars['cnt_{0}'.format(n)].value,
            pars['width_{0}'.format(n)].value,
            pars['ampl_{0}'.format(n)].value
            ))




    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Ratio lost (counts)')

    fig.suptitle('Scan: {0} - U2: {1} - RF Power: {2} dBm'.format(timestamp, s['compensation']['U2'], s['rf_power']))

    fig.tight_layout()



    # save results

    results.append({
        'timestamp'    : timestamp,
        'scan'         : s,
        'fit_results'  : fit_result.params,
        'par_str'      : my_str,
        'x'            : x,
        'y'            : y,
        'xf'           : xf,
        'yf'           : yf
        })




save_all_plots('fits')


# save results

with open('fit_results.pkl', 'wb') as file:
    pickle.dump(results, file)


# save ascii file

f = open('fit_results.txt', 'w')

for n in range(len(results)):

    f.write('{0} - U2: {1} - RF Power: {2} dBm\n'.format(results[n]['timestamp'], results[n]['scan']['compensation']['U2'], results[n]['scan']['rf_power']))

    for k in range(len(results[n]['par_str'])):

        f.write('   {0}\n'.format(results[n]['par_str'][k]))

    f.write('*' * 57)
    f.write('\n')

f.close()


