import sys
sys.path.append("../Analysis_Scripts/")

import pickle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pylab as pylab
from matplotlib.backends.backend_pdf import PdfPages


from fit_multi_peaks import fit_multi_peaks 
from pdf_functions import save_all_plots
from report_functions import print_dict

import copy
from prettytable import PrettyTable



########################################################
# Plot several scans in one plot
########################################################

def plot_overlapped_scans(d, info, shift_arr = [], ampl_arr = [], plot_fits = False):

    x_out = []
    y_out = []
    x_fit_out = []
    y_fit_out = []

    print()

    if len(shift_arr) != len(d):
        print('Shift arr has the wrong length. len(d) = {0}\n'.format(len(d)))
        asd

    cnt_freq_0 = d[0]['data']['output']['cnt_freq']
    x_min      = 0
    x_max      = 1

    if len(shift_arr) == 0:
        shift_arr = len(d) * [0]

    if len(ampl_arr) == 0:
        ampl_arr = len(d) * [1]

    fig = plt.figure()
    for n in range(len(d)):

        my_date = d[n]['data']['raw_data']['date']
        my_time = d[n]['data']['raw_data']['time']

        print('Processing {2} - {0}/{1} ...\n'.format(
            my_date,
            my_time,
            d[n]['data']['options']['info']))

        x        = d[n]['data']['output']['freq']
        y        = d[n]['data']['output']['y_freq']
        cnt_freq = d[n]['data']['output']['cnt_freq']

        print("Difference of center_freq relative to {0:.6f} THz: {1:6.1f} MHz".format(cnt_freq_0/1e12, (cnt_freq - cnt_freq_0)/1e6))

        x_plot   =  (x + cnt_freq - cnt_freq_0 + shift_arr[n]) / 1e6

        x_min = min( np.append(x_plot, x_min) )
        x_max = max( np.append(x_plot, x_max) )
       
        plt.plot( x_plot, ampl_arr[n] * y, label = "{0}/{1} - shift: {2:.2f} MHz".format(my_date, my_time, shift_arr[n]/1e6))

        if plot_fits and 'fit' in d[n].keys():

            xf = d[n]['fit']['xfit'] + (cnt_freq - cnt_freq_0 + shift_arr[n]) / 1e6
            yf = d[n]['fit']['yfit']

            plt.plot( xf, yf, '-' )
        
            x_fit_out.append(xf)
            y_fit_out.append(yf)


        plt.xlabel('Frequency (MHz) + {0:.6f} THz'.format(cnt_freq_0/1e12))
        plt.ylabel('Signal (a.u.)')
        plt.legend(fontsize = 8)

        print()


        x_out.append(x_plot)
        y_out.append(ampl_arr[n] * y)


    for k in range(len(info['txt'])):
        fig.text(info['xpos'][k], info['ypos'][k], info['txt'][k], fontsize = 1.5*info['fontsize'])

    plt.xlim(x_min, x_max)
   
    print()

    return ( x_out, y_out, x_fit_out, y_fit_out )



################################################################
# Combine datasets
################################################################


def combine_datasets(d_arr, inds):

    # start by taking the first element in the list

    d_new = copy.deepcopy(d_arr[inds[0]])

    cnt_freq_0 = d_new['output']['cnt_freq']

    # append the rest and skip the first item
    for k in range(1, len(inds)):

        freq_offset = d_arr[inds[k]]['output']['cnt_freq'] - cnt_freq_0

        d_new['output']['freq'] = np.append( d_new['output']['freq'], d_arr[inds[k]]['output']['freq'] + freq_offset )
        
        # amplitudes
        d_new['output']['y_freq'] = np.append( d_new['output']['y_freq'], d_arr[inds[k]]['output']['y_freq'] )
        
        # time and date
        try:
            d_new['raw_data']['date'].append(d_arr[inds[k]]['raw_data']['date'])
            d_new['raw_data']['time'].append(d_arr[inds[k]]['raw_data']['time'])
        except:
            d_new['raw_data']['date'] = [ d_new['raw_data']['date'], d_arr[inds[k]]['raw_data']['date']]
            d_new['raw_data']['time'] = [ d_new['raw_data']['time'], d_arr[inds[k]]['raw_data']['time']]

    return d_new







##################################################################
# Plot multiple lines with broken axes
##################################################################

def plot_multi_lines(d_arr, info, marker_style = '.', line_style = '-'):

    d = .01
    l = .015

    fontsize = info['fontsize']
    number_of_subplots = len(d_arr)
    
    fig, axes = plt.subplots(
            nrows = 1, 
            ncols = number_of_subplots, 
            sharex = False,
            sharey = True,
            )
    
    
    axes[0].yaxis.tick_left()
    axes[0].spines['right'].set_visible(False)
    
    kwargs = dict(transform = axes[0].transAxes, color='k', linewidth = 1, 
             clip_on = False)
    
    # plot axes breaks
    axes[0].plot((1-d/1.5,1+d/1.5), (-d,+d), **kwargs)
    axes[0].plot((1-d/1.5,1+d/1.5),(1-d,1+d), **kwargs)
    kwargs.update(transform = axes[-1].transAxes)
    axes[-1].plot((-l,+l), (1-d,1+d), **kwargs)
    axes[-1].plot((-l,+l), (-d,+d), **kwargs)


    # sort by center frequency

    ind = np.argsort( [x['data']['output']['cnt_freq'] for x in d_arr] )

    plot_numbers = np.arange(number_of_subplots)

    for i in plot_numbers:

        i_d = ind[i]

        x_data = d_arr[i_d]['data']['output']['freq']/1e6
        y_data = d_arr[i_d]['data']['output']['y_freq']
    
        x_fit  = d_arr[i_d]['fit']['xfit']
        y_fit  = d_arr[i_d]['fit']['yfit']
    
        axes[i].plot( x_data, y_data, marker = marker_style, linestyle = line_style, color = 'b' )
        axes[i].plot( x_fit,  y_fit,  marker = None, linestyle = '-',        color = 'r' )
    
        axes[i].set_xlim( 0.90*min(x_data), 1.10*max(x_data) )
    
        if i != 0:
            axes[i].spines['left'].set_visible(False)
            axes[i].tick_params(axis = 'y', which = 'both', labelright = 
                    False, length = 0)
        
        if i != number_of_subplots - 1 and i != 0:
            axes[i].spines['right'].set_visible(False)
            
            kwargs = dict(transform = axes[i].transAxes, color='k', 
                     linewidth = 1, clip_on=False)
            
            axes[i].plot((1-l,1+l), (-d,+d), **kwargs)
            axes[i].plot((1-l,1+l),(1-d,1+d), **kwargs)
            kwargs.update(transform = axes[i].transAxes)
            axes[i].plot((-l,+l), (1-d,1+d), **kwargs)
            axes[i].plot((-l,+l), (-d,+d), **kwargs)
    
        axes[i].set_title(d_arr[i_d]['data']['options']['info'])
    
    
    fig.supxlabel(info['xlabel'], fontsize = fontsize)
    fig.supylabel(info['ylabel'], fontsize = fontsize, x = 0.065)
    
    for k in range(len(info['txt'])):
        fig.text(info['xpos'][k], info['ypos'][k], info['txt'][k], fontsize = 1.5*info['fontsize'])


    return




def plot_rotational_lines():

    ##################################################################
    # Plot splitting vs rotational line
    ##################################################################
    
    x = []
    y = []
    
    
    for k in range(len(fit_results)):
    
        x.append(results[k]['options']['info'])
        
        par = fit_results[k]['fit_result'].params
    
        y.append( [par['cnt_' + str(n)].value for n in range(fit_results[k]['no_of_peaks'])] )
    
    
    plt.figure()
    
    for k in range(len(x)):
    
        cnts = np.array(y[k])
    
        plt.plot( len(y[k])*[x[k]], cnts - cnts[0], 'x')
    
    plt.ylabel('Frequency Difference (MHz)')




