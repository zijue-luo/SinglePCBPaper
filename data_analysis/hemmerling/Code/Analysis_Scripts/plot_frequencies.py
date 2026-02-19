import matplotlib.pyplot as plt
import numpy as np
from data_manipulation import get_time_indeces, get_time_index, get_freq_indeces, filter_data
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from math_functions import moving_average

from my_fit_functions import fit_2D_line, fit_gauss_with_slope

import copy






def peak_search_trace_python(x, y, min_height = 1):

    peaks, _ = find_peaks(y, height = (min_height, None))#, prominence = 4)
 
    return peaks
    if len(peaks) != 2:
        peaks = [np.nan] * 2
        return peaks
    else:
        return x[peaks]


def peak_search_trace(x_raw, y_raw, options):


    # find background between threshold

    x = moving_average(x_raw, n = 2)
    y = moving_average(y_raw, n = 2)

    #x = x_raw
    #y = y_raw

    peak_threshold = options['comb_settings']['peak_detection']['peak_threshold']

    width = options['comb_settings']['peak_detection']['peak_width']

    dead_zone = 6

    peaks = []

    hlp1 = []
    hlp2 = []

    begin_peak = False
    end_peak   = False

    peak_rising_edge = False
    
    new_peak = []

    #for k in range(len(y)-width):
    for k in range(int(width/2), len(y)-width):
        if (x[k] > dead_zone and x[k] < 200 - dead_zone):
           
            x_fit = x[k-int(width/2):k+int(width/2)]
            y_fit = y[k-int(width/2):k+int(width/2)]

            
            hlp2.append([x[k], np.std(y_fit)])

            peak_detected = (np.std(y_fit)*(y[k] - np.mean(y_fit))) > peak_threshold[0]
            
            if peak_detected and not peak_rising_edge:
                peak_rising_edge = True
                
                new_peak = [x[k]]

            else:

                if peak_rising_edge and not peak_detected:

                    new_peak.append(x[k])

                    peak_rising_edge = False

            if len(new_peak) == 2:                
                hlp1.append(np.mean(new_peak))
                new_peak = []



    # remove all peaks close to 0, 100 and 200 MHz

    #plt.plot(x, y, 'b')

    #for k in range(len(hlp1)):

    #    plt.axvline(hlp1[k])
    ###

    ###print(hlp1)
    #plt.show()


    #asd
   
    if len(hlp1)!=2:
        peaks = [np.nan, np.nan]
    else:
        peaks = hlp1
   
    return peaks 



def make_line_with2points(x, y):

    # make function such that it starts positive and just ramps up
    # then we beat only with one comb node

    f = lambda t : (y[0] + (t - x[0]) * (y[1]-y[0])/(x[1]-x[0]))

    g = lambda t : f(t) - 200 * (np.ceil(f(x[0])/200)-1)

    return g


def extrapolate_line(xs, g, n = 1000):

    x = np.linspace(min(xs), max(xs), n)
    
    return (x, g(x))


def plot_laser_frequencies(d, opts, plot_fac = 1e6):

    # get scan number
    scan_number = str(d['scan_info']['date']) + '/' + str(d['scan_info']['time'])

    comb = d['comb_scans']
    
    wr = comb['comb_rep_rate']
    
    y2d = comb['comb_beat_node_spectrum']
    
    background_ind = opts['comb_settings']['comb_background_ind']

    # hardcoded: Spectrum analyzer scans between 1 - 211 MHz
    freq        = np.linspace(1, 211, 1000)


    # get setpoints and wavemeter readout in IR frequencies
    sp          = d['set_points'] / 4.0
    
    wm_freqs    = d['act_freqs'] / 4.0

    cnt_freq    = np.mean(sp)

    # calculate laser error
    laser_wm_err = sp - wm_freqs

    ####################################################
    # subtract background of beat node
    ####################################################
    
    bg = copy.deepcopy(y2d[background_ind, :])

    for k in range(y2d.shape[0]):

        y2d[k, :] -= bg
        y2d[k, :] -= np.min(y2d[k, -10:])

    # amplify peaks and reduce noise
    y2d = y2d**3


    ####################################################
    # get peaks
    ####################################################
    
    peaks = []
    for k in range(len(y2d)):
         peaks.append( peak_search_trace(freq, y2d[k, :], opts) )
         #peaks.append( peak_search_trace_python(freq[20:-20], y2d[k, 20:-20], min_height = 500) )
    
    peaks = np.array(peaks) 



    ####################################################
    # fit line to 2D data for scan
    ####################################################

    no_of_folds = opts['comb_settings']['peak_detection']['no_of_folds']

    # combine all peaks in one array
    
    # order:
    # x = [10, 10, 10, 10, 10, 10, ...]
    # y = [p1, p1+200, p1+400, p2, p2+200, p2+400, ...]

    y = peaks.flatten()    

    hlp = []
    # attach the second loads of peaks shifted by 1 comb tooth
    for k in range(-no_of_folds, no_of_folds+1):
        hlp.append(list(y + k*200))

    y = np.array(hlp).transpose().flatten()

    x = sp.repeat((2*no_of_folds+1) * 2)
    x = (x - cnt_freq)/1e6
    
    limit_points = opts['comb_settings']['peak_detection']['limit_points']

    xind1 = 2 * no_of_folds * limit_points[0][0] + limit_points[0][1]
    yind1 = xind1
    
    xind2 = 2 * no_of_folds * limit_points[1][0] + limit_points[1][1]
    yind2 = xind2

    f_comb_fit = make_line_with2points([x[xind1], x[xind2]], [y[yind1], y[yind2]])

    #print(x.shape)
    #print(y.shape)
    #plt.plot(x, y, 'o')
    #plt.plot(x, f_comb_fit(x), 'r')
    #

    #plt.plot(x[xind1], y[yind1], 'gx')
    #plt.plot(x[xind2], y[yind2], 'rx')

    #plt.show()
    #asd

    ####################################################
    # Calculate absolute frequencies
    ####################################################

    # We beat the whole scan with the same comb tooth since we unfolded the beat node
    abs_freqs = []

    n_tooth = int( wm_freqs[0] // wr )
    
    for k in range(len(sp)):
        
        abs_freqs.append( n_tooth * wr + f_comb_fit( (sp[k] - cnt_freq)/1e6 ) * 1e6 )


    # calculate wavemeter - comb error
    # positive values mean wavemeter frequency is too high
    wavemeter_comb_error = wm_freqs - abs_freqs


    ####################################################
    # Plotting
    ####################################################
    
    if opts['do_plot']:
        
        (fig, ax) = plt.subplots(2, 3, constrained_layout = True, figsize = (12, 8))
            
        fig.suptitle("{0}: {1}".format(opts['info'], scan_number))

        # plot data
       
        # top left
        plot_2d = ax[0,0].pcolor(freq, (sp - cnt_freq)/1e6, y2d)
        
        ax[0,0].axhline((sp[background_ind] - cnt_freq)/1e6, color = 'r')
        
        ax[0,0].plot(peaks, (sp - cnt_freq)/1e6, color = 'r', ls = '--')
        
        # top right
        ax[0,1].plot(freq, y2d[0, :], 'r-')
        ax[0,1].plot(freq, y2d[10, :], 'g-')
        ax[0,1].plot(freq, y2d[-1, :], 'b-')
      
        # bottom left
        ax[1,0].plot((sp - cnt_freq)/1e6, laser_wm_err/1e6, color = 'r')
        
        ax[1,0].set_xlabel('Set point frequency (relative, IR) (MHz)')
        ax[1,0].set_ylabel('Set - Act Wavemeter Freq (IR) (MHz)')
        ax[1,0].set_ylim( -1.2*np.max(np.abs(laser_wm_err)/1e6), 1.2*np.max(np.abs(laser_wm_err)/1e6) )

        # bottom middle
        for k in range(-no_of_folds, no_of_folds):
            ax[1,1].plot((sp - cnt_freq)/1e6, peaks[:, 0] + 200 * k, color = 'r')
            ax[1,1].plot((sp - cnt_freq)/1e6, peaks[:, 1] + 200 * k, color = 'g')
       
        ax[1,1].plot((sp - cnt_freq)/1e6, f_comb_fit((sp - cnt_freq)/1e6), '--', label = 'BN fit')
      
        # plot used fit points
        ax[1,1].plot(x[xind1], y[yind1], 'gx', markersize = 10)
        ax[1,1].plot(x[xind2], y[yind2], 'rx', markersize = 10)


        ax[1,1].legend()
        
        # bottom right
        ax[1,2].plot((sp - cnt_freq)/1e6, ((abs_freqs - cnt_freq)/1e6), 'k-', label = 'FC')
        ax[1,2].plot((sp - cnt_freq)/1e6, ((wm_freqs - cnt_freq)/1e6),  'm-', label = 'WM')

        ax[1,2].legend()


        # top right
        ax[0,2].plot((sp - cnt_freq)/1e6, ((wavemeter_comb_error)/1e6), 'b-')


        # set x/y-labels
        if d['scan_info']['type'] == 'laser_scan':
            ax[0,0].set_xlabel("Spectrum Analyzer Frequency (IR) (MHz)")
            ax[0,1].set_xlabel("Spectrum Analyzer Frequency (IR) (MHz)")
            ax[1,1].set_xlabel('Set point frequency (relative, IR) (MHz)')
            ax[1,2].set_xlabel('Set point frequency (relative, IR) (MHz)')
            ax[0,2].set_xlabel('Set point frequency (relative, IR) (MHz)')

        ax[0,0].set_ylabel('Set point frequency (relative, IR) (MHz)')
        ax[1,1].set_ylabel('Spectrum Analyzer Frequency (IR) (MHz)')
        ax[1,2].set_ylabel('Frequency - center freq (IR) (MHz)')
        ax[0,2].set_ylabel('WM - FC frequency (IR) (MHz)')

        
        results = {
            #'f_comb_fit'        : f_comb_fit, # can't pickle lambda function
            'f_comb_fit'        : extrapolate_line((sp - cnt_freq)/1e6, f_comb_fit),
            'laser_err_wm'      : laser_wm_err,
            'wm_comb_err'       : (4*sp, 4*wavemeter_comb_error)
            }


    return results



