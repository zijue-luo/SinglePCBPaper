import sys
sys.path.append("../Analysis_Scripts/")

import pickle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pylab as pylab
from matplotlib.backends.backend_pdf import PdfPages

from fit_functions     import fit_gauss
from report_functions  import print_dict
from plot_functions    import my_scatter_plot
from constants         import *
from data_manipulation import threshold_filter



#################################################################
# General parameters
################################################################

from latex     import *
from constants import *

################################################################


def plot_comb_measurements(filename, fontsize = 8):
 
    ################################################################
    # Load data
    ################################################################

    with open('results/{0}.pckl'.format(filename), 'rb') as f:

        data_raw = pickle.load(f)


    data = data_raw[0]['data']
    
    channels = list(data.keys())

    # get setpoints
    freqs = data[channels[0]]['freq']
    times = data[channels[0]]['times']
    
    comb = data_raw[0]['comb_scans']


    print_dict(comb)

    
    (fig, ax) = plt.subplots(4, 1, constrained_layout = True, figsize = (8, 8))
    
    ############################
    # Plot rep rate
    ############################
    
    vrep         = comb['comb_rep_rate']
    vrep_nominal = comb['comb_rep_rate_nominal'][0]


    y_dev = (vrep - vrep_nominal)

    y_max = max(abs(y_dev))

    # data and fit
    my_scatter_plot(ax[0], freqs, y_dev, connect = True)

    ax[0].axhline(0, ls = '--', color = 'k')

    ax[0].set_ylim( -1.05 * y_max, 1.05 * y_max )

    ax[0].set_ylabel(r'$\nu_{rep}$ deviation (Hz)', fontsize = fontsize)

    
    #############################
    # Fit measured beat nodes
    #############################

    spec_freq = comb['comb_beat_node_freq'][0, :]/1e6
    beat_node = comb['comb_beat_node_psd']

    ax[1].pcolor(freqs, spec_freq, np.transpose(beat_node))

    ax[1].set_ylabel('Beat node frequency (MHz)', fontsize = fontsize)



    # beat node fit vs time

    beat_node_fit = fit_beat_nodes(spec_freq, beat_node, func = 'gaussian')


    bn_arr =     [ x['fit_result'].params['p2'].value for x in beat_node_fit ]
    bn_arr_err = [ x['fit_result'].params['p3'].value for x in beat_node_fit ] # using the width of the peak as an error

    bn_avg = np.mean(bn_arr)
    

    #ax[2].errorbar(x, bn_arr, yerr = bn_arr_err)
    
    ax[2].axhline( bn_avg, ls = '--', color = 'k')
    
    my_scatter_plot(ax[2], freqs, bn_arr, connect = True)

    
    ax[2].set_ylim( bn_avg - 2.0, bn_avg + 2.0 )

    ax[2].set_ylabel('Beat Node Frequency (MHz)', fontsize = fontsize)



    # wavemeter readings of Moglabs
    v_moglabs_wm = comb['ref_freqs']
    # wavemeter readings of scanning laser
    v_laser_wm   = comb['act_freqs']


    v_moglabs_wm_avg = np.mean(v_moglabs_wm)
    v_laser_wm_avg = np.mean(v_laser_wm)


    my_scatter_plot(ax[3], freqs, (v_moglabs_wm - v_moglabs_wm_avg)*1e12/1e6, label = 'Moglabs = {0:.6f} THz'.format(v_moglabs_wm_avg), connect = True)
    
    my_scatter_plot(ax[3], freqs, (v_laser_wm - np.mean(v_laser_wm))*1e12/1e6, label = 'Laser x2  = {0:.6f} THz'.format(v_laser_wm_avg), color = 'r', connect = True)

    ax[3].set_ylabel('Wavemeter deviation (MHz)', fontsize = fontsize)
    ax[3].set_ylim(-2,2)

    ax[3].legend(fontsize = fontsize)



    #############################
    # Calculate true frequency
    #############################

    y_moglabs = (v_moglabs_wm - v_moglabs_wm_avg) * 1e12/1e6

    n_comb = np.floor( v_moglabs_wm_avg*1e12 / vrep_nominal )

    moglabs_true_freq = (n_comb * vrep_nominal + bn_avg * 1e6)/1e12

        
    ax[3].axhline(0, ls = '--', color = 'k')
        
    my_scatter_plot(ax[3], freqs, y_moglabs)

    ax[3].set_ylabel('Frequency Moglabs (MHz) \n + {0:.6f} THz'.format(v_moglabs_wm_avg))

    #ax[3].set_ylim( -1.05 * np.max(abs(y_moglabs)), 1.05 * np.max(abs(y_moglabs)) )
        
    ax[3].set_ylim( -10, 10 )


        

    ax[3].text(min(freqs), 1.5, r'$\nu_{{Moglabs}} = {0:.0f} \cdot {1:.0f}\,{{MHz}} + {2:.1f}\,{{MHz}} = {3:.6f}\,{{THz}}$  ;  Wavemeter dev: ${4:.1f}$ MHz'.format(
        n_comb, 
        vrep_nominal/1e6, 
        bn_avg, 
        moglabs_true_freq, 
        (moglabs_true_freq * 1e12 - v_moglabs_wm_avg * 1e12)/1e6)
               )


    fig.tight_layout()



    return


################################################################
# Fit beat nodes
################################################################

def fit_beat_nodes(spec_freq, beat_node, cnt_guess = 71.0, func = 'gaussian'):

    cut1 = 100
    cut2 = 400

    beat_node_fit = []

    for k in range(beat_node.shape[0]):

        x = spec_freq[cut1:cut2]
        y = beat_node[k, cut1:cut2]

        (fit_result, xf, yf, residuals) = fit_gauss(
                                                       x, 
                                                       y, 
                                                       cnt_guess = cnt_guess,
                                                    )

        hlp = {
                'fit_result'  : fit_result,
                'residuals'   : residuals, 
                'x_fit'       : xf, 
                'y_fit'       : yf,
                'x_data'      : x,
                'y_data'      : y,
              }

        beat_node_fit.append(hlp)
        

    return beat_node_fit



#######################################################################################################
# Plot transfer cavity measurements
#######################################################################################################

def plot_cavity_measurements(filename, fontsize = 8):
 
    ################################################################
    # Load data
    ################################################################

    with open('results/{0}.pckl'.format(filename), 'rb') as f:

        data_raw = pickle.load(f)


    data = data_raw[0]['data']
    
    channels = list(data.keys())

    # get setpoints
    freqs = data[channels[0]]['freq']
    times = data[channels[0]]['times']
    
    comb = data_raw[0]['comb_scans']


    print_dict(comb)

    
    (fig, ax) = plt.subplots(4, 1, constrained_layout = True, figsize = (8, 8))


    cav_t               = comb['transfer_cavity_times']
    cav_ramp            = comb['transfer_cavity_traces_ch1']
    cav_moglabs_offset  = comb['transfer_cavity_traces_ch3']
    cav_moglabs         = comb['transfer_cavity_traces_ch2']
    cav_tisaph          = comb['transfer_cavity_traces_ch4']


    # take both channels and cut off below the threshold to simplify the fitting

    # filter with a threshold to enable fitting
    
    (cav_moglabs, cav_ml_cnt_ind) = threshold_filter(cav_moglabs, 0.15, width = 30)
    
    (cav_tisaph, cav_tisaph_cnt_ind) = threshold_filter(cav_tisaph, 0.15, width = 30)

    # fit each peak

    moglabs_pos = []
    
    tisaph_pos = []
    
    for k in range(len(freqs)):
        
        # fit moglabs transition

        x = cav_t
        y = cav_moglabs[k, :]
        
        cnt_guess = cav_t[cav_ml_cnt_ind]
        
        (fit_result, xf, yf, residuals) = fit_gauss(
                                                       x, 
                                                       y, 
                                                       cnt_guess = cnt_guess,
                                                    )

        moglabs_pos.append(fit_result.params['p2'].value)

        # fit tisaph position

        x = cav_t
        y = cav_tisaph[k, :]
        
        cnt_guess = cav_t[cav_tisaph_cnt_ind]

        (fit_result, xf, yf, residuals) = fit_gauss(
                                                       x, 
                                                       y, 
                                                       cnt_guess = cnt_guess,
                                                    )

        tisaph_pos.append(fit_result.params['p2'].value)


    ax[0].plot(np.transpose(cav_moglabs[:, :]))

    ax[1].plot(np.transpose(cav_tisaph[:, :]))


    # plot frequency deviations
    # pos is in seconds
    # slope gives us MHz/ms

    slope = 37.23/1e-3

    moglabs_freq_dev = (moglabs_pos - np.mean(moglabs_pos)) * slope
    tisaph_freq_dev = (tisaph_pos - np.mean(tisaph_pos)) * slope

    my_scatter_plot(ax[2], freqs, moglabs_freq_dev)

    my_scatter_plot(ax[3], freqs, tisaph_freq_dev)

    return















#######################################################################################################

def plot_old_comb_measurements( fit_results,
                            channel = 2,
                            freq_offset = 0.0, # in MHz
                            myfontsize = 12,
                            func = 'gaussian'
                         ):


    for n in range(len(fit_results)):



        # Determine comb tooth


        moglabs_wm_freqs = c['ref_freqs']
        moglabs_cnt_freqs = np.mean(moglabs_wm_freqs)

        y_moglabs = (moglabs_wm_freqs - moglabs_cnt_freqs) * 1e12/1e6

        n_comb = np.floor( moglabs_cnt_freqs*1e12 / vrep_nominal )


        moglabs_true_freq = (n_comb * vrep_nominal + bn_avg * 1e6)/1e12

        
        ax[3].axhline(0, ls = '--', color = 'k')
        
        my_scatter_plot(ax[3], x, y_moglabs)

        ax[3].set_ylabel('Frequency Moglabs (MHz) \n + {0:.6f} THz'.format(moglabs_cnt_freqs))

        #ax[3].set_ylim( -1.05 * np.max(abs(y_moglabs)), 1.05 * np.max(abs(y_moglabs)) )
        
        ax[3].set_ylim( -2, 2 )


        

        ax[3].text(min(x), 1.5, r'$\nu_\textrm{{Moglabs}} = {0:.0f} \cdot {1:.0f}\,\textrm{{MHz}} + {2:.1f}\,\textrm{{MHz}} = {3:.6f}\,\textrm{{THz}}$  ;  Wavemeter dev: ${4:.1f}$ MHz'.format(n_comb, vrep_nominal/1e6, bn_avg, moglabs_true_freq, (moglabs_true_freq * 1e12 - moglabs_cnt_freqs * 1e12)/1e6))


        fig.tight_layout()


        print_dict(c)

    return





##############################################################################################################

def plot_comb_comparison(d, opts, plot_fac = 1e6):

    fac = 4.0

    # get scan number
    scan_number = str(d['scan_info']['date']) + '/' + str(d['scan_info']['time'])

    comb = d['comb_scans']
    
    wr       = comb['comb_rep_rate']
    freq_raw = comb['comb_beat_node_freq_raw']
    y2d_raw  = comb['comb_beat_node_psd_raw']
 
    freq = comb['comb_beat_node_freq']
    y2d  = comb['comb_beat_node_psd']
    

    # get setpoints and wavemeter readout in IR frequencies
    
    sp          = d['set_points']/fac
    sp_raw      = d['set_points_raw']/fac
    
    wm_freqs     = d['absolute_freqs']
    wm_freqs_raw = d['absolute_freqs_raw']

    cnt_freq    = np.mean(sp)


    offset_cnt_freq = float(d['conf']['offset_laser_Daenerys']['val'])


    baseline = np.mean(np.mean( y2d ))

    wm_mod_raw = np.mod( wm_freqs_raw*1e12, wr )/1e6


    ##########################
    # Find peaks
    ##########################

    peaks_pos = []
    for k in range(len(sp_raw)):

        peaks, _ = find_peaks( y2d_raw[k, :], prominence = 20, height = (-45, 0))

        peaks_pos.append(freq_raw[k, peaks]/1e6)


    ##########################
    # Plotting
    ##########################

    if opts['do_plot']:

        (fig, ax) = plt.subplots(3, 1, constrained_layout = True, figsize = (8, 8))
                
        fig.suptitle("{0}: {1} - param: {2}".format(opts['info'], scan_number, d['scan_info']['par']))

        ############################
        # Plot data
        ############################

        # Comb data

        do_plot_2D_map = False

        if do_plot_2D_map:
            plot_2d = ax[0].pcolor(freq[0, :]/1e6, sp, y2d)

            ax[0].plot( np.mod( wm_freqs*1e12, wr )/1e6, sp, 'r.')

            ax[0].set_xlabel('Spectrum Analyzer Frequency (MHz)')
            ax[0].set_ylabel('Set Point IR (MHz)', fontsize = 8)
            plt.colorbar(plot_2d)


        # Comb and wavemeter comparison

        no = 0

        ax[no].plot( wm_mod_raw, sp_raw, 'r.', label = 'Wavemeter Mod 200 MHz')

        for k in range(len(sp_raw)):

            ax[no].scatter( peaks_pos[k], [sp_raw[k]] * len(peaks_pos[k]), s = 80, facecolors = 'none', edgecolors = STD_COL )

        ax[no].set_xlabel('Spectrum Analyzer Frequency (MHz) (red = wavemeter reading)')
        ax[no].set_ylabel('Set Point Frequency IR (MHz)', fontsize = 8)
        #plt.legend()


        # Comb and wavemeter comparison

        no = 1

        setpoints_comb_difference = []

        for k in range(len(sp_raw)):

            hlp = []
            
            # region in MHz to avoid where beat node peaks become ambiguous
            dead_zone_threshold = 10.0 # in MHz

            # check for dead zones
            if     (np.abs(wm_mod_raw[k] - 100.0) < dead_zone_threshold)   \
                or (np.abs(wm_mod_raw[k] -   0.0) < dead_zone_threshold)   \
                or (np.abs(wm_mod_raw[k] - 200.0) < dead_zone_threshold):

                my_color = 'r'
                my_style = 'x'

                for n in range(len(peaks_pos[k])):

                    hlp.append( np.nan )

            else:

                my_color = STD_COL
                my_style = '.'

                for n in range(len(peaks_pos[k])):

                    # positive value means: wavemeter shows lower frequency than frequency comb
                    # negative value means: wavemeter shows higher frequency than frequency comb
                    
                    hlp.append( peaks_pos[k][n] - wm_mod_raw[k] )     

            ax[no].plot( [sp_raw[k]] * len(hlp), hlp, my_style, color = my_color )

            setpoints_comb_difference.extend( hlp )

        setpoints_comb_difference = np.array(setpoints_comb_difference)

        ax[no].set_xlabel('Set Point Frequency IR (MHz)')
        ax[no].set_ylabel('Diff. to Comb IR (MHz)')

        ## region in MHz to avoid where beat node peaks become ambiguous
        #frequency_blind_spot = 10.0

        #my_label = '{0:.2f} +/- {1:.2f} MHz (for < {2} MHz)'.format( 
        #                                np.nanmean(setpoints_comb_difference[abs(setpoints_comb_difference) < frequency_blind_spot]), 
        #                                np.nanstd(setpoints_comb_difference[abs(setpoints_comb_difference) < frequency_blind_spot]),
        #                                                            frequency_blind_spot
        #                                                            )
        
        ax[no].set_ylim(-20.0, 20.0)


        # Wavemeter and set point comparison

        no = 2

        setpoints_wavemeter_difference = wm_freqs*1e12 - (sp * 1e6 + offset_cnt_freq * 1e12)

        ax[no].plot( sp, setpoints_wavemeter_difference/1e6, 'o-', label = '{0:.2f} +/- {1:.2f} MHz'.format( np.mean(setpoints_wavemeter_difference)/1e6, np.std(setpoints_wavemeter_difference)/1e6))
        
        ax[no].legend()

        ax[no].set_xlabel('Set Point Frequency IR (MHz)')
        ax[no].set_ylabel('Set Point - Rel. WM Freq. (MHz)', fontsize = 8)
        
        ax[no].set_ylim(-3, 3) 


    return {
            'setpoints_raw'                     : sp_raw,
            'wavemeter_mod_raw'                 : wm_mod_raw,
            'setpoints'                         : sp,
            'setpoints_wavemeter_difference'    : setpoints_wavemeter_difference,
            'setpoints_comb_difference'         : setpoints_comb_difference
           }

   


