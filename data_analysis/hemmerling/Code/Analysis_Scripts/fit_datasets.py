import sys
sys.path.append("../Analysis_Scripts/")

import pickle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pylab as pylab
from matplotlib.backends.backend_pdf import PdfPages

from fit_functions     import fit_multi_peaks, fit_gauss, fit_Rb87_sas
from save_data         import create_scan_list_str
from report_functions  import print_dict
from plot_functions    import get_plot_label
from plot_functions    import my_scatter_plot
from data_manipulation import *
from constants         import *

from models import get_Rb_87_freqs, get_Rb_85_freqs

#######################################################

def freq2vel(freq, wavelength, angle = 45.0, offset = 0.0):
    
    # converts frequency detuning to velocity with angle between k and velocity vectors
    vel = wavelength * (freq - offset) * 1e6 / np.cos( angle * np.pi / 180 )

    return vel


################################################################
# Fit saturated absorption
################################################################

def add_single_line(ax, c, cnt_freq, offset):

    txt_offset = 5

    if (c > -800) and (c < -400): 
        ax.axvline(c, ls = '-', color = 'g')
        ax.axvline(c + offset, ls = '--', color = 'r')

        ax.text(c + txt_offset, 0.05, '{0} {1:.6f} THz'.format( 
                 'L:',
                 (c*1e6 + cnt_freq*1e12)/1e12, 
                 ), fontsize = 8, rotation = 90)

        ax.text(c + offset + txt_offset, 0.05, '{0} {1:.6f} THz'.format( 
                 'M:',
                 (c*1e6 + cnt_freq*1e12 + offset*1e6)/1e12, 
                 ), fontsize = 8, rotation = 90)
 
    return


def add_Rb_lines(ax, cnt_freq, freq, y_f, wavemeter_offset, atom = 'Rb87'):

    # get the literature values for the frequencies
    if atom == 'Rb87':

        (f, A, cnt_Rb_lit) = get_Rb_87_freqs()
 
    elif atom == 'Rb85':
        
        (f, A, cnt_Rb_lit) = get_Rb_85_freqs()


    # plot the main lines

    for k in range(len(f)):

        Rb_line_freq = cnt_Rb_lit*1e12 + f[k] * 1e6 # in Hz
        
        c = Rb_line_freq/1e6 - cnt_freq*1e12/1e6

        add_single_line(ax, c, cnt_freq, wavemeter_offset)

           
        
    # plot the cross over lines

    Rb_line_freq = cnt_Rb_lit*1e12 + (f[0] + f[1])/2 * 1e6
    
    c = Rb_line_freq/1e6 - cnt_freq*1e12/1e6 # + wavemeter_offsets[scan_no]

    add_single_line(ax, c, cnt_freq, wavemeter_offset)



    Rb_line_freq = cnt_Rb_lit*1e12 + (f[0] + f[2])/2 * 1e6
    
    c = Rb_line_freq/1e6 - cnt_freq*1e12/1e6 #+ wavemeter_offsets[scan_no]

    add_single_line(ax, c, cnt_freq, wavemeter_offset)
   

    return



def fit_saturated_absorption(
                             filename, 
                             yag_ablation_time = 5.3,
                             filter_size = 0,
                             myfontsize = 12,
                             atom = 'Rb87',
                             aom = 0.0,
                             ):


    # plots the saturated absorption scans
    # needs two scans: probe+pump and pump only

    ################################################################
    # Load data
    ################################################################

    with open('results/{0}.pckl'.format(filename), 'rb') as f:

        data_raw = pickle.load(f)


    fit_results = []

    for scan_no in range(len(data_raw)):

        ########################################
        # Get data
        ########################################

        (times, freq, y2d, cnt_freq) = get_data(data_raw[scan_no], 7)

        y2d = np.transpose(y2d)

        t1_int = 1 #28 #12 #7
        t2_int = 39 #40 #11

        (t1_cut, t2_cut) = get_time_indeces_raw(times, [t1_int, t2_int])
        

        y_f = np.mean( y2d[t1_cut:t2_cut, :], axis = 0 )

        
        ###############################################################
        # Shifting the laser frequency by the AOM offset
        # since we are using the -1. order of the AOM
        ###############################################################
        
        cnt_freq = (cnt_freq * 1e12 + aom * 1e6)/1e12

        
        ########################################
        # Fit SAS
        ########################################

        # get data on three lines only

        ind = np.where( (freq > -730) & (freq < -400) )

        # rescale amplitude for better fit

        freq = freq[ind]
        y_f = y_f[ind]

        y_f = y_f - np.min(y_f)
        y_f = y_f/np.max(y_f)


        x_to_fit = freq #freq[ind]
        y_to_fit = y_f #y_f[ind]

        (result, xf, yf, residuals) = fit_Rb87_sas(x_to_fit, y_to_fit, 
                                              cnt_guesses = [[-520, -450], []], 
                                              width_guess = 250.0,
                                              vary = True)# False)
        # extract wavemeter offset

        cnt_fit = result.params['cnt_l_0'].value

        (f, A, cnt_Rb_lit) = get_Rb_87_freqs()

        wavemeter_offset_fit = ((cnt_freq + cnt_fit/1e6) - (cnt_Rb_lit + (f[0]+f[1])/2/1e6)) * 1e6

        ########################################
        # Plot data and SAS fit
        ########################################

        # set up figure
        
        (fig, ax2) = plt.subplots(1, 1, constrained_layout = True, figsize = (6, 6))

        ax2 = [ax2]


        # plot data

        my_scatter_plot(ax2[0], freq, y_f, 
                        label = 'Wavemeter offset fit {0:.1f} MHz + AOM {1:.0f} MHz'.format(
                            wavemeter_offset_fit, 
                            aom)
                        )
 
        # plot fit

        ax2[0].plot(xf, yf, 'g-')
        
        # axes

        ax2[0].set_xlabel('Detuning (MHz) + {0:.6f} THz'.format(cnt_freq))
        ax2[0].set_ylabel('Absorption (a.u.)')

        for k in range(len(ax2)):
            ax2[k].set_xlim(min(freq), max(freq))
       
        ax2[0].legend(fontsize = 8) 

        ax2[0].set_title('{0}/{1}'.format(data_raw[scan_no]['meta_data']['date'],data_raw[scan_no]['meta_data']['time']))


        ###########################################
        # add literature lines and fitted lines
        ###########################################
        
        add_Rb_lines(ax2[0], cnt_freq, freq, y_f, wavemeter_offset = wavemeter_offset_fit, atom = 'Rb87')        
        #add_Rb_lines(ax2[0], cnt_freq, freq, y_f, wavemeter_offsets = wavemeter_offsets, atom = 'Rb85')        

        ax2[0].set_xlim(-850, -350)

        fig.tight_layout()


        # Save results

        fit_results.append({
                    'date'              : data_raw[scan_no]['meta_data']['date'],
                    'time'              : data_raw[scan_no]['meta_data']['time'],
                    'wavemeter_offset'  : wavemeter_offset_fit
                })


    return (ax2, fit_results)



################################################################
# Fit velocity
################################################################

def fit_velocity(filename, scan_id = 0, detection_distance = 1.0,
                 yag_ablation_time = 5.3,
                 cell_exit_time = 0.5,
                 myfontsize = 12):

    ################################################################
    # Load data
    ################################################################

    with open('results/{0}.pckl'.format(filename), 'rb') as f:

        data_raw = pickle.load(f)

    scan = data_raw[scan_id]

    (times, freq, y2d_cell) = get_data(scan, 0)
    (times, freq, y2d) = get_data(scan, 2)

    t1_int = 7
    t2_int = 11

    (t1_cut, t2_cut) = get_time_indeces_raw(times, [t1_int, t2_int])

    y_f = np.mean( y2d_cell[:, t1_cut:t2_cut], axis = 1 )


    # change freq to velocities
    # omega = vec(k) * vec(v)
    # nu = v/lambda * cos(alpha)
    # v = lambda * nu / cos(alpha)
    # v = sqrt(2) * lambda * nu


    cnt_freq = scan['data'][2]['cnt_freq'] * 1e12
    wavelength = c_light/cnt_freq



    offset = 0

    vel = freq2vel(freq, wavelength, offset = offset)

    # plot

    (fig, ax) = plt.subplots(3, 1, constrained_layout = True, figsize = (4, 8))

    fig.suptitle('Velocity scan', fontsize = myfontsize)
 
    plot2d_cell = ax[0].pcolor(vel, times, np.transpose(y2d_cell))
    
    ax[1].plot(vel, y_f)

    plot2d = ax[2].pcolor(vel, times, gaussian_filter_data(np.transpose(y2d), filter_edge = 1))



    # add velocity curves

    velocities = -detection_distance/(times * 1e-3)

    ax[2].plot(velocities, times + yag_ablation_time + cell_exit_time, color = 'r', ls = '--')
    
    #ax[2].plot(velocities + freq2vel(300.0, wavelength), times + yag_ablation_time + cell_exit_time, color = 'r', ls = '--')



    
    ax[0].set_ylim(5, 12)
    
    ax[0].set_ylabel('Time')

    plot2d_cell.set_clim(-0.01, 0)
    
    #plot2d.set_clim(0, 0.7)

    ax[0].axhline(t1_int, ls = '--')
    ax[0].axhline(t2_int, ls = '--')

    
    ax[2].set_ylabel('Time (ms)')

    ax[2].set_xlim(min(vel), max(vel))
    ax[2].set_ylim(0, 18)


    return




################################################################
# Fit spectral features
################################################################

def fit_spectrum(
                    filename, 
                    fit_func, 
                    cnt_guesses, 
                    channels = [0], 
                    do_plot = False, 
                    do_fit = True, 
                    combine_datasets_arr = None, 
                    width_guess = 30.0, 
                    func_opt = 'gaussian', 
                    vary = True
                ):

    
    print()

    ################################################################
    # Load data
    ################################################################

    with open('results/{0}.pckl'.format(filename), 'rb') as f:

        data_raw = pickle.load(f)


    ###################################################################
    ## Combine datasets
    ###################################################################

    #data_results = []
    #if do_fit and (not combine_datasets_arr == None):

    #    all_datasets_ind = range(len(data_raw))

    #    processed_indeces = []

    #    for k in range(len(combine_datasets_arr)):
    #        # toggle through each combination event

    #        combination_inds = combine_datasets_arr[k]

    #        data_results.append(combine_datasets(data_raw, combination_inds))

    #        processed_indeces.extend(combination_inds)

    #    # attach remaining datasets
    #    for k in range(len(all_datasets_ind)):

    #        if not all_datasets_ind[k] in processed_indeces:
    #            data_results.append(data_raw[k])

    #else:
    #    data_results = data_raw
    
    data_results = data_raw


    if do_fit and len(cnt_guesses[channels[0]]) < len(data_results):
            print('Cnt guesses has the wrong length. len(cnt_guesses) = {0}, len(data) = {1}\n'.format(len(cnt_guesses), len(data_results)))
            asd



    ##################################################################
    # Fit data
    ##################################################################
    
    results = []

    for n in range(len(data_results)):

        scan_stamp = '{0}_{1}'.format(data_results[n]['meta_data']['date'], data_results[n]['meta_data']['time'])

        scan_info  = '{0}: {1}{2}'.format(
                data_results[n]['options']['info'],
                data_results[n]['meta_data']['par'][0],
                data_results[n]['meta_data']['par'][1],
                                          )
        hlp = { 'fit_results' : {} }
        
        hlp.update(data_results[n])

        # update some metadata
        hlp['meta_data']['scan_stamp'] = scan_stamp
        hlp['meta_data']['scan_info']  = scan_info


        for channel in channels:

            print('Fitting {1} - {0} - Channel {2}...\n'.format(scan_stamp, scan_info, channel))

            d = data_results[n]['data'][channel]

            if channel == 0:
                sign_factor = -1
            else:
                sign_factor = 1

            x = d['freq']
            y = d['y_freq'] * sign_factor
            

            ##################################################################
            # Do the actual fit
            ##################################################################

            (fit_result, xf, yf, residuals) = fit_func(
                                                       x, 
                                                       y, 
                                                       cnt_guesses = cnt_guesses[channel][n], 
                                                       width_guess = width_guess,
                                                       func_opt = func_opt,
                                                       vary = vary
                                                       )
            hlp_fit_dict = {
                    channel : {
                        'fit_result'  : fit_result,
                        'no_of_peaks' : len(cnt_guesses[channel][n]),
                        'residuals'   : residuals, 
                        'x_fit'       : xf, 
                        'y_fit'       : yf,
                        'x_data'      : x,
                        'y_data'      : y,
                        'cnt_freq'    : d['cnt_freq'],
                        'plot'        : []
                        }
                   }

            hlp['fit_results'].update(hlp_fit_dict)
               
            
            ##################################################################
            # Plot the fit
            ##################################################################

            if do_plot:
            
                (fig, ax) = plt.subplots()
            
                if do_fit and len(cnt_guesses[channel][n]) > 0:
                    ax.plot(xf, yf, 'r-')
                
                ax.plot(x,  y, 'k.-', label = "{0}".format(scan_info))
                ax.legend()

                ax.set_xlabel('Frequency (MHz)')
                ax.set_ylabel('Signal (a.u.)')

                ax.set_title('Scan: {0}'.format(scan_stamp))

                #results[-1]['plot'] = [fig, ax]

        # attach hlp to results
        results.append(hlp)

    if do_plot:
        plt.show()


    return results





