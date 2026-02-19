import matplotlib.pyplot as plt
import numpy as np
from data_manipulation import get_time_indeces, get_time_index, get_freq_indeces, integrate_freq_time, filter_data
from report_functions import print_dict

channel_translate = { 
            0 : 'absorption',   # in-cell
            1 : 'fire_check',   # yag photodiode check
            2 : 'pmt',          # pmt
            3 : 'hodor_pickup', # Hodor blue pickup
            4 : 'davos_pickup', # Davos blue pickup
            5 : 'yag_sync',     # Yag sync
            6 : 'daenerys_pickup', # Daenerys pickup
            7 : 'in_cell_ref',      # in-cell ref
            
            8 : 'slow_absorption',   # in-cell
            9 : 'slow_fire_check',   # yag photodiode check
            10 : 'slow_pmt',          # pmt
            11 : 'slow_hodor_pickup', # Hodor blue pickup
            12 : 'slow_davos_pickup', # Davos blue pickup
            13 : 'slow_yag_sync',     # Yag sync
            14 : 'slow_daenerys_pickup', # Daenerys pickup
            15 : 'slow_in_cell_ref',      # in-cell ref
            }


##############################################################################################################

def prepare_molecule_data(d, opts):
    
    scan_number = str(d['scan_info']['date']) + '/' + str(d['scan_info']['time'])
    
    t_max = opts['t_max_plot']

    t     = d['times']
    x     = d['set_points']
    x_act = d['act_freqs']
    y2d   = d['channels'][opts['channel']]

    # apply Fourier filter
    if opts['apply_fft_filter']:
        y2d = filter_data(y2d, filter_edge = opts['fft_cut_off_freq'])

    # cnt_freq is just the mean of the frequencies
    cnt_freq = np.mean(x_act)

    x_act -= np.mean(x_act)
    x_act *= 1e6

    t_max_plot_ind = get_time_index(t, t_max)

    if d['scan_info']['type'] == 'laser_scan':
        
        # subtract center frequency offset
        x = x - np.mean(x)
        plot_fac = 1.0

        # calculate laser frequency error = wavemeter readout - set frequency
        laser_err = (x - x_act)/plot_fac
    
    elif d['scan_info']['type'] == 'param_scan':
        
        plot_fac = 1.0
        laser_err = (x_act - np.mean(x_act))*1e12/1e6

    elif d['scan_info']['type'] == 'microwave_scan':
        
        x = x - cnt_freq
        plot_fac = 1.0e6
        laser_err = (x_act - np.mean(x_act))*1e12/1e6

    return (scan_number, t, t_max_plot_ind, x, x_act, y2d, cnt_freq, laser_err) 


##############################################################################################################

def plot_molecule_scan(d, opts, plot_fac = 1.0):

    # prepare data for plotting
    (scan_number, t, t_max_plot_ind, x, x_act, y2d, cnt_freq, laser_err) = prepare_molecule_data(d, opts)

    # calculate integration limits
    xmin = np.min(x)/plot_fac
    xmax = np.max(x)/plot_fac

    (t1_cut, t2_cut) = get_time_indeces(t, opts)
    (x1_cut, x2_cut) = get_freq_indeces(x, opts, scale_factor = plot_fac)

    (y_f, y_t) = integrate_freq_time(t, x, y2d, opts, scale_factor = plot_fac)


    ##################################################
    # Apply comb correction
    ##################################################

    if 'apply_comb_correction' in opts.keys():

        if opts['apply_comb_correction']:

            (fx, fy) = d['comb_scans']['comb_comparison']['wm_comb_err']

            x           += fy
            x_act       += fy
            cnt_freq    += np.mean(fy)

    # plotting
    if opts['do_plot']:
        
        (fig, ax) = plt.subplots(2, 2, constrained_layout = True, figsize = (8, 8))

        if opts['channel'] == 0:
            fig.suptitle("{0}: {1} - He flow: {4} sccm - Channel: {2} - {3} - reference beam".format(d['scan_info']['title'], scan_number, opts['channel'], channel_translate[opts['channel']], d['conf']['he_flow']['val']))
        else:
            fig.suptitle("{0}: {1} {5} - He: {4} sccm - Ch: {2} - {3}".format(d['scan_info']['title'], scan_number, opts['channel'], channel_translate[opts['channel']], d['conf']['he_flow']['val'], opts['info']))

        # plot data

        plot_2d = ax[0,0].pcolor(x/plot_fac, t[0:t_max_plot_ind], np.transpose(y2d[:, 0:t_max_plot_ind]))
        ax[0,1].plot(x/plot_fac, y_f, 'o-')
        ax[1,0].plot(x/plot_fac, laser_err, '.')
        ax[1,1].plot(t, y_t, '-')

        # set limits
        #ax[0,0].set_xlim(xmin, xmax)
        ax[0,0].set_ylim(opts['t_min_plot'], opts['t_max_plot'])

        plt.colorbar(plot_2d)
        
        # set colorbar limits
        if opts['channel'] == 0:
            #plot_2d.set_clim(np.min(np.min(y2d[x1_cut:x2_cut, t1_cut:t2_cut])), np.max(np.max(y2d[x1_cut:x2_cut, t1_cut:t2_cut])))
            plot_2d.set_clim(2*np.min(y_t), 2*np.abs(np.min(y_t)))
        
        ax[1,0].set_xlim(xmin, xmax)
        ax[1,0].set_ylim( -1.2*np.max(np.abs(laser_err)), 1.2*np.max(np.abs(laser_err)) )
        
        ax[0,1].set_xlim(xmin, xmax)
        
        ax[1,1].set_xlim(0, opts['t_max_plot'])
        if opts['channel'] == 0:
            ax[1,1].set_ylim(2*np.min(y_t), 2*np.abs(np.min(y_t)))
        
 
        # set x/y-labels
        if d['scan_info']['type'] == 'laser_scan':
            ax[0,0].set_xlabel("Frequency (MHz) + {0:12.6f} THz".format(cnt_freq))
            ax[1,0].set_xlabel("Frequency (MHz) + {0:12.6f} THz".format(cnt_freq))
            ax[0,1].set_xlabel("Frequency (MHz) + {0:12.6f} THz".format(cnt_freq))
            ax[1,1].set_xlabel('Time (ms)')

        elif d['scan_info']['type'] == 'param_scan':
            ax[0,0].set_xlabel("{0}".format(d['scan_info']['xlabel']))
            ax[1,0].set_xlabel("{0}".format(d['scan_info']['xlabel']))
            ax[0,1].set_xlabel("{0}".format(d['scan_info']['xlabel']))
            ax[1,1].set_xlabel('Time (ms)')
 
        elif d['scan_info']['type'] == 'microwave_scan':
            ax[0,0].set_xlabel("Frequency (MHz) + {0:5.1f} MHz".format(cnt_freq/1e6))
            ax[1,0].set_xlabel("Frequency (MHz) + {0:5.1f} MHz".format(cnt_freq/1e6))
            ax[0,1].set_xlabel("Frequency (MHz) + {0:5.1f} MHz".format(cnt_freq/1e6))
            ax[1,1].set_xlabel('Time (ms)')
        
       

        ax[0,0].set_ylabel('Time (ms)')
        ax[0,1].set_ylabel('Int. Signal (a.u.)')
        ax[1,1].set_ylabel('Int. Signal (a.u.)')
        ax[1,0].set_ylabel('Set - Act Freq (MHz)')

        ## plot limit lines
        ax[0,0].axhline(t[t1_cut], ls = '--', color = 'r')
        ax[0,0].axhline(t[t2_cut], ls = '--', color = 'r')

        ax[0,0].axvline(x[x1_cut]/plot_fac, ls = '--', color = 'r')
        ax[0,0].axvline(x[x2_cut]/plot_fac, ls = '--', color = 'r')
        
        # plot average line of laser error
        ax[1,0].axhline(np.mean(x + cnt_freq - x_act)/1e6, ls = '--', color = 'r')
        ax[1,0].axhline(0, ls = '-', color = 'k')

    results = {
            'freq'              : x,
            'y_freq'            : y_f,
            'times'             : t,
            'y_times'           : y_t,
            'y2d'               : y2d,
            'cnt_freq'          : cnt_freq,
            'no_of_setpoints'   : d['no_of_setpoints'],
            'no_of_averages'    : d['no_of_avg']
            }


    return results


##############################################################################################################

def prepare_electron_data(d, opts):

    scan_number = str(d['scan_info']['date']) + '/' + str(d['scan_info']['time'])
    
    t_max = opts['t_max_plot']

    t       = d['times']
    x       = d['set_points']
    x_act   = d['act_freqs']
    y       = d['channels'][opts['channel']]

    cnt_freq = np.mean(x)

    t_max_plot_ind = 0

    if d['scan_info']['type'] == 'laser_scan':
        # center frequencies
        x = x - cnt_freq

    x *= 1e12
    cnt_freq *= 1e12

    return (scan_number, t, t_max_plot_ind, x, x_act, y, cnt_freq) 


##############################################################################################################

def plot_electron_scan(d, opts, plot_fac = 1):

    (scan_number, t, t_max_plot_ind, x, x_act, y_t, cnt_freq) = prepare_electron_data(d, opts)

    xmin = np.min(x)/plot_fac
    xmax = np.max(x)/plot_fac

    if d['scan_info']['type'] == 'laser_scan':
        laser_err = (x + cnt_freq - x_act)/plot_fac
    elif d['scan_info']['type'] == 'param_scan':
        laser_err = (x_act - np.mean(x_act))*1e12/1e6

    if opts['do_plot']:
        
        (fig, ax) = plt.subplots(2, 2, constrained_layout = True, figsize = (12, 12))

        fig.suptitle("{0}: {1} - Channel: {2} - {3}".format(d['scan_info']['title'], scan_number, opts['channel'], channel_translate[opts['channel']]))

        # plot data

        #ax[0,0].pcolor(x/plot_fac, t[0:t_max_plot_ind], np.transpose(y2d[:, 0:t_max_plot_ind]))
        ax[0,1].plot(x/plot_fac, y_t)
        ax[1,0].plot(x/plot_fac, laser_err, '.')

        # set limits
        ax[0,0].set_xlim(xmin, xmax)
        ax[1,0].set_xlim(xmin, xmax)
        ax[0,1].set_xlim(xmin, xmax)
        
        ax[1,0].set_ylim( -1.2*np.max(np.abs(laser_err)), 1.2*np.max(np.abs(laser_err)) )
 
        # set x/y-labels
        if d['scan_info']['type'] == 'laser_scan':
            ax[0,0].set_xlabel("Frequency (MHz) + {0:12.6f} THz".format(cnt_freq/1e12))
            ax[1,0].set_xlabel("Frequency (MHz) + {0:12.6f} THz".format(cnt_freq/1e12))
            ax[0,1].set_xlabel("Frequency (MHz) + {0:12.6f} THz".format(cnt_freq/1e12))

        elif d['scan_info']['type'] == 'param_scan':
            ax[0,0].set_xlabel("{0}".format(d['scan_info']['xlabel']))
            ax[1,0].set_xlabel("{0}".format(d['scan_info']['xlabel']))
            ax[0,1].set_xlabel("{0}".format(d['scan_info']['xlabel']))
        

        ax[0,0].set_ylabel('Signal (a.u.)')
        ax[0,1].set_ylabel('Int. Signal (a.u.)')
        ax[1,1].set_ylabel('Int. Signal (a.u.)')
        ax[1,0].set_ylabel('Set - Act Freq (MHz)')

        # plot limit lines
        #ax[0,0].axvline(x[x1_cut]/plot_fac, ls = '--', color = 'r')
        #ax[0,0].axvline(x[x2_cut]/plot_fac, ls = '--', color = 'r')
        
        # plot average line of laser error
        ax[1,0].axhline(np.mean(x + cnt_freq - x_act)/1e6, ls = '--', color = 'r')
        ax[1,0].axhline(0, ls = '-', color = 'k')

    return (x, y_t)


##############################################################################################################


def plot_full_scan(d, opts, plot_fac = 1.0, which_exp = 'molecules'):

    if d['scan_info']['type'] == 'laser_scan':
        plot_fac = 1.0
    elif d['scan_info']['type'] == 'param_scan':
        plot_fac = 1.0

    if   which_exp == 'molecules':
        results = plot_molecule_scan(d, opts, plot_fac = plot_fac)

    elif which_exp == 'electron':
        results = plot_electron_scan(d, opts, plot_fac = plot_fac)

    return results



