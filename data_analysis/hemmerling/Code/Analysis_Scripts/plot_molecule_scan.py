import matplotlib.pyplot as plt
import numpy as np

from data_manipulation import get_time_indeces, get_time_index, get_freq_indeces, integrate_freq_time, filter_data, gaussian_filter_data
from report_functions  import print_dict
from plot_functions    import my_scatter_plot


channel_translate = { 
            0 : 'absorption',   # in-cell
            1 : 'fire_check',   # yag photodiode check
            2 : 'pmt',          # pmt
            3 : 'hodor_pickup', # Hodor blue pickup
            4 : 'davos_pickup', # Davos blue pickup
            5 : 'yag_sync',     # Yag sync
            6 : 'daenerys_pickup', # Daenerys pickup
            7 : 'HV monitor',      # in-cell ref
            
            8 : 'slow_absorption',   # in-cell
            9 : 'slow_fire_check',   # yag photodiode check
            10 : 'slow_pmt',          # pmt
            11 : 'slow_hodor_pickup', # Hodor blue pickup
            12 : 'slow_davos_pickup', # Davos blue pickup
            13 : 'slow_yag_sync',     # Yag sync
            14 : 'slow_daenerys_pickup', # Daenerys pickup
            15 : 'slow_in_cell_ref',      # in-cell ref
            }


c_light = 299792458

STD_BLUE   = '#1f77b4'
STD_RED    = '#d62728'
STD_ORANGE = '#ff7f0e'




##############################################################################################################

def prepare_molecule_data(d, meta_data, opts):
    
    scan_number = str(meta_data['date']) + '/' + str(meta_data['time'])
    
    t_max = opts['t_max_plot']

    t     = d['times']
    #x     = d['set_points']
    x     = d['absolute_set_points'] # need to use the setpoints with the scanning laser offset to combine multiple scans
    x_act = d['act_freqs']
    y2d   = d['channels'][opts['channel']]

    # apply Fourier filter
    if opts['apply_fft_filter']:
        y2d = filter_data(y2d, filter_edge = opts['fft_cut_off_freq'])

    # apply Fourier filter
    if opts['apply_gaussian_filter']:
        y2d = gaussian_filter_data(y2d, filter_edge = opts['gaussian_filter_size'])

    #print(x[0]/1e6, x[-1]/1e6)

    #asd


    # center frequency
    # cnt_freq is just the mean of the frequencies
    # No, this works only for symmetric scans
 
    # subtract offset of scan for plotting

    # since x is now in terms of absolute frequency set points, we get the center frequency by taking the mean
    cnt_freq = np.mean(x)/1e6 # in THz

    x = x - cnt_freq*1e6
    
    #print(cnt_freq)
    #print(x)

    ## cnt frequency for set point 0 MHz
    #cnt_freq = (max(x_act) - min(x_act))/(max(x) - min(x)) * ( 0.0 - min(x) ) + min(x_act)

    

    
    x_act = x_act - np.mean(x_act)
    x_act = x_act * 1e6

    t_max_plot_ind = get_time_index(t, t_max)

    if meta_data['type'] == 'laser_scan':
        
        plot_fac = 1

        # calculate laser frequency error = wavemeter readout - set frequency
        # set points - (wavemeter frequencies - center_frequency)
        laser_err = (x - np.mean(x) - x_act)/plot_fac
   
    elif meta_data['type'] == 'cavity_scan':

        plot_fac = 1
        laser_err = (x - np.mean(x) - x_act)/plot_fac


    elif meta_data['type'] == 'param_scan':
        
        plot_fac = 1.0
        laser_err = (x_act - np.mean(x_act))*1e12/1e6

    elif meta_data['type'] == 'microwave_scan':
        
        x = x - cnt_freq
        plot_fac = 1.0e6
        laser_err = (x_act - np.mean(x_act))*1e12/1e6

    else:

        pass
    #print(x)
    #print(x_act)
    #print(cnt_freq)

    return (scan_number, t, t_max_plot_ind, x, x_act, y2d, cnt_freq, laser_err) 


##############################################################################################################

def plot_molecule_scan(d, meta_data, opts, plot_fac = 1.0):


    ##################################################
    # Prepare data for plotting
    ##################################################

    (scan_number, t, t_max_plot_ind, x, x_act, y2d, cnt_freq, laser_err) = prepare_molecule_data(d, meta_data, opts)

    channel = opts['channel']

    ##################################################
    # Calculate integration limits
    ##################################################

    xmin = np.min(x)/plot_fac
    xmax = np.max(x)/plot_fac

    (t1_cut, t2_cut) = get_time_indeces(t, opts['t_integrate'][channel])
    (x1_cut, x2_cut) = get_freq_indeces(x, opts, scale_factor = plot_fac)

    (y_f, y_t) = integrate_freq_time(t, x, y2d, [[t1_cut, t2_cut], [x1_cut, x2_cut]], scale_factor = plot_fac)


    ##################################################
    # Apply comb correction
    ##################################################

    #if 'apply_comb_correction' in opts.keys():

    #    if opts['apply_comb_correction']:

    #        (fx, fy) = d['comb_scans']['comb_comparison']['wm_comb_err']

    #        x           += fy
    #        x_act       += fy
    #        cnt_freq    += np.mean(fy)


    ##################################################
    # Plotting
    ##################################################

    if opts['do_plot'] and (channel ==2 or channel == 0 or channel == 7):
       
        myfontsize = 12

        scan_param = meta_data['par']

        (fig, ax) = plt.subplots(2, 2, figsize = (8, 8) , layout = 'tight') #constrained_layout = True, 

        fig.suptitle("{0}: {1} {5} - He: {4} sccm - Ch: {2} - {3} - Param: {6}".format(
                meta_data['title'], 
                scan_number, 
                channel, 
                channel_translate[channel], 
                d['conf']['he_flow']['val'], 
                opts['info'],
                scan_param)
                         , fontsize = myfontsize)

        #fig.tight_layout()

        ##################
        # Plot data
        ##################
        
        # 2D
        plot_2d = ax[0,0].pcolor(x/plot_fac, t[0:t_max_plot_ind], np.transpose(y2d[:, 0:t_max_plot_ind]))

        # Integration over time
        my_scatter_plot(ax[0, 1], x/plot_fac, y_f)

        # Integration over frequency
        ax[1,1].plot(t, y_t, '-')        
        #my_scatter_plot(ax[1, 1], t, y_t)

        # Laser error
        #ax[1,0].plot(x/plot_fac, laser_err, '.')
 
        laser_err[laser_err > 50.0] = np.nan
        
        laser_err -= np.nanmean(laser_err)

       
        my_scatter_plot(ax[1, 0], x/plot_fac, laser_err)

        
        ##################
        # Set limits
        ##################

        plt.colorbar(plot_2d)        
        
        # Set x-limits

        ax[1,0].set_xlim(xmin, xmax)
        ax[0,1].set_xlim(xmin, xmax)
        ax[1,1].set_xlim(0, opts['t_max_plot'])

        # Set y-limits

        ax[1,0].set_ylim( -1.2*np.nanmax(np.abs(laser_err)), 1.2*np.nanmax(np.abs(laser_err)) )
        
        #ax[1,0].set_ylim( -20.0, 20.0 )



        ax[0,0].set_ylim(opts['t_min_plot'], opts['t_max_plot'])

        if channel == 0:

            ax[1,1].set_ylim(2*np.nanmin(y_t), 2*np.abs(np.nanmin(y_t)))

            # set colorbar limits
            plot_2d.set_clim(opts['y_cmin_ch0'], opts['y_cmax_ch0'])

        elif channel == 2:

            ax[1,1].set_ylim(opts['yt_min'], opts['yt_max'])
        
            plot_2d.set_clim(opts['y_cmin_ch2'], opts['y_cmax_ch2'])

        # Set x-labels

        if meta_data['type'] == 'laser_scan' or meta_data['type'] == 'cavity_scan':
            ax[0,0].set_xlabel("Frequency (MHz) + {0:12.6f} THz".format(cnt_freq), fontsize = myfontsize)
            ax[1,0].set_xlabel("Frequency (MHz) + {0:12.6f} THz".format(cnt_freq), fontsize = myfontsize)
            ax[0,1].set_xlabel("Frequency (MHz) + {0:12.6f} THz".format(cnt_freq), fontsize = myfontsize)
            ax[1,1].set_xlabel('Time (ms)', fontsize = myfontsize)

        elif meta_data['type'] == 'param_scan':
            ax[0,0].set_xlabel("{0}".format(meta_data['xlabel']))
            ax[1,0].set_xlabel("{0}".format(meta_data['xlabel']))
            ax[0,1].set_xlabel("{0}".format(meta_data['xlabel']))
            ax[1,1].set_xlabel('Time (ms)')
 
        elif meta_data['type'] == 'microwave_scan':
            ax[0,0].set_xlabel("Frequency (MHz) + {0:5.1f} MHz".format(cnt_freq/1e6), fontsize = myfontsize)
            ax[1,0].set_xlabel("Frequency (MHz) + {0:5.1f} MHz".format(cnt_freq/1e6), fontsize = myfontsize)
            ax[0,1].set_xlabel("Frequency (MHz) + {0:5.1f} MHz".format(cnt_freq/1e6), fontsize = myfontsize)
            ax[1,1].set_xlabel('Time (ms)', fontsize = myfontsize)


        # Set y-labels 

        ax[0,0].set_ylabel('Time (ms)', fontsize = myfontsize)
        ax[0,1].set_ylabel('Int. Signal (a.u.)', fontsize = myfontsize)
        ax[1,1].set_ylabel('Int. Signal (a.u.)', fontsize = myfontsize)
        ax[1,0].set_ylabel('Set - Act Freq (MHz)', fontsize = myfontsize)

        
        # Plot limit lines

        ax[0,0].axhline(t[t1_cut], ls = '--', color = 'r')
        ax[0,0].axhline(t[t2_cut], ls = '--', color = 'r')

        ax[0,0].axvline(x[x1_cut]/plot_fac, ls = '--', color = 'r')
        ax[0,0].axvline(x[x2_cut]/plot_fac, ls = '--', color = 'r')
        
        
        # Plot average line of laser error

        ax[1,0].axhline(0, ls = '--', color = 'k')
        ax[1,0].axhline(np.mean(x + cnt_freq - x_act)/1e6, ls = '--', color = 'r')

        #fig.tight_layout()


    ##################################################
    # Return results
    ##################################################

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







