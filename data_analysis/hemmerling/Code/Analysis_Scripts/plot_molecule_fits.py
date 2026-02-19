import matplotlib.pyplot as plt
import numpy as np

from data_manipulation import get_time_indeces, get_time_index, get_freq_indeces, filter_data, gaussian_filter_data
from report_functions  import print_dict, print_params
from plot_functions    import *
#from fit_datasets      import fit_beat_nodes

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



#######################################################################################################

def plot_cavity_scans( fit_results,
                       channel = 2,
                       freq_offset = 0.0, # in MHz
                       myfontsize = 12
                     ):


    for n in range(len(fit_results)):

        fr = fit_results[n]

        meta_data = fr['meta_data']

        scan_param = meta_data['par']

        (fig, ax) = plt.subplots(2, 1, constrained_layout = True, figsize = (8, 8))

        fig.suptitle(get_plot_label(meta_data), fontsize = myfontsize)
 

        hlp_d = fr['fit_results'][channel]

        x  = hlp_d['x_data']
        y  = hlp_d['y_data']

        xf = hlp_d['x_fit']
        yf = hlp_d['y_fit']
        res = hlp_d['residuals']

        # data and fit
        my_scatter_plot(ax[0], x - freq_offset, y, connect = False)
        ax[0].plot(xf - freq_offset, yf, color = STD_RED)
        
        # residuals
        my_scatter_plot(ax[1], x - freq_offset, res, connect = True)

        ax[1].axhline(0.0, ls = '--', color = 'k')

        my_ylim = 1.025 * max(abs(res))

        ax[1].set_ylim( -my_ylim, my_ylim )


        # axes label

        cnt_freq = hlp_d['cnt_freq'] + freq_offset*1e6/1e12 # this needs a plus

        my_xlabel = "{0} + {1:12.6f} THz".format(meta_data['xlabel'], cnt_freq)

        ax[0].set_xlabel(my_xlabel)

        ax[1].set_xlabel(my_xlabel)




        fig.tight_layout()

        print('Center frequencies of peaks do not include offset')

        pars = print_params(hlp_d['fit_result'].params)

        # extract centers

        for n, k in enumerate(pars.keys()):

            if 'cnt' in k:

                ax[1].axvline(pars[k]['val'] - freq_offset, ls = '--', color = STD_RED)

                ax[1].text(pars[k]['val'], 0.90 * my_ylim - my_ylim/len(pars.keys()) * n, 
                           '{0:.2f} +/- {1:.2f}'.format(
                               pars[k]['val'] - freq_offset,
                               pars[k]['err']
                               ), fontsize = 8)



    return



