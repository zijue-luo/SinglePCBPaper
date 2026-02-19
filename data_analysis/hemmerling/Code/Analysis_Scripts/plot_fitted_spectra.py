################################################################
# General parameters
################################################################

from latex     import *
from constants import *


import numpy as np
import matplotlib.pyplot as plt

from plot_functions    import my_scatter_plot, get_plot_label, myc
from report_functions  import print_dict, print_fit_results


################################################################

def plot_fitted_spectra(fit_results, channels = [0], fontsize = 10, y_offsets = None, x_offset = 0.0, plot_all_in_one = True, color_scheme = 'single'):

    axs  = {}
    figs = {}

    for channel in channels:

        axs[channel] = []
        figs[channel] = []
        
        if plot_all_in_one:
            offset_freq = fit_results[0]['fit_results'][channel]['cnt_freq'] * 1e12/1e6 + x_offset
   
            print("Frequency: {0:.6f} THz".format(offset_freq * 1e6/1e12))
            print("Wavelength: {0:.6f} nm".format(3e8/(offset_freq * 1e6) / 1e-9))
        else:
            offset_freq = x_offset

        if plot_all_in_one:
            
            (fig, ax) = plt.subplots(1, 1, figsize = (8, 8) , layout = 'tight') #constrained_layout = True, 
        
            meta_data = fit_results[0]['meta_data']
        
            fig.suptitle("{0} - Channel: {1}".format(
                        meta_data['title'],
                        channel
                        )
                        , fontsize = fontsize)

            axs[channel]  = [ax]
            figs[channel] = [figs]

        else:

            for k in range(len(fit_results)):
                
                (fig, ax) = plt.subplots(1, 1, figsize = (8, 8) , layout = 'tight') #constrained_layout = True, 
        
                meta_data = fit_results[0]['meta_data']
        
                fig.suptitle("{0} - Channel: {1}".format(
                            meta_data['title'],
                            channel
                            )
                            , fontsize = fontsize)

                axs[channel].append(ax)
                figs[channel].append(fig)


        ################################
        # Toggle over all fit_results
        ################################

        for k in range(len(fit_results)):
        
            d = fit_results[k]
    
            meta_data = d['meta_data']
    
            cnt = d['fit_results'][channel]['cnt_freq']
        
            cnt = cnt * 1e12/1e6
            
            if plot_all_in_one:
                x_plot_offset = cnt - offset_freq
            else:
                x_plot_offset = - offset_freq

            # data
            x  = d['fit_results'][channel]['x_data']
            y  = d['fit_results'][channel]['y_data']
            
            # fits
            xf = d['fit_results'][channel]['x_fit']
            yf = d['fit_results'][channel]['y_fit']
           
            if not y_offsets == None:
                y_offset = y_offsets[k]
            else:
                y_offset = 0.0

            


            ###############
            # Plotting
            ###############

            if plot_all_in_one:
                ax  = axs[channel][0]
                fig = figs[channel][0]
            else:
                ax  = axs[channel][k]
                fig = figs[channel][k]

            # data
            my_scatter_plot(ax, x + x_plot_offset, y + y_offset, color = myc(k, 0, color_scheme = color_scheme))

            # fit
            ax.plot(xf + x_plot_offset, yf + y_offset, color = myc(k, 1, color_scheme = color_scheme), label = '{0} {1} Offset: {2} MHz'.format(
                get_plot_label(meta_data),
                d['meta_data']['scan_info'],
                x_offset)
                )

            ax.set_xlim([min(x + x_plot_offset), max(x + x_plot_offset)])

                        
            if plot_all_in_one:
                ax.set_xlabel('Detuning (MHz) + {0:.6f} THz'.format((offset_freq)/1e12*1e6))
            else:
                ax.set_xlabel('Detuning (MHz) + {0:.6f} THz'.format((cnt + offset_freq)/1e12*1e6))

            ax.set_ylabel('Signal (a.u.)')
        
            ax.legend(fontsize = 7)
   

        
    return (figs, axs)


