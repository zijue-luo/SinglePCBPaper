##############################################
# Read in target scan image data
##############################################

import numpy as np
import matplotlib.pyplot as plt

from data_manipulation import get_time_indeces, get_time_index, get_freq_indeces, filter_data, gaussian_filter_data
from math_functions import moving_average


def integrate_image_absorption(img, inter_x, inter_y, t_start, t_stop):

    abs_img = np.zeros([len(inter_x), len(inter_y)])

    for nx in range(len(inter_x)):
        for ny in range(len(inter_y)):
    
            lin_ind = nx * len(inter_y) + ny
            
            absorption = np.mean(img[lin_ind, t_start:t_stop])
            
            #absorption = np.min(img[lin_ind, t_start:t_stop])
            
            abs_img[nx, ny] = np.abs(absorption)

    return np.transpose(abs_img)


def integrate_patches(img, inter_x, inter_y, t, patches):

    abs_traces = []

    for k in range(len(patches)):
        abs_traces.append(np.zeros(len(t)))

    for k in range(len(patches)):

        xmin = patches[k][0][0] - patches[k][1][0]
        xmax = patches[k][0][0] + patches[k][1][0]
        ymin = patches[k][0][1] - patches[k][1][1] 
        ymax = patches[k][0][1] + patches[k][1][1]

        cnt = 0
        for nx in range(len(inter_x)):
            for ny in range(len(inter_y)):

                if      ((xmin <= inter_x[nx]) and (inter_x[nx] <= xmax)) \
                    and ((ymin <= inter_y[ny]) and (inter_y[ny] <= ymax)):
                
                    lin_ind = nx * len(inter_y) + ny
            
                    abs_traces[k] += img[lin_ind, :]
                    cnt += 1

        if not cnt == 0:
            abs_traces[k] /= cnt

    return abs_traces


def plot_rectangle(arr, c):

    xmin = arr[0][0] - arr[1][0]
    xmax = arr[0][0] + arr[1][0]
    ymin = arr[0][1] - arr[1][1] 
    ymax = arr[0][1] + arr[1][1]
   
    plt.plot([xmin, xmax], [ymin, ymin], c)
    plt.plot([xmin, xmax], [ymax, ymax], c)
    plt.plot([xmin, xmin], [ymin, ymax], c)
    plt.plot([xmax, xmax], [ymin, ymax], c)




###################################################################################

def plot_target_img(d, opts, title, do_plot = True):

    t          = d['times']
    abs_traces = d['channels'][opts['channel']]

    he_flow = d['conf']['he_flow']['val']

    #for k in range(len(abs_traces)):

    #    abs_traces[k, :] = moving_average(abs_traces[k, :], n = 35)


    ##################################################
    # Calculate integration limits
    ##################################################

    (t1_cut, t2_cut) = get_time_indeces(t, opts)


    ##################################################
    # Integrate absorption
    ##################################################

    # get absorption image
    img           = integrate_image_absorption(abs_traces, d['x_pos'], d['y_pos'], t1_cut, t2_cut)

    # get average time trace for each patch
    target_yields = integrate_patches(abs_traces, d['x_pos'], d['y_pos'], t, opts['img_integrate'])
    
    #img = gaussian_filter_data(img, filter_edge = 0.75)

    ##################################################
    # Plot data
    ##################################################

    # plot absorption image
    if do_plot:
        plt.figure(figsize = (10,6))
        plt.subplot(1,2,1)
        plt.pcolor(d['x_pos'], d['y_pos'], img)#, shading = 'auto')
        #plt.colorbar()
   
        my_color_arr = 'rkbyg'

        for k in range(len(opts['img_integrate'])):
            plot_rectangle(opts['img_integrate'][k], my_color_arr[k])

        plt.xlabel('x pos')
        plt.ylabel('y pos')

        plt.gca().invert_yaxis()

        #plt.clim(np.mean(np.mean(img)) * 0.9, np.mean(np.mean(img)) * 1.1)
        
        plt.clim(0, 0.02)

        plt.colorbar()

        ax = plt.subplot(1,2,2)

    absorption_yields = []

    # plot average time traces
    for k in range(len(target_yields)):
        
        # sig == 0 : 0%
        # sig == -offset : 100%
        # sig/offset - 1

        #absorption = 100 * (-target_yields[k])/(d['no_absorption_level'][k] - opts['photodiode_offset'])

        absorption = target_yields[k]
        
        absorption_yields.append(np.mean(absorption[t1_cut:t2_cut]))

        if do_plot:
            plt.plot(t, absorption, color = my_color_arr[k], label = opts['img_integrate'][k][2])
   
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            
            #plt.ylim(1.1*np.min(absorption), np.mean(absorption[0:10]))
            
            #plt.ylim(np.min(absorption), 0.0)
            #plt.ylim(-5.0,100.5)
            #plt.ylabel('Absorption (%)')
            plt.ylabel('Absorption (a.u.)')

    if do_plot:
        plt.legend()
        plt.title("{2} {0}_{1} He flow: {3} sccm".format(d['date'], d['timestamp'], title, he_flow))

    return (img, absorption_yields)


