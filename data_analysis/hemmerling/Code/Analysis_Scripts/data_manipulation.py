import numpy as np
import matplotlib.pyplot as plt
from math_functions import moving_average
from scipy.fft      import fft, ifft
from scipy          import fftpack
from scipy.ndimage  import gaussian_filter

from report_functions import print_dict

#######################################################

def get_data(s, channel):

    times = s['data'][channel]['times']
    freq  = s['data'][channel]['freq']
    y2d   = s['data'][channel]['y2d']

    #print_dict(s)

    cnt   = s['data'][channel]['cnt_freq']

    return (times, freq, y2d, cnt)


#######################################################

def remove_high_frequency_noise(t, y, freq_cut_off = 0):
    
    # low pass filters data

    if freq_cut_off > 0:

        #y = np.sin(2*np.pi *2.7e3 * t * 1e-3)

        y_fft = fftpack.fft(y, t.size)
        freq = fftpack.fftfreq(t.size, max(t)/len(t))
 
        # low pass filter
        y_fft_filtered = y_fft
        y_fft_filtered[freq > freq_cut_off] = 0.0
        y_fft_filtered[freq < -freq_cut_off] = 0.0

        y_filtered = np.real(fftpack.ifft(y_fft))
        
        #plt.figure()
        #plt.subplot(2,1,1)
        #plt.plot(t, y)
        #plt.plot(t, y_filtered)
        #plt.subplot(2,1,2)
        #plt.plot(freq, abs(y_fft))
        #plt.plot(freq, abs(y_fft_filtered))
        #plt.show()
        #asd

        return y_filtered

    else:

        return y



#######################################################

def threshold_filter(d, threshold, width = 1):

    hlp = np.copy(d)

    for k in range(d.shape[0]):
        ind = np.where(d[k, :] > threshold)

        ind = ind[0]

        # remove offset
        hlp[k, :] = hlp[k, :] - np.mean(d[k, 0:10])


        try:
            cnt_ind = int(np.mean(ind))
        except:
            cnt_ind = 0

        hlp[k, 0:cnt_ind - width] = 0.0
        hlp[k, cnt_ind + width:] = 0.0

    # this just returns the last center index

    return (hlp, cnt_ind)



#######################################################

def gaussian_filter_data(y2d, filter_edge = 0):
    
    # applies a gaussian filter to the 2D data array

    if filter_edge > 0:
        return gaussian_filter(y2d, sigma = filter_edge)
    else:
        return y2d


def filter_data(y2d, filter_edge = 0, plot = False):
    
    # applies a low pass filter to the data to get rid of high frequency noise

    #y2d = np.transpose(y2d)

    f_y2d = fft(y2d)

    # low pass filter
    # this filteres each time trace
    f_y2d[:, filter_edge:-filter_edge] = 0.0

    y2d_filtered = np.real(ifft(f_y2d))


    if filter_edge == 0:
        y2d_filtered = y2d

    if plot:
        plt.figure()
        plt.subplot(3,1,1)

        plt.pcolor(y2d)
        
        plt.subplot(3,1,2)
        plt.pcolor(np.log(np.abs(f_y2d)))

        plt.subplot(3,1,3)
        plt.pcolor(y2d_filtered)


        plt.figure()
        
        plt.plot(y2d[65, :], 'b-')
        plt.plot(y2d_filtered[65, :], 'r.-')
        #plt.plot(moving_average(y2d[65, :], 10), 'g.-')


        plt.show()

    return y2d_filtered



def integrate_freq_time(t, x, y2d, cut_ind, scale_factor = 1.0):

    t1_cut = cut_ind[0][0]
    t2_cut = cut_ind[0][1]

    x1_cut = cut_ind[1][0]
    x2_cut = cut_ind[1][1]

    # integrate over the frequency range 
    y_t = np.mean( y2d[x1_cut:x2_cut, :], axis = 0 )

    # integrate over the time range
    y_f = np.mean( y2d[:, t1_cut:t2_cut], axis = 1 )

    return (y_f, y_t)



def get_time_index(t, ts):

    dt = np.abs(t[1] - t[0])

    t_cut = np.where( np.abs(t - ts) <= dt )[0][0]

    return (t_cut)


def get_freq_index(v, vs):

    dv = np.abs(v[1] - v[0])

    # in case the first two points are the same
    # needs to be improved
    if dv == 0:
        dv = np.abs(v[2] - v[1])

    try:
        v_cut = np.where( np.abs(v - vs) <= dv )[0][0]
    except:
        print("Can't find frequency index: min {0} - max {1} - search for {2} MHz, dv = {3} MHz".format(min(v), max(v), vs/1e6, dv))
        print(v)
        print( np.abs(v - vs)/1e6  )
        print(vs)
        v_cut = 0

    return (v_cut)


def get_time_indeces_raw(t, arr):

    t1 = arr[0]
    t2 = arr[1]
    
    dt = np.abs(t[1] - t[0])

    t1_cut = np.where( np.abs(t - t1) <= dt )[0][0]
    t2_cut = np.where( np.abs(t - t2) <= dt )[0][0]

    return (t1_cut, t2_cut)



def get_time_indeces(t, t_arr):

    t1 = t_arr[0]
    t2 = t_arr[1]

    t1_cut = get_time_index(t, t1)
    t2_cut = get_time_index(t, t2)

    return (t1_cut, t2_cut)



def get_freq_indeces(v, opts, scale_factor = 1.0):

    if opts['freq_integrate'] == 'all':

        v1_cut = 0
        v2_cut = len(v) - 1
       
    else:

        # also used for other scans than frequency
        v1 = opts['freq_integrate'][0]*scale_factor
        v2 = opts['freq_integrate'][1]*scale_factor

        v1_cut = get_freq_index(v, v1)
        v2_cut = get_freq_index(v, v2)



    return (v1_cut, v2_cut)



def get_all_indeces(t, v, opts, scale_factor = 1.0):

    (t1_cut, t2_cut) = get_time_indeces(t, opts)
    (v1_cut, v2_cut) = get_freq_indeces(v, opts, scale_factor = scale_factor)

    inds = [[t1_cut, t2_cut], [v1_cut, v2_cut]]

    return inds


