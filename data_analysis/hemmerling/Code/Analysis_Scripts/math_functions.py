import numpy as np


####################################################
# Cross section functions
####################################################

def get_number_of_molecules(I, I0, lamb):

    T = 10
    delta = 0.0

    my_gamma = 2*np.pi/lamb * np.sqrt(2*kB*T/(massAl+massCl_35))

    G_D = 1/(np.sqrt(np.pi) * my_gamma) * np.exp(-delta**2/my_gamma**2) 

    sigma = 1.0/4.0 * lamb**2 * my_gamma * G_D

    z = 2e-2 # cell length

    n = -1.0/(sigma * z) * np.log(I/I0)

    return n


####################################
# sort 2D data
####################################

def datasort(arr):
    
    # for 1D array
    if len(arr.shape)==1:
        return np.sort(arr)

    # for 2D array
    if len(arr.shape)==2:     

        ind = np.argsort(arr[0, :])
        arr[0, :] = arr[0, ind]
        arr[1, :] = arr[1, ind]

        return arr


####################################
# averages data
####################################

def av(arr, no_of_avg, remove_nan = False):
    
    if not remove_nan:
    
        # for 1D array
        if len(arr.shape) == 1:
            #hlp = np.zeros([int(arr.shape[0]/no_of_avg)])

            #for k in range(len(hlp)):
            #    for m in range(no_of_avg):
            #        hlp[k] += arr[no_of_avg*k + m]

            #return hlp/no_of_avg

            hlp = np.reshape(arr, [int(arr.shape[0]/no_of_avg), no_of_avg])
       
            return np.mean(hlp, axis = 1)

        # for 2D array
        if len(arr.shape) == 2:     

            hlp = np.zeros([int(arr.shape[0]/no_of_avg), arr.shape[1]])

            for k in range(len(hlp)):
                for m in range(no_of_avg):
                    hlp[k] += arr[no_of_avg*k + m, :]

            return hlp/no_of_avg
    else:

        # for 2D array
        if len(arr.shape) == 2:     

            hlp = np.zeros([int(arr.shape[0]/no_of_avg), arr.shape[1]])

            for k in range(len(hlp)):
                
                hlp_nan = []

                for m in range(no_of_avg):                    
                
                    hlp_nan.append(arr[no_of_avg*k + m, :])

                # check case if all points are np.nan since np.nan throws an error when averaging over a nan-only array
                if not np.all(np.isnan(hlp_nan)):
                    hlp[k, :] = np.nanmean(hlp_nan, axis = 0)
                else:
                    #replace with np.nan
                    hlp[k, :] = np.array([np.nan]*len(hlp_nan[0]))

            return hlp
        
        elif len(arr.shape) == 1:
            
            hlp = np.reshape(arr, [int(arr.shape[0]/no_of_avg), no_of_avg])
            
            if not np.all(np.isnan(hlp)):
                hlp = np.nanmean(hlp, axis = 1)
            else:
                hlp = np.array([np.nan]*len(hlp))

            return hlp

        else:
            print('Only 2D array available')
            return


####################################
# returns moving average
####################################

def moving_average(a, n = 3, pad = True):
    
    if not n == 0:

        ret = np.cumsum(a, dtype = float)
        ret[n:] = ret[n:] - ret[:-n]

        if pad:
            result = np.append((n-1) * [np.nan], ret[n - 1:])
        else:
            result = re[n - 1, :]

        return result / n

    else:

        return a


####################################
# combines data sets
####################################

def combine_data(x_arr, y_arr, arr = [], sort = True):

    # concatenates data arrays

    if len(arr) == 0:
        arr = range(len(x_arr))

    x = []
    y = []
    for n in arr:

        x.extend(x_arr[n])
        y.extend(y_arr[n])

    x = np.array(x)
    y = np.array(y)

    if sort:

        ind = np.argsort(x)

        x = x[ind]
        y = y[ind]

    return (x, y)


