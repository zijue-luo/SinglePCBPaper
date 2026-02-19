import numpy as np

import matplotlib.pyplot as plt

import matplotlib


font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)


####################################
# returns moving average
####################################

def moving_average(a, n = 3) :
    
    if not n == 0:
        ret = np.cumsum(a, dtype = float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    else:
        return a



def get_data(date, time, filename, timestamps = False):

    #path = '/Users/boerge/Software/offline_electrons_data/'
    path = '/home/electrons/software/data/'

    f = open(path + date + '/' + date + '_' + time + '_' + filename)

    lines = f.readlines()

    result = []
    for k in range(len(lines)):

        if not timestamps:
            
            result.append( float(lines[k].strip()) )

        else:

            result.append( [ float(x) for x in lines[k][1:-2].split(',') ] )

    f.close()

    return result


date = '20231211'

time = '163614'


parameter = 'Tickle frequency (MHz)'

sp = np.array(get_data(date, time, 'arr_of_setpoints'))

d = []
#d.append(np.array(get_data(date, '145309', 'spectrum')))
#d.append(np.array(get_data(date, '154311', 'spectrum')))
#d.append(np.array(get_data(date, '154836', 'spectrum')))
#d.append(np.array(get_data(date, '155628', 'spectrum')))

#my_label = ['0', '-10', '0', '0, -0.59']

d.append(np.array(get_data(date, '163246', 'spectrum')))
d.append(np.array(get_data(date, '163433', 'spectrum')))
d.append(np.array(get_data(date, '163614', 'spectrum')))



my_label = ['.69', '.5', '.4']


no = 5

d[0] /= 10.0

for k in range(len(d)):

    x    = moving_average(sp, n = no)
    d[k] = moving_average(d[k], n = no)

plt.figure()

for k in range(len(d)):
    plt.plot(x, d[k], '.-', label = my_label[k])

plt.xlabel(parameter)

plt.ylabel('Trapped Electron Counts (rel.)')

plt.legend()

plt.show()



