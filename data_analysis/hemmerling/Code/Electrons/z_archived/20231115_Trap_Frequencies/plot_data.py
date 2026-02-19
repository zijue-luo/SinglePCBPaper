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

    path = '/Users/boerge/Software/offline_electrons_data/'

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


date = '20231115'

time = '140201'


parameter = 'Tickle frequency (MHz)'

sp = np.array(get_data(date, time, 'arr_of_setpoints'))


tstamps = get_data(date, time, 'arr_of_timestamps', timestamps = True)


sig = []

trapped_no = []

for k in range(len(tstamps)):

    (h, edges) = np.histogram(tstamps[k], bins = np.linspace(0, 150, 200))

    sig.append(h)

sig = np.array(sig)

plt.pcolor(edges[:-1], sp, sig)

plt.xlabel('Time (us)')
plt.ylabel(parameter)

plt.clim(0,300)


offset_cnts = np.mean( sig[:, 10:40], axis = 1 )




plt.figure()


for k in [3, 12]:
    plt.plot(edges[:-1], sig[k, :], label = parameter + "{0:5.0f}".format(sp[k]))

plt.xlabel('Time (us)')

plt.ylabel('Electron MCP Counts #')

plt.legend()





plt.figure()

x = sp

y = []

print(len(sp))
print(sig.shape)
print(len(offset_cnts))

for k in range(sig.shape[0]):

    trapped_signal = np.max(sig[k, 150:]) + 0*1/ offset_cnts[k]

    y.append( trapped_signal )


y = np.array(y)



x = moving_average(x, n = 10)
y = moving_average(y, n = 10)


plt.plot(x, y, 'o-')

plt.xlabel(parameter)

plt.ylabel('Trapped Electron Counts (rel.)')

plt.show()



