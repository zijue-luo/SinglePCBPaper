import numpy as np

import matplotlib.pyplot as plt

import matplotlib


font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)


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


date = '20231113'

time = '145258'
time = '164343'


ts = np.array(get_data(date, time, 'arr_of_extraction_times'))


tstamps = get_data(date, time, 'arr_of_timestamps', timestamps = True)


sig = []

trapped_no = []

for k in range(len(tstamps)):

    (h, edges) = np.histogram(tstamps[k], bins = np.linspace(0, 10000, 500))

    sig.append(h)

    trapped_no.append(max(h[10:]))

sig = np.transpose(np.array(sig))

print(sig.shape)

plt.pcolor(ts, edges[:-1], sig)

plt.ylabel('Detection time (us)')
plt.xlabel('Extraction time (us)')



plt.figure()


for k in [3, 12]:
    plt.plot(edges[:-1]/1e3, sig[:, k], label = "Extraction pulse at {0:5.0f} us".format(ts[k]))

plt.xlabel('Detection time (ms)')

plt.ylabel('Electron MCP Counts #')

plt.legend()



plt.figure()

plt.plot(ts[1:]/1e3, trapped_no[1:], 'o-')
#plt.semilogy(ts[1:]/1e3, trapped_no[1:], 'o-')

plt.xlabel('Detection time (ms)')

plt.ylabel('Electron MCP Counts #')

plt.show()



