import pylab

fontsize = 15

params = {
            'text.usetex'       : True,     # for latex fonts
            'legend.fontsize'   : fontsize,
            'figure.figsize'    : (12, 5),
            'axes.labelsize'    : fontsize,
            'axes.titlesize'    : fontsize,
            'xtick.labelsize'   : fontsize,
            'ytick.labelsize'   : fontsize
            }

pylab.rcParams.update(params)



