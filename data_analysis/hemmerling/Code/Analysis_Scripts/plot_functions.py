STD_BLUE   = '#1f77b4'
STD_RED    = '#d62728'
STD_ORANGE = '#ff7f0e'
STD_CYAN   = 'palegreen'
STD_BLACK  = 'k'

colordict  = { 0 : STD_BLUE, 1 : STD_RED, 2 : STD_ORANGE, 3 : STD_CYAN, 4 : STD_BLACK }
colordict2  = colordict
#colordict2 = { 1 : STD_BLUE, 0 : STD_RED, 2 : STD_ORANGE, 3 : STD_CYAN }


#######################################################################################################

def myc(k, order, color_scheme = 'multi'):

    if color_scheme == 'multi':
        if order == 0:
            
            return colordict[k % len(colordict.keys())]

        else:
            
            return colordict2[k % len(colordict.keys())]
    else:
        if order == 0:
            return STD_BLUE
        else:
            return STD_RED

    return

#######################################################################################################

def get_plot_label(meta_data):

    return "{0}: {1}".format(
                meta_data['title'], 
                meta_data['scan_stamp']
                )


##############################################################################################################

def my_scatter_plot(ax, x, y, color = None, label = None, connect = True):
    
    if color == None:

        if connect:
            ax.plot(   x, y, color = STD_BLUE, ls = '-', label = label)
        ax.scatter(x, y, marker = 'o', facecolor = 'w', edgecolor = STD_BLUE, zorder = 2)

    else:

        ax.plot(   x, y, color = color, label = label, ls = '-')
        ax.scatter(x, y, marker = 'o', facecolor = 'w', edgecolor = color, zorder = 2)

    return



