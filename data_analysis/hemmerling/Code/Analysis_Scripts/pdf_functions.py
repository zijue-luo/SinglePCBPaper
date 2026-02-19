import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def save_all_plots(filename):
   
    pp = PdfPages(filename + '.pdf')
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
          fig.savefig(pp, format='pdf')
    pp.close()

    return


