import numpy as np
from configparser import ConfigParser
from os import path

import matplotlib
#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import lmfit
from lmfit import Minimizer, Parameters, report_fit
from prettytable import PrettyTable
import pickle
import scipy

import sys
sys.path.append("Analysis_Scripts/")
from constants import *
from energy_functions import get_scaled_dunham, get_energy
#from plotWindow import plotWindow

cnt_freq_00 = 1146.330000e12
cnt_freq_11 = 1145.330000e12 - 100e9





###########################################################################################
# Print fit results
###########################################################################################

def print_fit_results(fit_results):

    for n in range(len(fit_results)):

        print_scan_info(fit_results[n])

        # get the fitted channels
        channels = fit_results[n]['fit_results'].keys()

        for channel in channels:

            print('\nChannel: {0}'.format(channel))

            print_params(fit_results[n]['fit_results'][channel]['fit_result'].params)

    return


###############################################

def print_params(params):

    par_arr = {}

    print()

    for k in sorted(params.keys()):

        par = params[k]

        #print("{0:7} : {2:8.2f} <= {1:8.2f} +/- {4:5.2f} <= {3:8.2f}".format(k, par.value, par.min, par.max, par.stderr))

        print_single_param(params, k)

        par_arr[k] = {
                'min' : par.min,
                'max' : par.max,
                'val' : par.value,
                'err' : par.stderr
                }

    print()

    return par_arr


###############################################

def print_single_param(params, key):

    par = params[key]
    
    if par.stderr == None:
        err = 'N/A'
    else:
        err = '{0:8.3f}'.format(par.stderr)

    my_str = "{0:7} : {2:8.3f} <= {1:8.3f} +/- {4} <= {3:8.3f}".format(key, par.value, par.min, par.max, err)

    print(my_str)

    return my_str


###############################################

def get_single_param(params, par_key):

    par_arr = []
    for k in sorted(params.keys()):

        par = params[k]
        
        if par_key in k:

            par_arr.append(par.value)

    return np.array(sorted(par_arr))


###########################################################################################
# Print scan info
###########################################################################################

def print_scan_info(scan):

    m = scan['meta_data']

    print('Scan:  {0}/{1} Title: {2} Parameter: {3} = {4} '.format(m['date'], m['time'], m['title'], m['par'][1], m['par'][0]))

    return


###########################################################################################
# Print nested dictionary structure
###########################################################################################

def print_dict(d, tab = 0, level = 0):

    # prints nested dictionaries and their keys and data types

    key_list_unsorted = d.keys()

    # split key list in dictionaries and rest

    key_list_1 = []
    key_list_2 = []

    for k in key_list_unsorted:

        if isinstance(d[k], dict):
            key_list_1.append(k)
        else:
            key_list_2.append(k)

    key_list_1.sort()
    key_list_2.sort()

    key_list = []
    key_list.extend(key_list_1)
    key_list.extend(key_list_2)

    #print(key_list)
    if len(key_list) == 0:
        return

    longest_key = 2+max([len(str(x)) for x in key_list])

    if tab == 0:
        print("*" * 60)
    
    print("{0}{1}".format(" " * tab,"{"))
    for k in key_list:

        # check what type the entry is
        if isinstance(d[k], dict):
        
            print("\033[92m {2}{0:{3}} \033[0m: {1}".format("'" + str(k) + "'", type(d[k]), " " * tab, longest_key))

            if level == 0:
                print_dict(d[k], tab = tab + 3)

        elif isinstance(d[k], int) or isinstance(d[k], str) or isinstance(d[k], float):
            
            print("\033[91m {2}{0:{3}} \033[0m: {1}".format("'" + str(k) + "'", d[k], " " * tab, longest_key))
        
        elif isinstance(d[k], list):
            
            print("\033[91m {2}{0:{3}} \033[0m: {1}".format("'" + str(k) + "'", str(type(d[k])) + " len: " + str(len(d[k])), " " * tab, longest_key))

        elif isinstance(d[k], np.ndarray):
            
            print("\033[94m {2}{0:{3}} \033[0m: {1}".format("'" + str(k) + "'", str(type(d[k])) + " shape: " + str(d[k].shape), " " * tab, longest_key))

        else:
            
            print("\033[93m {2}{0:{3}} \033[0m: {1}".format("'" + str(k) + "'", type(d[k]), " " * tab, longest_key))


    print("{0}{1}".format(" " * tab,"}"))

    if tab == 0:
        print("*" * 60)
        print()
    
    return


###########################################################################################
# Print complete Dunham fit report
###########################################################################################

def create_latex_table(M, title, caption = ''):

    no_col = 0
    for x in range(len(M)):
        no_col = np.max([no_col, len(M[x])])

    s = '\\begin{table}[h]\n\\begin{tabular}{c' + 'r'*no_col + '}\n'

    s += '\\toprule\n' + title + '\\\\ \midrule \n'

    for x in range(len(M)):
        for y in range(len(M[x])):
            
            val = np.float(M[x][y])

            if np.abs(val) > 1.0:
                s += "& {0:2.3f}".format(M[x][y]) + ' '
            else:
                s += "& {0:2.3E}".format(M[x][y]) + ' '
            
        s += '\\\\\n'

    s += '\\bottomrule\n\\end{tabular}\caption{' + caption + '}\\end{table}\n'

    return s


def save_latex_table(filename, M, my_title, caption = ''):
    f = open(filename + '.tex', 'w')
    s = create_latex_table(M, my_title, caption = caption)
    f.write(s)
    f.close()


def plot_errors(comparison, my_max = 200):

    d = {'0' : {'35' : { 'm' : [], 'p' : [] }, '37' : { 'm' : [], 'p' : [] }}, '1' : {'35' : { 'm' : [], 'p' : [] }, '37' : { 'm' : [], 'p' : [] }}}


    for k in range(1, len(comparison)):
        
        fm = np.float(comparison[k][4])
        fp = np.float(comparison[k][5])

        nu = str(comparison[k][0])
        
        iso = str(comparison[k][7])

        d[nu][iso]['m'].append(fm)
        d[nu][iso]['p'].append(fp)

    for k1 in d.keys():
        for k2 in d[k1].keys():
            for k3 in d[k1][k2].keys():
                d[k1][k2][k3] = np.array(d[k1][k2][k3])

    sub = lambda k1, k2 : d[k1][k2]['m'] - d[k1][k2]['p']

    plt.figure()

    plt.subplot(2,1,1)
    plt.plot((d['0']['35']['m']*1e12 - cnt_freq_00)/1e9, sub('0', '35')*1e6, 'ob', label = '35')
    plt.plot((d['0']['37']['m']*1e12 - cnt_freq_00)/1e9, sub('0', '37')*1e6, 'or', label = '37')

    plt.xlabel("Meas. Frequency Detuning (MHz) + {0:2.6f} THz".format(cnt_freq_00/1e12))
    plt.ylabel("Meas. - Cal. Freq. (MHz)")

    if not my_max == -1:
        plt.ylim(-my_max, my_max)

    plt.legend()

    plt.tight_layout()

    plt.subplot(2,1,2)
    plt.plot((d['1']['35']['m']*1e12 - cnt_freq_11)/1e9, sub('1', '35')*1e6, 'ob', label = '35')
    plt.plot((d['1']['37']['m']*1e12 - cnt_freq_11)/1e9, sub('1', '37')*1e6, 'or', label = '37')
    
    plt.xlabel("Meas. Frequency Detuning (MHz) + {0:2.6f} THz".format(cnt_freq_11/1e12))
    plt.ylabel("Meas. - Cal. Freq. (MHz)")
    
    if not my_max == -1:
        plt.ylim(-my_max, my_max)
    
    plt.legend()
    
    plt.tight_layout()
    
   
    return


def compare_exp_theory(data, Ug, Ue, Dg, De):
    
    (Yg35, Ye35, Yg37, Ye37) = get_scaled_dunham(Ug, Ue, Dg, De)
    
    all_result = [['vg', 'Jg', 've', 'Je', 'measured', 'predicted', 'difference', 'isotope', 'line_type']]
            
    avg_err = 0.0        
             
    for k in range(len(data)):
        
        # d = [ [gs_v, gs_J, ex_v, ex_J, freq, 35/37] ]

        vg = data[k][0]
        Jg = data[k][1]
        ve = data[k][2]
        Je = data[k][3]
        meas_freq = data[k][4]
        isotope = data[k][5]
        
        if isotope == 35:
            eng = get_energy(Yg35, Ye35, vg, Jg, ve, Je)
        elif isotope == 37:
            eng = get_energy(Yg37, Ye37, vg, Jg, ve, Je)
        else:
            print('Error')
            asd

        my_type = get_line_type(Jg, Je) 
        
        hlp = [vg, Jg, ve, Je, meas_freq/1e12, eng/1e12, (meas_freq - eng)/1e6, isotope, my_type]

        avg_err += np.abs((meas_freq - eng)/1e6)

        all_result.append(hlp)

    avg_err /= len(data)

    return (all_result, avg_err)


def make_report(data, Ug, Ue, Dg, De, latex_dunham_file = None, latex_prediction_file = None): #, vmax = 1, Jmax = 1, save_filename = 'report.txt'):
    
    (Yg35, Ye35, Yg37, Ye37) = get_scaled_dunham(Ug, Ue, Dg, De)
    
    # save all Dunham matrices
    save_matrices({ 'Ug' : Ug, 'Dg' : Dg, 'Ue' : Ue, 'De' : De, 'Yg35' : Yg35, 'Ye35' : Ye35, 'Yg37' : Yg37, 'Ye37' : Ye37 }, 'dunham_matrices_fit')


    # Latex output matrices

    if not latex_dunham_file is None:
           
       save_latex_table(latex_dunham_file + '_Ue', Ue, '$U (A^1\Pi)$', caption = 'Mass-reduced Dunham A-state. Units (1/cm).')
       save_latex_table(latex_dunham_file + '_De', De, '$\Delta_\\textrm{Cl} (A^1\Pi)$', caption = 'Born-Oppenheimer Corrections A-State, Cl. Unit (1/cm).') 
       save_latex_table(latex_dunham_file + '_Ye35', Ye35, '$Y_{35} (A^1\Pi)$', caption = 'Dunham A-state, Cl-35. Unit (1/cm).') 
       save_latex_table(latex_dunham_file + '_Ye37', Ye37, '$Y_{37} (A^1\Pi)$', caption = 'Dunham A-state, Cl-37. Unit (1/cm).') 
       
       save_latex_table(latex_dunham_file + '_Ug', Ug, '$U (X^1\Sigma)$', caption = 'Mass-reduced Dunham X-state. Unit (1/cm).') 
       save_latex_table(latex_dunham_file + '_Dg', Dg, '$\Delta_\\textrm{Cl} (X^1\Sigma)$', caption = 'Born-Oppenheimer Corrections X-State, Cl. Unit (1/cm).') 
       save_latex_table(latex_dunham_file + '_Yg35', Yg35, '$Y_{35} (X^1\Sigma)$', caption = 'Dunham X-state, Cl-35. Unit (1/cm).') 
       save_latex_table(latex_dunham_file + '_Yg37', Yg37, '$Y_{37} (X^1\Sigma)$', caption = 'Dunham X-state, Cl-37. Unit (1/cm).') 
       

    print()
    print("*"*80)
    print("Report")
    print("*"*80)

    # table to compare measured and calculated lines 
    tab_title = ['vg', 'Jg', 've', 'Je', 'Exp. Freq. (THz)', 'Cal. Freq. (THz)', 'Diff. (MHz)', 'AlCl', 'Type']
    t = PrettyTable(tab_title)

    s = '\\begin{table}\n\\begin{tabu}{' + 'r'*(len(tab_title)) + '}\n'
    s += '\\toprule\n' # + title + '\\\\ \midrule \n'
    s += ' & '.join(tab_title) + '\\\\ \\midrule \n'

    t.float_format['Exp. Freq. (THz)'] = ".6"
    t.float_format['Cal. Freq. (THz)'] = ".6"
    t.float_format['Diff. (MHz)'] = "8.2"

    # compare experiment and theory
    all_result, avg_err = compare_exp_theory(data, Ug, Ue, Dg, De)

    for k in range(len(data)):
        
        hlp = all_result[k+1]
        t.add_row(hlp)

        # add to latex string
        hlp[4] = "{0:2.6f}".format(hlp[4])
        hlp[5] = "{0:2.6f}".format(hlp[5])
        hlp[6] = "{0:0.1f}".format(hlp[6])

        my_type = hlp[-1]

        if my_type == 'R':
            s += "\\rowfont{\\color{red}}" + " & ".join(list(map(str, hlp))) + '\\\\\n'
        elif my_type == 'P':
            s += "\\rowfont{\\color{blue}}" + " & ".join(list(map(str, hlp))) + '\\\\\n'
        else:
            s += " & ".join(list(map(str, hlp))) + '\\\\\n'

    t.add_row(['-']*6 + ['--------'] + ['-']*2)
    t.add_row(['','','','','','','avg:' + "{0:2.2f}".format(avg_err),'',''])
    
    print(t)

    s += '\\midrule\n'
    s += ' & '* 6 + ' avg: {0:2.1f}\\\\\n'.format(avg_err)
    s += '\\bottomrule\n\\end{tabu}\caption{Comparison of measured and predicted (calculated) transition frequencies.}\\end{table}\n'

    if not latex_prediction_file is None:
        f = open(latex_prediction_file + '.tex', 'w')
        f.write(s)
        f.close()

    return all_result

###############################################
# Save Dunham matrix to file
###############################################

def save_matrices(Ms, filename):

    # Save matrices in one file
    f = open(filename + '.txt', 'w')

    for k in Ms.keys():
        M = Ms[k]
        f.write(k + " = [\n")
        for k in range(len(M)):
            f.write("[ ")
            f.write(",".join(map(str, M[k])))
            f.write(" ],\n")
        f.write("]\n\n")

    f.close()


    # Save each matrices in pickle file

    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(Ms, f)

    return

###############################################
# Save line centers in csv file
###############################################

def spectrum2csv(data_original, save_filename = 'data_lines.txt', add_q = False, q_lines = []):

    # table to compare measured and calculated lines

    
    #t  = ",".join(['vg', 'Jg', 've', 'Je', 'Measured Fitted Freq. (THz)', 'Calculated Freq. (THz)', 'Difference (MHz)', 'AlCl', 'Type'])
    t  = ",".join(['vg', 'Jg', 've', 'Je', 'Measured Fitted Freq. (THz)', 'AlCl', 'Type'])
    t += "\n"

    if add_q:
        data = data_original.copy()
        data.append(q_lines[0])
        data.append(q_lines[1])
    else:
        data = data_original

    #print(data)

    #print(data)
    for k in range(len(data)):
        
        # d = [ [gs_v, gs_J, ex_v, ex_J, freq, 35/37] ]

        vg = data[k][0]
        Jg = data[k][1]
        ve = data[k][2]
        Je = data[k][3]
        meas_freq = data[k][4]
        isotope = data[k][5]

        my_type = get_line_type(Jg, Je) 
        
        #t += "{0}, {1}, {2}, {3}, {4:6.6f}, {5:6.6f}, {6:6.2f}, {7}, {8}\n".format(vg, Jg, ve, Je, meas_freq/1e12, eng/1e12, (eng - meas_freq)/1e6, isotope, my_type)
        t += "{0}, {1}, {2}, {3}, {4:6.6f}, {5}, {6}\n".format(vg, Jg, ve, Je, meas_freq/1e12, isotope, my_type)


    f = open(save_filename, 'w')
    f.write(t)
    f.close()
    
    #print(t)


###############################################

def print_matrix(U, txt = None):

    if not txt is None:
        print(txt + ' = [')
    else:
        print('[')
    for k in range(len(U)):
        print(U[k], end = '')
        #for l in range(len(U[k])):
        #    print(U[k][l])
        if k < len(U):
            print(',')

    print(']\n')


###############################################

def print_dunham(U):

    print()
    print('Dunham coefficients')
    print('-'*30)
    for v in range(len(U)):
        for l in range(len(U[v])):

            val = U[v][l]

            if np.abs(val) > 1.0:
                print("U_{0}{1} = {2:15.6f}".format(v, l, val))
            else:
                print("U_{0}{1} = {2:15.6e}".format(v, l, val))

    print()
    return



####################################

def get_line_type(Jg, Je):
    
    if Je == Jg:
        my_type = 'Q'
    elif Je == Jg - 1:
        my_type = 'P'
    elif Je == Jg + 1:
        my_type = 'R'

    return my_type

def get_line_color(Jg, Je):
    
    color_scheme = {'P' : 'b', 'Q' : 'k', 'R' : 'r'}
    
    return color_scheme[get_line_type(Jg, Je)]


