import numpy as np
import copy
import pickle

from prettytable import PrettyTable

from pdf_functions    import save_all_plots
from report_functions import print_dict



######################################################################

def load_pickle_file(filename):

    with open('results/{0}.pckl'.format(filename), 'rb') as f:

        data_raw = pickle.load(f)

    return data_raw


######################################################################

def save_pickle_file(filename, arr):

    with open('results/' + filename + '.pckl', 'wb') as f:
    
        pickle.dump(arr, f)
    
    f.close()

    return


######################################################################

def create_scan_list_str(d):

    scan_list = ""

    if not isinstance(d['time'], int):

        for k in range(len(d['time'])):
            scan_list += "{0}/{1},".format(d['date'][k], d['time'][k])

        scan_list = scan_list.strip(",")

    else:
        scan_list = "{0}/{1}".format(d['date'], d['time'])

    return scan_list


######################################################################

def save_fit_results(d_arr, info):

    ##################################################################
    # Save data and figures
    ##################################################################
   
    print()
    print("Saving to txt files ...\n")

    # Save all open figures
    save_all_plots('{0}/{1}'.format('results', info['pdf_filename']))
 
    # Create overview table
    t = PrettyTable(['Scan Date/Time(s)', 'Scan Info', 'Cnt freq (THz)', 'Scan min (MHz)', 'Scan max (MHz)'])

    t.float_format['Cnt freq (THz)'] = ".6"
    t.float_format['Scan min (MHz)'] = ".2"
    t.float_format['Scan max (MHz)'] = ".2"

    for k in range(len(d_arr)):
        
        d = d_arr[k]

        print("Scan {0} - Info: {1}".format(create_scan_list_str(d['meta_data']), d['meta_data']['scan_info']))

        ###########################################################
        # Save all data
        ###########################################################
       
        for channel in d_arr[k]['fit_results'].keys():
        
            df = d_arr[k]['fit_results'][channel]
        
            x_data      = df['x_data']
            y_data      = df['y_data']
            
            cnt_freq    = df['cnt_freq']
    
            x_fit       = df['x_fit']
            y_fit       = df['y_fit']
    
            par         = df['fit_result'].params

            filename    = 'results/' + info['csv_filename'] + '_ch' + str(channel) + '_' + str(k)


            ###########################################################
            # Save relative frequencies
            ###########################################################

            data = np.array([x_data, y_data])
    
            np.savetxt(filename + '_relative_frequencies.txt', np.transpose(data), delimiter = ',')


            ###########################################################
            # Save absolute frequencies
            ###########################################################

            data = np.array([x_data*1e6 + cnt_freq*1e12, y_data])
    
            np.savetxt(filename + '_absolute_frequencies.txt', np.transpose(data), delimiter = ',')


            ###########################################################
            # Save fit
            ###########################################################

            data = np.array([x_fit, y_fit])
    
            np.savetxt(filename + '_fit.txt', np.transpose(data), delimiter = ',')


            ###########################################################
            # Save all meta data
            ###########################################################
    
            f = open(filename + '_info.txt', 'w')
    
            f.write('Info on data files\n\n\
    --------------------------------------------------------\n\
    Scan date/time stamp: {0}\n\n\
    --------------------------------------------------------\n\
    Scan info:            {1}\n\n\
    Parameter:            {2}{3}\n\n\
    Absolute offset frequency : {4:10.6f} THz\n\n\
    Fit results:\n\
    Fitting function: offset + sum_i ( ampl_i * w_i**2/((x - cnt_i)**2 + w_i**2) )\n\n\
    ampl_i : amplitude\n\
    w_i    : width\n\
    cnt_i  : relative peak center\n\n\
            '.format( \
                create_scan_list_str(d['meta_data']),
                d['meta_data']['scan_info'],
                d['meta_data']['par'][0],
                d['meta_data']['par'][1],
                d['data'][channel]['cnt_freq']
            ))
    
            fit_str = ""
            fit_w_str = ""
            fit_a_str = ""
            fit_c_str = ""
    
            for mykey in par.keys():
    
                if mykey == 'offset':
    
                    fit_str += "{0:10} : {1:8.4f}\n".format(mykey, par[mykey].value)
    
                elif 'ampl_' in mykey:
    
                    fit_a_str += "{0:10} : {1:8.4f}\n".format(mykey, par[mykey].value)
    
                elif 'width_' in mykey:
    
                    fit_w_str += "{0:10} : {1:8.2f} MHz\n".format(mykey, par[mykey].value)
    
                elif 'cnt_' in mykey:
    
                    fit_c_str += "{0:10} : {1:8.2f} MHz\n".format(mykey, par[mykey].value)
    
            f.write(fit_str + "\n" + fit_a_str + "\n" + fit_w_str + "\n" + fit_c_str)
    
            f.close()

            
        t.add_row([create_scan_list_str(d['meta_data']), d['meta_data']['scan_info'], cnt_freq, min(x_data), max(x_data)])


    # Print pretty table
    print()
    print(t)

    print("\n")

    ##################################
    # Save table in file
    ##################################

    f = open(filename[:-2] + 'overview.txt', 'w')
    f.write(t.get_string())
    f.close()


    return


