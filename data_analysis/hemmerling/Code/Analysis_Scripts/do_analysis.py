from read_molecule_scan import *
from plot_molecule_scan import *

from pdf_functions import save_all_plots
from report_functions import print_dict

import copy
import pickle
import dill

from matplotlib.backends.backend_pdf import PdfPages



def do_analysis(scan_list, single_scan, filename, save_plots = True, save_data = True):

    ###############################################
    # Make scans array
    ###############################################
    
    scans = []
    
    for k in range(len(scan_list)):
    
        single_scan['meta_data']['time']            = scan_list[k]['time']
        single_scan['meta_data']['date']            = scan_list[k]['date']
        single_scan['meta_data']['par']             = scan_list[k]['par']
        single_scan['options']['info']              = scan_list[k]['info']
        single_scan['options']['freq_integrate']    = scan_list[k]['freq_int']
        
        ## add comb_settings
        #if 'comb_settings' in scan_list[k].keys():
        #    single_scan['options']['comb_settings'] = scan_list[k]['comb_settings']
        #else:
        #    single_scan['options']['comb_settings'] = {}

        scans.append(copy.deepcopy(single_scan))





    ###############################################
    # Toggle through all scans
    ###############################################

    output_arr = []
    
    for k in range(len(scans)):
    
        ###############################################
        # Read in data
        ###############################################

        options      = scans[k]['options']
        meta_data    = scans[k]['meta_data']
    
        result = read_molecule_scan(meta_data, options)

        
        ################################################
        ## Comparison with frequency comb
        ################################################

        #if options['compare_to_comb']:

        #    comb_comparison = plot_comb_comparison(result, options)
        #    
        #    result['comb_scans']['comb_comparison'] = comb_comparison

        #else:
        #    comb_comparison = []
        #    
        #    result['comb_scans']['comb_comparison'] = comb_comparison


        ######################################################
        # Plot all frequency and time scans for each channel
        ######################################################
        
        hlp_dict = {
            'data'            : {},
            'options'         : options,
            'meta_data'       : meta_data,
            'conf'            : result['conf'],
            'comb_scans'      : result['comb_scans']
            }

        # toggle through each channel
        for n in range(len(scans[k]['channels'])):
    
            options['channel'] = scans[k]['channels'][n]

            out = plot_molecule_scan(result, meta_data, options)
    
            hlp_dict['data'][options['channel']] = out
        
        #print_dict(hlp_dict)
        
        output_arr.append(hlp_dict)



    ####################################
    # Save data for further analysis
    ####################################
    
    if save_data:

        with open('results/' + filename + '.pckl', 'wb') as f:
    
            pickle.dump(output_arr, f)
    
        f.close()


    ####################################
    # Save all figures
    ####################################
   
    if save_plots:
        save_all_plots('results/' + filename + '_figures')

    return output_arr




