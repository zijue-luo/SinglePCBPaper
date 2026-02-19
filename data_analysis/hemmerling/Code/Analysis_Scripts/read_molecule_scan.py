import numpy as np
import scipy
from configparser import ConfigParser
from os import path
import sys


from math_functions     import *
from data_manipulation  import *
from read_in_data       import *
from report_functions   import print_dict


#######################################################################################################

def get_sampler_data(t, options, basefilename, extensions, no_of_avg, freq_errors):

    all_channels = []

    # toggle through all channels
    for ext in range(len(extensions)):
        
        #if extensions[k] == '_ch0_arr' and options['subtract_absorption_reference']:
        #    # ok this is quite a hack
        #    # to subtract the reference signal from the absorption signal

        #    ch0     = read_in_file(basefilename + '_ch0_arr', 1)
        #    ch5     = read_in_file(basefilename + '_ch5_arr', 1)            

        #    channel = ch0 - ch5

        #    channel = av(channel, no_of_avg)
        #    
        #else:
        #    channel = read_in_file(basefilename + extensions[k], no_of_avg)


        # remove dropouts
        if options['UV_dropouts_remove'] or options['frequency_dropouts_remove']:

            ch6_raw     = read_in_file(basefilename + '_ch6_arr', 1)
            
            channel_raw = read_in_file(basefilename + extensions[ext], 1)

            for n in range(len(channel_raw)):
                # toggle over each ablation shot

                if options['UV_dropouts_remove']:
                    # remove data points where ch6 dropped below a threshold
                    (t1_check, t2_check) = get_time_indeces_raw(t, options['UV_dropouts_time_check'])
                    if np.mean(ch6_raw[n, t1_check:t2_check]) < options['UV_dropouts_threshold']:
                        channel_raw[n, :] = np.nan

                if options['frequency_dropouts_remove']:
                    # remove data points where the laser was offlock
                    if np.abs(freq_errors[n]) > options['frequency_dropouts_threshold']:
                        channel_raw[n, :] = np.nan
                    
 
            # average data taking the nan into account
            channel = av(channel_raw, no_of_avg, remove_nan = True)
            

        else:

            channel = read_in_file(basefilename + extensions[ext], 1)
            channel = av(channel, no_of_avg)


        # subtracting the DC offset
        if options['subtract_dc_offset']:
            
            (t1_cut, t2_cut) = get_time_indeces_raw(t, options['offset_avg_times'])
            
            # only for in-cell and PMT
            if    extensions[ext] == '_ch0_cfg0_arr' \
               or extensions[ext] == '_ch2_cfg0_arr' \
               or extensions[ext] == '_ch0_cfg1_arr' \
               or extensions[ext] == '_ch2_cfg1_arr':
                for k in range(channel.shape[0]):
                    
                    bg = np.mean(channel[k, t1_cut:t2_cut]) 

                    channel[k, :] = channel[k, :] - bg

                    if ('ch2' in extensions[ext]) and (np.mean(bg) > 0.5):

                        print('\nATTENTION: Background on PMT too large! avg offset = {0:.2f}\n'.format(bg))

        # apply moving average
        if options['moving_avg_no'] > 0:

            for k in range(channel.shape[0]):
                #channel[k, :-options['moving_avg_no']+1] = moving_average(channel[k, :], n = options['moving_avg_no'])
                #channel[k,  -options['moving_avg_no']:] = 0.0
                
                channel[k, :] = moving_average(channel[k, :], n = options['moving_avg_no'])

        # cut off data after set time
        if options['cut_off_time_data'] > 0:
            for k in range(channel.shape[0]):
                channel[k, get_time_index(t, options['cut_off_time_data']):] = 0

        # remove high-frequency noise
        if options['remove_high_frequency_noise'] > 0:
            for k in range(channel.shape[0]):
                
                channel[k, :] = remove_high_frequency_noise(t, channel[k, :], freq_cut_off = options['remove_high_frequency_noise'])


        all_channels.append(channel)

    return (t, all_channels)


#######################################################################################################

def get_molecules_laser_frequencies(frequency_domain, conf, set_freqs, act_freqs):

    # returns the set and act points in Hz

    # set_freqs = detunings
    # act_freqs = wavemeter absolute frequencies

    # the factor to multiply the wavemeter frequencies with
    # sometimes we use UV, sometimes blue, sometimes IR

    if frequency_domain == 'UV_QHG':
        frequency_factor = 4.0
    elif frequency_domain == 'UV_THG':
        frequency_factor = 3.0
    elif frequency_domain == 'Blue':
        frequency_factor = 2.0
    elif frequency_domain == 'IR':
        frequency_factor = 1.0    
    elif frequency_domain == 'Microwave':
        frequency_factor = 1.0
    else:
        print('Frequency domain not defined.')
        stop
    

    old_data_set = False
    
    # get frequency scan interval in terms of absolute frequencies

    if frequency_domain == 'Microwave':
        
        laser_offset = 0.0

        #set_freqs = set_freqs / 1e6
        act_freqs = set_freqs #/ 1e6

    else:

        # check which laser was scanned
        if 'which_scanning_laser' in conf.keys():
            # old data sets

            old_data_set = True

            if conf['which_scanning_laser']['val'] == '1':
                laser_offset = frequency_factor * float(conf['offset_laser1']['val']) * 1e12
            elif conf['which_scanning_laser']['val'] == '2':
                laser_offset = frequency_factor * float(conf['offset_laser2']['val']) * 1e12
        
        else:
            if conf['scanning_laser']['val'] == 'Davos':
                laser_offset = frequency_factor * float(conf['offset_laser_Davos']['val']) * 1e12
            elif conf['scanning_laser']['val'] == 'Hodor':
                laser_offset = frequency_factor * float(conf['offset_laser_Hodor']['val']) * 1e12
            elif conf['scanning_laser']['val'] == 'Daenerys':
                laser_offset = frequency_factor * float(conf['offset_laser_Daenerys']['val']) * 1e12
   
        set_freqs_offset = laser_offset


        # This addresses a potential systematic measurements offset

        if 'wavemeter_offset' in conf.keys():
            wavemeter_offset = frequency_factor * float(conf['wavemeter_offset']['val']) * 1e6
        else:
            wavemeter_offset = 0.0


        if old_data_set:
            # convert to the Blue/IR/UV and add the offset
            set_freqs = frequency_factor * set_freqs * 1e6 + laser_offset #+ wavemeter_offset # in Hz    

            act_freqs = frequency_factor * act_freqs * 1e12

        else:
            # in new data set, the set points are saved in terms of absolute frequency
            # convert to the Blue/IR/UV and add the offset        
            #set_freqs = frequency_factor * set_freqs * 1e12

            #act_freqs = frequency_factor * act_freqs * 1e12
            
            if conf['scanning_parameter']['val'] == 'cavity_ramp':

                #set_freqs = frequency_factor * set_freqs 

                set_freqs = frequency_factor * set_freqs * 0.98 # here we can build in a correction factor for the ramp voltage to frequency conversion
 
                ref_laser_freqs = act_freqs[:, 1]
                
                # only use first column which is the scanning laser, second column is the Moglabs
                act_freqs = frequency_factor * act_freqs[:, 0]
                
                
            else:
 
                set_freqs = frequency_factor * set_freqs

                act_freqs = frequency_factor * act_freqs
                
                ref_laser_freqs = act_freqs



    return (set_freqs, act_freqs, ref_laser_freqs, set_freqs_offset)


#######################################################################################################

def remove_list_items(lst, inds):
    
    # removes multiple items from a list

    return np.array([lst[i] for i in range(len(lst)) if i not in inds])


#######################################################################################################

def clean_data(data, options):

    # this function removes all setpoints where the channel data is set to np.nan

    n = 0
    remove_ind = []

    for k in range(len(data['channels'][n])):
        # find all set points that are np.nan
        if np.all(np.isnan(data['channels'][n][k])):
            remove_ind.append(k)
    
    print('Removing {0} setpoints completely ...'.format(len(remove_ind)))

    # remove slices

    cut_data = data

    cut_data['set_points']      = remove_list_items(data['set_points'], remove_ind)
    cut_data['act_freqs']       = remove_list_items(data['act_freqs'],  remove_ind)
    cut_data['no_of_setpoints'] = data['no_of_setpoints'] - len(remove_ind)

    for k in range(len(cut_data['channels'])):
        cut_data['channels'][k] = remove_list_items(data['channels'][k], remove_ind)

    cut_data['channels'] = np.array(cut_data['channels'])

    return cut_data


#######################################################################################################

def print_scan_summary(s):

    #if do_print:
        #    print('Found {0} averages ...'.format(no_of_avg))
        #    print('Found {0} setpoints ...'.format(conf['setpoint_count']['val']))

    print('Found {0} averages ...'.format(s['no_of_avg']))
    print('Found {0} setpoints ...'.format(s['no_of_setpoints']))
    print('Freq setpoint   first - last: {0} : {1} MHz'.format(s['set_points'][0], s['set_points'][-1]))
    print('Set points Freq first - last: {0:.6f} : {1:.6f} THz'.format(s['absolute_set_points'][0]/1e6, s['absolute_set_points'][-1]/1e6))
    print('Center frequency of scan    : {0:.6f} THz'.format(np.mean(s['absolute_set_points'])*1e6/1e12))
    print('Wavemeter Freq first - last : {0:.6f} : {1:.6f} THz'.format(s['wavemeter_freqs'][0], s['wavemeter_freqs'][-1]))

    return


#######################################################################################################

def read_single_molecule_scan(data, options, do_print = True):

    ######################################################
    # Get base config
    ######################################################

    (basefilename, conf, no_of_avg) = get_base_config(data)


    convscan2type = { 
                     'cavity_scan' : 'cavity_scan',
                     'cavity_ramp' : 'cavity_scan',
                     'offset_laser_Hodor' : 'laser_scan',
                     'offset_laser_Davos' : 'laser_scan'
                     }


    try:
        scan_type = convscan2type[conf['scanning_parameter']['val']]
    except:
        print('Need to define scan type in read_molecule_scan')
        asd

    data['type'] = scan_type


    ######################################################
    # Data files to read out
    ######################################################

    if data['type'] == 'param_scan':
        list_of_data_files         = ['_ch' + str(s) + '_cfg0_arr' for s in range(8)]
        list_of_data_slowing_files = ['_ch' + str(s) + '_cfg1_arr' for s in range(8)]
    else:
        list_of_data_files         = ['_ch' + str(s) + '_cfg0_arr' for s in range(8)]
        list_of_data_slowing_files = ['_ch' + str(s) + '_cfg1_arr' for s in range(8)]


    ######################################################
    # Get setpoints frequencies
    ######################################################
    
    if data['type'] == 'param_scan':

        set_points_raw = read_in_file(basefilename + '_freqs', 1, delimiter = None)

        set_points_raw = set_points_raw.repeat(no_of_avg)

    else:

        set_points_raw = read_in_file(basefilename + '_set_points', 1, delimiter = None)

    no_of_setpoints = int(len(set_points_raw)/no_of_avg)

    ######################################################
    # Get wavemeter readings
    ######################################################

    try:
        
        act_freqs_raw = read_in_file(basefilename + '_act_freqs', 1)

        if data['type'] == 'cavity_scan':
            absolute_freqs_raw = act_freqs_raw[:, 0]
        else:
            absolute_freqs_raw = act_freqs_raw        

    except:
        
        print('Actual frequency file not found ...')
        act_freqs_raw = np.copy(set_points)


    ######################################################
    # Get comb beat node measurements
    ######################################################

    try:
        comb_hlp = read_in_file(basefilename + '_beat_node_fft', 1)
        
        comb_beat_node_freq_raw = comb_hlp[0::2, :]
        comb_beat_node_psd_raw  = comb_hlp[1::2, :]

    except:
        print('Comb beat node frequency spectrum file not found ...')

        comb_beat_node_freq_raw = np.zeros([len(set_points_raw), 2])
        comb_beat_node_psd_raw  = np.zeros([len(set_points_raw), 2])


    try:
        comb_vrep_raw = read_in_file(basefilename + '_frequency_comb_frep', 1)
    except:
        print('Comb vrep measurement file not found ...')
        comb_vrep_raw = np.array([0.0] * len(set_points_raw))

    ######################################################
    # Get transfer cavity lock readout
    ######################################################

    try:
        transfer_cavity_times  = read_in_file(basefilename + '_transfer_lock_times', 1)
        transfer_cavity_traces_raw = read_in_file(basefilename + '_transfer_lock_traces', 1)
    
        transfer_cavity_traces_ch1_raw = transfer_cavity_traces_raw[0::4, :]
        transfer_cavity_traces_ch2_raw = transfer_cavity_traces_raw[1::4, :]
        transfer_cavity_traces_ch3_raw = transfer_cavity_traces_raw[2::4, :]
        transfer_cavity_traces_ch4_raw = transfer_cavity_traces_raw[3::4, :]

    except:
        print('Transfer cavity files not found ...')

        transfer_cavity_times      = np.zeros(1000)
        transfer_cavity_traces_ch1_raw = np.zeros([no_of_setpoints * no_of_avg, 1000])
        transfer_cavity_traces_ch2_raw = np.zeros([no_of_setpoints * no_of_avg, 1000])
        transfer_cavity_traces_ch3_raw = np.zeros([no_of_setpoints * no_of_avg, 1000])
        transfer_cavity_traces_ch4_raw = np.zeros([no_of_setpoints * no_of_avg, 1000])
   
  
    ######################################################
    # Read in times
    ######################################################
    
    times = read_in_file(basefilename + '_times', 1)


    #######################################################
    # Average data
    ######################################################

    if data['type'] == 'laser_scan' or data['type'] == 'cavity_scan' or data['type'] == 'microwave_scan':

        # convert laser frequencies to absolute values and Hz
        (set_points_raw, act_freqs_raw, ref_freqs_raw, scanning_laser_offset) = get_molecules_laser_frequencies(data['frequency_domain'], conf, set_points_raw, act_freqs_raw)

        if data['type'] == 'cavity_scan':        
            freq_errors_raw  = set_points_raw - (act_freqs_raw - np.mean(act_freqs_raw)) # in Hz
        else:
            freq_errors_raw  = set_points_raw - act_freqs_raw # in Hz
        


        # Remove dropouts

        if options['frequency_dropouts_remove']:

            inds = np.where(np.abs(freq_errors_raw) > options['frequency_dropouts_threshold'])[0]
            
            set_points_raw[inds]      = np.nan
            act_freqs_raw[inds]       = np.nan
            absolute_freqs_raw[inds]  = np.nan
            
            comb_beat_node_freq_raw[inds, :] = np.nan
            comb_beat_node_psd_raw[inds, :]  = np.nan

            set_points      = av(set_points_raw,  no_of_avg, remove_nan = True)
            act_freqs       = av(act_freqs_raw,   no_of_avg, remove_nan = True)
            freq_errors     = av(freq_errors_raw, no_of_avg, remove_nan = True)
           
            absolute_freqs = av(absolute_freqs_raw, no_of_avg, remove_nan = True)

            comb_beat_node_freq  = av(comb_beat_node_freq_raw,  no_of_avg, remove_nan = True)
            comb_beat_node_psd   = av(comb_beat_node_psd_raw,   no_of_avg, remove_nan = True)

        else:

            set_points   = av(set_points_raw,  no_of_avg)
            act_freqs    = av(act_freqs_raw,   no_of_avg)
            freq_errors  = av(freq_errors_raw, no_of_avg)
            
            ref_freqs    = av(ref_freqs_raw, no_of_avg)
            
            absolute_freqs = av(absolute_freqs_raw, no_of_avg)
            
            comb_beat_node_freq = av(comb_beat_node_freq_raw,  no_of_avg)
            comb_beat_node_psd  = av(comb_beat_node_psd_raw,   no_of_avg)
            
            comb_vrep  = av(comb_vrep_raw,   no_of_avg)


            transfer_cavity_traces_ch1 = av(transfer_cavity_traces_ch1_raw, no_of_avg) # factor of four comes from the four traces
            transfer_cavity_traces_ch2 = av(transfer_cavity_traces_ch2_raw, no_of_avg) # factor of four comes from the four traces
            transfer_cavity_traces_ch3 = av(transfer_cavity_traces_ch3_raw, no_of_avg) # factor of four comes from the four traces
            transfer_cavity_traces_ch4 = av(transfer_cavity_traces_ch4_raw, no_of_avg) # factor of four comes from the four traces


    elif data['type'] == 'param_scan':

        freq_errors_raw = []
        set_points   = av(set_points_raw, no_of_avg, remove_nan = True)
        act_freqs    = av(act_freqs_raw,  no_of_avg, remove_nan = True)

        #comb_beat_node  = av(comb_beat_node_raw,  no_of_avg, remove_nan = True)
        comb_beat_node_freq = av(comb_beat_node_freq_raw,  no_of_avg)
        comb_beat_node_psd  = av(comb_beat_node_psd_raw,   no_of_avg)
            
        absolute_freqs = av(absolute_freqs_raw, no_of_avg)




    # get each data array
    times, all_channels = get_sampler_data(times, options, basefilename, list_of_data_files, no_of_avg, freq_errors_raw)

    # check if there is slowing data
    if path.exists(basefilename + '_ch0_cfg1_arr'):
        times, slow_channels = get_sampler_data(times, options, basefilename, list_of_data_slowing_files, no_of_avg, freq_errors_raw)

        all_channels.extend(slow_channels)

    # save all data in dictionary
    result = {
            'times'                   : times,
            'scanning_laser_offset'   : scanning_laser_offset,
            'absolute_set_points'     : set_points + scanning_laser_offset/1e6,     # in MHz
            'absolute_set_points_raw' : set_points_raw + scanning_laser_offset/1e6,
            'set_points'              : set_points,
            'set_points_raw'          : set_points_raw,
            'act_freqs'               : act_freqs,
            'act_freqs_raw'           : act_freqs_raw,
            'wavemeter_freqs'         : absolute_freqs,        # wavemeter readings
            'wavemeter_freqs_raw'     : absolute_freqs_raw,
            'no_of_avg'               : no_of_avg,
            'no_of_setpoints'         : no_of_setpoints,
            'conf'                    : conf,
            'comb_scans'              : {
                'comb_beat_node_freq'            : comb_beat_node_freq,
                'comb_beat_node_freq_raw'        : comb_beat_node_freq_raw,
                'comb_beat_node_psd'             : comb_beat_node_psd,
                'comb_beat_node_psd_raw'         : comb_beat_node_psd_raw,
                'comb_rep_rate_nominal'          : [200.0e6],
                'comb_rep_rate_raw'              : comb_vrep_raw,
                'comb_rep_rate'                  : comb_vrep,
                'ref_freqs'                      : ref_freqs,     # Moglabs wavemeter frequencies
                'ref_freqs_raw'                  : ref_freqs_raw,
                'act_freqs'                      : act_freqs,     # Scanning laser wavemeter frequencies
                'act_freqs_raw'                  : act_freqs_raw,
                'transfer_cavity_times'          : transfer_cavity_times,
                'transfer_cavity_traces_ch1'     : transfer_cavity_traces_ch1,
                'transfer_cavity_traces_ch2'     : transfer_cavity_traces_ch2,
                'transfer_cavity_traces_ch3'     : transfer_cavity_traces_ch3,
                'transfer_cavity_traces_ch4'     : transfer_cavity_traces_ch4,
                'transfer_cavity_traces_ch1_raw' : transfer_cavity_traces_ch1_raw,
                'transfer_cavity_traces_ch2_raw' : transfer_cavity_traces_ch2_raw,
                'transfer_cavity_traces_ch3_raw' : transfer_cavity_traces_ch3_raw,
                'transfer_cavity_traces_ch4_raw' : transfer_cavity_traces_ch4_raw,
                }
           }
    
    result['channels'] = all_channels

    # remove setpoints that were cut, e.g., due to UV dropouts, etc ...    
    result = clean_data(result, options)

    # printing summary of scan
    print_scan_summary(result)

    return result


#######################################################################################################

def read_molecule_scan(data, options, do_print = True):

    # combine multiple scans

    time_arr = data['time']
    date_arr = data['date']

    full_result = {}

    my_comb_list = [ 
                    'comb_beat_node_freq',     
                    'comb_beat_node_freq_raw',
                    'comb_beat_node_psd',      
                    'comb_beat_node_psd_raw',  
                    'comb_rep_rate_nominal',   
                    'comb_rep_rate_raw',       
                    'comb_rep_rate',           
                    'ref_freqs',               
                    'ref_freqs_raw',
                    'act_freqs',               
                    'act_freqs_raw',
                    'transfer_cavity_times', 
                    'transfer_cavity_traces_ch1',
                    'transfer_cavity_traces_ch2',   
                    'transfer_cavity_traces_ch3',   
                    'transfer_cavity_traces_ch4',   
                    'transfer_cavity_traces_ch1_raw',
                    'transfer_cavity_traces_ch2_raw',
                    'transfer_cavity_traces_ch3_raw',
                    'transfer_cavity_traces_ch4_raw'
                    ]
    
    my_append_list = [
                    'no_of_setpoints',
                    'no_of_avg',
                    'scanning_laser_offset',
                    'absolute_set_points',
                    'absolute_set_points_raw',
                    'set_points',
                    'set_points_raw',
                    'act_freqs',
                    'act_freqs_raw',
                    'wavemeter_freqs',
                    'wavemeter_freqs_raw'
                  ]


    #####################
    # Loop over all scans
    #####################

    for k in range(len(time_arr)):
        
        data['time'] = time_arr[k]
        data['date'] = date_arr[k]

        # Read in single scan

        result = read_single_molecule_scan(data, options, do_print = do_print)

        if k == 0:

            full_result = result

        else:
                       
            for my_key in my_append_list:

                full_result[my_key]  = np.append(full_result[my_key], result[my_key])

            full_result['channels'] = np.append(full_result['channels'], result['channels'], axis = 1)

            for my_key in my_comb_list:
                full_result['comb_scans'][my_key] = np.append(full_result['comb_scans'][my_key], result['comb_scans'][my_key], axis = 0)


    ############################
    # Sort data
    ############################

    # need to check the act frequency since the offset can have changed

    # get the sorting indeces using the set points + laser offset
    inds     = np.argsort(full_result['absolute_set_points'])
    inds_raw = np.argsort(full_result['absolute_set_points_raw'])

    
    # sort all data arrays

    full_result['absolute_set_points'] = full_result['absolute_set_points'][inds]
    full_result['absolute_set_points_raw'] = full_result['absolute_set_points_raw'][inds]
    
    full_result['set_points'] = full_result['set_points'][inds]
    full_result['set_points_raw'] = full_result['set_points_raw'][inds_raw]
    
    full_result['act_freqs']  = full_result['act_freqs'][inds]
    full_result['act_freqs_raw']  = full_result['act_freqs_raw'][inds_raw]

    full_result['channels']   = full_result['channels'][:, inds, :]
   

    for my_key in my_comb_list:

        if not my_key == 'comb_rep_rate_nominal' and not my_key == 'transfer_cavity_times':
        
            if '_raw' in my_key:
                re_ind = inds_raw
            else:
                re_ind = inds

            #print_dict(full_result['comb_scans'])
            hlp = full_result['comb_scans'][my_key].shape

            if len(hlp) == 1:
                full_result['comb_scans'][my_key]   = full_result['comb_scans'][my_key][inds]
            elif len(hlp) == 2:
                full_result['comb_scans'][my_key]   = full_result['comb_scans'][my_key][inds, :]

    #plt.figure()
    #plt.plot(full_result['absolute_set_points'])
    #
    #plt.figure()
    #plt.plot(full_result['set_points'])
    #
    #plt.figure()

    #plt.plot(full_result['act_freqs'])

    #plt.show()

    #asd

    return full_result






