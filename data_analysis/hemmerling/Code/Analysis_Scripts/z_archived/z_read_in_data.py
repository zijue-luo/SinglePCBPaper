import numpy as np
import scipy
from configparser import ConfigParser
from os import path
from math_functions import *
from data_manipulation import *
import sys


####################################
# reads in config file
####################################

def read_in_config(f):
    
    config = ConfigParser()
    config.read(f)

    sensor_ids = config.sections()
    # make dictionary out of config

    sensors = {}

    for s in sensor_ids:
        opts = config.options(s)
        
        sensors[s] = {}
        for o in opts:
            sensors[s][o] = config.get(s, o)

    return sensors



####################################
# reads in a scan
####################################



def read_in_file(filename, no_of_avg, delimiter = ','):

    if not delimiter == None:
        x = np.genfromtxt(filename, delimiter=",")

        if no_of_avg > 1:
            x = av(x, no_of_avg)

    else:
        x = np.genfromtxt(filename)

    return x


def check_which_data_folder(which_exp = 'molecule'):

    if which_exp == 'molecule':
        folder1 = '/home/molecules/software/data/'
        folder2 = '/Users/boerge/Software/offline_molecules_data/'
        folder3 = '/home/lab-user/software/data/'
    
    if which_exp == 'electron':
        folder1 = '/home/electrons/software/data/'
        folder2 = '/Users/boerge/Software/offline_electrons_data/'
        folder3 = ''

    if path.exists(folder1):
        datafolder = folder1
    elif path.exists(folder2):
        datafolder = folder2
    elif path.exists(folder3):
        datafolder = folder3
    else:
        print('Data path not found')
        print(folder1)
        print(folder2)
        print(folder3)
        asd

    return datafolder



def get_filename_and_conf(my_date, my_time, which_exp = 'molecule'):

    datafolder = check_which_data_folder(which_exp = which_exp)

    basefolder = str(my_date)

    basefilename = datafolder + basefolder + '/' + basefolder + '_' + str(my_time)

    # read in config file
    config_file = basefilename + '_conf'
    conf = read_in_config(config_file)

    if conf == {}:
        print('\nFile {0} not found.\n'.format(basefilename))
        sys.exit()

    return basefilename, conf



#######################################################################################################

def get_base_config(data, do_print = True, which_exp = 'molecule'):

    (basefilename, conf) = get_filename_and_conf(data['date'], data['time'], which_exp = which_exp)
    
    if do_print:
        print()
        print("-"*100)
        print('Analyzing file ... ' + str(data['date']) + '_' + str(data['time']))
        print("-"*100)
        print("Path: " + basefilename)
        print()
 
    if which_exp == 'molecule':

        # take old scans into account as well
        if 'no_of_averages' in conf.keys():
            # read in setpoints
            no_of_avg = int(conf['no_of_averages']['val'])
        elif 'scan_count' in conf.keys():
            # read in setpoints
            no_of_avg = int(conf['scan_count']['val'])
        else:
            print('No no_of_averages found in config file.')
            stop

        ## get number of set points / scan points by reading out set_points file
        #set_points = read_in_file(basefilename + '_set_points', no_of_avg, delimiter = None)

        #if which_exp == 'molecule':
        #    # remove the averages in the set_points interval
        #    set_points = av(set_points, no_of_avg)

        #no_of_setpoints = len(set_points)

    elif which_exp == 'electron':

        set_points = 0
        no_of_avg = conf['no_of_repeats']['val']
        no_of_setpoints = 0

    if do_print:
        print('Found {0} averages ...'.format(no_of_avg))
        #print('Found {0} setpoints ...'.format(no_of_setpoints))

    return (basefilename, conf, no_of_avg)


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
                    channel[k, :] = channel[k, :] - np.mean(channel[k, t1_cut:t2_cut]) 

        # apply moving average
        if options['moving_avg_no'] > 0:

            for k in range(channel.shape[0]):
                channel[k, :-options['moving_avg_no']+1] = moving_average(channel[k, :], n = options['moving_avg_no'])
                channel[k,  -options['moving_avg_no']:] = 0.0

        # cut off data after set time
        if options['cut_off_time_data'] > 0:
            for k in range(channel.shape[0]):
                channel[k, get_time_index(t, options['cut_off_time_data']):] = 0

        all_channels.append(channel)

    return (t, all_channels)



#######################################################################################################

def get_molecules_laser_frequencies(frequency_domain, conf, set_freqs, act_freqs):

    # returns the set and act points in Hz

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
            
            set_freqs = frequency_factor * set_freqs

            act_freqs = frequency_factor * act_freqs

    return (set_freqs, act_freqs)


#######################################################################################################

def remove_list_items(lst, inds):
    
    # removes multiple items from a list

    return np.array([lst[i] for i in range(len(lst)) if i not in inds])



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



def read_single_molecule_scan(data, options, do_print = True):

    # data files to read out

    if data['type'] == 'param_scan':
        list_of_data_files         = ['_ch' + str(s) + '_cfg1_arr' for s in range(8)]
        list_of_data_slowing_files = ['_ch' + str(s) + '_cfg1_arr' for s in range(8)]
    else:
        list_of_data_files         = ['_ch' + str(s) + '_cfg0_arr' for s in range(8)]
        list_of_data_slowing_files = ['_ch' + str(s) + '_cfg1_arr' for s in range(8)]

    # get base config
    (basefilename, conf, no_of_avg) = get_base_config(data)

    # get setpoint frequencies, absolute IR frequencies
    # NEEDS FIXING
    
    if data['type'] == 'param_scan':

        set_points_raw = read_in_file(basefilename + '_freqs', 1, delimiter = None)

        set_points_raw = set_points_raw.repeat(no_of_avg)

    else:

        set_points_raw = read_in_file(basefilename + '_set_points', 1, delimiter = None)

    # check if there are wavemeter measurements of the frequencies
    # get wavemeter readings, absolute IR frequencies
    try:
        act_freqs_raw = read_in_file(basefilename + '_act_freqs', 1)

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

    try:
        comb_vrep_raw = read_in_file(basefilename + '_frequency_comb_frep', 1)
    except:
        print('Comb vrep measurement file not found ...')
        comb_vrep_raw = np.array([0.0] * len(set_points_raw))



    # read in times
    times = read_in_file(basefilename + '_times', 1)


    if data['type'] == 'laser_scan' or data['type'] == 'microwave_scan':
        # convert laser frequencies to absolute values and Hz
        (set_points_raw, act_freqs_raw) = get_molecules_laser_frequencies(data['frequency_domain'], conf, set_points_raw, act_freqs_raw)
        
        freq_errors_raw  = set_points_raw - act_freqs_raw # in Hz
        
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
            
            absolute_freqs = av(absolute_freqs_raw, no_of_avg)
            
            comb_beat_node_freq = av(comb_beat_node_freq_raw,  no_of_avg)
            comb_beat_node_psd  = av(comb_beat_node_psd_raw,   no_of_avg)


    elif data['type'] == 'param_scan':

        freq_errors_raw = []
        set_points   = av(set_points_raw, no_of_avg, remove_nan = True)
        act_freqs    = av(act_freqs_raw,  no_of_avg, remove_nan = True)

        comb_beat_node  = av(comb_beat_node_raw,  no_of_avg, remove_nan = True)
    
    # get each data array
    times, all_channels = get_sampler_data(times, options, basefilename, list_of_data_files, no_of_avg, freq_errors_raw)

    # check if there is slowing data
    if path.exists(basefilename + '_ch0_cfg1_arr'):
        times, slow_channels = get_sampler_data(times, options, basefilename, list_of_data_slowing_files, no_of_avg, freq_errors_raw)

        all_channels.extend(slow_channels)

    # save all data in dictionary
    result = {
            'times'                 : times,
            'set_points'            : set_points,
            'set_points_raw'        : set_points_raw,
            'act_freqs'             : act_freqs,
            'act_freqs_raw'         : act_freqs_raw,
            'absolute_freqs'        : absolute_freqs,
            'absolute_freqs_raw'    : absolute_freqs_raw,
            'no_of_avg'             : no_of_avg,
            'no_of_setpoints'       : len(set_points),
            'conf'                  : conf,
            'scan_info'             : data,
            'comb_scans'            : {
                'comb_beat_node_freq'     : comb_beat_node_freq,
                'comb_beat_node_freq_raw' : comb_beat_node_freq_raw,
                'comb_beat_node_psd'      : comb_beat_node_psd,
                'comb_beat_node_psd_raw'  : comb_beat_node_psd_raw,
                'comb_rep_rate'           : 200.0e6,
                'comb_rep_rate_raw'       : comb_vrep_raw
                }
           }

    result['channels'] = all_channels

    # remove setpoints that were cut, e.g., due to UV dropouts, etc ...    
    result = clean_data(result, options)

    return result




def read_molecule_scan(data, options, do_print = True):

    time_arr = data['time']
    date_arr = data['date']

    full_result = {}

    my_comb_list = [ 
                    'comb_beat_node_freq',
                    'comb_beat_node_freq_raw',
                    'comb_beat_node_psd',
                    'comb_beat_node_psd_raw'
                    ]
    my_append_list = [
                    'no_of_setpoints',
                    'set_points',
                    'set_points_raw',
                    'act_freqs',
                    'act_freqs_raw',
                    'absolute_freqs',
                    'absolute_freqs_raw'
                  ]

    for k in range(len(time_arr)):
        
        data['time'] = time_arr[k]
        data['date'] = date_arr[k]

        result = read_single_molecule_scan(data, options, do_print = do_print)

        if k == 0:

            full_result = result

        else:
            
            for my_key in my_append_list:

                full_result[my_key]  = np.append(full_result[my_key], result[my_key])

            full_result['channels']         = np.append(full_result['channels'], result['channels'], axis = 1)
            
            for my_key in my_comb_list:
                full_result['comb_scans'][my_key] = np.append(full_result['comb_scans'][my_key], result['comb_scans'][my_key], axis = 0)

    ## sort data

    #inds = np.argsort(full_result['set_points'])

    #full_result['set_points'] = full_result['set_points'][inds]
    #full_result['act_freqs']  = full_result['act_freqs'][inds]
    #full_result['channels']   = full_result['channels'][:, inds, :]
    #
    #for my_key in my_comb_list:
    #    full_result['comb_scans'][my_key]   = full_result['comb_scans'][my_key][inds, :]


    return full_result






def read_electron_scan(data, options, do_print = True):

    # data = {
    # 'date' : 20200720,
    # 'time' : 1702329_100
    # }

    # all files that Artiq produces
    # 20200807_184357_1004_act_freqs    # readout of wavemeter frequencies in THz
    # 20200807_184357_1004_ch0_arr      # transient data
    # 20200807_184357_1004_ch1_arr
    # 20200807_184357_1004_ch2_arr
    # 20200807_184357_1004_ch3_arr
    # 20200807_184357_1004_ch4_arr
    # 20200807_184357_1004_conf         # config file
    # 20200807_184357_1004_freqs        # unique setpoint frequencies in MHz
    # 20200807_184357_1004_sequence     # sequence file
    # 20200807_184357_1004_set_points   # setpoint frequencies including averages
    # 20200807_184357_1004_times        # time array
    

    #(basefilename, conf, set_points, no_of_avg, no_of_setpoints) = get_base_config(data, which_exp = 'electron')
    (basefilename, conf, no_of_avg) = get_base_config(data, which_exp = 'electron')
    
    set_points = read_in_file(basefilename + '_set_points', 1, delimiter = None)
    no_of_setpoints = len(set_points)

    scan_points = set_points
    
    try:
        act_freqs = read_in_file(basefilename + '_act_freqs', no_of_avg, delimiter = None)

        act_freqs = act_freqs/1e12

    except:
        print('Actual frequency file not found ...')
        act_freqs = np.copy(set_points)
    

    #times = read_in_file(basefilename + '_times', 1)

    # get each data array
    
    ch0 = read_in_file(basefilename + '_scan_result', no_of_avg, delimiter = None)

    laser_offset_390 = float(conf['frequency_390']['val'])
    laser_offset_422 = float(conf['frequency_422']['val'])

    # convert to the UV and add the offset
    #set_freqs = frequency_factor * set_freqs * 1e6 + laser_offset + wavemeter_offset # in Hz

    act_freqs = act_freqs * 1e12



    # save all data in dictionary
    result = {
            'times' : 0,
            'set_points' : set_points,
            'act_freqs' : act_freqs,
            'no_of_avg' : no_of_avg,
            'no_of_setpoints' : no_of_setpoints,
            'act_freqs' : act_freqs,
            'conf' : conf,
            'scan_info' : data
           }
 
    result['channels'] = [ch0]
    
    #result = {'times' : 0, 'set_points' : set_points, 'act_freqs' : act_freqs, 'channels' : ch0, 'laser_offset_422' : laser_offset_422, 'laser_offset_390' : laser_offset_390, 'scan_points' : scan_points, 'conf' : conf}

    return result


def read_electron_histogram(data, options, do_print = True):

    # data = {
    # 'date' : 20200720,
    # 'time' : 1702329_100
    # }

    # all files that Artiq produces
    # 20200807_184357_1004_act_freqs    # readout of wavemeter frequencies in THz
    # 20200807_184357_1004_ch0_arr      # transient data
    # 20200807_184357_1004_ch1_arr
    # 20200807_184357_1004_ch2_arr
    # 20200807_184357_1004_ch3_arr
    # 20200807_184357_1004_ch4_arr
    # 20200807_184357_1004_conf         # config file
    # 20200807_184357_1004_freqs        # unique setpoint frequencies in MHz
    # 20200807_184357_1004_sequence     # sequence file
    # 20200807_184357_1004_set_points   # setpoint frequencies including averages
    # 20200807_184357_1004_times        # time array
    

    (basefilename, conf, set_points, no_of_avg, no_of_setpoints) = get_base_config(data, which_exp = 'electron')
    
    #times = read_in_file(basefilename + '_times', 1)

    # get each data array
    
    ch0 = read_in_file(basefilename + '_timestamps', no_of_avg, delimiter = None)


    # save all data in dictionary
    result = {
            'no_of_avg' : no_of_avg,
            'conf' : conf,
            'scan_info' : data
           }
 
    result['channels'] = [ch0]

    return result






def read_img_data(data, options):

    basefilename, conf = get_filename_and_conf(data['date'], data['time'], which_exp = 'molecule')

    # get number of averages
    no_of_avg = int(conf['no_of_averages']['val'])
    print('Found ' + str(no_of_avg) + ' averages.')
   
    times = read_in_file(basefilename + '_times', 1)
    
    ch0 = read_in_file(basefilename + '_ch0_arr', no_of_avg)
    posx = read_in_file(basefilename + '_posx', 1)
    posy = read_in_file(basefilename + '_posy', 1)

    # subtracting the DC offset
    offset_levels = []
    
    if options['subtract_dc_offset']:
        for k in range(ch0.shape[0]):
            
            #offset_levels.append(np.mean(ch0[k, -options['offset_avg_points']:-1]))
            offset_levels.append(np.mean(ch0[k, 0:options['offset_avg_points']]))

            ch0[k, :] = ch0[k, :] - offset_levels[-1]


    inter_x = np.unique(posx)
    inter_y = np.unique(posy)

    result = {
            'date'      : data['date'], 
            'timestamp' : data['time'], 
            'times'     : times, 
            'conf'      : conf, 
            'x_pos'     : inter_x, 
            'y_pos'     : inter_y, 
            'channels'  : [ch0], 
            'no_absorption_level' : offset_levels
            }

    return result


