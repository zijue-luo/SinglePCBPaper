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


####################################
# check data folder
####################################

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



#######################################################################################################

def get_filename_and_conf(my_date, my_time, which_exp = 'molecule'):

    datafolder = check_which_data_folder(which_exp = which_exp)

    basefolder = str(my_date)

    # if time is smaller than 10am, then there is a 0 missing
    if my_time < 100000:
        my_time = '0' + str(my_time)
    else:
        my_time = str(my_time)

    basefilename = datafolder + basefolder + '/' + basefolder + '_' + my_time

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

    #if do_print:
    #    print('Found {0} averages ...'.format(no_of_avg))
    #    print('Found {0} setpoints ...'.format(conf['setpoint_count']['val']))

    return (basefilename, conf, no_of_avg)



#######################################################################################################

def read_img_data(data, options):

    basefilename, conf = get_filename_and_conf(data['date'], data['time'], which_exp = 'molecule')

    # get number of averages
    no_of_avg = int(conf['no_of_averages']['val'])
    print('Found ' + str(no_of_avg) + ' averages.')
   
    times = read_in_file(basefilename + '_times', 1)
    
    ch0  = read_in_file(basefilename + '_ch0_cfg0_arr', no_of_avg)
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


