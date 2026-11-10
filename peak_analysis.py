import numpy as np
import pandas as pd 
from math import floor
import os
import time

from HV_analysis import get_HV
from data_read import fast_folder_reader
from mean_pulse import sort_multiple_lists, get_folder_name, group_multiple_lists

def get_pulse (data_signal : pd.DataFrame, len_seg : int, pulse_number : int = 0) -> tuple[np.ndarray, np.ndarray] :
    '''
    Extracts one pulse from data_signal.

    Arguments :
    data_signal - pandas.DataFrame : dataframe with signal data from one file
    len_seg - int : number of frame per segment
    pulse_number - (optional) int : index of desired pulse (default is 0)

    Returns : ([float], [float])
    t - numpy.array of float : array of time
    i - numpy.array of float : array of values
    '''
    begin = pulse_number*len_seg
    end = (pulse_number+1)*len_seg
    t = np.array(data_signal['Time'])[begin : end]
    i = np.array(data_signal['Ampl'])[begin : end]
    return(t,i)

def void_parameters (global_folder_path : str, channel : str | tuple[str, ...], save_path : str, invert : bool | tuple[bool, ...] = False, verbose : bool = False, folder_list : list = [], n_seg=-1, len_seg=-1) -> tuple :
    return ()

def void_parameters_tbeam (t_list, signal_full_list, save_path, channel) :
    return()

def mean_max (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], parameters : tuple, global_name : str = "", HV_value : int = 0, invert : bool | tuple[bool, ...]  = False, verbose : bool = False, n_seg : int = -1, len_seg : int = -1, reader_result : tuple = ()) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str], list[str]] :
    '''
    Calculate information about the maxima of each pulse, from a folder containing data files. Can be used with get_stats function.

    Arguments :
    folder_path - str or tuple of str : path of the folder(s)
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    parameters - tuple : None
    global_name - (optional) str : name of the global working folder
    HV_value - (optional) int : HV value corresponding to the working folder
    invert - (optional) bool : whether you want the signal to be inverted (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment
    reader_result - (optional) tuple : result of the fast_folder_reader function

    Returns : ([np.array], [np.array], [str], [str], [str])
    stat1_list - list of np.array : array of HV values for each pulse
    stat2_list - list of np.array : array of stats for max, argmax, time_max, name of the data file
    stat1_name_list - list of str : "HV"
    stat2_name_list - list of str : names of the corresponding array
    save_name_list - list of str : names of the save file
    '''
    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)

    if type(invert) == bool :
        invert = (invert,) * nb_channels

    if len(reader_result) == 0 :
        n_seg_list, len_seg_list, data_signal_list, name_list = fast_folder_reader(folder_path, channel, n_seg=n_seg, len_seg=len_seg)
    else :
        n_seg_list, len_seg_list, data_signal_list, name_list = reader_result

    max_list = [[] for i in range (nb_channels)]
    argmax_list = [[] for i in range (nb_channels)]
    time_max_list = [[] for i in range (nb_channels)]
    name_full_list = [[] for i in range (nb_channels)]

    for j in range(nb_channels) :
        if invert[j] :
            for df in data_signal_list[j] :
                df["Ampl"] = - df["Ampl"]

    for i in range (nb_channels) :
        for j in range(len(n_seg_list[i])) :
            df = data_signal_list[i][j]
            for k in range (int(df.shape[0]/len_seg_list[i][j])) :
                curve = get_pulse(df, len_seg_list[i][j], k)
                signal = curve[1]
                max_list[i].append(max(signal) - np.mean(signal[:len(signal)//10]))
                argmax_list[i].append(np.argmax(signal))
                time_max_list[i].append(curve[0][argmax_list[i][-1]])
                name_full_list[i].append(name_list[i][j])

    stat1_list = []
    stat2_list = []
    stat1_name_list = ["HV"] * (4*nb_channels)
    stat2_name_list = ["max", "argmax", "time_max", "name"] * nb_channels
    save_name_list = []
    for i in range (nb_channels) :
        stat1_list += [np.array([HV_value] * len(max_list[i]))] * 4
        stat2_list += [np.array(max_list[i]), np.array(argmax_list[i]), np.array(time_max_list[i]), np.array(name_full_list[i])]
        save_name_list += ["max_" + global_name + "_" + channel[i], "argmax_" + global_name + "_" + channel[i], "time_max_" + global_name + "_" + channel[i], "file_names_" + global_name + "_" + channel[i]]
    return (stat1_list, stat2_list, stat1_name_list, stat2_name_list, save_name_list)


def get_stats (global_folder_path : str, channel : str | tuple[str, ...], stat_func, get_parameters_func, folder_list : list[str] = [], HV_list : list[int] = [], save : bool = False, save_path : str = "saves_test", invert : bool | tuple[bool, ...] = False, filter_noise : float | tuple[float, ...] = 1.2, filter_saturation : float | tuple[float, ...] = 100.0,  verbose : bool = False, n_seg : int = -1, len_seg : int = -1, multiple_channel_analysis = False) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str]] :
    '''
    Get information about the stats of each pulse, from a global folder.

    Arguments :
    global_folder_path - str : path of the folder containing the data folders for each HV value
    channel - str : desired channel of the oscilloscope (ex : "C1" or "C3") (only one channel accepted)
    stat_func - function or tuple of functions : function(s) that extract the desired stats from files (max / noise_std / mean_pulse). 
        Structure : (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], parameters : tuple, global_name : str, HV_value : int, verbose : bool, n_seg : int, len_seg : int, reader_result : tuple) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str], list[str]]
    get_parameters_func - function or tuple of functions : function(s) that ask the user to give values to the parameters used in the corresponding stat_func
        Structure : (global_folder_path : str, channel : str | tuple[str, ...], save_path : str, invert : bool | tuple[bool, ...], verbose : bool) -> tuple
    folder_list - (optional) list of str : the names of the folders for each HV value
    HV_list - (optional) list of int : the HV values (needs to be in the same order as folder_list)
    save - (optional) bool : whether you want to save the results in a new .csv file (defaut is False)
    save_path - (optional) str : where you want to save (default is "saves_test")
    invert - (optional) bool : whether you want the signal to be inverted (default is False)
    verbose - (optional) bool : whether you want the function to print its status (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment

    Returns : ([np.array], [np.array], [str], [str])
    stat1_list - list of np.array : array of stats (column 1)
    stat2_list - list of np.array : array of stats (column 2)
    stat1_name_list - list of str : names of the corresponding array (column 1)
    stat2_name_list - list of str : names of the corresponding array (column 2)
    '''
    t_start = time.time()
    stat1_list = []
    stat2_list = []
    stat1_name_list = []
    stat2_name_list = []
    save_name_list = []

    folder_name = get_folder_name(global_folder_path)
    if verbose :
        print("Starting calculations on " + folder_name)

    if folder_list == [] :
        for d in os.scandir(global_folder_path) :
            folder_list.append(d.name)
    if HV_list == [] :
        for f in folder_list :
            print(f)
            HV_list.append(get_HV(f))

    if global_folder_path[-1] != "/" :
        global_folder_path += "/"

    if save_path[-1] != "/" :
        save_path += "/"
    
    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)
    
    if type(stat_func) != tuple :
        stat_func = (stat_func,)
    nb_func = len(stat_func)

    if type(get_parameters_func) != tuple :
        get_parameters_func = (get_parameters_func,)
    
    if type(invert) == bool :
        invert = (invert,) * nb_channels
    
    if type(filter_noise) == float :
        filter_noise = (filter_noise,) * nb_channels

    if type(filter_saturation) == float :
        filter_saturation = (filter_saturation,) * nb_channels

    sort_multiple_lists(HV_list, [folder_list])
    group_multiple_lists(HV_list, [folder_list])

    nb_folders = len(HV_list)

    param = []
    for f in get_parameters_func :
        param.append(f(global_folder_path, channel, save_path, invert= invert, verbose=verbose, folder_list=folder_list, n_seg=n_seg, len_seg=len_seg))
    
    for i in range (nb_folders) :
        if verbose :
            completion = int((i/nb_folders)*100)
            print("Calculations completed by : " + str(completion) + "%")
            if completion != 0 :
                remaining = (100 - completion) * (time.time() - t_start) / completion
                print("Estimated remaining time : " + str(round(remaining, 2)) + " seconds")
            print("Current folder for calculations : " + str(folder_list[i]))
        n_seg_list, len_seg_list, data_signal_list, name_list = fast_folder_reader(tuple([global_folder_path + f for f in folder_list[i]]), channel, n_seg=n_seg, len_seg=len_seg, invert=invert, filter_noise=filter_noise, filter_saturation=filter_saturation, multiple_channel_analysis=multiple_channel_analysis)
    
    for j in range(nb_func) :
        stat1, stat2, stat1_name, stat2_name, save_name = stat_func[j](tuple([global_folder_path + f for f in folder_list[i]]), channel, param[j], global_name=folder_name, HV_value=HV_list[i], verbose = verbose, reader_result = (n_seg_list, len_seg_list, data_signal_list, name_list), n_seg=n_seg, len_seg=len_seg)
        stat1_list += stat1
        stat2_list += stat2
        stat1_name_list += stat1_name
        stat2_name_list += stat2_name
        save_name_list += save_name

    i = 0
    while i < len(stat1_list):
        j = i+1
        while j < len(stat1_list) : # stacks stats with the same save_name (to integrate with stat_func ??)
            if save_name_list[i] == save_name_list[j] :
                stat1 = stat1_list.pop(j)
                stat2 = stat2_list.pop(j)
                stat1_name = stat1_name_list.pop(j)
                stat2_name = stat2_name_list.pop(j)
                save_name = save_name_list.pop(j)
                stat1_list[i] = np.hstack((stat1_list[i], stat1))
                stat2_list[i] = np.hstack((stat2_list[i], stat2))
            else :
                j += 1
        if save :
            if len(stat2_list[i]) == 0 :
                df = pd.DataFrame(stat1_list[i], columns=[stat1_name_list[i]])
                df.to_csv(save_path + save_name_list[i])
            else :
                df = pd.DataFrame(np.stack((stat1_list[i], stat2_list[i]), axis=1), columns=[stat1_name_list[i], stat2_name_list[i]])
                df.to_csv(save_path + save_name_list[i])
        i += 1

    if verbose :
        print("Calculations completed ")

    return (stat1_list, stat2_list, stat1_name_list, stat2_name_list)