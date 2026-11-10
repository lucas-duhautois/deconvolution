import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from math import floor

from mean_pulse import get_folder_name

def pulse_distribution (global_folder_path : str, channel : str, HV_value : int, intervals : list[float] = [], save_path : str = "saves_test", bins : int = 100, log : bool = False, range_ = None, invert : bool = False, verbose : bool = False, folder_list : list[str] = [], HV_list : list[int] = [], n_seg : int = -1, len_seg : int = -1) -> None :
    '''
    Plots the pulse distribution of maxima for a given HV value.

    Arguments :
    global_folder_path - str : path of the folder containing the data folders for each HV value
    channel - str : desired channel of the oscilloscope (ex : "C1" or "C3") (only one channel accepted)
    HV_value - int : the desired HV value
    save_path - (optional) str : path of the folder containing the save files from get_mean_max (default is "./saves_test")
    bins - (optional) int : number of sections in the histogram (default is 100)
    log - (optional) bool : whether you want to plot on a log scale (default is False)
    range_ - (optional) tuple of float : minimum and maximum value of amplitude (in V)
    invert - (optional) bool : whether you want the signal to be inverted (default is False)
    verbose - (optional) bool : whether you want the function to print its status (default is False)
    folder_list - (optional) list of str : the names of the folders for each HV value, used for get_mean_max if the mean values of all maxima are not already calculated
    HV_list - (optional) list of int : the HV values, used for get_mean_max if the mean values of all maxima are not already calculated (needs to be in the same order as folder_list)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment

    Return : None
    '''
    folder_name = get_folder_name(global_folder_path)

    HV_full_list = np.array(pd.read_csv(save_path + "/max_" + folder_name + "_" + channel)["HV"])
    max_list = np.array(pd.read_csv(save_path + "/max_" + folder_name + "_" + channel)["max"])

    deciles = np.percentile(max_list[HV_full_list == HV_value], np.arange(0, 100, 10))
    print(deciles)

    for i in intervals :
        plt.axvline(i*1e-3, color = "black")

    hist = plt.hist(max_list[HV_full_list == HV_value], bins = bins, log=log, range = range_)
    plt.title(folder_name + " " + channel + " HV" + str(HV_value))
    plt.xlabel("Amplitude (V)")
    plt.ylabel("Number of pulses")
    #plt.grid()
    #print(hist[0])
    #print(hist[1][:len(hist[1])-1][hist[0] >=10])
    plt.show()