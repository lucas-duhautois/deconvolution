import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import cblind as cb

from mean_pulse import get_mean_pulse, plot_pulse, plot_save, standardize, sort_multiple_lists, get_folder_name

def get_HV (name : str) -> int :
    '''
    Extracts the HV value from the name of a data folder.

    Argument :
    name - str : name of the data folder

    Return : int
    after_HV - int : value of the number just after "HV" in the name (-1 if fail)
    '''
    if "HV" in name :
        after_HV = name.split("HV")[1]
        for i in range(len(after_HV)) :
            if not("0" <= after_HV[i] <= "9") :
                return(int(after_HV[:i]))
            elif i == len(after_HV)-1 :
                return(int(after_HV))
    elif "hv" in name :
        after_HV = name.split("hv")[1]
        for i in range(len(after_HV)) :
            if not("0" <= after_HV[i] <= "9") :
                return(int(after_HV[:i]))
            elif i == len(after_HV)-1 :
                return(int(after_HV))
    elif "V" in name :
        before_V = name.split("V")[0]
        for i in range(len(before_V)-1, 0, -1) :
            if not("0" <= before_V[i] <= "9") :
                return(int(before_V[i+1:]))
            elif i == 0 :
                return(int(before_V))
    else :
        raise Exception("No HV reading possible in folder name.")
    return (-1)

def plot_HV (global_folder_path : str, channel : str | tuple[str, ...], folder_list : list[str] = [], HV_list : list[int] = [], version : int = 4, standard : bool = False, save : bool = False, n_seg : int = -1, len_seg : int = -1) -> None :
    '''
    Plots the mean curves for different HV values.

    Arguments :
    global_folder_path - str : path of the folder containing the data folders for each HV value
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    folder_list - (optional) list of str : the names of the folders for each HV value
    HV_list - (optional) list of int : the HV values (needs to be in the same order as folder_list)
    version - (optional) int : version of the mean pulse calculation desired (default is 4)
    standard - (optional) bool : whether you want the data to be standardized (default is False)
    save - (optional) bool : whether you want to save the results in a new .csv file (defaut is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment

    Return : None
    '''
    if folder_list == [] :
        for d in os.scandir(global_folder_path) :
            folder_list.append(d.name)
    if HV_list == [] :
        for f in folder_list :
            HV_list.append(get_HV(f))

    nb_folders = len(folder_list)
    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)
    color, linestyle = cb.Colorplots().extreme_rainbow(nb_folders)

    sort_multiple_lists(HV_list, [folder_list])
    folder_name = get_folder_name(global_folder_path)

    if global_folder_path[-1] != "/" :
        global_folder_path += "/"

    results = []
    for i in range(len(folder_list)) :
        results.append(get_mean_pulse(global_folder_path + folder_list[i], channel = channel, version = version, plot = False, save = save, n_seg = n_seg, len_seg = len_seg))
    for j in range(nb_channels) :
        plt.clf()
        for i in range(nb_folders) :
            if standard :
                results[i][j][1] = standardize(results[i][j][1])
            if len(results[i][j][0]) != 0 :
                c = color[i%34]
                l = linestyle[i//34]
                plot_pulse(results[i][j][0], results[i][j][1], title = folder_name + " " + channel[j], show = False, label = str(HV_list[i]), color=c, linestyle=l)
        plt.grid()
        plt.legend(title = "HV (V)")
        plt.show()

def plot_HV_from_save (save_folder_path : str, channel : str | tuple[str, ...], title : str = "", sub_string : str = "", save_list : list[str] = [], HV_list : list[int] = [], standard : bool = False) -> None :
    '''
    Plots the mean curves for different HV values, from save files.

    Arguments :
    save_folder_path - str : path of the folder containing the data files for each HV value
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    sub_string - str : needs to be in the save file name for it to be considered
    save_list - (optional) list of str : the names of the files for each HV value
    HV_list - (optional) list of int : the HV values (needs to be in the same order as save_list)
    standard - (optional) bool : whether you want the data to be standardized (default is False)

    Return : None
    '''
    if save_list == [] :
        for f in os.scandir(save_folder_path) :
            save_list.append(f.name)
    if HV_list == [] :
        for f in save_list :
            HV_list.append(get_HV(f))
    nb_saves = len(save_list)
    
    sort_multiple_lists(HV_list, [save_list])
    folder_name = get_folder_name(save_folder_path)
    
    if type(channel) == str :
        channel = (channel,)

    if save_folder_path[-1] != "/" :
        save_folder_path += "/"

    for ch in channel :
        plt.clf()
        i_plot = []
        for i in range(nb_saves) :
            if (ch in save_list[i]) and (sub_string in save_list[i]) :
                i_plot.append(i)
        nb_plot = len(i_plot)
        color, linestyle = cb.Colorplots().extreme_rainbow(nb_plot)
        for i in range(nb_plot) :
            c = color[i%34]
            l = linestyle[i//34]
            plot_save(save_folder_path+save_list[i_plot[i]], title = "Convolved signal " + title, show = False, label = str(HV_list[i_plot[i]]), standard=standard, color=c, linestyle=l)
        if standard :
            plt.ylabel("normalized amplitude")
        else :
            plt.ylabel("amplitude (V)")
        plt.grid()
        plt.legend(title = "HV (V)")
        plt.show()

def function_max (t : np.ndarray, signal : np.ndarray) -> tuple[int, str] :
    '''
    Gives the maximum amplitude of the signal.

    Arguments :
    t - array of float : time points
    signal - array of float : values of the signal

    Return :
    int :  maximum
    ylabel - str : label of the y axis
    '''
    return (max(signal) - np.mean(signal[:len(signal)//10]), "amplitude (V)")

def function_integrate (t : np.ndarray, signal : np.ndarray) -> tuple[int, str] :
    '''
    Gives the area under the curve of the signal.

    Arguments :
    t - array of float : time points
    signal - array of float : values of the signal

    Return :
    int : area
    '''
    return (np.trapezoid(signal,t), "area under the curve (V.s)")

def function_time_up (t : np.ndarray, signal : np.ndarray) -> tuple[int, str] :
    '''
    Gives the time needed for the signal to go up (from 10% of max to 90% of max).

    Arguments :
    t - array of float : time points
    signal - array of float : values of the signal

    Return :
    int : time
    ylabel - str : label of the y axis
    '''
    start_point = signal[0]
    delta = max(signal) - start_point
    signal_10_percent = start_point + 0.1*delta
    signal_90_percent = start_point + 0.9*delta
    t_10_percent = t[signal > signal_10_percent][0]
    t_90_percent = t[signal > signal_90_percent][0]
    return (t_90_percent - t_10_percent, "time up (s)")

def function_time_down (t : np.ndarray, signal : np.ndarray) -> tuple[int, str] :
    '''
    Gives the time needed for the signal to go down (from 90% of max to 10% of max).

    Arguments :
    t - numpy.array of float : time points
    signal - numpy.array of float : values of the signal

    Return :
    int : time (0 if not reached)
    ylabel - str : label of the y axis
    '''
    start_point = signal[0]
    delta = max(signal) - start_point
    i_max = np.argmax(signal)
    signal_10_percent = start_point + 0.1*delta
    signal_90_percent = start_point + 0.9*delta
    t_after_max = t[i_max:]
    signal_after_max = signal[i_max:]
    t_after_10_percent = t_after_max[signal_after_max < signal_10_percent]
    t_after_90_percent = t_after_max[signal_after_max < signal_90_percent]
    if len(t_after_10_percent) > 0 :
        t_10_percent = t_after_10_percent[0]
    else :
        return (0, "Time down (s)")
    if len(t_after_90_percent) > 0 :
        t_90_percent = t_after_90_percent[0]
    else :
        return (0, "Time down (s)")
    return (t_10_percent - t_90_percent, "time down (s)")

def plot_function_HV_from_save (save_path : str, channel : str | tuple[str, ...], func,  title : str = "", sub_string : str = "", plot : bool = True, save_list : list[str] = [], HV_list : list[int] = [], n_pulse : int = 1) -> tuple[list[list[int]], list[list[float]]] :
    '''
    Plots the result of a given function on the mean curves for different HV values, from save files.

    Arguments :
    save_path - str : path of the folder containing the data files for each HV value
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    func - function : function that calculates what we want to plot and that takes 2 arguments : an array of time points and an array of signal values
    ylabel - str : label of the y axis 
    sub_string - str : needs to be in the save file name for it to be considered
    plot - (optional) bool : whether you actually want to plot
    save_list - (optional) list of str : the names of the files for each HV value
    HV_list - (optional) list of int : the HV values (needs to be in the same order as save_list)

    Returns : ([[int]], [[float]])
    HV_list_channel - list of list of int : HV values, for each channel
    result - list of list of float : result of the function for each HV values, for each channel

    examples :
    [0][i] : HV values for channel i
    [1][i] : result of the function for channel i
    '''
    if save_list == [] :
        for f in os.scandir(save_path) :
            save_list.append(f.name)
    if HV_list == [] :
        for f in save_list :
            HV_list.append(get_HV(f))
    nb_saves = len(save_list)
    
    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)

    sort_multiple_lists(HV_list, [save_list])
    folder_name = get_folder_name(save_path)

    if save_path[-1] != "/" :
        save_path += "/"
    
    HV_list_channel = []
    result = []

    for j in range(nb_channels) :
        result.append([])
        HV_list_channel.append([])
        for i in range(nb_saves) :
            if (channel[j] in save_list[i]) and (sub_string in save_list[i]) :
                res_list = []
                for k in range (n_pulse):
                    df = pd.read_csv(save_path+save_list[i])
                    t_array = np.array(df['Time'])
                    signal_array = np.array(df['Ampl'])
                    n = len(t_array)
                    res, ylabel = func(t_array[int(n * k / n_pulse) : int(n * (k+1) / n_pulse)], signal_array[int(n * k / n_pulse) : int(n * (k+1) / n_pulse)])
                    if res != 0 :
                        res_list.append(res)

                res = np.mean(res_list)
                if res != 0 :
                    result[j].append(res)
                    HV_list_channel[j].append(HV_list[i])
        if plot :
            print(HV_list_channel[j])
            if "time" in ylabel :
                print([int(x*1e12)/1000 for x in result[j]])
            else :
                print([int(x*1e6)/1000 for x in result[j]])
            plt.clf()
            plt.title(title)
            plt.xlabel("HV (V)")
            plt.ylabel(ylabel)
            plt.scatter(HV_list_channel[j], result[j], color="#0303A0")
            plt.grid()
            plt.show()
    return(HV_list_channel, result)

def plot_function_HV (t_list : list[np.ndarray], signal_list : list[np.ndarray], HV_list : list[int], func, title : str = "", plot : bool = True) -> tuple[list[list[int]], list[list[float]]] :
    '''
    Plots the result of a given function on the mean curves for different HV values, from save files.

    Arguments :
    save_path - str : path of the folder containing the data files for each HV value
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    func - function : function that calculates what we want to plot and that takes 2 arguments : an array of time points and an array of signal values
    ylabel - str : label of the y axis 
    sub_string - str : needs to be in the save file name for it to be considered
    plot - (optional) bool : whether you actually want to plot
    save_list - (optional) list of str : the names of the files for each HV value
    HV_list - (optional) list of int : the HV values (needs to be in the same order as save_list)

    Returns : ([[int]], [[float]])
    HV_list_channel - list of list of int : HV values, for each channel
    result - list of list of float : result of the function for each HV values, for each channel

    examples :
    [0][i] : HV values for channel i
    [1][i] : result of the function for channel i
    '''
    nb_saves = len(signal_list)
    result = []

    for i in range(nb_saves) :
        res, ylabel = func(t_list[i], signal_list[i])
        if res != 0 :
            result.append(res)
        else :
            HV_list.pop(i)

    if plot :
        plt.clf()
        plt.title(title)
        plt.xlabel("HV (V)")
        plt.ylabel(ylabel)
        print(HV_list)
        if "time" in ylabel :
            print([int(x*1e10)/10 for x in result])
        else :
            print([int(x*1e4)/10 for x in result])
        plt.scatter(HV_list, result, color="#0303A0")
        plt.grid()
        plt.show()
    return(HV_list, result)

def plot_function_from_save_strontium (save_path : str, channel : str | tuple[str, ...], func, HV_value : int, title : str = "", sub_string : str = "", plot : bool = True, save_list : list[str] = [], window : list =[]) -> tuple[list[list[int]], list[list[float]]] :
    '''
    Plots the result of a given function on the mean curves for different HV values, from save files.

    Arguments :
    save_path - str : path of the folder containing the data files for each HV value
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    func - function : function that calculates what we want to plot and that takes 2 arguments : an array of time points and an array of signal values
    ylabel - str : label of the y axis 
    sub_string - str : needs to be in the save file name for it to be considered
    plot - (optional) bool : whether you actually want to plot
    save_list - (optional) list of str : the names of the files for each HV value
    HV_list - (optional) list of int : the HV values (needs to be in the same order as save_list)

    Returns : ([[int]], [[float]])
    HV_list_channel - list of list of int : HV values, for each channel
    result - list of list of float : result of the function for each HV values, for each channel

    examples :
    [0][i] : HV values for channel i
    [1][i] : result of the function for channel i
    '''
    if save_list == [] :
        for f in os.scandir(save_path) :
            file_name = f.name
            if (str(HV_value) + "V" in file_name) or ("HV" + str(HV_value) in file_name) :
                save_list.append(file_name)
    nb_saves = len(save_list)
    
    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)

    low_ampl = []
    high_ampl = []
    for f in save_list :
        file_split = f.split("_")[-1].split("-")
        low_ampl.append(float(file_split[0]))
        high_ampl.append(float(file_split[1].split("mV")[0]))
    sort_multiple_lists(low_ampl, [high_ampl, save_list])

    folder_name = get_folder_name(save_path)

    if save_path[-1] != "/" :
        save_path += "/"
    
    ampl_list_channel = []
    result = []

    for j in range(nb_channels) :
        result.append([])
        ampl_list_channel.append([])
        for i in range(nb_saves) :
            if (channel[j] in save_list[i]) and (sub_string in save_list[i]) :
                df = pd.read_csv(save_path+save_list[i])
                t = np.array(df['Time'])
                signal = np.array(df['Ampl'])
                if len(window) != 0 :
                    t = t[window[0]:window[1]]
                    signal = signal[window[0]:window[1]]
                res, ylabel = func(t, signal)
                if res != 0 :
                    result[j].append(res)
                    ampl_list_channel[j].append(low_ampl[i])
        if plot :
            plt.clf()
            plt.title(title)
            plt.xlabel("lower value of the amplitude interval (V)")
            plt.ylabel(ylabel)
            plt.scatter(ampl_list_channel[j], result[j], color="#0303A0")
            plt.grid()
            plt.show()
    return(ampl_list_channel, result)

def plot_function_strontium (t_list : list[np.ndarray], signal_list : list[np.ndarray], low_ampl_list : list[float], func, title : str = "", plot : bool = True) -> tuple[list[list[int]], list[list[float]]] :
    '''
    Plots the result of a given function on the mean curves for different HV values, from save files.

    Arguments :
    save_path - str : path of the folder containing the data files for each HV value
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    func - function : function that calculates what we want to plot and that takes 2 arguments : an array of time points and an array of signal values
    ylabel - str : label of the y axis 
    sub_string - str : needs to be in the save file name for it to be considered
    plot - (optional) bool : whether you actually want to plot
    save_list - (optional) list of str : the names of the files for each HV value
    HV_list - (optional) list of int : the HV values (needs to be in the same order as save_list)

    Returns : ([[int]], [[float]])
    HV_list_channel - list of list of int : HV values, for each channel
    result - list of list of float : result of the function for each HV values, for each channel

    examples :
    [0][i] : HV values for channel i
    [1][i] : result of the function for channel i
    '''
    nb_saves = len(signal_list)
    result = []

    for i in range(nb_saves) :
        res, ylabel = func(t_list[i], signal_list[i])
        if res != 0 :
            result.append(res)
        else :
            low_ampl_list.pop(i)
    if plot :
        plt.clf()
        plt.title(title)
        plt.xlabel("lower value of the amplitude interval (V)")
        plt.ylabel(ylabel)
        plt.scatter(low_ampl_list, result, color="#0303A0")
        plt.grid()
        plt.show()
    return(low_ampl_list, result)