import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import cblind as cb

from data_read import fast_folder_reader

global_dir = "../Data19-05/chip3"
chip_dir = "cactus_gl_chip3-HRHG-HV0"

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

def get_folder_name (folder_path : str) -> str :
    '''
    Extracts the name of a folder from its path.

    Argument :
    folder_path - str : path of the folder

    Return :
    folder_name - str : name of the folder
    '''
    folder_path_split = folder_path.split("/")
    if folder_path_split[-1] == "" :
        folder_name = folder_path_split[-2]
    else :
        folder_name = folder_path_split[-1]
    return (folder_name)

def sort_multiple_lists (list_sort : list | np.ndarray, list_other : list[list | np.ndarray]) -> None :
    '''
    Sorts in place multiple lists or arrays at the same time.

    Arguments :
    list_sort - list or numpy.array : list/array on which you want to sort
    list_other - list of list or numpy.array : other lists/arrays you want to sort simultaneously
    '''
    nb_saves = len(list_sort)
    for i in range (nb_saves) :
        minimum = min(list_sort[i:])
        for j in range (i,nb_saves) :
            if list_sort[j] == minimum :
                list_sort[i], list_sort[j] = list_sort[j], list_sort[i]
                for l in list_other :
                    l[i], l[j] = l[j], l[i]
                break

def group_multiple_lists (list_group : list, list_other : list[list]) -> None :
    i = 0
    while i < len(list_group) :
        j = i+1
        j_group = []
        while (j < len(list_group)) and (list_group[i] == list_group[j]) :
            j_group.append(j)
            list_group.pop(j)
        for l in list_other :
            l_group = []
            for k in j_group :
                l_group.append(l.pop(k))
            l[i] = tuple([l[i]] + l_group)
        i += 1

def standardize (curve : np.ndarray) -> np.ndarray :
    ''' 
    Sets the values of a curve between 0 and 1.

    Argument :
    curve - numpy.array of float (or int) : initial values

    Return :
    curve - numpy.array of float (or int) : standardized values
    '''
    min_value = min(curve)
    max_value = max(curve)
    for i in range(len(curve)) :
        curve[i] = (curve[i]-min_value) / (max_value - min_value)
    return (curve)

def plot_pulse (t : np.ndarray, signal : np.ndarray, title : str = "", show : bool = True, label : str = "", color = None, linestyle = None) -> None :
    '''
    Plot a given mean pulse.

    Arguments :
    t - numpy.array of float : time in seconds
    signal - numpy.array of float : signal in V
    title - (optional) str : name that will appear in the title
    show - (optional) bool : whether you want to show the plot
    label - (optional) str : label you might want on your curve
    color - (optional) : color of the curve
    linestyle - (optional) : linestyle of the curve

    Return : None
    '''
    plt.xlabel("time (s)")
    plt.ylabel("amplitude (V)")
    plt.title(title)
    plt.plot(t, signal, label = label, color=color, linestyle=linestyle)
    if show :
        plt.grid()
        plt.show()

def plot_save (file_path : str, title : str = "", show : bool = True, label : str = "", standard : bool = False, color = None, linestyle = None, window : list = []) -> None :
    '''
    Plot the mean pulse from a save.

    Argument :
    file_path - str : path of the save file
    title - (optional) str : name that will appear in the title
    show - (optional) bool : whether you want to show the plot
    label - (optional) str : label you might want on your curve
    standard - (optional) bool : whether you want the data to be standardized (default is False)
    color - (optional) : color of the curve
    linestyle - (optional) : linestyle of the curve

    Return : None
    '''
    df = pd.read_csv(file_path)
    t = np.array(df["Time"])
    signal = np.array(df["Ampl"])
    if len(window) != 0 :
        t = t[window[0]:window[1]]
        signal = signal[window[0]:window[1]]
    if standard :
        signal = standardize(signal)
    if title == "" :
        title = file_path.split("/")[-1].split(".")[0] # get the file name by taking the last string after "/" and removing possible extensions
    plot_pulse (t, signal, title = title, show = show, label = label, color=color, linestyle=linestyle)

def mean_pulse_v1 (data_signal_list : list[pd.DataFrame], n_seg : list[int], len_seg : int) -> tuple[np.ndarray, np.ndarray] :
    '''
    Calculate the mean pulse of a experimentation on cactus. First version that disregards small differences of time origin between different pulses.

    Arguments :
    data_signal_list - list of pandas.DataFrame : list of all signals read from the .txt files
    n_seg - list of int : number of segments per file for each file
    len_seg - int : number of frames per segment (needs to be the same for each segment)

    Returns : ([float], [float])
    t - numpy.array of float : time scale of the first segment
    mean_pulse - numpy.array of float : mean signal of all segments
    '''
    t = np.array(data_signal_list[0])[0:len_seg,0] # Using the time scale of the first pulse
    
    nb_files = len(data_signal_list)
    total_n_seg = sum(n_seg)
    pulse_list = np.zeros((len_seg, total_n_seg))
    for i in range(nb_files) :
        for j in range(n_seg[i]) :
            begin = j*len_seg # beginning of the pulse
            end = (j+1)*len_seg # ending of the pulse
            pulse_list[:,sum(n_seg[:i])+j] = np.array(data_signal_list[i])[begin:end,1]
    
    mean_pulse = np.mean(pulse_list, axis=1)
    
    return (t, mean_pulse)

def linear_interpolation (t : float, t1 : float, x1 : float, t2 : float, x2 : float) -> float :
    '''
    Calculates the value of a point using linear interpolation with two other points.

    Arguments :
    t - float : time of the point wanted
    t1 - float : time of the first given point
    x1 - float : value of the first given point
    t2 - float : time of the second given point
    x2 - float : value of the second given point

    Returns : float
    '''
    return ((t-t1)*(x2-x1)/(t2-t1) + x1)

def mean_pulse_v2 (data_signal_list : list[pd.DataFrame], n_seg : list[int], len_seg : list[int]) -> tuple[np.ndarray, np.ndarray] :
    '''
    More advanced way to calculate the mean pulse using linear interpolation on time points from the first segment.
    No limitation on time scale structure.

    Arguments :
    data_signal_list - list of pandas.DataFrame : list of all signals read from the .txt files
    n_seg - list of int : number of segments per file for each file
    len_seg - list of int : number of frames per segment for each file

    Returns : ([float], [float])
    t - numpy.array of float : time scale of the first segment
    mean_pulse - numpy.array of float : mean signal of all segments
    '''
    t = np.array(data_signal_list[0])[0:len_seg[0],0] # Using the time scale of the first pulse

    nb_files = len(data_signal_list)
    total_n_seg = sum(n_seg)
    pulse_list = np.zeros((len_seg[0], total_n_seg))
    for i in range(nb_files) :
        t_full = np.array(data_signal_list[i])[:,0]
        signal = np.array(data_signal_list[i])[:,1]
        for k in range(len_seg[i]) : # calculating the correct interpolation for each time value
            dt = np.abs(t_full - t[k])
            for j in range(n_seg[i]) :
                begin = j*len_seg[i] # beginning of the pulse
                end = (j+1)*len_seg[i] # ending of the pulse
                dt_crop = dt[begin:end]
                i1 = np.argmin(dt_crop) + begin # index of the time value of the current pulse, that is the closest to the current time value of the first pulse
                t1 = t_full[i1]
                x1 = signal[i1]
                dt_crop[i1-begin] = np.inf
                i2 = np.argmin(dt_crop) + begin # index of the time value of the current pulse, that is the second closest to the current time value of the first pulse
                t2 = t_full[i2]
                x2 = signal[i2]
                x = linear_interpolation(t[k], t1, x1, t2, x2)
                pulse_list[k,sum(n_seg[:i])+j] = x
    
    mean_pulse = np.mean(pulse_list, axis=1)

    return (t, mean_pulse)

def mean_pulse_v3 (data_signal_list : list[pd.DataFrame], n_seg : list[int], len_seg : int) -> tuple[np.ndarray, np.ndarray] :
    '''
    More advanced way to calculate the mean pulse using linear interpolation on time points from the first segment. Faster than v2 (using the construction of time scales).
    Time scales need to be sequenced : a point of time in a given time scale needs to be in between of two points of time in the other time scale.
    
    Arguments :
    data_signal_list - list of pandas.DataFrame : list of all signals read from the .txt files
    n_seg - list of int : number of segments per file for each file
    len_seg - int : number of frames per segment (needs to be the same for each segment)

    Returns : ([float], [float])
    t - numpy.array of float : time scale of the first segment
    mean_pulse - numpy.array of float : mean signal of all segments
    '''
    t = np.array(data_signal_list[0])[0:len_seg,0] # Using the time scale of the first pulse

    nb_files = len(data_signal_list)
    total_n_seg = sum(n_seg)
    pulse_list = np.zeros((len_seg, total_n_seg))
    for i in range(nb_files) :
        t_full = np.array(data_signal_list[i])[:,0]
        signal = np.array(data_signal_list[i])[:,1]
        for j in range(n_seg[i]) :
            begin = j*len_seg # beginning of the pulse
            end = (j+1)*len_seg # ending of the pulse
            index = sum(n_seg[:i])+j # index of the working segment
            if t_full[begin] > t[0] : # when the first time value is from the first pulse, we need to calculate the first time value of the first pulse differently
                pulse_list[0,index] = linear_interpolation(t[0], t_full[begin], signal[begin], t_full[begin+1], signal[begin+1])
                for k in range (1,len_seg) :
                    pulse_list[k,index] = linear_interpolation(t[k], t_full[begin+k-1], signal[begin+k-1], t_full[begin+k], signal[begin+k])
            else : # when the first time value is from the working pulse, we need to calculate the last time value of the first pulse differently
                for k in range (len_seg-1) :
                    pulse_list[k,index] = linear_interpolation(t[k], t_full[begin+k], signal[begin+k], t_full[begin+k+1], signal[begin+k+1])
                pulse_list[len_seg-1,index] = linear_interpolation(t[len_seg-1], t_full[begin+len_seg-2], signal[begin+len_seg-2], t_full[begin+len_seg-1], signal[begin+len_seg-1])
                
    mean_pulse = np.mean(pulse_list, axis=1)

    return (t, mean_pulse)

def mean_pulse_v4 (data_signal_list : list[pd.DataFrame], n_seg : list[int], len_seg : list[int]) -> tuple[np.ndarray, np.ndarray] :
    '''
    More advanced way to calculate the mean pulse using linear interpolation on time points from the first segment. Faster than v2 (using the construction of time scales but less than v3).
    Time scales need to be in ascending order.
    
    Arguments :
    data_signal_list - list of pandas.DataFrame : list of all signals read from the .txt files
    n_seg - list of int : number of segments per file for each file
    len_seg - list of int : number of frames per segment for each file
        WARNING : the first file needs to have the most number of points per segment

    Returns : ([float], [float])
    t - numpy.array of float : time scale of the first segment
    mean_pulse - numpy.array of float : mean signal of all segments
    '''
    nb_files = len(data_signal_list)
    t = np.array(data_signal_list[0])[0:len_seg[0],0]
    total_n_seg = sum(n_seg)
    pulse_list = np.zeros((len_seg[0], total_n_seg))
    for i in range(nb_files) :
        t_full = np.array(data_signal_list[i])[:,0]
        signal = np.array(data_signal_list[i])[:,1]
        for j in range(n_seg[i]) :
            begin = j*len_seg[i] # beginning of the pulse
            end = (j+1)*len_seg[i] # ending of the pulse
            index = sum(n_seg[:i])+j # index of the working segment
            i1 = begin
            i2 = begin + 1
            for k in range (len_seg[0]) :
                if t[k] < t_full[i2] : # we are still before i2 (between i1 and i2, except for initialisation)
                    pulse_list[k,index] = linear_interpolation(t[k], t_full[i1], signal[i1], t_full[i2], signal[i2])
                elif i2 < end-1 : # we are after i2 but before the end of the segment
                    i1 += 1
                    i2 += 1
                    pulse_list[k,index] = linear_interpolation(t[k], t_full[i1], signal[i1], t_full[i2], signal[i2])
                else : # we reach the end of the segment
                    pulse_list[k,index] = linear_interpolation(t[k], t_full[i1], signal[i1], t_full[i2], signal[i2])

    mean_pulse = np.mean(pulse_list, axis=1)

    return (t, mean_pulse)

def mean_pulse (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], parameters : tuple, global_name : str = "", HV_value : int = 0, version : int = 4, invert : bool | tuple[bool, ...] = False, verbose : bool = False, n_seg : int = -1, len_seg : int = -1, reader_result : tuple = ()) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str], list[str]] :
    '''
    Calculate the mean pulse of a given set of experimentations on cactus. Can be used with get_stats function.

    Arguments :
    folder_path - str or tuple of str : path of the folder(s)
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    parameters - tuple : None
    global_name - (optional) str : name of the global working folder
    HV_value - (optional) int : HV value corresponding to the working folder
    version - (optional) int : version of the mean pulse calculation desired (1, 2, 3 or 4) (default is 4)
    invert - (optional) bool or tuple of bool : whether you want the signal to be inverted, for each channel (default is False)
    verbose - (optional) bool : whether you want the function to print its status (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment
    reader_result - (optional) tuple : result of the fast_folder_reader function

    Returns : ([np.array], [np.array], [str], [str], [str])
    stat1_list - list of np.array : time (s)
    stat2_list - list of np.array : signal (V)
    stat1_name_list - list of str : "Time"
    stat2_name_list - list of str : "Ampl"
    save_name_list - list of str : names of the save file
    '''
    if verbose :
            print("Starting mean pulse calculations for " + str(folder_path))

    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)

    if type(folder_path) == str :
        folder_path = (folder_path,)

    if type(invert) == bool :
        invert = (invert,) * nb_channels
    
    if len(reader_result) == 0 :
        n_seg_list, len_seg_list, data_signal_list, name_list = fast_folder_reader(folder_path, channel, n_seg=n_seg, len_seg=len_seg)
    else :
        n_seg_list, len_seg_list, data_signal_list, name_list = reader_result
    
    for i in range (nb_channels) :
        if invert[i] :
            for df in data_signal_list[i] :
                df["Ampl"] = - df["Ampl"]
    
    stat1_list = []
    stat2_list = []
    stat1_name_list = ["Time"] * nb_channels
    stat2_name_list = ["Ampl"] * nb_channels
    save_name_list = []

    for i in range (nb_channels) :
        if verbose :
            print("Starting channel " + channel[i])
        if len(data_signal_list[i]) == 0 :
            if verbose :
                print("No signal")
        else :
            if version == 1 :
                mean_pulse = mean_pulse_v1(data_signal_list[i], n_seg_list[i], len_seg_list[i][0])
            elif version == 2 :
                mean_pulse = mean_pulse_v2(data_signal_list[i], n_seg_list[i], len_seg_list[i])
            elif version == 3 :
                mean_pulse = mean_pulse_v3(data_signal_list[i], n_seg_list[i], len_seg_list[i][0])
            elif version == 4 :
                mean_pulse = mean_pulse_v4(data_signal_list[i], n_seg_list[i], len_seg_list[i])
            else :
                raise Exception("Unexpected version")

            stat1_list.append(mean_pulse[0])
            stat2_list.append(mean_pulse[1])
            save_name_list.append(get_folder_name(folder_path[0]) + "_" + channel[i] + "_mean_pulse")
    
    return(stat1_list, stat2_list, stat1_name_list, stat2_name_list, save_name_list)

def get_mean_pulse (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], version : int = 4, plot : bool = False, save : bool = False, save_path : str = "saves", invert : bool | tuple[bool, ...] = False, verbose : bool = False, n_seg : int = -1, len_seg : int = -1, reader_result : tuple = ()) -> list[tuple[np.ndarray, np.ndarray]] :
    '''
    Calculate the mean pulse of a given set of experimentations on cactus. Data needs to be in csv-like files in the given folder.

    Arguments :
    folder_path - str or tuple of str : path of the folder(s)
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    version - (optional) int : version of the mean pulse calculation desired (1, 2, 3 or 4) (default is 4)
    plot - (optional) bool : whether you want to plot the results (defaut is False) 
    save - (optional) bool : whether you want to save the results in a new .csv file (defaut is False)
    save_path - (optional) str : where you want to save (default is "saves")
    invert - (optional) bool or tuple of bool : whether you want the signal to be inverted, for each channel (default is False)
    verbose - (optional) bool : whether you want the function to print its status (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment
    reader_result - (optional) tuple : result of the fast_folder_reader function

    Returns : [([float], [float])]
    t - numpy.array of float : time scale of the first segment
    mean_pulse - numpy.array of float : mean signal of all segments

    examples :
    [i][0] : array of time for channel i
    [i][1] : array of values for channel i
    '''
    mean_pulse_list = []
    folder_name = ""

    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)

    if type(folder_path) == str :
        folder_path = (folder_path,)

    if type(invert) == bool :
        invert = (invert,) * nb_channels

    if save_path[-1] != "/" :
        save_path += "/"
    
    n_seg_list = [[] for i in range (nb_channels)]
    len_seg_list = [[] for i in range (nb_channels)]
    data_signal_list = [[] for i in range (nb_channels)]
    name_list = [[] for i in range (nb_channels)]
    for path in folder_path :
        folder_name = get_folder_name(path)
        if verbose :
            print("Starting mean pulse calculations for " + folder_name)
        if len(reader_result) == 0 :
            n_seg_list_, len_seg_list_, data_signal_list_, name_list_ = fast_folder_reader(path, channel, n_seg=n_seg, len_seg=len_seg)
        else :
            n_seg_list_, len_seg_list_, data_signal_list_, name_list_ = reader_result
        for i in range (nb_channels) :
            n_seg_list[i] = n_seg_list[i] + n_seg_list_[i]
            len_seg_list[i] = len_seg_list[i] + len_seg_list_[i]
            data_signal_list[i] = data_signal_list[i] + data_signal_list_[i]
            name_list[i] = name_list[i] + name_list_[i]
    for i in range (nb_channels) :
        if invert[i] :
            for df in data_signal_list[i] :
                df["Ampl"] = - df["Ampl"]
    
    for i in range (nb_channels) :
        if verbose and type(channel) == str :
            print("Starting channel " + channel[i])
        if len(data_signal_list[i]) == 0 :
            mean_pulse_list.append((np.array([]), np.array([])))
        else :
            if version == 1 :
                mean_pulse = mean_pulse_v1(data_signal_list[i], n_seg_list[i], len_seg_list[i][0])
            elif version == 2 :
                mean_pulse = mean_pulse_v2(data_signal_list[i], n_seg_list[i], len_seg_list[i])
            elif version == 3 :
                mean_pulse = mean_pulse_v3(data_signal_list[i], n_seg_list[i], len_seg_list[i][0])
            elif version == 4 :
                mean_pulse = mean_pulse_v4(data_signal_list[i], n_seg_list[i], len_seg_list[i])
            else :
                raise Exception("Unexpected version")

            if plot :
                plot_pulse(mean_pulse[0], mean_pulse[1], title = folder_name + " " + channel[i])
            if save :
                stack = np.stack(mean_pulse, axis=1)
                df = pd.DataFrame(stack, columns=["Time", "Ampl"])
                df.to_csv(save_path + folder_name + "_" + channel[i] + "_mean_pulse")
                
            mean_pulse_list.append(mean_pulse)
    return(mean_pulse_list)

def mean_pulse_strontium_v4 (data_signal_list : list[pd.DataFrame], n_seg : list[int], len_seg : list[int], intervals : list[float] = [], size_interval : int = 10) -> tuple[np.ndarray, list[tuple[np.ndarray, int, int]]] :
    '''
    More advanced way to calculate the mean pulse using linear interpolation on time points from the first segment. Faster than v2 (using the construction of time scales but less than v3).
    Time scales need to be in ascending order.
    
    Arguments :
    data_signal_list - list of pandas.DataFrame : list of all signals read from the .txt files
    n_seg - list of int : number of segments per file for each file
    len_seg - list of int : number of frames per segment for each file
        WARNING : the first file needs to have the most number of points per segment
    intervals - (optional) list of floats : limits of the intervals (in mV), including both ends
    size_interval - (optional) int : size of amplitude intervals, in mV (defaut is 10mV)

    Returns : ([float], [([float], float, float)])
    t - numpy.array of float : time scale of the first segment
    mean_pulse - numpy.array of float : mean signal of all segments
    start_interval - int : starting point of the amplitude interval (in mV)
    end_interval - int : ending point of the amplitude interval (in mV)

    examples :
    [0] : array of time (same for all intervals)
    [1][i][0] : array of values for interval i
    [1][i][1] : minimum of amplitude for interval i
    [1][i][2] : maximum of amplitude for interval i
    '''
    nb_files = len(data_signal_list)
    t = np.array(data_signal_list[0])[0:len_seg[0],0]
    result = []
    if len(intervals) > 0 :
        pulse_list = []
        for i in range(1, len(intervals)) :
            pulse_list.append(([], intervals[i-1], intervals[i])) # each interval is a tuple in the list
    else :
        pulse_list = [([], 0, size_interval)] # each interval is a tuple in the list
    for i in range(nb_files) :
        t_full = np.array(data_signal_list[i])[:,0]
        signal = np.array(data_signal_list[i])[:,1]
        for j in range(int(len(signal) / len_seg[i])) :
            begin = j*len_seg[i] # beginning of the pulse
            end = (j+1)*len_seg[i] # ending of the pulse

            maximum = max(signal[begin:end])*1e3 # max of the signal in mV

            i_signal = 0
            pulse_list_low = pulse_list[i_signal][1]
            pulse_list_high = pulse_list[i_signal][2]
            while not(pulse_list_low <= maximum < pulse_list_high) : # searching for the correct interval
                i_signal += 1
                if (len(intervals) == 0) and (i_signal == len(pulse_list)) :
                    pulse_list.append(([], pulse_list_high, pulse_list_high + size_interval))
                elif (i_signal == len(pulse_list)) :
                    break
                pulse_list_low = pulse_list[i_signal][1]
                pulse_list_high = pulse_list[i_signal][2]
            pulse_list[i_signal][0].append(np.zeros(len_seg[0]))

            i1 = begin
            i2 = begin + 1
            for k in range (len_seg[0]) :
                if t[k] < t_full[i2] : # we are still before i2 (between i1 and i2, except for initialisation)
                    pulse_list[i_signal][0][-1][k] = linear_interpolation(t[k], t_full[i1], signal[i1], t_full[i2], signal[i2])
                elif i2 < begin+len_seg[i]-1 : # we are after i2 but before the end of the segment
                    i1 += 1
                    i2 += 1
                    pulse_list[i_signal][0][-1][k] = linear_interpolation(t[k], t_full[i1], signal[i1], t_full[i2], signal[i2])
                else : # we reach the end of the segment
                    pulse_list[i_signal][0][-1][k] = linear_interpolation(t[k], t_full[i1], signal[i1], t_full[i2], signal[i2])

    for p in pulse_list :
        print(str(p[1]) + "mV-" + str(p[2]) + "mV : " + str(len(p[0])))

    for i in range (len(pulse_list)) :
        if not(len(pulse_list[i][0]) == 0) :
            result.append((np.mean(np.array(pulse_list[i][0]), axis = 0), pulse_list[i][1], pulse_list[i][2]))

    return (t, result)

def get_mean_pulse_strontium (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], version : int = 4, max_ampl : int = -1, min_ampl : int = 0, intervals : list[float] = [], size_interval : int | tuple[int, ...] = 10, plot : bool = False, save : bool = False, save_path : str = "saves", verbose : bool = False, invert : bool | tuple[bool, ...] = False, filter_noise : float | tuple[float, ...] = 1.0, filter_saturation : float | tuple[float, ...] = 100.0, n_seg : int = -1, len_seg : int = -1, reader_result : tuple = ()) -> list[tuple[np.ndarray, list[tuple[np.ndarray, int, int]]]] :
    '''
    Calculate the mean pulse of a given set of experimentations on cactus. Data needs to be in csv-like files in the given folder.

    Arguments :
    folder_path - str or tuple of str : path of the folder(s)
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    version - int : version of the mean pulse calculation desired (1, 2, 3 or 4)
    max_ampl - (optional) int : amplitude max wanted (in mV) (default : keep all)
    min_ampl - (optional) int : amplitude min wanted (in mV) (default is 0)
    intervals - (optional) list of floats : limits of the intervals (in mV), including both ends
    size_interval - (optional) int or tuple of int : size of amplitude intervals for each channel, in mV (defaut is 10mV)
    plot - (optional) bool : whether you want to plot the results (defaut is False)
    save - (optional) bool : whether you want to save the results in a new .csv file (defaut is False)
    save_path - (optional) str : where you want to save (default is "saves")
    verbose - (optional) bool : whether you want the function to print its status (default is False)
    invert - (optional) bool or tuple of bool : whether you want the signal to be inverted, for each channel (default is False)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment
    reader_result - (optional) tuple : result of the fast_folder_reader function

    Returns : [([float], [([float], float, float)])]
    t - numpy.array of float : time scale of the first segment
    mean_pulse - numpy.array of float : mean signal of all segments
    start_interval - int : starting point of the amplitude interval (in mV)
    end_interval - int : ending point of the amplitude interval (in mV)

    examples :
    [i][0] : array of time for channel i
    [i][1][j][0] : array of values for channel i and interval j
    [i][1][j][1] : minimum of amplitude for channel i and interval j
    [i][1][j][2] : maximum of amplitude for channel i and interval j
    '''

    mean_pulse_list = []
    folder_name = ""

    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)

    if type(folder_path) == str :
        folder_path = (folder_path,)

    if type(invert) == bool :
        invert = (invert,) * nb_channels

    if type(filter_noise) == float :
        filter_noise = (filter_noise,) * nb_channels

    if type(filter_saturation) == float :
        filter_saturation = (filter_saturation,) * nb_channels

    if type(size_interval) == int :
        size_interval = (size_interval,) * nb_channels

    if save_path[-1] != "/" :
        save_path += "/"
    
    n_seg_list = [[] for i in range (nb_channels)]
    len_seg_list = [[] for i in range (nb_channels)]
    data_signal_list = [[] for i in range (nb_channels)]
    name_list = [[] for i in range (nb_channels)]
    for path in folder_path :
        folder_name = get_folder_name(path)
        if verbose :
            print("Starting mean pulse calculations for " + folder_name)
        if len(reader_result) == 0 :
            n_seg_list_, len_seg_list_, data_signal_list_, name_list_ = fast_folder_reader(path, channel, n_seg=n_seg, len_seg=len_seg, filter_noise=filter_noise, filter_saturation=filter_saturation, invert=invert)
        else :
            n_seg_list_, len_seg_list_, data_signal_list_, name_list_ = reader_result
        for i in range (nb_channels) :
            n_seg_list[i] = n_seg_list[i] + n_seg_list_[i] # pb out of range ??? (HRHG_struct2 pour les 2 channels)
            len_seg_list[i] = len_seg_list[i] + len_seg_list_[i]
            data_signal_list[i] = data_signal_list[i] + data_signal_list_[i]
            name_list[i] = name_list[i] + name_list_[i]
    # for i in range (nb_channels) :
    #     if invert[i] :
    #         for df in data_signal_list[i] :
    #             df["Ampl"] = - df["Ampl"]
    #     for m in range(len(data_signal_list[i])) :
    #         k_drop = 0
    #         for k in range (n_seg_list[i][m]) :
    #             df = data_signal_list[i][m]
    #             t, signal = get_pulse(df, len_seg_list[i][m], k - k_drop)
    #             print(filter_noise[i])
    #             if not(test_noise(signal, filter_noise[i])) or not(test_saturation(signal, filter_saturation[i])) :
    #                 begin = k*len_seg_list[i][m]
    #                 end = (k+1)*len_seg_list[i][m]
    #                 data_signal_list[i][m] = df.drop(index=list(range(begin, end)))
    #                 k_drop += 1
                

    for i in range (nb_channels) :
        if verbose :
            print("Starting channel " + channel[i])
        if len(data_signal_list[i]) == 0 :
            mean_pulse_list.append((np.array([]), np.array([]), 0, 0))
        else :
            #if version == 1 :
            #    mean_pulse = mean_pulse_v1(data_signal_list[i], n_seg_list[i], len_seg_list[i][0])
            #elif version == 2 :
            #    mean_pulse = mean_pulse_v2(data_signal_list[i], n_seg_list[i], len_seg_list[i])
            #elif version == 3 :
            #    mean_pulse = mean_pulse_v3(data_signal_list[i], n_seg_list[i], len_seg_list[i][0])
            if version == 4 :
                mean_pulse = mean_pulse_strontium_v4(data_signal_list[i], n_seg_list[i], len_seg_list[i], size_interval=size_interval[i], intervals = intervals)
            else :
                raise Exception("Unexpected version")

            if plot :
                to_plot = []
                for interval in mean_pulse[1] :
                    if (min_ampl <= interval[1]) and (max_ampl == -1 or interval[2] <= max_ampl) :
                        to_plot.append(interval)
                nb_plot = len(to_plot)
                color, linestyle = cb.Colorplots().extreme_rainbow(nb_plot)
                for interval in to_plot :
                    c = color[i%34]
                    l = linestyle[i//34]
                    plot_pulse(mean_pulse[0], interval[0], title = folder_name + " " + channel[i], label="["+str(interval[1])+"mV, "+str(interval[2])+"mV]", show=False, color=c, linestyle=l)
                plt.legend()
                plt.grid()
                plt.show()
            if save :
                for interval in mean_pulse[1] :
                    stack = np.stack((mean_pulse[0], interval[0]), axis=1)
                    df = pd.DataFrame(stack, columns=["Time", "Ampl"])
                    df.to_csv(save_path + folder_name + "_" + channel[i] + "_mean_pulse_" + str(interval[1]) + "-" + str(interval[2]) + "mV")
            mean_pulse_list.append(mean_pulse)
    
    return(mean_pulse_list)

def plot_save_strontium (folder_path : str, channel : str, HV_value : int, title : str = "", max_ampl : int = -1, min_ampl : int = 0, show : bool = True, label : str = "", standard : bool = False, window : list = []) -> None :
    '''
    Plot the mean pulse from a save.

    Argument :
    folder_path - str : path of the save folder
    channel - str : which channel you want to plot
    HV_value - int : which value of HV you want to plot
    max_ampl - (optional) int : amplitude max wanted (in mV) (default : keep all)
    min_ampl - (optional) int : amplitude min wanted (in mV) (default is 0)
    show - (optional) bool : whether you want to show the plot
    label - (optional) str : label you might want on your curve
    standard - (optional) bool : whether you want the data to be standardized (default is False)

    Return : None
    '''
    file_path = []
    file_name = []
    low_ampl = []
    high_ampl = []
    for f in os.scandir(folder_path) :
        file_path.append(f.path)
        file_name.append(f.name)
        file_split = file_name[-1].split("_")[-1].split("-")
        low_ampl.append(float(file_split[0]))
        high_ampl.append(float(file_split[1].split("mV")[0]))

    folder_name = get_folder_name(folder_path)

    sort_multiple_lists(low_ampl, [high_ampl, file_name, file_path])
    nb_saves = len(file_path)

    i_plot = []
    for i in range (nb_saves):
        if ((str(HV_value)+"V" in file_name[i]) or ("HV"+str(HV_value) in file_name[i])) and (channel in file_name[i]) and (min_ampl <= low_ampl[i]) and (max_ampl == -1 or high_ampl[i] <= max_ampl) :
            i_plot.append(i)
    nb_plot = len(i_plot)
    color, linestyle = cb.Colorplots().extreme_rainbow(nb_plot)
    for i in range (nb_plot) :
        c = color[i%34]
        l = linestyle[i//34]
        #plot_save(file_path[i_plot[i]], title=folder_name + " " + channel + " HV" + str(HV_value) , show=False, label="["+str(low_ampl[i_plot[i]])+"mV, "+str(high_ampl[i_plot[i]])+"mV]", standard=standard, color=c, linestyle=l)
        plot_save(file_path[i_plot[i]], title="Convolved signal " + title, show=False, label="["+str(low_ampl[i_plot[i]])+"mV, "+str(high_ampl[i_plot[i]])+"mV]", standard=standard, color=c, linestyle=l, window=window)
    if show :
        if standard :
            plt.ylabel("normalized amplitude")
        else :
            plt.ylabel("amplitude (V)")
        plt.grid()
        plt.legend(title = "amplitude interval")
        plt.show()

def combine_mean_pulse (paths : list, nb_events : list, save_path : str) :
    total_events = sum(nb_events)
    t = np.array(pd.read_csv(paths[0])["Time"])
    for i in range(len(paths)) :
        mean_pulse = np.array(pd.read_csv(paths[i])["Ampl"])
        if i == 0 :
            total_mean_pulse = np.zeros(mean_pulse.shape)
        total_mean_pulse += (mean_pulse * nb_events[i] / total_events)
    df = pd.DataFrame(np.stack((t, total_mean_pulse), axis=1), columns=["Time", "Ampl"])
    df.to_csv(save_path)