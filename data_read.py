import numpy as np
import pandas as pd 
import os
import zipfile

from utils import test_noise, test_saturation, test_noise_old

len_header = 2 #number of lines before the header of the first data set (default is 2)
header_info_line = 1 #index (starting from 0) of the line of the header containing information about the number of segments and the number of frame per segment
n_seg_i = 1 #index (starting from 0) of the number of segments in the header (default is 1)
len_seg_i = 3 #index (starting from 0) of the number of frame per segment in the header (default is 3)

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

def get_serial_number (file_name : str) -> str :
    '''
    Extracts the serial number from a file name.

    Argument :
    file_name - str : the name of the file

    Return :
    serial_number - str : the serial number of the file (it needs to contain a "0")
    '''
    serial_number = ""
    for x in file_name.split(".")[0].split("--") : # Splitting the extension .txt, then splitting file name with "--"
        if "0" in x : # Searching for "0" in the name to get the serial number (problem if no "0" in the serial number)
            serial_number = x
    if serial_number == "" :
        raise Exception ("serial number not found")
    return (serial_number)

def file_reader (file_path : str, n_seg : int = -1, len_seg : int = -1) -> tuple[int, int, pd.DataFrame, pd.DataFrame, pd.DataFrame] :
    '''
    Reads a file from the cactus experimentation with header, time data and signal data.

    Arguments :
    file_path - str : path of the .txt data file
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment

    Returns : (int, int, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
    n_seg - int : number of segments
    len_seg - int : number of frame per segment
    data_header - pandas.DataFrame : header of the data file
    data_time - pandas.DataFrame : data frame about time
    data_signal - pandas.DataFrame : data frame with time and signal
    '''

    data_header = pd.read_csv(file_path, header = 0, low_memory=False)[:len_header-1] # Reading the header of the file
    data_header_array = np.array(data_header)

    if n_seg == -1 :
        n_seg = int(data_header_array[header_info_line-1,n_seg_i-1]) # Searching for the number of segments if needed
    if len_seg == -1 :
        len_seg = int(data_header_array[header_info_line-1,len_seg_i-1]) # Searching for the number of frames per segment if needed

    data_time = pd.read_csv(file_path, header = len_header)[:n_seg]
    data_signal = pd.read_csv(file_path, header = n_seg+len_header+1)

    return (n_seg, len_seg, data_header, data_time, data_signal)

def folder_reader (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], n_seg : int = -1, len_seg : int = -1) -> tuple[list[list[int]], list[list[int]], list[list[pd.DataFrame]], list[list[pd.DataFrame]], list[list[pd.DataFrame]], list[list[str]]] :
    '''
    Reads a folder from the cactus experimentation.

    Arguments :
    folder_path - str or tuple of str : path of the folder(s)
    channel - str or tuple of str : name of the channel desired in the file names (ex : "C1" or "C3") (1 or more channels accepted)
    n_seg_ - (optional) int : number of segments
    len_seg_ - (optional) int : number of frame per segment

    Returns : ([[int]], [[int]], [[pandas.DataFrame]], [[pandas.DataFrame]], [[pandas.DataFrame]], [[str]])
    n_seg_list - list of list of int : number of segments for each file from each channel
    len_seg_list - list of list of int : number of frame per segment for each file from each channel
    data_header_list - list of list of pandas.DataFrame : header of each data file from each channel
    data_time_list - list of list of pandas.DataFrame : data frame about time for each file from each channel
    data_signal_list - list of list of pandas.DataFrame : data frame with time and signal for each file from each channel
    serial_number_list - list of list of str : serial number of each file from each channel

    examples :
    [0][i][j] : number of segments for channel i in file j
    [4][i][j] : pandas.Dataframe with signal for channel i in file j
    '''

    # Initialisation
    if type(channel) == str :
        channel = (channel,)
    nb_channels = len(channel)

    if type(folder_path) == str :
        folder_path = (folder_path,)

    n_seg_list = [[] for i in range (nb_channels)]
    len_seg_list = [[] for i in range (nb_channels)]
    data_header_list = [[] for i in range (nb_channels)]
    data_time_list = [[] for i in range (nb_channels)]
    data_signal_list = [[] for i in range (nb_channels)]
    serial_number_list = [[] for i in range (nb_channels)]

    n_seg_given = False
    len_seg_given = False
    if n_seg != -1 :
        n_seg_given = True
    if len_seg != -1 :
        len_seg_given = True

    # Reading each file in the folder
    for path in folder_path :
        if path[-4:] == ".zip" :
            with zipfile.ZipFile(path) as z:
                for file in z.namelist():
                    with z.open(file) as f:
                        file_name = f.name
                        for i in range(nb_channels) :
                            if channel[i] in file_name :
                                serial_number = get_serial_number(file_name)
                                if not(n_seg_given) :
                                    n_seg = -1 # Reinitiate n_seg between each file, if it wasn't given at the start
                                if not(len_seg_given) :
                                    len_seg = -1 # Reinitiate len_seg between each file, if it wasn't given at the start
                                n_seg, len_seg, data_header, data_time, data_signal = file_reader(f, n_seg = n_seg, len_seg = len_seg)
                                n_seg_list[i].append(n_seg)
                                len_seg_list[i].append(len_seg)
                                data_header_list[i].append(data_header)
                                data_time_list[i].append(data_time)
                                data_signal_list[i].append(data_signal)
                                serial_number_list[i].append(serial_number)
        else :
            for f in os.scandir(path) :
                if f.is_file() :
                    file_name = f.name
                    file_path = f.path
                    for i in range(nb_channels) :
                        if channel[i] in file_name :
                            serial_number = get_serial_number(file_name)
                            if not(n_seg_given) :
                                n_seg = -1 # Reinitiate n_seg between each file, if it wasn't given at the start
                            if not(len_seg_given) :
                                len_seg = -1 # Reinitiate len_seg between each file, if it wasn't given at the start
                            n_seg, len_seg, data_header, data_time, data_signal = file_reader(file_path, n_seg = n_seg, len_seg = len_seg)
                            n_seg_list[i].append(n_seg)
                            len_seg_list[i].append(len_seg)
                            data_header_list[i].append(data_header)
                            data_time_list[i].append(data_time)
                            data_signal_list[i].append(data_signal)
                            serial_number_list[i].append(serial_number)
    
    return (n_seg_list, len_seg_list, data_header_list, data_time_list, data_signal_list, serial_number_list)

def fast_file_reader (file_path : str, n_seg : int = -1, len_seg : int = -1) -> tuple[int, int, pd.DataFrame] :
    '''
    Reads a file from the cactus experimentation with header, time data and signal data. Faster to get only the signal.

    Arguments :
    file_path - str : path of the .txt data file
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment

    Returns : (int, int, pandas.DataFrame)
    n_seg - int : number of segments
    len_seg - int : number of frame per segment
    data_signal - pandas.DataFrame : data frame with time and signal
    '''

    if n_seg == -1 or len_seg == -1 :
        data_header = pd.read_csv(file_path, header = 0, low_memory=False)[:len_header-1] # Reading the header of the file
        data_header_array = np.array(data_header)

        if n_seg == -1 :
            n_seg = int(data_header_array[header_info_line-1,n_seg_i-1]) # Searching for the number of segments if needed
        if len_seg == -1 :
            len_seg = int(data_header_array[header_info_line-1,len_seg_i-1]) # Searching for the number of frames per segment if needed

    data_signal = pd.read_csv(file_path, header = n_seg+len_header+1)

    return (n_seg, len_seg, data_signal)

def fast_folder_reader (folder_path : str | tuple[str, ...], channel : str | tuple[str, ...], invert : bool | tuple[bool, ...] = False, filter_noise : float | tuple[float, ...] = 1.2, filter_saturation : float | tuple[float, ...] = 100.0, multiple_channel_analysis = False, n_seg : int = -1, len_seg : int = -1) -> tuple[list[list[int]], list[list[int]], list[list[pd.DataFrame]], list[list[str]]] :
    '''
    Reads a folder from the cactus experimentation. Faster to get only the signal.

    Arguments :
    folder_path - str or tuple of str : path of the folder(s)
    channel - str or tuple of str : name of the channel desired in the file names (ex : "C1" or "C3") (1 or more channels accepted)
    n_seg - (optional) int : number of segments
    len_seg - (optional) int : number of frame per segment

    Returns : ([[int]], [[int]], [[pandas.DataFrame]], [[str]])
    n_seg_list - list of list of int : number of segments for each file from each channel
    len_seg_list - list of list of int : number of frame per segment for each file from each channel
    data_signal_list - list of list of pandas.DataFrame : data frame with time and signal for each file from each channel
    name_list - list of list of str : name of the file

    example :
    [0][i][j] : number of segments for channel i in file j
    [2][i][j] : pandas.Dataframe with signal for channel i in file j
    '''

    # Initialisation
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

    n_seg_list = [[] for i in range (nb_channels)]
    len_seg_list = [[] for i in range (nb_channels)]
    data_signal_list = [[] for i in range (nb_channels)]
    name_list = [[] for i in range (nb_channels)]

    n_seg_given = False
    len_seg_given = False
    if n_seg != -1 :
        n_seg_given = True
    if len_seg != -1 :
        len_seg_given = True

    nb_files = 0

    # Reading each file in the folder
    for path in folder_path :
        if path[-4:] == ".zip" :
            with zipfile.ZipFile(path) as z:
                for file in z.namelist():
                    # if nb_files <= 3000:
                    #     nb_files += 1
                    #     continue
                    # if nb_files > 4000:
                    #     continue
                    with z.open(file) as f:
                        file_name = f.name
                        if channel[0] in file_name :
                            is_file = [True] + [False for i in range (nb_channels-1)]
                            print(file_name)
                            if not(n_seg_given) : 
                                n_seg = -1 # Reinitiate n_seg between each file, if it wasn't given at the start
                            if not(len_seg_given) :
                                len_seg = -1 # Reinitiate len_seg between each file, if it wasn't given at the start
                            n_seg, len_seg, data_signal = fast_file_reader(f, n_seg = n_seg, len_seg = len_seg)
                            nb_files += 1
                            n_seg_list[0].append(n_seg)
                            len_seg_list[0].append(len_seg)
                            data_signal_list[0].append(data_signal)
                            name_list[0].append(file_name)
                            with zipfile.ZipFile(path) as z2:
                                for file in z2.namelist():
                                    with z.open(file) as f2:
                                        for i in range (1, nb_channels):
                                            if f2.name == file_name.replace(channel[0], channel[i]):
                                                is_file[i] = True
                                                if not(n_seg_given) : 
                                                    n_seg = -1 # Reinitiate n_seg between each file, if it wasn't given at the start
                                                if not(len_seg_given) :
                                                    len_seg = -1 # Reinitiate len_seg between each file, if it wasn't given at the start
                                                print(f2.name)
                                                n_seg, len_seg, data_signal = fast_file_reader(f2, n_seg = n_seg, len_seg = len_seg)
                                                nb_files += 1
                                                n_seg_list[i].append(n_seg)
                                                len_seg_list[i].append(len_seg)
                                                data_signal_list[i].append(data_signal)
                                                name_list[i].append(file_name)

                            if multiple_channel_analysis :
                                for i in range(nb_channels) :
                                    if invert[i] :
                                        data_signal_list[i][-1]["Ampl"] = - data_signal_list[i][-1]["Ampl"]
                    
                                k_drop = []
                                for i in range (nb_channels):
                                    for k in range (0, n_seg_list[i][-1]) :
                                        if k not in k_drop :
                                            df = data_signal_list[i][-1]
                                            t, signal = get_pulse(df, len_seg_list[i][-1], k - len(np.array(k_drop)[np.array(k_drop) < k]))
                                            if not(test_noise(signal, filter_noise[i])) or not(test_saturation(signal, filter_saturation[i])) :
                                                for j in range (nb_channels):
                                                    if k < n_seg_list[j][-1] :
                                                        begin = (k)*len_seg_list[j][-1]
                                                        end = (k+1)*len_seg_list[j][-1]
                                                        data_signal_list[j][-1] = data_signal_list[j][-1].drop(index=list(range(begin, end)))
                                                k_drop.append(k)
                                            else :
                                                for j in range (nb_channels):
                                                    if k >= n_seg_list[j][-1] :
                                                        begin = (k)*len_seg_list[i][-1]
                                                        end = (k+1)*len_seg_list[i][-1]
                                                        data_signal_list[i][-1] = data_signal_list[i][-1].drop(index=list(range(begin, end)))
                                for i in range(nb_channels):
                                    print(channel[i])
                                    print(data_signal_list[i][-1].shape)
                            else :
                                for i in range(nb_channels) :
                                    if is_file[i]:
                                        if invert[i] :
                                            data_signal_list[i][-1]["Ampl"] = - data_signal_list[i][-1]["Ampl"]
        
                                        k_drop = 0
                                        for k in range (n_seg_list[i][-1]) :
                                            df = data_signal_list[i][-1]
                                            t, signal = get_pulse(df, len_seg_list[i][-1], k - k_drop)
                                            if not(test_noise(signal, filter_noise[i])) or not(test_saturation(signal, filter_saturation[i])) :
                                                begin = (k)*len_seg_list[i][-1]
                                                end = (k+1)*len_seg_list[i][-1]
                                                data_signal_list[i][-1] = df.drop(index=list(range(begin, end)))
                                                k_drop += 1
        else :
            for f in os.scandir(path) :
                if f.is_file() :
                    file_name = f.name
                    file_path = f.path
                    if channel[0] in file_name :
                        test_all_channels = True
                        for i in range (1, nb_channels) :
                            if not(os.path.isfile(file_path.replace(channel[0], channel[i]))) :
                                test_all_channels = False
                                break
                        if test_all_channels :
                            print(file_path)
                            if not(n_seg_given) : 
                                n_seg = -1 # Reinitiate n_seg between each file, if it wasn't given at the start
                            if not(len_seg_given) :
                                len_seg = -1 # Reinitiate len_seg between each file, if it wasn't given at the start
                            n_seg, len_seg, data_signal = fast_file_reader(file_path, n_seg = n_seg, len_seg = len_seg)
                            n_seg_list[0].append(n_seg)
                            len_seg_list[0].append(len_seg)
                            data_signal_list[0].append(data_signal)
                            name_list[0].append(file_path.split("/")[-2] + "/" + file_name)
                            for i in range(1, nb_channels) :
                                file_path_bis = file_path.replace(channel[0], channel[i])
                                print(file_path_bis)
                                if not(n_seg_given) :
                                    n_seg = -1 # Reinitiate n_seg between each file, if it wasn't given at the start
                                if not(len_seg_given) :
                                    len_seg = -1 # Reinitiate len_seg between each file, if it wasn't given at the start
                                n_seg, len_seg, data_signal = fast_file_reader(file_path_bis, n_seg = n_seg, len_seg = len_seg)
                                n_seg_list[i].append(n_seg)
                                len_seg_list[i].append(len_seg)
                                if invert[i] and len(data_signal)>0:
                                    data_signal["Ampl"] = - data_signal["Ampl"]
                                data_signal_list[i].append(data_signal)
                                name_list[i].append(file_path.split("/")[-2] + "/" + file_name)

                            if multiple_channel_analysis :
                    
                                k_drop = []
                                for i in range (nb_channels):
                                    for k in range (0, n_seg_list[i][-1]) :
                                        if k not in k_drop :
                                            df = data_signal_list[i][-1]
                                            t, signal = get_pulse(df, len_seg_list[i][-1], k - len(np.array(k_drop)[np.array(k_drop) < k]))
                                            if not(test_noise(signal, filter_noise[i])) or not(test_saturation(signal, filter_saturation[i])) :
                                                for j in range (nb_channels):
                                                    if k < n_seg_list[j][-1] :
                                                        begin = (k)*len_seg_list[j][-1]
                                                        end = (k+1)*len_seg_list[j][-1]
                                                        data_signal_list[j][-1] = data_signal_list[j][-1].drop(index=list(range(begin, end)))
                                                k_drop.append(k)
                                            else :
                                                for j in range (nb_channels):
                                                    if k >= n_seg_list[j][-1] :
                                                        begin = (k)*len_seg_list[i][-1]
                                                        end = (k+1)*len_seg_list[i][-1]
                                                        data_signal_list[i][-1] = data_signal_list[i][-1].drop(index=list(range(begin, end)))
                                for i in range(nb_channels):
                                    print(channel[i])
                                    print(data_signal_list[i][-1].shape)
                            else :
                                for i in range(nb_channels) :
                                    if invert[i] :
                                        data_signal_list[i][-1]["Ampl"] = - data_signal_list[i][-1]["Ampl"]

                                    k_drop = 0
                                    for k in range (n_seg_list[i][-1]) :
                                        df = data_signal_list[i][-1]
                                        t, signal = get_pulse(df, len_seg_list[i][-1], k - k_drop)
                                        if not(test_noise(signal, filter_noise[i])) or not(test_saturation(signal, filter_saturation[i])) :
                                            begin = (k)*len_seg_list[i][-1]
                                            end = (k+1)*len_seg_list[i][-1]
                                            data_signal_list[i][-1] = df.drop(index=list(range(begin, end)))
                                            k_drop += 1
                        
            
    return (n_seg_list, len_seg_list, data_signal_list, name_list)