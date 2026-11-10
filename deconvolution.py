import numpy as np
import scipy as sp
import pandas as pd 
import matplotlib.pyplot as plt
import os
import cblind as cb

from HV_analysis import get_HV
from mean_pulse import standardize, sort_multiple_lists, get_folder_name, linear_interpolation

def plot_fft_from_save (save_folder_path : str, channel : str | tuple[str, ...], title : str = "", zoom : int = 1, sub_string : str = "", save_list : list[str] = [], HV_list : list[int] = [], standard : bool = False) -> None :
    '''
    Plots the fft curves for different HV values, from save files.

    Arguments :
    save_folder_path - str : path of the folder containing the data files for each HV value
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    zoom - int : how much you want the classic frequence window to be zoomed in
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
    
    if type(channel) == str :
        channel = (channel,)
    
    folder_name = get_folder_name(save_folder_path)
    
    sort_multiple_lists(HV_list, [save_list])
    if save_folder_path[-1] != "/" :
        save_folder_path += "/"

    fft_list = []
    fftfreq_list = []
    for i in range(nb_saves) :
        df = pd.read_csv(save_folder_path+save_list[i])
        t = np.array(df["Time"])
        signal = np.array(df["Ampl"])
        n = len(signal)
        fft_list.append(sp.fft.rfft(signal))
        fftfreq_list.append(sp.fft.rfftfreq(n,t[1]-t[0]))
        if standard :
            fft_list[i] = standardize(np.abs(fft_list[i]))
    n_freq = len(fftfreq_list[0])

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
            plt.plot(fftfreq_list[i_plot[i]][:int(n_freq/zoom)], np.log10(np.abs(fft_list[i_plot[i]]))[:int(n_freq/zoom)], label=HV_list[i_plot[i]], color=c, linestyle=l) #IR
            
        plt.title("FFT " + title)
        plt.xlabel("frequency (Hz)")
        plt.ylabel("decimal logarithm")
        plt.legend(title = "HV (V)")
        plt.grid()
        plt.show()

def plot_fft_from_save_strontium (save_folder_path : str, channel : str | tuple[str, ...], title : str = "", zoom : int = 1, sub_string : str = "", save_list : list[str] = [], HV_list : list[int] = [], standard : bool = False, window : list =[]) -> None :
    '''
    Plots the fft curves for different HV values, from save files.

    Arguments :
    save_folder_path - str : path of the folder containing the data files for each HV value
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    zoom - int : how much you want the classic frequence window to be zoomed in
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
    
    if type(channel) == str :
        channel = (channel,)
    
    folder_name = get_folder_name(save_folder_path)

    low_ampl = []
    high_ampl = []
    for f in save_list :
        file_split = f.split("_")[-1].split("-")
        low_ampl.append(float(file_split[0]))
        high_ampl.append(float(file_split[1].split("mV")[0]))
    sort_multiple_lists(low_ampl, [high_ampl, save_list]) #particles
    
    #sort_multiple_lists(HV_list, [save_list]) #IR
    if save_folder_path[-1] != "/" :
        save_folder_path += "/"

    fft_list = []
    fftfreq_list = []
    for i in range(nb_saves) :
        df = pd.read_csv(save_folder_path+save_list[i])
        t = np.array(df["Time"])
        signal = np.array(df["Ampl"])
        if len(window) != 0 :
            t = t[window[0]:window[1]]
            signal = signal[window[0]:window[1]]
        n = len(signal)
        fft_list.append(sp.fft.rfft(signal))
        fftfreq_list.append(sp.fft.rfftfreq(n,t[1]-t[0]))
        if standard :
            fft_list[i] = standardize(np.abs(fft_list[i]))
    n_freq = len(fftfreq_list[0])

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
            #plt.plot(fftfreq_list[i_plot[i]][:int(n_freq/zoom)], np.log10(np.abs(fft_list[i_plot[i]]))[:int(n_freq/zoom)], label=HV_list[i_plot[i]], color=c, linestyle=l) #IR
            plt.plot(fftfreq_list[i_plot[i]][:int(n_freq/zoom)], np.log10(np.abs(fft_list[i_plot[i]]))[:int(n_freq/zoom)], label="["+str(low_ampl[i_plot[i]])+"mV, "+str(high_ampl[i_plot[i]])+"mV]", color=c, linestyle=l) #particles
            
        plt.title("FFT " + title)
        plt.xlabel("frequency (Hz)")
        plt.ylabel("decimal logarithm")
        plt.legend(title = "amplitude interval")
        plt.grid()
        plt.show()

def h (t : float | np.ndarray) -> float | np.ndarray :
    '''
    Transfer function of the system.

    Argument :
    t - float or numpy.array of float : time value(s)

    Return :
    float or numpy.array of float : value(s) of the transfer function in given time value(s)
    '''
    V0 = 10.4e-3
    tau = 28.9e-9

    # plt.plot(t, V0*np.exp(-t/tau))
    # plt.xlabel("Time (s)")
    # plt.ylabel("Ampl (V)")
    # plt.show()

    return (V0*np.exp(-t/tau))

def h_spice (t : float | np.ndarray) -> float | np.ndarray :
    '''
    Transfer function of the system, from Spice.

    Argument :
    t - float or numpy.array of float : time value(s)

    Return :
    float or numpy.array of float : value(s) of the transfer function in given time value(s)
    '''
    with open("LT_TIA.txt", "r") as f :
        data = f.read()
        data_lines = data.split("\n")[1:-1]
        data_table = [l.split("\t") for l in data_lines]
        data_table = np.array(data_table, dtype=float)
    
    result = np.zeros(np.shape(t))
    t = t - t[0]

    t_h = data_table[70:,0] # 42 / 60 / 70
    signal_h = - data_table[70:,1]

    i1 = 0
    i2 = 1
    for i in range (len(result)) :
        while (i2 < len(t_h)-1) and (t[i] > t_h[i2]) :
            i1 += 1
            i2 +=1
        result[i] = linear_interpolation(t[i], t_h[i1], signal_h[i1], t_h[i2], signal_h[i2])
    
    # plt.plot(t, result)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Ampl (V)")
    # plt.show()

    return (result)


def h_exp (t : float | np.ndarray) -> float | np.ndarray :
    '''
    Transfer function of the system, from experimentation.

    Argument :
    t - float or numpy.array of float : time value(s)

    Return :
    float or numpy.array of float : value(s) of the transfer function in given time value(s)
    '''
    df = pd.read_csv("new_saves/without_IR/mean_pulse/epi_HG_Saclay_HV140_noise_C3_mean_pulse")
    result = np.zeros(np.shape(t))
    t_exp = np.array(df["Time"])[200:] # 100 / 120 / 140 / 150 / 200
    signal_exp = np.array(df["Ampl"])[200:]

    t = t - t[0]
    t_exp = t_exp - t_exp[0]

    start = np.argmin(np.abs(t - t_exp[0]))

    result[start:start+len(t_exp)] = signal_exp

    # plt.plot(t, result)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Ampl (V)")
    # plt.show()

    return(result)

def h_exp_santacruz (t : float | np.ndarray) -> float | np.ndarray :
    '''
    Transfer function of the system, from experimentation.

    Argument :
    t - float or numpy.array of float : time value(s)

    Return :
    float or numpy.array of float : value(s) of the transfer function in given time value(s)
    '''
    df = pd.read_csv("new_saves/without_IR/mean_pulse/epi_HG_SantaCruz_HV120_noise_C3_mean_pulse")
    result = np.zeros(np.shape(t))
    t_exp = np.array(df["Time"])[95:]
    signal_exp = np.array(df["Ampl"])
    signal_exp = signal_exp - signal_exp[0]
    signal_exp = signal_exp[95:]

    t = t - t[0]
    t_exp = t_exp - t_exp[0]

    start = np.argmin(np.abs(t - t_exp[0]))

    result[start:start+len(t_exp)] = signal_exp

    # plt.plot(t, result)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Ampl (V)")
    # plt.show()

    return(result)

def deconvoluate_from_save_strontium (save_folder_path : str, channel : str | tuple[str, ...], transfer_function, title : str = "", sub_string : str = "", save_list : list[str] = [], HV_list : list[int] = [], standard : bool = False, plot : bool = True, start : int = 0, window : list = []) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    '''
    Deconvoluates the signal using a given transfer function in the Fourier domain.

    Arguments :
    save_folder_path - str : path of the folder containing the data files for each HV value
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    transfert_function - function : transfer function of the system
    sub_string - str : needs to be in the save file name for it to be considered
    save_list - (optional) list of str : the names of the files for each HV value
    HV_list - (optional) list of int : the HV values (needs to be in the same order as save_list)
    standard - (optional) bool : whether you want the data to be standardized (default is False)
    plot - (optional) bool : whether you want to plot the results (defaut is True)
    start - (optional) int : index that you want the plot to start on (default is 0)

    Returns :
    t_list - list of np.ndarray : time values for each channel
    output_list - list of np.ndarray : deconvoluated signal values for each channel
    '''
    if type(channel) == str :
        channel = (channel,)

    if save_list == [] :
        for f in os.scandir(save_folder_path) :
            for ch in channel :
                if ch in f.name :
                    save_list.append(f.name)
    if HV_list == [] :
        for f in save_list :
            HV_list.append(get_HV(f))
    nb_saves = len(save_list)

    folder_name = get_folder_name(save_folder_path)

    low_ampl = []
    high_ampl = []
    for f in save_list :
        file_split = f.split("_")[-1].split("-")
        low_ampl.append(float(file_split[0]))
        high_ampl.append(float(file_split[1].split("mV")[0]))
    sort_multiple_lists(low_ampl, [high_ampl, save_list])

    if save_folder_path[-1] != "/" :
        save_folder_path += "/"

    t_list = []
    fft_list = []
    H_list = []
    output_list = []
    for i in range(nb_saves) :
        df = pd.read_csv(save_folder_path+save_list[i])
        t = np.array(df["Time"])
        signal = np.array(df["Ampl"])
        if len(window) != 0 :
            t = t[window[0]:window[1]]
            signal = signal[window[0]:window[1]]
        t_list.append(t)
        fft_list.append(sp.fft.rfft(signal))
        H_list.append(sp.fft.rfft(transfer_function(t)))
        output_list.append((sp.fft.irfft(np.array(fft_list[i]) / np.array(H_list[i])))[start:])
        t_list[-1] = t_list[-1][start:]
        if standard :
            output_list[-1] = standardize(output_list[-1])
    
    if plot :
        for ch in channel :
            plt.clf()
            i_plot = []
            for i in range(nb_saves) :
                if (ch in save_list[i]) and (sub_string in save_list[i]) :
                    i_plot.append(i)
            nb_plot = len(i_plot)
            color, linestyle = cb.Colorplots().extreme_rainbow(nb_plot)
            for i in range (nb_plot) :
                c = color[i%34]
                l = linestyle[i//34]
                plt.plot(t_list[i_plot[i]], output_list[i_plot[i]], label="["+str(low_ampl[i_plot[i]])+"mV, "+str(high_ampl[i_plot[i]])+"mV]", color=c, linestyle=l) #particles
                
            plt.legend(title = "amplitude interval")
            #plt.title(folder_name + " " + ch)
            plt.title("Deconvolved signal " + title)
            plt.xlabel("time (s)")
            if standard :
                plt.ylabel("normalized amplitude")
            else :
                plt.ylabel("amplitude (V)")
            plt.grid()
            plt.show()
    return (t_list[:], output_list[:], low_ampl)

def deconvoluate_from_save (save_folder_path : str, channel : str | tuple[str, ...], transfer_function, title : str = "", sub_string : str = "", save_list : list[str] = [], HV_list : list[int] = [], standard : bool = False, plot : bool = True, start : int = 0) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    '''
    Deconvoluates the signal using a given transfer function in the Fourier domain.

    Arguments :
    save_folder_path - str : path of the folder containing the data files for each HV value
    channel - str or tuple of str : desired channel of the oscilloscope (ex : "C1" or "C3") (1 or more channels accepted)
    transfert_function - function : transfer function of the system
    sub_string - str : needs to be in the save file name for it to be considered
    save_list - (optional) list of str : the names of the files for each HV value
    HV_list - (optional) list of int : the HV values (needs to be in the same order as save_list)
    standard - (optional) bool : whether you want the data to be standardized (default is False)
    plot - (optional) bool : whether you want to plot the results (defaut is True)
    start - (optional) int : index that you want the plot to start on (default is 0)

    Returns :
    t_list - list of np.ndarray : time values for each channel
    output_list - list of np.ndarray : deconvoluated signal values for each channel
    '''
    if type(channel) == str :
        channel = (channel,)

    if save_list == [] :
        for f in os.scandir(save_folder_path) :
            for ch in channel :
                if ch in f.name :
                    save_list.append(f.name)
    if HV_list == [] :
        for f in save_list :
            HV_list.append(get_HV(f))
    nb_saves = len(save_list)

    folder_name = get_folder_name(save_folder_path)
    
    sort_multiple_lists(HV_list, [save_list])
    if save_folder_path[-1] != "/" :
        save_folder_path += "/"

    t_list = []
    fft_list = []
    H_list = []
    output_list = []
    for i in range(nb_saves) :
        df = pd.read_csv(save_folder_path+save_list[i])
        t = np.array(df["Time"])
        signal = np.array(df["Ampl"])
        t_list.append(t)
        fft_list.append(sp.fft.rfft(signal))
        H_list.append(sp.fft.rfft(transfer_function(t)))
        output_list.append((sp.fft.irfft(np.array(fft_list[i]) / np.array(H_list[i])))[start:])
        t_list[-1] = t_list[-1][start:]
        if standard :
            output_list[-1] = standardize(output_list[-1])
    
    if plot :
        for ch in channel :
            plt.clf()
            i_plot = []
            for i in range(nb_saves) :
                if (ch in save_list[i]) and (sub_string in save_list[i]) :
                    i_plot.append(i)
            nb_plot = len(i_plot)
            color, linestyle = cb.Colorplots().extreme_rainbow(nb_plot)
            for i in range (nb_plot) :
                c = color[i%34]
                l = linestyle[i//34]
                plt.plot(t_list[i_plot[i]], output_list[i_plot[i]], label=HV_list[i_plot[i]], color=c, linestyle=l)
                
            plt.legend(title = "HV (V)")
            #plt.title(folder_name + " " + ch)
            plt.title("Deconvolved signal " + title)
            plt.xlabel("time (s)")
            if standard :
                plt.ylabel("normalized amplitude")
            else :
                plt.ylabel("amplitude (V)")
            plt.grid()
            plt.show()
    
    return (t_list, output_list, HV_list)