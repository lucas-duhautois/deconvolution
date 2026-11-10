from data_read import fast_file_reader
from mean_pulse import get_mean_pulse_strontium, plot_pulse, plot_save, plot_save_strontium, get_folder_name, mean_pulse, standardize
from HV_analysis import plot_HV_from_save, function_max, function_time_down, function_integrate, function_time_up, plot_function_HV_from_save, get_HV, plot_function_from_save_strontium, plot_function_HV, plot_function_strontium
from peak_analysis import get_pulse, get_stats, mean_max, void_parameters
from deconvolution import plot_fft_from_save, deconvoluate_from_save, h, deconvoluate_from_save_strontium, plot_fft_from_save_strontium, h_spice, h_exp, h_exp_santacruz
from strontium import pulse_distribution
from utils import test_noise, test_saturation, test_noise_old

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import zipfile

# General informations
n_seg = -1
len_seg = -1
# n_seg = 100           # always use this (with the correct value) for zip
# len_seg = 50002        # always use this (with the correct value) for zip
verbose = 2                                 # 0 / 1 / 2

folder_list = []
HV_list = []


# Parameters for some plots
title_name = ""               # for the titles of plots
sub_string = ""                             # if particles, indicates the segment of amplitude for HV_analysis
normalized = False                          # whether you want some results from peak_analysis/noise_analysis to be divided by the max of the mean pulse
standard = False                            # whether you want the curves to be standardized between 0 and 1
min_ampl = 0                                # minimal amplitude for plots of particle mean pulses 
max_ampl = -1                              # maximal amplitude for plots of particle mean pulses (-1 to see all curves)
zoom = 1                                    # for fft
start = 1                                  # for deconvoluated signal
bins = 100                                  # nb of segments in histogram for pulse distribution
log = True                                  # whether the scale of the histogram for pulse distribution is in log scale

transfer_function = h_exp                   # h, h_spice, h_exp, h_exp_santacruz

window = []                                 # minimum and maximum index to be computed
# window = [4000,6000] #HG_EPI_struct2 HV100

intervals = []                              # intervals of amplitude distribution (every end included)
#intervals = [20.0,29.0,35.0,41.0,49.0,58.0,73.0,100.0,158.0,255.0,770.0] #HG_EPI_struct5 HV100
#intervals = [9.0,14.5,18.0,25.0,28.0,31.0,34.0,37.5,43.0,53.0,365.0] #HG_EPI_struct2 HV100

#intervals = [22.0,29.0,35.0,42.0,67.0,68.0] # without_IR HV150
#intervals = [1.1,1.5,1.7,1.9,2.2,3.3] #without_IR HV120
#intervals = [22.0,24.0,25.0,27.0,30.0,60.0] #without_IR HV140

#intervals = [0.4,2.0,2.5,3.0,3.5,4.2,5.2,7.0,10.5,18.0,69.0] # Santa Cruz Sr 1mVp


# To get the transfer function

# global_folder_path = "../Data/Saclay/without_IR"
# save_path = "new_saves/without_IR"
# save_path_mean = "new_saves/without_IR/mean_pulse"
# channel = ("C3")
# exp = "IR"


#deconvoluate chip4 IR
global_folder_path = "../Data/Saclay/ledIR/chip4"
save_path = "new_saves/chip4"
save_path_mean = "new_saves/chip4/mean_pulse"
channel = ("C3")
exp = "IR"

#deconvoluate epi struct2 Sr
# global_folder_path = "../Data/Saclay/strontium/HG_EPI_struct2"
# save_path = "new_saves/HG_EPI_struct2"
# save_path_mean = "new_saves/HG_EPI_struct2/mean_pulse_ter"
# channel = ("C3")
# exp="particles"
# window = [4000,6000]

#deconvoluate epi struct5 Sr
# global_folder_path = "../Data/Saclay/strontium/HG_EPI_struct5"
# save_path = "new_saves/HG_EPI_struct5"
# save_path_mean = "new_saves/HG_EPI_struct5/mean_pulse_new"
# channel = ("C3")
# exp="particles"

#deconvoluate SantaCruz IR
# global_folder_path = "../Data/SantaCruz/ledIR/epi_HG_SantaCruz"
# save_path = "new_saves/epi_HG_SantaCruz"
# save_path_mean = "new_saves/epi_HG_SantaCruz/mean_pulse"
# channel = ("C3")
# exp = "IR"

#deconvoluate SantaCruz Sr 1mV (HV=100V)
# global_folder_path = "../Data/SantaCruz/strontium/SantaCruz_100V_trig_1mVp"
# save_path = "new_saves/SantaCruz_Sr/trig_1mVp"
# save_path_mean = "new_saves/SantaCruz_Sr/trig_1mVp/mean_pulse"
# channel = ("C2")
# exp = "particles"

#deconvoluate SantaCruz Sr 5mV (HV=100V)
# global_folder_path = "../Data/SantaCruz/strontium/SantaCruz_100V_trig_5mVp"
# save_path = "new_saves/SantaCruz_Sr/trig_5mVp"
# save_path_mean = "new_saves/SantaCruz_Sr/trig_1mVp/mean_pulse"
# channel = ("C2")
# exp = "particles"

# Check list
check_invert = False                        # plots some pulses for each channel to check if it is upside down + tune the filter for testbeam
check_noise = False
check_noise_old = False
check_sat = False
calculations = False                        # mean pulse calculations + amplitude distribution + noise standard deviation
plot_mean = True                           # plots the mean pulse(s)
HV_analysis = True                         # plots HV related curves : mean pulse, max, area, time up and time down
fourier_analysis = True                    # plots fft and deconvoluated signal + time up and time down of deconvolved signal
pulse_distrib = False                       # plots the distribution of pulse's amplitude

#########################################################################################################################################################

if global_folder_path[-1] != "/" and global_folder_path[-4:] != ".zip" :
    global_folder_path += "/"
folder_name = get_folder_name(global_folder_path)

if save_path[-1] != "/" :
    save_path += "/"

if type(channel) == str :
    channel = (channel,)
nb_channel = len(channel)

invert = np.array([False] * nb_channel)
t_limit = np.array([0] * nb_channel)
filter_noise = np.array([1.0] * nb_channel)
filter_noise_old = np.array([0.0] * nb_channel)
filter_saturation = np.array([100.0] * nb_channel)

if check_invert :
    folder_test = os.listdir(global_folder_path)[0]

    if os.path.isfile(global_folder_path + folder_test) :
        folder_test = ""
    if len(folder_list) > 0 :
        folder_test = folder_list[0]
    for i in range(nb_channel) :
        if (global_folder_path + folder_test)[-4:] == ".zip" :
            with zipfile.ZipFile(global_folder_path + folder_test) as z:
                for file in z.namelist():
                    with z.open(file) as f:
                        if channel[i] in f.name :

                            n_seg, len_seg, data_signal = fast_file_reader(f, n_seg=n_seg, len_seg=len_seg)
                            pulse0 = get_pulse(data_signal, len_seg, 0)
                            pulse1 = get_pulse(data_signal, len_seg, 1)
                            pulse2 = get_pulse(data_signal, len_seg, 2)
                            pulse3 = get_pulse(data_signal, len_seg, 3)
                            pulse4 = get_pulse(data_signal, len_seg, 4)
                            plot_pulse(pulse0[0], pulse0[1], title="Test " + channel[i], show = False)
                            plot_pulse(pulse1[0], pulse1[1], title="Test " + channel[i], show = False)
                            plot_pulse(pulse2[0], pulse2[1], title="Test " + channel[i], show = False)
                            plot_pulse(pulse3[0], pulse3[1], title="Test " + channel[i], show = False)
                            plot_pulse(pulse4[0], pulse4[1], title="Test " + channel[i], show = True)

                            test = input("Is the signal upside down ? [y/n] ")
                            if test == "y" :
                                invert[i] = True
                            break
        else :
            for f in os.scandir (global_folder_path + folder_test) :
                if channel[i] in f.name :

                    n_seg, len_seg, data_signal = fast_file_reader(f.path)
                    pulse0 = get_pulse(data_signal, len_seg, 0)
                    pulse1 = get_pulse(data_signal, len_seg, 1)
                    pulse2 = get_pulse(data_signal, len_seg, 2)
                    pulse3 = get_pulse(data_signal, len_seg, 3)
                    pulse4 = get_pulse(data_signal, len_seg, 4)
                    plot_pulse(pulse0[0], pulse0[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse1[0], pulse1[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse2[0], pulse2[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse3[0], pulse3[1], title="Test " + channel[i], show = False)
                    plot_pulse(pulse4[0], pulse4[1], title="Test " + channel[i], show = True)
                    test = input("Is the signal upside down ? [y/n] ")
                    if test == "y" :
                        invert[i] = True
                    break
    print(invert)

if check_noise :
    if len(folder_list) == 0 :
        folder_test_list = [os.listdir(global_folder_path)[0]]
    else :
        folder_test_list = folder_list
    plt.clf()
    test_channel = [False] * nb_channel
    if os.path.isfile(global_folder_path + folder_test_list[0]) and (global_folder_path + folder_test_list[0])[-4:] != ".zip" :
            folder_test_list = [""] * len(folder_test_list)
    for i in range (nb_channel) :
        print("Starting check_noise for channel " + channel[i])

        test_filter_noise = False
        useless_files =  []
        while not(test_filter_noise) :
            for folder_test in folder_test_list :
                if (global_folder_path + folder_test)[-4:] == ".zip" :
                    with zipfile.ZipFile(global_folder_path + folder_test) as z:
                        for file in z.namelist():
                            with z.open(file) as f:
                                if (channel[i] in f.name) and not(test_channel[i]) and not(str(folder_test) + str(f.name) in useless_files) :
                                    file_name = f.name
                                    file_folder = folder_test
                                    
                                    # check the noise filter by plotting correct signals
                                    n_seg, len_seg, data_signal = fast_file_reader(f, n_seg=n_seg, len_seg=len_seg)
                                    if invert[i] :
                                        data_signal = - data_signal
                                    nb_plot = 0
                                    for j in range (min(n_seg,100)):
                                        t, signal = get_pulse(data_signal, len_seg, j)
                                        if test_noise(signal, filter_noise[i]) :
                                            signal = standardize(signal)
                                            plot_pulse(t, signal, title="Test filter noise " + channel[i], show = False)
                                            nb_plot += 1

                                    if nb_plot > 0 :
                                        print(nb_plot)
                                        plt.show()
                                        test_channel[i] = True
                else :
                    for f in os.scandir(global_folder_path + folder_test) :
                        if (channel[i] in f.name) and not(test_channel[i]) and not(str(folder_test) + str(f.name) in useless_files) :
                            file_name = f.name
                            file_folder = folder_test
                            
                            # check the noise filter by plotting correct signals
                            n_seg, len_seg, data_signal = fast_file_reader(f.path)
                            if invert[i] :
                                data_signal = - data_signal
                            nb_plot = 0
                            for j in range (min(n_seg,100)):
                                t, signal = get_pulse(data_signal, len_seg, j)
                                if test_noise(signal, filter_noise[i]) :
                                    signal = standardize(signal)
                                    plot_pulse(t, signal, title="Test filter noise " + channel[i], show = False)
                                    nb_plot += 1

                            # check the noise filter by plotting incorrect signals
                            # n_seg, len_seg, data_signal = fast_file_reader(f.path)
                            # j = 0
                            # nb_plot = 0
                            # while (j < n_seg) and (nb_plot < 50) :
                            #     t, signal = get_pulse(data_signal, len_seg, j)
                            #     if not(test_noise(signal, filter_noise[i])) :
                            #         signal = standardize(signal)
                            #         plot_pulse(t, signal, title="Test filter noise " + channel[i], show = False)
                            #         nb_plot += 1
                            #     j += 1
                            if nb_plot > 0 :
                                print(nb_plot)
                                plt.show()
                                test_channel[i] = True
                        
            print("Current filter value : " + str(filter_noise[i]))
            test = input("Was this file useful ? [y/n] ")
            if test == "n" :
                useless_files.append(str(file_folder) + str(file_name))
                print(useless_files)
                test_channel[i] = False
            else :
                test = input("What value do you want filter to be ? ")
                filter_noise[i] = float(test)
    
                test = input("Do you want to check the filter again ? [y/n] ")
                if test == "n" :
                    test_filter_noise = True
                else :
                    test_channel[i] = False
    print(filter_noise)

if check_noise_old :
    if len(folder_list) == 0 :
        folder_test_list = [os.listdir(global_folder_path)[0]]
    else :
        folder_test_list = folder_list
    plt.clf()
    test_channel = [False] * nb_channel
    if os.path.isfile(global_folder_path + folder_test_list[0]) and (global_folder_path + folder_test_list[0])[-4:] != ".zip" :
            folder_test_list = [""] * len(folder_test_list)
    for i in range (nb_channel) :
        print("Starting check_noise for channel " + channel[i])

        test_filter_noise = False
        useless_files =  []
        while not(test_filter_noise) :
            for folder_test in folder_test_list :
                print(folder_test)
                if (global_folder_path + folder_test)[-4:] == ".zip" :
                    with zipfile.ZipFile(global_folder_path + folder_test) as z:
                        for file in z.namelist():
                            with z.open(file) as f:
                                if (channel[i] in f.name) and not(test_channel[i]) and not(str(folder_test) + str(f.name) in useless_files) :
                                    file_name = f.name
                                    file_folder = folder_test
                                    
                                    # check the noise filter by plotting correct signals
                                    n_seg, len_seg, data_signal = fast_file_reader(f, n_seg=n_seg, len_seg=len_seg)
                                    if invert[i] :
                                        data_signal = - data_signal
                                    nb_plot = 0
                                    for j in range (min(n_seg,100)):
                                        t, signal = get_pulse(data_signal, len_seg, j)
                                        if test_noise_old(signal, filter_noise_old[i]) :
                                            signal = standardize(signal)
                                            plot_pulse(t, signal, title="Test filter noise " + channel[i], show = False)
                                            nb_plot += 1

                                    if nb_plot > 0 :
                                        print(nb_plot)
                                        plt.show()
                                        test_channel[i] = True
                else :
                    for f in os.scandir(global_folder_path + folder_test) :
                        if (channel[i] in f.name) and not(test_channel[i]) and not(str(folder_test) + str(f.name) in useless_files) :
                            file_name = f.name
                            file_folder = folder_test
                            
                            # check the noise filter by plotting correct signals
                            n_seg, len_seg, data_signal = fast_file_reader(f.path)
                            if invert[i] :
                                data_signal = - data_signal
                            nb_plot = 0
                            for j in range (min(n_seg,100)):
                                t, signal = get_pulse(data_signal, len_seg, j)
                                if test_noise_old(signal, filter_noise_old[i]) :
                                    signal = standardize(signal)
                                    plot_pulse(t, signal, title="Test filter noise " + channel[i], show = False)
                                    nb_plot += 1

                            # check the noise filter by plotting incorrect signals
                            # n_seg, len_seg, data_signal = fast_file_reader(f.path)
                            # j = 0
                            # nb_plot = 0
                            # while (j < n_seg) and (nb_plot < 50) :
                            #     t, signal = get_pulse(data_signal, len_seg, j)
                            #     if not(test_noise(signal, filter_noise[i])) :
                            #         signal = standardize(signal)
                            #         plot_pulse(t, signal, title="Test filter noise " + channel[i], show = False)
                            #         nb_plot += 1
                            #     j += 1
                            if nb_plot > 0 :
                                print(nb_plot)
                                plt.show()
                                test_channel[i] = True
                        
            print("Current filter value : " + str(filter_noise_old[i]))
            test = input("Was this file useful ? [y/n] ")
            if test == "n" :
                useless_files.append(str(file_folder) + str(file_name))
                print(useless_files)
                test_channel[i] = False
            else :
                test = input("What value do you want filter to be ? ")
                filter_noise_old[i] = float(test)
    
                test = input("Do you want to check the filter again ? [y/n] ")
                if test == "n" :
                    test_filter_noise = True
                else :
                    test_channel[i] = False
    print(filter_noise)


if check_sat :
    if len(folder_list) == 0 :
        folder_test_list = [os.listdir(global_folder_path)[0]]
    else :
        folder_test_list = folder_list
    plt.clf()
    test_channel = [False] * nb_channel
    if os.path.isfile(global_folder_path + folder_test_list[0]) and (global_folder_path + folder_test_list[0])[-4:] != ".zip" :
            folder_test_list = [""] * len(folder_test_list)
    for i in range (nb_channel) :
        print("Starting check_sat for channel " + channel[i])
        test_filter_saturation = False
        useless_files = []
        while not(test_filter_saturation) :
            for folder_test in folder_test_list:
                if (global_folder_path + folder_test)[-4:] == ".zip" :
                    with zipfile.ZipFile(global_folder_path + folder_test) as z:
                        for file in z.namelist():
                            with z.open(file) as f:
                                if (channel[i] in f.name) and not(test_channel[i]) and not(str(folder_test) + str(f.name) in useless_files) :
                                    file_name = f.name
                                    file_folder = folder_test
                                    
                                    # check the saturation filter by plotting correct signals
                                    n_seg, len_seg, data_signal = fast_file_reader(f, n_seg=n_seg, len_seg=len_seg)
                                    if invert[i] :
                                        data_signal = - data_signal
                                    nb_plot = 0
                                    for j in range (min(n_seg,100)):
                                        t, signal = get_pulse(data_signal, len_seg, j)
                                        if test_saturation(signal, filter_saturation[i]) :
                                            plot_pulse(t, signal, title="Test filter saturation " + channel[i], show = False)
                                            nb_plot += 1
        
                                    # check the saturation filter by plotting incorrect signals
                                    # n_seg, len_seg, data_signal = fast_file_reader(f.path)
                                    # j = 0
                                    # nb_plot = 0
                                    # while (j < n_seg) and (nb_plot < 50) :
                                    #     t, signal = get_pulse(data_signal, len_seg, j)
                                    #     if not(test_saturation(signal, filter_saturation[i])) :
                                    #         plot_pulse(t, signal, title="Test filter saturation " + channel[i], show = False)
                                    #         nb_plot += 1
                                    #     j += 1
            
                                    if nb_plot > 0 :
                                        print(nb_plot)
                                        plt.show()
                                        test_channel[i] = True
                else :
                    for f in os.scandir(global_folder_path + folder_test) :
                        if (channel[i] in f.name) and not(test_channel[i]) and not(str(folder_test) + str(f.name) in useless_files) :
                            file_name = f.name
                            file_folder = folder_test
                            
                            # check the saturation filter by plotting correct signals
                            n_seg, len_seg, data_signal = fast_file_reader(f.path)
                            if invert[i] :
                                data_signal = - data_signal
                            nb_plot = 0
                            for j in range (min(n_seg,100)):
                                t, signal = get_pulse(data_signal, len_seg, j)
                                if test_saturation(signal, filter_saturation[i]) :
                                    plot_pulse(t, signal, title="Test filter saturation " + channel[i], show = False)
                                    nb_plot += 1

                            # check the saturation filter by plotting incorrect signals
                            # n_seg, len_seg, data_signal = fast_file_reader(f.path)
                            # j = 0
                            # nb_plot = 0
                            # while (j < n_seg) and (nb_plot < 50) :
                            #     t, signal = get_pulse(data_signal, len_seg, j)
                            #     if not(test_saturation(signal, filter_saturation[i])) :
                            #         plot_pulse(t, signal, title="Test filter saturation " + channel[i], show = False)
                            #         nb_plot += 1
                            #     j += 1

                            if nb_plot > 0 :
                                print(nb_plot)
                                plt.show()
                                test_channel[i] = True
                        
            print("Current filter value : " + str(filter_saturation[i]))

            test = input("Was this file useful ? [y/n] ")
            if test == "n" :
                useless_files.append(str(file_folder) + str(file_name))
                print(useless_files)
                test_channel[i] = False
            else :
                test = input("What value do you want filter to be ? ")
                filter_saturation[i] = float(test)
                
                test = input("Do you want to check the filter again ? [y/n] ")
                if test == "n" :
                    test_filter_saturation = True
                else :
                    test_channel[i] = False
    print(filter_saturation)

if calculations :
    if exp == "IR" :
        get_stats(global_folder_path, channel, (mean_pulse), (void_parameters), folder_list=folder_list, HV_list=HV_list, save = True, save_path=save_path, invert=tuple(invert), verbose=(verbose>=1), n_seg=n_seg, len_seg=len_seg)
    elif exp == "particles" :
        get_stats(global_folder_path, channel, (mean_max), (void_parameters), folder_list=folder_list, HV_list=HV_list, save = True, save_path=save_path, filter_noise = tuple(filter_noise), filter_saturation=tuple(filter_saturation), invert=tuple(invert), verbose=(verbose>=1), multiple_channel_analysis=True, n_seg=n_seg, len_seg=len_seg)
        
        # mean pulse
        if len(intervals) != 0 :    # use the pulse distribution to get the intervals
            t_start = time.time()
            if folder_list == [] :
                for d in os.scandir(global_folder_path) :
                    folder_list.append(d.name)
            if HV_list == [] :
                for f in folder_list :
                    HV_list.append(get_HV(f))
            nb_folder = len(HV_list)

            for i in range(nb_folder) :

                if verbose >= 1 :
                    completion = int((i/nb_folder)*100)
                    print("Mean pulse calculations completed by : " + str(completion) + "%")
                    if completion != 0 :
                        remaining = (100 - completion) * (time.time() - t_start) / completion
                        print("Estimated remaining time : " + str(round(remaining, 2)) + " seconds")

                get_mean_pulse_strontium(tuple([global_folder_path + f for f in folder_list]), channel, 4, intervals=intervals, plot=False, save=True, invert = tuple(invert), save_path=save_path, min_ampl=min_ampl, max_ampl=max_ampl, verbose=(verbose>=2), filter_noise=tuple(filter_noise), filter_saturation=tuple(filter_saturation), len_seg=len_seg, n_seg=n_seg)
        
if plot_mean :
    if exp == "IR" :
        HV_value = input ("Which HV value do you want to plot ? ")
        for ch in channel :
            for f in os.scandir(save_path_mean) :
                file_name = f.name
                if (ch in file_name) and (int(HV_value) == get_HV(file_name)) :
                    plot_save(str(f.path), title=file_name, standard=standard)
    
    elif exp == "particles" :
        for ch in channel :
            HV_value = input ("Which HV value do you want to plot ? ")
            plot_save_strontium(save_path_mean, ch, int(HV_value), title = title_name, standard=standard, min_ampl=min_ampl, max_ampl=max_ampl, window=window)


if HV_analysis :
    if exp == "particles" :
        HV_value = input ("Which HV value do you want to test ? ")
        plot_function_from_save_strontium(save_path_mean, channel, function_time_up, int(HV_value), title="Convolved signal " + title_name, save_list=[], window=window)
        plot_function_from_save_strontium(save_path_mean, channel, function_time_down, int(HV_value), title="Convolved signal " + title_name, save_list=[], window=window)
    else :
        plot_HV_from_save(save_path_mean, channel, title = title_name, sub_string=sub_string, standard=standard, save_list=[], HV_list=[])
        plot_function_HV_from_save(save_path_mean, channel, function_max, title="Convolved signal " + title_name, sub_string=sub_string, save_list=[], HV_list=[], n_pulse=1)
        plot_function_HV_from_save(save_path_mean, channel, function_integrate, title="Convolved signal " + title_name, sub_string=sub_string, save_list=[], HV_list=[])
        plot_function_HV_from_save(save_path_mean, channel, function_time_up, title="Convolved signal " + title_name, sub_string=sub_string, save_list=[], HV_list=[], n_pulse=1)
        plot_function_HV_from_save(save_path_mean, channel, function_time_down, title="Convolved signal " + title_name, sub_string=sub_string, save_list=[], HV_list=[])


if fourier_analysis :
    if exp == "IR" :
        plot_fft_from_save(save_path_mean, channel, zoom=zoom, title=title_name, sub_string = "", standard=standard, save_list=[], HV_list=[])
        t_list, output_list, HV_list = deconvoluate_from_save(save_path_mean, channel, transfer_function, title=title_name, sub_string="", standard=standard, start = start, save_list=[], HV_list=[], plot=True)
        plot_function_HV(t_list, output_list, HV_list, function_max, title="Deconvolved signal " + title_name)
        plot_function_HV(t_list, output_list, HV_list, function_time_up, title="Deconvolved signal " + title_name)
        plot_function_HV(t_list, output_list, HV_list, function_time_down, title="Deconvolved signal " + title_name)

    if exp == "particles" :
        plot_fft_from_save_strontium(save_path_mean, channel, zoom=zoom, title=title_name, sub_string = "", standard=standard, save_list=[], HV_list=[], window=window)
        t_list, output_list, ampl_list = deconvoluate_from_save_strontium(save_path_mean, channel, transfer_function, title=title_name, sub_string="", standard=standard, start = start, save_list=[], HV_list=[], window=window, plot=True)
        plot_function_strontium(t_list, output_list, ampl_list, function_time_up, title="Deconvolved signal " + title_name)
        plot_function_strontium(t_list, output_list, ampl_list, function_time_down, title="Deconvolved signal " + title_name)


if pulse_distrib :
    for i in range (nb_channel) :
        test_distribution = False
        while not(test_distribution) :
            HV_value = input ("Which HV value do you want to test the amplitude distribution ? ")
            pulse_distribution(global_folder_path, channel[i], int(HV_value), save_path=save_path, bins=int(bins), log=log, invert = invert[i], folder_list=folder_list, HV_list=HV_list, verbose=(verbose>=2), intervals=intervals)
            test = input("Do you need to run the histogram again ? [y/n] ")
            if test == "n" :
                test_distribution = True