import numpy as np

def test_noise_old (signal, threshold) :
    return (np.max(signal) > threshold)

def test_noise (signal, threshold) :
    return ((np.std(signal) / (np.max(signal) - np.median(signal))) < threshold)

def test_saturation (signal, threshold) :
    return (np.max(signal) < threshold)