import autograd.numpy as np
from scipy.signal import sawtooth
from synthetic_measurement import signals


###################
##### Defines #####
###################

lag1 = 1
lag2 = 5

#####################
##### Functions #####
#####################

signal = signals[0]

def generate_signal(length):
    # Generate a triangle signal using scipy's sawtooth function
    # a time array covering one period of the waveform, with values ranging from 0 to 2π radians
    # 0.5 for DC = 50%
    return sawtooth(np.linspace(0, 1, length) * 2 * np.pi, width=0.5)

def first_order_autocorrelation(signal):
    # Calculate the first-order autocorrelation
    return np.mean(signal)

def second_order_autocorrelation(signal, lag, M):
    # Calculate the second-order autocorrelation
    autocorr_sum = 0
    # summing up z[i] * z[i + lag] in the range: max(0, -lag) --> M - 1 + min(0, -lag)
    for i in range(max(0, -lag), M + min(0, -lag)):
        autocorr_sum += signal[i] * signal[i + lag]
    return autocorr_sum / M

def third_order_autocorrelation(signal, lag1, lag2, M):
    # Calculate the third-order autocorrelation
    autocorr_sum = 0
    # summing up z[i] * z[i + lag1] * z[i + lag2] in the range: max(0, -lag1, -lag2) --> M - 1 + min(0, -lag1, -lag2)
    for i in range(max(0, -lag1, -lag2), M + min(0, -lag1, -lag2)):
        autocorr_sum += signal[i] * signal[i + lag1] * signal[i + lag2]
    return autocorr_sum / M
