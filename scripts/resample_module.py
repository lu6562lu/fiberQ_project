import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt

def resample_linear_window_average(freq_list, amp_list, delta_logf=0.2, window=0.1):
    """
    Resample data by averaging in linear scale within a window, then convert to log10 for visualization.

    Parameters:
        freq_list (2D array): Original frequency data (each row represents a set of frequencies).
        amp_list (2D array): Original amplitude data corresponding to freq_list.
        delta_logf (float): Interval in log10 frequency space for resampling.
        window (float): Width of the averaging window (in linear scale).

    Returns:
        log_freq_resampled (array): Resampled log10 frequency points.
        log_amp_resampled (2D array): Resampled amplitude data in log10 scale.
    """
    # Convert to numpy arrays
    freq_list = np.array(freq_list)
    amp_list = np.array(amp_list)

    # Calculate log10 of frequency range
    log_f_min, log_f_max = np.log10(freq_list.min()), np.log10(freq_list.max())
    target_log_freq = np.arange(log_f_min, log_f_max, delta_logf)  # Fixed log10 intervals
    target_freq = 10**target_log_freq  # Convert to linear scale

    resampled_amp_list = []
    
    for i in range(freq_list.shape[0]):  # Process each row of frequency and amplitude
        freq = freq_list[i]
        amp = amp_list[i]
        resampled_amp = []

        for f in target_freq:
            # Find values within the linear window around the target frequency
            mask = (freq >= f * (1 - window / 2)) & (freq <= f * (1 + window / 2))
            values_in_window = amp[mask]

            # Compute the average if there are values, otherwise assign NaN
            if len(values_in_window) > 0:
                resampled_amp.append(np.mean(values_in_window))
            else:
                resampled_amp.append(np.nan)

        resampled_amp_list.append(resampled_amp)

    # Convert resampled amplitudes to log10 scale, ignoring NaNs
    resampled_amp_array = np.array(resampled_amp_list)
    log_amp_resampled = np.log10(resampled_amp_array, where=~np.isnan(resampled_amp_array))
    
    return target_log_freq, log_amp_resampled


def resample_frequency_data_interpolation(freq, amplitude, delta_logf=0.1):
    """
    Resample frequency domain data with a fixed delta log(f) for linear input data.

    Parameters:
        freq (array-like): Original frequency data (linearly spaced, must be sorted in ascending order).
        amplitude (array-like): Original amplitude or spectral data corresponding to freq.
        delta_logf (float): Desired fixed interval in logarithmic frequency (default: 0.2).

    Returns:
        freq_resampled (numpy.ndarray): Resampled frequency points in logarithmic scale.
        amplitude_resampled (numpy.ndarray): Resampled amplitude points corresponding to the new frequency points.
    """
    # Ensure the input is numpy array
    freq = np.array(freq)
    amplitude = np.array(amplitude)
    
    # Convert frequency to log space
    log_f_min, log_f_max = np.log10(freq.min()), np.log10(freq.max())
    
    # Calculate the number of resampled points
    num_samples = int((log_f_max - log_f_min) / delta_logf) + 1
    
    # Generate resampled log-frequency points
    log_freq_resampled = np.linspace(log_f_min, log_f_max, num_samples)
    freq_resampled = 10**log_freq_resampled
    
    # Interpolate amplitude data to match linear-to-log spacing
    amplitude_interpolated = np.interp(np.log10(freq_resampled), np.log10(freq), amplitude)
    
    # Resample amplitude data using scipy.signal.resample
    amplitude_resampled = resample(amplitude_interpolated, num_samples)

    # Plot to compare
    plt.figure(figsize=(10, 6))
    plt.loglog(freq, amplitude, label='Original Data', alpha=0.7)
    plt.loglog(freq_resampled, amplitude_resampled, 'o-', color='orange', label=f'Resampled Data ($\Delta \log f = {delta_logf}$)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    # plt.xlim(0,20)
    plt.title('Frequency Domain Resampling with Fixed $\Delta \log f$ (Linear Input)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()
    
    return freq_resampled, amplitude_resampled

def resample_log_with_log_window(freq_list, amp_list, delta_logf=0.1, log_window=0.4):
    """
    Resample data using log-transformed amplitude and log-scaled window for frequency.

    Parameters:
        freq_list (2D array): Original frequency data (each row represents a set of frequencies).
        amp_list (2D array): Original amplitude data corresponding to freq_list.
        delta_logf (float): Interval in log10 frequency space for resampling.
        log_window (float): Width of the averaging window in log10 scale.

    Returns:
        resampled_freq (array): Resampled frequency points in linear scale.
        log_amp_resampled (2D array): Resampled amplitude data in log10 scale.
    """
    # Convert to numpy arrays
    freq_list = np.array(freq_list)
    amp_list = np.array(amp_list)

    # Convert amplitude to log10 scale
    log_amp_list = np.log10(amp_list)

    # Calculate log10 of frequency range
    log_f_min, log_f_max = np.log10(freq_list.min()), np.log10(freq_list.max())
    target_log_freq = np.arange(log_f_min, log_f_max, delta_logf)  # Fixed log10 intervals
    target_freq = 10**target_log_freq  # Convert back to linear space

    resampled_amp_list = []

    for i in range(freq_list.shape[0]):  # Process each row of frequency and amplitude
        freq = freq_list[i]
        log_amp = log_amp_list[i]
        resampled_amp = []

        for log_f in target_log_freq:
            # Define the window range in log10 scale
            lower_bound_log = log_f - log_window / 2
            upper_bound_log = log_f + log_window / 2

            # Convert bounds back to linear scale and find values within the range
            lower_bound = 10**lower_bound_log
            upper_bound = 10**upper_bound_log
            mask = (freq >= lower_bound) & (freq <= upper_bound)
            values_in_window = log_amp[mask]

            # Compute the average if there are values, otherwise assign NaN
            if len(values_in_window) > 0:
                resampled_amp.append(np.mean(values_in_window))
            else:
                resampled_amp.append(np.nan)

        resampled_amp_list.append(resampled_amp)

    # Convert the resampled amplitudes to a numpy array
    log_amp_resampled = np.array(resampled_amp_list)
    
    return target_freq, log_amp_resampled