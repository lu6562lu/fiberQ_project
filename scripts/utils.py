## modules for spectral anaylsis
import numpy as np
import pandas as pd
from obspy import read, UTCDateTime, Trace, Stream 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares
from scipy.signal import resample
from scipy.interpolate import interp1d

def get_integrated_stream(stream):
    # taper, integrate, highpass=0.075Hz
    stream_intergrated = stream.copy()
    stream_intergrated.taper(max_percentage=0.05, type="cosine")
    stream_intergrated.integrate()
    stream_intergrated.filter("highpass", freq=0.075)
    return stream_intergrated


def get_integrated_stream_second(stream):
    # integrate, highpass=0.075Hz
    stream_intergrated = stream.copy()
    stream_intergrated.integrate()
    stream_intergrated.filter("highpass", freq=0.075)
    return stream_intergrated


def integrate_stream(data, waveform_type):
    stream = data
    stream.detrend(type="demean")  # baseline correction
    # stream.filter("lowpass", freq=20)  # filter

    if waveform_type == "acc":
        pass
    elif waveform_type == "vel":
        stream = get_integrated_stream(stream)
    elif waveform_type == "dis":
        stream = get_integrated_stream(stream)
        stream = get_integrated_stream_second(stream)
    return stream


def fft(wv, N, dt): 
    y = wv.data
    y_fft = np.fft.fft(y)
    freq = np.fft.fftfreq(N, d=dt)
    Amp = abs(y_fft/(N/2))
    # log_amp = np.log10(Amp[1:int(N/2)])
    amp = Amp[1:int(N/2)]
    f = freq[1:int(N/2)]
    # print(freq)
    return amp, f


velocity_data = pd.read_csv("../v.csv")
def average_velocity(depth_pair):
    
    start_depth, end_depth = depth_pair
    depth_range_data = velocity_data[(velocity_data['depth'] >= start_depth) & (velocity_data['depth'] <= end_depth)]
    average_velocity = depth_range_data['velocity'].mean()

    return average_velocity
    

df_map = pd.read_csv("../fiber_depth_mapping.csv")  # columns: depth, fiber, borehole
def define_depth_combinations(depth_pair):
    d1, d2 = depth_pair

    def find_nearest(depth):
        idx = (df_map["depth"] - depth).abs().idxmin()
        row = df_map.loc[idx]
        return {
            "depth": row["depth"],
            "fiber": str(int(row["fiber"])),
            "borehole": row["borehole"] if pd.notna(row["borehole"]) else None,
            "error": abs(row["depth"] - depth)
        }

    info1 = find_nearest(d1)
    info2 = find_nearest(d2)

    print(f"[FIBER] {d1}m → #{info1['fiber']} (misfit: {info1['error']:.1f}m)")
    print(f"[FIBER] {d2}m → #{info2['fiber']} (misfit: {info2['error']:.1f}m)")
    print(f"[BOREHOLE] {d1}m → {info1['borehole']} (misfit: {info1['error']:.1f}m)")
    print(f"[BOREHOLE] {d2}m → {info2['borehole']} (misfit: {info2['error']:.1f}m)")

    return {
        "fiber": (info1["fiber"], info2["fiber"]),
        "borehole": (info1["borehole"], info2["borehole"])
    }

# Function to generate node pairs with sliding window
def generate_node_pairs(start_node_idx, end_node_idx, window_length, step):
    """
    Generate node pairs using a sliding window approach.
    
    Parameters:
        start_node_idx (int): Index of the starting node.
        end_node_idx (int): Index of the ending node.
        window_length (int): Length of the window analyzed (number of nodes).
        step (int): Step size for sliding the window (number of nodes).
    
    Returns:
        List[Tuple[int, int]]: List of node pairs.
    """

    # Load the meter-to-node mapping from CSV
    mapping_file = "meter_to_node_map.csv" 
    meter_to_node_df = pd.read_csv(mapping_file)
    
    node_list = meter_to_node_df["Node"].tolist()
    pairs = []
    for i in range(start_node_idx, end_node_idx - window_length + 1, step):
        start_node = node_list[i]
        end_node = node_list[i + window_length]
        pairs.append((start_node, end_node))
    return pairs



def generate_node_pairs_with_depth(start_node_idx, end_node_idx, window_length, step, mapping_file):
    """
    Generate node pairs with their corresponding meter pairs in a simplified format.

    Parameters:
        start_node_idx (int): Index of the starting node.
        end_node_idx (int): Index of the ending node.
        window_length (int): Length of the window analyzed (number of nodes).
        step (int): Step size for sliding the window (number of nodes).
        mapping_file (str): Path to the meter-to-node mapping file.
    
    Returns:
        List[Dict]: List of node and meter pairs in the specified format.
    """

    # Load the meter-to-node mapping from CSV
    meter_to_node_df = pd.read_csv(mapping_file)
    
    # Extract node and meter mappings
    node_list = meter_to_node_df["Node"].tolist()
    meter_list = meter_to_node_df["Meter"].tolist()
    
    pairs = []
    for i in range(start_node_idx, end_node_idx - window_length + 1, step):
        # Get the node pair
        start_node = node_list[i]
        end_node = node_list[i + window_length]
        
        # Get the corresponding meter pair
        start_meter = meter_list[i]
        end_meter = meter_list[i + window_length]
        
        # Append the node and meter pairs in the specified format
        pairs.append({
            "Node": (start_node, end_node),
            "Meter": (start_meter, end_meter)
        })
    
    return pairs


# Function to process and save data for each pair
def process_and_save_data_for_pairs(node_pairs, fiber_data_path, start_time, end_time, waveform_type):
    """
    Process fiber data for each pair of nodes and save to a variable.
    """
    fiber_data = read(fiber_data_path, starttime=start_time, endtime=end_time)  # Read the entire fiber data
    processed_data = {}  # Dictionary to store data for each pair

    for start_node, end_node in node_pairs:
        print(f"Processing data for nodes {start_node} and {end_node}...")
        
        # Extract traces for the pair
        try:
            tr_u = fiber_data[start_node]
            tr_l = fiber_data[end_node]
        except IndexError:
            print(f"Error: Node indices {start_node} or {end_node} not found in data.")
            continue
        
        # Combine into a Stream
        st_strainrate = Stream(traces=[tr_u, tr_l])
        
        # Apply waveform processing based on the waveform_type
        st_processed = integrate_stream(st_strainrate, waveform_type)
        
        # Integrate to strain
        # st_strain = utils.integrate_stream(st_used, 'vel')
        
        # Save processed data to dictionary
        processed_data[(start_node, end_node)] = st_processed
        print(f"Data for nodes {start_node} and {end_node} processed and saved.")

    return processed_data
    

def process_signal(trace, signal_start_time, signal_duration, noise_duration, picking, save_path=None):
    """
    處理指定的 station trace，根據時間範圍提取信號和噪聲數據。

    :trace: data used
    :param signal_start: str, 信號開始時間 (格式: "YYYY-MM-DDTHH:MM:SS")
    :param signal_duration: float, 信號持續時間 (秒)
    :param noise_duration: float, 噪聲持續時間 (秒)
    :return: None
    """
    # 轉換輸入的時間
    signal_end_time = signal_start_time + signal_duration

    # 檢查時間範圍是否合理
    if signal_start_time < trace.stats.starttime or signal_end_time > trace.stats.endtime:
        print("指定的 signal 時間範圍超出 Trace 時間範圍")
        return

    # 設定噪聲時間範圍 (1 sec before p arrival)
    noise_end_time = picking['P_pick'] - 3
    noise_start_time = noise_end_time - noise_duration

    if noise_start_time < trace.stats.starttime:
        print("指定的 noise 時間範圍超出 Trace 時間範圍")
        return

    # 擷取信號和噪聲
    signal_trace = trace.slice(starttime=signal_start_time, endtime=signal_end_time)
    noise_trace = trace.slice(starttime=noise_start_time, endtime=noise_end_time)

    # 計算信號與噪聲的 RMS
    signal_rms = (signal_trace.data ** 2).mean() ** 0.5
    noise_rms = (noise_trace.data ** 2).mean() ** 0.5
    
    # 計算 SNR
    snr = signal_rms / noise_rms if noise_rms > 0 else float("inf")
    
    # 顯示結果
    # print(f"Signal RMS: {signal_rms:.4f}")
    # print(f"Noise RMS: {noise_rms:.4f}")
    print(f"SNR: {snr:.4f}")

    # 計算 FFT
    dt = trace.stats.delta  # 時間間隔
    signal_amp, signal_freq = fft(signal_trace, len(signal_trace.data), dt)
    noise_amp, noise_freq = fft(noise_trace, len(noise_trace.data), dt)
    print(f"Signal length: {len(signal_trace.data)/1000} second")

    # 繪製信號與噪聲
    plt.figure(figsize=(12, 6))

    # 原始數據（局部放大）
    plt.subplot(2, 1, 1)
    plt.plot(trace.times("matplotlib"), trace.data, label="Original Trace", color='black', linewidth = 0.6)
    plt.axvspan(noise_start_time.matplotlib_date, noise_end_time.matplotlib_date, color="red", alpha=0.3, label="Noise Region")
    plt.axvspan(signal_start_time.matplotlib_date, signal_end_time.matplotlib_date, color="blue", alpha=0.3, label="Signal Region")
    plt.legend()
    plt.title(f"Trace for Station {trace.stats.station}")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Amplitude")
    plt.xlim(noise_start_time.matplotlib_date - 0.00001, signal_end_time.matplotlib_date + 0.0001)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    # FFT 頻譜圖
    plt.subplot(2, 1, 2)
    plt.plot(signal_freq, signal_amp, label="Signal FFT", color="blue")
    plt.plot(noise_freq, noise_amp, label="Noise FFT", color="red", alpha=0.6)
    plt.legend()
    # plt.xlim(0,50)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Signal and Noise FFT")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    label_text = f"SNR = {snr:.4f}"
    plt.text(0.01, 0.01, label_text, transform=plt.gca().transAxes,
         fontsize=12, ha='left', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/spec_vel_{trace.stats.station[1:]}.png', dpi=300)
        print(f"Plot saved to {save_path}")
    plt.show()
   
    return signal_amp, signal_freq, noise_amp, noise_freq, snr


def plot_velocity_spectra(
    signal_freq_list, 
    signal_amp_list, 
    noise_freq_list, 
    noise_amp_list, 
    st_strain, 
    f_range=(0, 20), 
    initial_guess=[0.5, 1.5, 0.03], 
    use_default_fc=False,  
    default_fc=1.5,        # Default value for f_c if use_default_fc=True
    wave='s',
    save_path=None,
    save_name='omega_fit'
):
    """
    Plot and fit velocity spectra using the ω-square source model.

    Parameters:
    - signal_freq_list: List of arrays, frequency data for signal spectra.
    - signal_amp_list: List of arrays, amplitude data for signal spectra.
    - noise_freq_list: List of arrays, frequency data for noise spectra.
    - noise_amp_list: List of arrays, amplitude data for noise spectra.
    - st_strain: List of objects with station info for labeling.
    - f_range: Tuple, frequency range for filtering data (default: (0, 25) Hz).
    - initial_guess: List, initial guesses for [Omega_0, f_c, t_star].
    - use_default_fc: Bool, whether to fix f_c to default_fc (default: False).
    - default_fc: Float, default value for f_c if use_default_fc=True.
    - save_path: String, optional path to save the output plot.

    Returns:
    - results: List of dictionaries with fitted parameters for each station.
    - A_fit
    """

    def velocity_spectra(f, Omega_0, f_c, t_star):
        return np.log10(2 * np.pi * f * Omega_0 * (f_c**2 / (f_c**2 + f**2)) * np.exp(-np.pi * f * t_star))
        # return 2 * np.pi * f * Omega_0 * (f_c**2 / (f_c**2 + f**2)) * np.exp(-np.pi * f * t_star)

    results = []  # To store results for each station
    fitting = []

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=True, constrained_layout=True)

    f = signal_freq_list
    for i, ax in enumerate(axes):
        # Extract and filter data
        mask = (f >= f_range[0]) & (f <= f_range[1])  
        f_filtered = f[mask]
        sig_filtered = signal_amp_list[i][mask]

        # Define fitting function with or without fixed f_c
        if use_default_fc:
            def residuals(params):
                Omega_0, t_star = params
                return velocity_spectra(f_filtered, Omega_0, default_fc, t_star) - sig_filtered
            initial_fit_params = [initial_guess[0], initial_guess[2]]
            bounds = ([0, 0], [np.inf, np.inf])  # 只對 Omega_0 和 t_star 設定正值範圍
        else:
            def residuals(params):
                return velocity_spectra(f_filtered, *params) - sig_filtered
            initial_fit_params = initial_guess
            bounds = ([0, 0.5, 0], [np.inf, 20, np.inf])  # 限制 f_c 在 1 Hz 到 100 Hz 之間

        # 擬合過程
        result = least_squares(residuals, initial_fit_params, loss='huber', f_scale=0.1, bounds=bounds)

        Omega_0_fit = result.x[0]
        if use_default_fc:
            f_c_fit = default_fc
            t_star_fit = result.x[1]
        else:
            f_c_fit = result.x[1]
            t_star_fit = result.x[2]

        A_fit = velocity_spectra(f_filtered, Omega_0_fit, f_c_fit, t_star_fit)
        
        # Store results
        results.append({
            "station": str(st_strain[i].stats.station[1:]),
            "Omega_0": Omega_0_fit,
            "f_c": f_c_fit,
            "t_star": t_star_fit,
        })

        fitting.append(A_fit)
        
        print(f"Results for Station #{str(st_strain[i].stats.station)[0:]}")
        print(f"  Fitted Omega_0: {Omega_0_fit}")
        print(f"  Fitted f_c: {f_c_fit}")
        print(f"  Fitted t*: {t_star_fit}")
        print()
        
        # Plot original data, noise, and fitting results
        ax.plot(signal_freq_list, 10**signal_amp_list[i], 'k', label=f'Signal')
        ax.plot(noise_freq_list, 10**noise_amp_list[i], 'gray', linestyle='--', label='Noise')
        ax.plot(f_filtered, 10**A_fit, 'r-', label='Fitting')
        ax.axvline(x=f_c_fit, color='y', linestyle='--', label='$f_c$', alpha=0.5)  # 加入 f_c 垂直線


        if wave == 's':
        # Add parameter labels to the plot
            ax.text(
                0.9, 0.95, 
                f"$f_c$ = {np.round(f_c_fit, 3)}\n$t_s^*$ = {t_star_fit:.3f}", 
                fontsize=12, ha='right', va='top', transform=ax.transAxes
            ) 
        else:
            ax.text(
                0.9, 0.95, 
                f"$f_c$ = {np.round(f_c_fit, 3)}\n$t_p^*$ = {t_star_fit:.3f}", 
                fontsize=12, ha='right', va='top', transform=ax.transAxes
            ) 

        # Set axis scales and labels
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.5,150)
        # ax.set_xlabel('Frequency (Hz)')
        ax.set_title(fr'$\omega^2$ Model Fit of #{str(st_strain[i].stats.station)[0:]}')
        ax.legend(loc='lower left')

    # Add shared Y-axis label
    axes[0].set_ylabel('Amplitude [Log]')
    axes[1].set_xlabel('Frequency (Hz)')

    # Show and optionally save the plot
    plt.show()
    if save_path:
        fig.savefig(f'{save_path}/{save_name}.png', dpi=300)
        print(f"Plot saved to {save_path}")
        
    return results, fitting, f_filtered

def fft_divide(wv1, wv2):
    if np.any(wv2 == 0):
        return "Error: wv2 contains zero values, division not possible."
    result_amp = wv1 / wv2
    return result_amp


def fit_and_plot_amplitude_ratio(signal_amp_list, signal_freq_list, save_path, delta_f, f_range=(3, 25), min_points=10, wave='s', save_name='amp_ratio'):
    """
    Performs amplitude ratio calculation, linear fitting, variance estimation, and plots the results.
    
    Parameters:
    - signal_amp_list: List of arrays, amplitude data for signal spectra.
    - signal_freq_list: List of arrays, frequency data for signal spectra.
    - save_path: String, path to save the output plot.
    - delta_f: Float, frequency step size for FFT.
    - f_range: Tuple, frequency range for filtering data (default: (0.5, 25) Hz).
    - wave: String, 's' for shear wave, 'p' for primary wave.

    Returns:
    - coefficients: Tuple, linear fitting coefficients (slope and intercept).
    - variance: Tuple, variances of the slope and intercept.
    - tps: Float, calculated Δt_p* or Δt_s* value.
    """
    f = signal_freq_list[0] if isinstance(signal_freq_list, list) else signal_freq_list

    try:
        # Compute amplitude ratio using FFT divide function
        div_amp = signal_amp_list[0] / signal_amp_list[1]  # 替換 fft_divide
    except ValueError as e:
        print(str(e))
        return None, None, None

    # 確保 log 計算不會遇到負數或 0
    if np.any(div_amp <= 0):
        raise ValueError("Error: div_amp contains non-positive values, cannot compute log.")

    # print(div_amp, f)

    # Apply frequency range filter
    mask = (f >= f_range[0]) & (f <= f_range[1])  
    filtered_frequency = f[mask]
    filtered_amplitude = np.log(div_amp)[mask]
    # print(filtered_frequency)
    
    if len(filtered_frequency) < min_points:
        interp_func = interp1d(filtered_frequency, filtered_amplitude, kind='linear', fill_value="extrapolate")
        new_x = np.linspace(filtered_frequency[0], f_range[1], min_points)  # 在範圍內補足 min_points 個點
        filtered_frequency = new_x
        filtered_amplitude = interp_func(new_x)
    # print(filtered_amplitude, filtered_frequency)

    # Perform linear fitting with variance calculation
    coefficients, covariance_matrix = np.polyfit(filtered_frequency, filtered_amplitude, 1, cov=True)
    slope, intercept = coefficients
    slope_var, intercept_var = np.diag(covariance_matrix)  # 取對角線元素作為方差

    # 計算 95% 置信區間
    slope_std = np.sqrt(slope_var)
    intercept_std = np.sqrt(intercept_var)
    ci_slope = 1.96 * slope_std  # 95% 置信區間
    ci_intercept = 1.96 * intercept_std

    # Calculate Δt_p* or Δt_s*
    tps = -slope / np.pi
    tps_std = ci_slope / np.pi  # Δt_p* 或 Δt_s* 的標準差
    tps_ci = 1.96 * tps_std  # 95% 置信區間

    # Compute fitted values and confidence interval
    fitted_values = np.polyval(coefficients, filtered_frequency)
    lnR_std = np.sqrt((filtered_frequency ** 2) * slope_var + intercept_var + 2 * filtered_frequency * covariance_matrix[0, 1])
    fitted_upper = fitted_values + 1.96 * lnR_std
    fitted_lower = fitted_values - 1.96 * lnR_std

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(f, np.log(div_amp), 'k-', label='Amplitude Ratio')
    plt.plot(filtered_frequency, fitted_values, 'r--', label='Fitted Line')
    plt.fill_between(filtered_frequency, fitted_lower, fitted_upper, color='red', alpha=0.2, label='95% Confidence Interval')

    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('ln(R)', fontsize=12)
    plt.xlim(f_range[0]-1, f_range[1]+1)
    y_min, y_max = min(filtered_amplitude), max(filtered_amplitude)
    margin = (y_max - y_min) * 0.3  # 增加 10% 的邊界
    plt.ylim(y_min - margin, y_max + margin)
    # plt.ylim(min(np.log(div_amp)), max(np.log(div_amp)))
    # plt.ylim(None, None)

    # 添加線性擬合公式和 Δt_p*
    # 添加線性擬合公式（不含誤差範圍）
    label_text = f"$lnR(f) = {intercept:.3f} + {slope:.3f}f$\n"
    
    # Δt_p* 或 Δt_s* 需要包含誤差範圍
    label_text += f"$\\Delta t_s^*$ = {tps:.4f} ± {tps_ci:.4f}" if wave == 's' else f"$\\Delta t_p^*$ = {tps:.4f} ± {tps_ci:.4f}"
    
    plt.text(f_range[1] * 0.95, y_min - margin, label_text,
             fontsize=12, ha='right', va='bottom')

    plt.legend(loc='upper right')

    if save_path:
        plt.savefig(f'{save_path}/{save_name}.png', dpi=300)
    # plt.show()

    print(f"斜率 a: {slope:.3f} ± {ci_slope:.3f}")
    print(f"截距 b: {intercept:.3f} ± {ci_intercept:.3f}")
    out = f"Δt_p*: {tps:.4f} ± {tps_ci:.4f}" if wave == 'p' else f"Δt_s*: {tps:.4f} ± {tps_ci:.4f}"
    print(out)
    tps_info = {"ts": tps, "ts_error": tps_ci}
    plt.close()

    return coefficients, (slope_var, intercept_var), tps_info


def check_array_validity(*arrays):
    """
    Check if any of the provided arrays contain NaN or negative infinity values.

    Parameters:
    - arrays: List of arrays to check.

    Returns:
    - A dictionary summarizing the validity of each array.
    """
    results = {}
    for i, arr in enumerate(arrays, start=1):
        has_nan = np.isnan(arr).any()
        has_neginf = np.isneginf(arr).any()
        results[f'Array_{i}'] = {'has_nan': has_nan, 'has_neginf': has_neginf}
    return results


def smooth_moving_average(data, window_size=5):
    """
    對資料做簡單移動平均平滑處理
    :param data: 一維 numpy array
    :param window_size: 平滑視窗大小，需為奇數（預設為5，即前後各兩筆）
    :return: 平滑後的資料 array，長度與輸入相同
    """
    assert window_size % 2 == 1, "Window size 必須為奇數"
    pad = window_size // 2
    padded_data = np.pad(data, pad_width=pad, mode='edge')  # 邊界補值
    smoothed = np.convolve(padded_data, np.ones(window_size)/window_size, mode='valid')
    return smoothed