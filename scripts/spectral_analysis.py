import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from signal_processing import fft


# ===============================
# === Data Plot in Time- and Frequency- Domain ===
# ===============================

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


# ===============================
# === Omega-Square Fitting ===
# ===============================

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
