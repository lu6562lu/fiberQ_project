import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scripts.signal_processing import fft



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


