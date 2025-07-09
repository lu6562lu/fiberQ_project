import os
import numpy as np
import pandas as pd
from obspy import Stream, read, UTCDateTime


from scripts.signal_processing import fft, integrate_stream, fft_divide, smooth_moving_average
from scripts.mapping import load_velocity_data, average_velocity, load_depth_mapping, define_depth_combinations
from scripts.spectral_analysis import process_signal
from scripts.resample_module import resample_log_with_log_window
from scripts.esd_module import calculate_esd_for_stations
from scripts.picking_module import process_event
from scripts.fitting import plot_velocity_spectra, plot_velocity_spectra_iteration
from scripts.surface_reflections_correction import (
    mseed_upgoing, 
    plot_up_down_original_mseed, 
    plot_channel_traces_mseed
)

from scripts.q_methods import (
    amplitude_ratio_original,
    amplitude_ratio_resampled,
    amplitude_ratio_omega_square,
)

def analyze_event(
    event_date,
    data_path,
    output_path,
    instrument_type,
    wave_type,
    depth_pair,
    velocity_csv="../data/v.csv",
    depth_map_csv="../data/fiber_depth_mapping.csv",
    plot=True
):
    """
    整合所有流程與三種Q分析方法，回傳所有Q值、誤差、繪圖路徑等。
    """
    # === 1. 路徑與參數設定 ===
    save_path = f"{output_path}/{event_date}/{instrument_type}-{wave_type}/{depth_pair}"
    os.makedirs(save_path, exist_ok=True)

    # === 2. 載入速度與深度對應表 ===
    velocity_data = load_velocity_data(velocity_csv)
    df_map = load_depth_mapping(depth_map_csv)
    ds = depth_pair[1] - depth_pair[0]
    if wave_type == 'p':
        v = average_velocity(depth_pair, velocity_data)
    else:
        v = average_velocity(depth_pair, velocity_data) / np.sqrt(3)

    # === 3. 對應儀器通道 ===
    depth_combinations = define_depth_combinations(depth_pair, df_map)
    if instrument_type not in ['fiber', 'borehole']:
        raise ValueError("Invalid instrument_type. Choose 'fiber' or 'borehole'.")
    station1, station2 = depth_combinations[instrument_type]

    # === 4. Picking ===
    picking = process_event(event_date, f"{data_path}/borehole", output_path=output_path)

    # === 5. 讀取資料 ===
    start_time = picking['P_pick'] - 10
    end_time = picking['S_pick'] + 5

    if instrument_type == 'borehole':
        st1 = read(f"{data_path}/{instrument_type}/{event_date}/VL.{station1}..GL*.SAC", starttime=start_time, endtime=end_time)
        st2 = read(f"{data_path}/{instrument_type}/{event_date}/VL.{station2}..GL*.SAC", starttime=start_time, endtime=end_time)
        st1[0].stats.channel = 'N'
        st1[1].stats.channel = 'E'
        st2[0].stats.channel = 'N'
        st2[1].stats.channel = 'E'

        if wave_type == 'p':
            trace1 = read(f"{data_path}/{instrument_type}/{event_date}/VL.{station1}..GLZ*.SAC", starttime=start_time, endtime=end_time)[0]
            trace2 = read(f"{data_path}/{instrument_type}/{event_date}/VL.{station2}..GLZ*.SAC", starttime=start_time, endtime=end_time)[0]
        else:
            df = pd.read_csv("inci30_ml2-4_s_borehole.csv")
            baz = float(df.loc[df["Event"] == event_date, "baz"].iloc[0])
            st1.rotate("NE->RT", back_azimuth=baz)
            st2.rotate("NE->RT", back_azimuth=baz)
            trace1 = st1.select(channel="T")[0]
            trace2 = st2.select(channel="T")[0]

    elif instrument_type == 'fiber':
        fiber_file = f"{data_path}/{instrument_type}/{event_date}.mseed"
        st = read(fiber_file, starttime=start_time, endtime=end_time)
        st_up, st, st_down = mseed_upgoing(st)
        trace1 = integrate_stream(Stream([st_up[int(station1)]]), 'vel')[0]
        trace2 = integrate_stream(Stream([st_up[int(station2)]]), 'vel')[0]

    # === 6. 信號/雜訊分段 ===
    if wave_type == 's':
        esd_results = calculate_esd_for_stations(event_date, f"{data_path}/borehole")
        signal_start = picking['S_pick'] - 0.5
        signal_duration = esd_results[0]['end_time'] - signal_start
    else:
        signal_start = picking['P_pick'] - 0.5
        signal_duration = picking['S_pick'] - picking['P_pick']
    noise_duration = 3

    wv = [trace1, trace2]
    signal_amp_list, signal_freq_list = [], []
    noise_amp_list, noise_freq_list = [], []
    snr_list = []

    for idx, trace in enumerate(wv):
        sig_amp, sig_freq, noi_amp, noi_freq, snr = process_signal(
            trace, signal_start, signal_duration, noise_duration, picking,
            save_path=save_path if plot else None
        )
        signal_amp_list.append(sig_amp)
        signal_freq_list.append(sig_freq)
        noise_amp_list.append(noi_amp)
        noise_freq_list.append(noi_freq)
        snr_list.append(snr)

    avg_snr = float(np.mean(snr_list))

    # === 7. 頻譜重取樣 ===
    delta_logf = 0.08
    signal_freq_resampled, signal_amp_resampled = resample_log_with_log_window(
        signal_freq_list, signal_amp_list, delta_logf=delta_logf, log_window=0.4
    )
    noise_freq_resampled, noise_amp_resampled = resample_log_with_log_window(
        noise_freq_list, noise_amp_list, delta_logf=delta_logf, log_window=0.4
    )

    # === 8. omega-square擬合需用到的spectral fitting ===
    omega_results_resample, fitting, f_filtered = plot_velocity_spectra_iteration(
        signal_freq_resampled,
        signal_amp_resampled,
        noise_freq_resampled,
        noise_amp_resampled,
        wv,
        initial_guess=[1e-10, 5, 0.01],
        f_range=(3, 20),
        wave=wave_type,
        save_path=save_path,
        save_name='omega_fit_resampled'
    )

    fc_avg = (omega_results_resample[0]['f_c'] + omega_results_resample[1]['f_c']) / 2

    # === 9. 三種Q分析方法 ===
    # 第一種：原始
    q_ori_result = amplitude_ratio_original(
        signal_amp_list, signal_freq_list, noise_amp_list, noise_freq_list,
        station1, station2, ds, v, wave_type, fc_avg,
        delta_f=signal_freq_list[0][1] - signal_freq_list[0][0], save_path=save_path, plot=plot
    )
    # 第二種：重取樣
    q_resampled_result = amplitude_ratio_resampled(
        signal_amp_resampled, signal_freq_resampled,
        signal_amp_list, signal_freq_list,
        noise_amp_list, noise_freq_list,
        station1, station2, ds, v, wave_type, fc_avg, save_path=save_path, plot=plot
    )
    # 第三種：omega-square擬合
    q_omega_result = amplitude_ratio_omega_square(
        signal_amp_resampled, signal_freq_resampled,
        noise_amp_resampled, noise_freq_resampled,
        omega_results_resample, fitting, f_filtered,
        station1, station2, ds, v, wave_type, fc_avg, save_path=save_path, plot=plot
    )

    # === 10. 統一回傳資料 ===
    result = {
        "event_date": event_date,
        "instrument_type": instrument_type,
        "wave_type": wave_type,
        "depth_pair": str(depth_pair),
        "station1": station1,
        "station2": station2,
        "v": v,
        "ds": ds,
        "avg_snr": avg_snr,
        "fit_fc": fc_avg,
        "plot_dir": save_path,
        # 三種方法結果
        **q_ori_result,
        **q_resampled_result,
        **q_omega_result,
    }
    return result