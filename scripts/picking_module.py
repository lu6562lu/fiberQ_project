# File: picking_module.py

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import read
from obspy.signal.trigger import ar_pick
import os


def process_event(event_date, data_path, output_path=None, filter_low=1.0, filter_high=15.0):
    """
    Process seismic event data for P and S wave picking and plot the results.

    Parameters:
        event_date (str): The event date in 'YYYYMMDD-HHMM' format.
        data_path (str): Path to the directory containing SAC files.
        output_path (str): Path to save the output plot.
        filter_low (float): Lower bound of the bandpass filter.
        filter_high (float): Upper bound of the bandpass filter.

    Returns:
        dict: A dictionary containing the P-pick and S-pick times.
    """
    # Load data
    MDSA5 = read(f'{data_path}/{event_date}/AC.MDSA5..GL*.SAC')
    tr_z = MDSA5[2]
    tr_n = MDSA5[1]
    tr_e = MDSA5[0]

    # Apply bandpass filter
    tr_z_filtered = tr_z.copy().filter("bandpass", freqmin=filter_low, freqmax=filter_high)
    tr_n_filtered = tr_n.copy().filter("bandpass", freqmin=filter_low, freqmax=filter_high)
    tr_e_filtered = tr_e.copy().filter("bandpass", freqmin=filter_low, freqmax=filter_high)

    df = tr_z.stats.sampling_rate
    
    # 假設 z, n, e 為三個分量的 numpy.ndarray，samp_rate 為採樣率
    # 濾波參數
    f1 = 1.0    # 下限頻率，單位 Hz
    f2 = 20.0   # 上限頻率，單位 Hz
    
    # STA/LTA 參數（以秒為單位）
    sta_p = 0.1   # P 波 STA
    lta_p = 2.0   # P 波 LTA
    sta_s = 1.0   # S 波 STA
    lta_s = 4.0   # S 波 LTA
    
    # AR 模型係數
    m_p = 2
    m_s = 2

    # 方差視窗長度（以秒為單位）
    l_p = 0.1   # P 波方差視窗
    l_s = 0.2   # S 波方差視窗


    # AR picker
    p_pick_filtered, s_pick_filtered = ar_pick(tr_z_filtered.data, tr_n_filtered.data, tr_e_filtered.data, df, 
                                               f1=f1, f2=f2, lta_p=lta_p, sta_p=sta_p, lta_s=lta_s, sta_s=sta_s, 
                                               m_p=m_p, m_s=m_s, l_p=l_p, l_s=l_s)

    # Convert picks to absolute times
    start_time = tr_z.stats.starttime
    p_pick_time_filtered = start_time + p_pick_filtered
    s_pick_time_filtered = start_time + s_pick_filtered

    # Plotting
    plt.figure(figsize=(14, 10))

    # Z-component
    plt.subplot(3, 1, 1)
    plt.plot(tr_z_filtered.times("matplotlib"), tr_z_filtered.data, label='Filtered Z-component')
    plt.axvline(p_pick_time_filtered.matplotlib_date, color='red', linestyle='--', label='P pick')
    plt.axvline(s_pick_time_filtered.matplotlib_date, color='blue', linestyle='--', label='S pick')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.title('Filtered Z-component with P and S Picks')
    plt.ylabel('Amplitude')

    # N-component
    plt.subplot(3, 1, 2)
    plt.plot(tr_n_filtered.times("matplotlib"), tr_n_filtered.data, label='Filtered N-component')
    plt.axvline(p_pick_time_filtered.matplotlib_date, color='red', linestyle='--', label='P pick')
    plt.axvline(s_pick_time_filtered.matplotlib_date, color='blue', linestyle='--', label='S pick')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.title('Filtered N-component with P and S Picks')
    plt.ylabel('Amplitude')

    # E-component
    plt.subplot(3, 1, 3)
    plt.plot(tr_e_filtered.times("matplotlib"), tr_e_filtered.data, label='Filtered E-component')
    plt.axvline(p_pick_time_filtered.matplotlib_date, color='red', linestyle='--', label='P pick')
    plt.axvline(s_pick_time_filtered.matplotlib_date, color='blue', linestyle='--', label='S pick')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.title('Filtered E-component with P and S Picks')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    if output_path:
        output_file = f'{output_path}/{event_date}/borehole_picking.png'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)  # 確保目錄存在
        plt.savefig(output_file, dpi=300)

    plt.close("all")

    # Return picks as dictionary
    return {
        "P_pick": p_pick_time_filtered,
        "S_pick": s_pick_time_filtered
    }
