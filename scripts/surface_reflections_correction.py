import numpy as np
import datetime
import scipy.constants
from tqdm import tqdm
import xarray as xr
from obspy import read

GAUGE_LENGTH = 10
SAMPLES = 1000
MULT = 4.0838 
DAS_CNST = 116/8192*10**(-9)*SAMPLES/GAUGE_LENGTH

def to_datetime64(t):
    """將字串或datetime轉為numpy datetime64[ns]"""
    if isinstance(t, str):
        return np.datetime64(datetime.datetime.fromisoformat(t), 'ns')
    return np.datetime64(t).astype('datetime64[ns]')

def time_scale(a, b, samples=SAMPLES):
    """根據起訖時間產生時間軸"""
    a = to_datetime64(a)
    b = to_datetime64(b)
    return np.arange(a, b, np.timedelta64(int(1/samples/scipy.constants.nano), 'ns'))

def load_das_profile(dasf, start, end, channels, das_cnst=DAS_CNST):
    """
    載入DAS資料檔案，回傳xarray.DataArray格式
    """
    st = read(dasf)
    st.trim(start, end)
    A = np.zeros([len(st), st[0].stats.npts])
    t = time_scale(start, end + 1/SAMPLES)[:A.shape[1]]
    
    for i in tqdm(np.arange(len(st))): 
        A[i, :] = st[i].data * das_cnst
    
    A = xr.DataArray(
        A, 
        dims=('channel', 'time'), 
        coords=dict(
            time=t,
            channel=channels,
            depth=('channel', (channels - channels[0]) * MULT),
        )
    )
    A.time.attrs['long_name'] = 'Time'
    A.depth.attrs['long_name'] = 'Depth'
    A.depth.attrs['units'] = 'meter'
    A.channel.attrs['long_name'] = 'Channel'
    A.attrs['long_name'] = 'Strain Rate'
    A.attrs['units'] = '1/sec'
    return A

def remove_reflections(A: np.ndarray) -> np.ndarray:
    """
    利用FFT將數據中的反射波移除（簡單高低頻域濾波）
    """
    Af = np.fft.fft2(A)
    x = Af.shape[0] // 2
    y = Af.shape[1] // 2
    Af[x:, :y] = 0
    Af[:x, y:] = 0
    return np.fft.ifft2(Af).real

def correct_surface_reflection(A: xr.DataArray):
    """
    對輸入的DAS資料進行surface reflections修正
    回傳 (upgoing_wave, original, downgoing_wave)
    """
    # full waveforms
    A = A.copy()
    # upcoming waves
    A_up = remove_reflections(A.data)
    # down-going waves
    A_down = A.data - A_up
    # 包成xarray（保持原本的label/metadata）
    A_up_da = xr.DataArray(A_up, dims=A.dims, coords=A.coords, attrs=A.attrs)
    A_down_da = xr.DataArray(A_down, dims=A.dims, coords=A.coords, attrs=A.attrs)
    return A_up_da, A, A_down_da

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)  # 假設最後一軸是時間

def filter_by_channel(xarr, fs=1000, lowcut=1, highcut=20, order=4):
    data = xarr.values
    filtered = np.zeros_like(data)
    for i in tqdm(range(data.shape[0]), desc="Filtering channels"):
        filtered[i] = bandpass_filter(data[i], fs, lowcut, highcut, order)
    return xr.DataArray(filtered, dims=xarr.dims, coords=xarr.coords, attrs=xarr.attrs)

def plot_up_down_original(
    A_up_da, A, A_down_da, 
    ch_start=1234, ch_end=1404, 
    weight=1, 
    fs=1000, lowcut=1, highcut=20, order=4,
    do_filter=True,
    vmin=-1e-7, vmax=1e-7,
    figsize=(16,12)
):
    """
    繪製 Upgoing, Original, Downgoing 資料的時頻圖，可選擇是否進行濾波。
    """
    # 濾波或直接使用
    if do_filter:
        A_up_filt = filter_by_channel(A_up_da, fs, lowcut, highcut, order)
        A_filt = filter_by_channel(A, fs, lowcut, highcut, order)
        A_down_filt = filter_by_channel(A_down_da, fs, lowcut, highcut, order)
    else:
        A_up_filt = A_up_da
        A_filt = A
        A_down_filt = A_down_da

    sel = slice(ch_start, ch_end + 1)
    data_list = [
        (A_up_filt.isel(channel=sel) * weight),
        (A_filt.isel(channel=sel) * weight),
        (A_down_filt.isel(channel=sel) * weight)
    ]
    titles = ['Upgoing', 'Original', 'Downgoing']

    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)
    ims = []
    for ax, data, label in zip(axs, data_list, titles):
        im = data.plot(
            x='time', y='channel',
            ax=ax,
            cmap='seismic',
            vmin=vmin, vmax=vmax,
            add_colorbar=False
        )
        ax.set_title(label)
        ax.set_ylabel('Channel')
        ax.invert_yaxis()
        ims.append(im)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(ims[0], cax=cbar_ax, label='Strain Rate (1/sec)')

    plt.xlabel('Time')
    plt.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08, hspace=0.3)
    plt.show()

def plot_channel_traces(
    A_up_da, A_da, A_down_da,
    channel_id,
    t0=None, seconds=3,
    do_filter=True,
    fs=1000, lowcut=1, highcut=20, order=4
):
    """
    繪製單一 channel 上 Upgoing, Original, Downgoing 的時序曲線，可選擇是否濾波。

    參數
    ----
    A_up_da, A_da, A_down_da : xarray.DataArray
        三種資料（未濾波）。
    channel_id : int
        要顯示的 channel ID。
    t0 : np.datetime64 or None
        起始時間，預設為資料最早時間。
    seconds : float
        從 t0 起持續幾秒的資料。
    do_filter : bool
        是否進行濾波。
    fs, lowcut, highcut, order : 濾波參數
    """
    if do_filter:
        A_up = filter_by_channel(A_up_da, fs, lowcut, highcut, order)
        A = filter_by_channel(A_da, fs, lowcut, highcut, order)
        A_down = filter_by_channel(A_down_da, fs, lowcut, highcut, order)
    else:
        A_up = A_up_da
        A = A_da
        A_down = A_down_da

    if t0 is None:
        t0 = A_up['time'].values[0]
    tt = t0 + np.timedelta64(int(seconds * 1e9), 'ns')
    time_sel = (A_up['time'] <= tt)

    y_up = A_up.sel(channel=channel_id).sel(time=time_sel)
    y_orig = A.sel(channel=channel_id).sel(time=time_sel)
    y_down = A_down.sel(channel=channel_id).sel(time=time_sel)

    t = y_up['time'].values

    plt.figure(figsize=(10,4))
    plt.plot(t, y_orig, label='Original', color='black', lw=1)
    plt.plot(t, y_up, label='Upgoing', color='red', lw=1)
    plt.plot(t, y_down, label='Downgoing', color='blue', lw=1)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'Channel {channel_id}: Upgoing, Original, Downgoing')
    plt.legend()
    plt.tight_layout()
    plt.show()