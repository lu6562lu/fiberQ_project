import numpy as np
import datetime
import scipy.constants
import tqdm
import xarray as xr
from obspy import read

GAUGE_LENGTH = 10
SAMPLES = 1000
MULT = 4.0838  # 若有不同倍率請自行修改
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
    
    for i in tqdm.tqdm(np.arange(len(st))): 
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