GAUGE_LENGTH = 10
SAMPLES = 1000
cnst=116/8192*10**(-9)*SAMPLES/GAUGE_LENGTH

# -- Data corrections- surface reflections --
def time_scale(a, b, samples=SAMPLES):
    a = to_datetime64(a)
    b = to_datetime64(b)
    return np.arange(a, b, np.timedelta64(int(1/samples/scipy.constants.nano), 'ns'))

def to_datetime64(t):
    if isinstance(t, str):
        return np.datetime64(datetime.datetime.fromisoformat(t), 'ns')
    return np.datetime64(t).astype('datetime64[ns]')
    
def load_das_profile(dasf,start,end,channels,das_cnst):
    st=read(dasf)
    st.trim(start,end)
    A=np.zeros([len(st),st[0].stats.npts])
    #t = np.arange(len(A[0,:])) / samples
    t=time_scale(start, end + 1/SAMPLES)[:A.shape[1]]
    
    for i in tqdm.tqdm(np.arange(len(st))): 
        A[i,:]=st[i].data*das_cnst
    
    A = xr.DataArray(A, dims=('channel', 'time'), coords=dict(
            time=t,
            channel=channels,
            depth=('channel', (channels - channels[0]) * 4*MULT),
        ))
    A.time.attrs['long_name'] = 'Time'
    A.depth.attrs['long_name'] = 'Depth'
    A.depth.attrs['units'] = 'meter'
    A.channel.attrs['long_name'] = 'Channel'
    A.attrs['long_name'] = 'Strain Rate'
    A.attrs['units'] = '1/sec'
    return A

def remove_reflections(A):
    Af = np.fft.fft2(A)
    x = Af.shape[0]//2
    y = Af.shape[1]//2
    Af[x:,:y] = 0
    Af[:x,y:] = 0
    return np.fft.ifft2(Af).real

def correct_reflections(A):
    # full waveforms
    A = A.copy()
    # upcoming waves
    A_up = remove_reflections(A.copy())
    # down-going waves
    A_down = A.copy() - A_up
    return A_up, A.data, A_down.data