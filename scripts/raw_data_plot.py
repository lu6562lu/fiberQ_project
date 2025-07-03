import numpy as np
import matplotlib.pyplot as plt

def t_vector_para(tr):
    N = tr.stats.npts
    fs = tr.stats.sampling_rate
    dt = 1 / fs
    t = np.arange(N) / fs
    return N, fs, dt, t

def plot_filled_timeseries(
    st,
    startnode=1234,
    endnode=1404,
    intv=1,
    y_plot_scale=3e-4,
    event_date="",
    picking=None,
    show_arrival=False,
    fig_name=None,
    figsize=(10, 20),
    show_progress=True,
    do_filter=False,
    filter_type="bandpass",
    filter_kwargs=None,
):
    """
    Plots timeseries using fill_between for a range of nodes with scaled and reversed y-axis.
    Positive amplitude (y>0) in blue, negative (y<0) in red. Optionally marks P/S arrivals.

    Args:
        st: dict or list of traces to plot (should support st[i].data)
        startnode: starting node index (int)
        endnode: ending node index (int)
        intv: interval between nodes (int)
        y_plot_scale: scale factor for y axis (float)
        event_date: date string, used in plot title and filename
        picking: dict with 'P_pick' and 'S_pick' as UTCDateTime objects for arrival times (optional)
        show_arrival: whether to show P/S arrival vertical lines (bool)
        fig_name: filename for saving the figure (str, default: f'profile{event_date}.png')
        figsize: size of the figure as (width, height)
        show_progress: whether to show a progress bar (default: True)
        do_filter: whether to apply filtering to each trace (default: False)
        filter_type: filter type to apply if do_filter is True (default: 'bandpass')
        filter_kwargs: dict, additional kwargs for filter (e.g., {'freqmin':1, 'freqmax':10})
    """
    if fig_name is None:
        fig_name = f'profile{event_date}.png'

    # Try to import tqdm for progress bar
    use_tqdm = False
    if show_progress:
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            print("tqdm not found, progress bar will not be shown.")

    if do_filter:
        try:
            # Import obspy for filtering
            from obspy import Trace
        except ImportError:
            raise ImportError("obspy is required for filtering. Please install obspy.")

    plt.rcParams["figure.figsize"] = figsize
    fig, ax1 = plt.subplots()

    node_range = np.arange(startnode, endnode + 1, intv)
    iterator = tqdm(node_range, desc="Plotting traces") if use_tqdm else node_range

    for i in iterator:
        tr = st[i]
        # Apply filtering if requested
        if do_filter:
            # Make a copy to avoid changing original data
            tr = tr.copy()
            if filter_kwargs is None:
                filter_kwargs = {}
            # Default bandpass: freqmin=1, freqmax=10
            if filter_type == "bandpass":
                filter_args = dict(freqmin=1, freqmax=10)
                filter_args.update(filter_kwargs)
                tr.filter('bandpass', **filter_args)
            elif filter_type == "lowpass":
                filter_args = dict(freq=10)
                filter_args.update(filter_kwargs)
                tr.filter('lowpass', **filter_args)
            elif filter_type == "highpass":
                filter_args = dict(freq=1)
                filter_args.update(filter_kwargs)
                tr.filter('highpass', **filter_args)
            elif filter_type == "notch":
                filter_args = dict(freq=50)
                filter_args.update(filter_kwargs)
                tr.filter('notch', **filter_args)
            else:
                raise ValueError(f"Unsupported filter_type: {filter_type}")
        N, fs, dt, t = t_vector_para(tr)
        y = tr.data * y_plot_scale * (-1)
        # Ensure t and y are the same length
        length = min(len(t), len(y))
        t, y = t[:length], y[:length]
        ax1.fill_between(t, i, i + y, where=(y > 0), facecolor='blue', alpha=0.3, linewidth=0)
        ax1.fill_between(t, i, i + y, where=(y < 0), facecolor='red', alpha=0.3, linewidth=0)

    ax1.set_xlim(0, 2.5)
    ax1.set_ylim(startnode - 1, endnode + 1)
    ax1.set_yticks(np.arange(startnode - intv, endnode + intv, intv * 3))
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Node # of trace')
    ax1.set_title(f'{event_date} Raw Data Plot')

    # Plot P/S arrivals if available
    if picking is not None and show_arrival:
        st0_start = st[0].stats.starttime
        if 'P_pick' in picking:
            p_arrival = (picking['P_pick'] - st0_start)
            ax1.axvline(p_arrival, color='red', linestyle='--', linewidth=1.5, label='P arrival')
        if 'S_pick' in picking:
            s_arrival = (picking['S_pick'] - st0_start)
            ax1.axvline(s_arrival, color='blue', linestyle='--', linewidth=1.5, label='S arrival')
        ax1.legend()

    ax1.invert_yaxis()
    plt.savefig(fig_name)
    plt.show()