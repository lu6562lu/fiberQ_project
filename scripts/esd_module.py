import numpy as np
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
import matplotlib.dates as mdates

def calculate_esd_for_stations(event_date, data_path, output_path=None):
    """
    Calculate ESD for all stations in a given directory, save visualizations, and return results.

    Parameters:
        data_dir (str): Directory containing SAC files.
        event_date (str): Event date for labeling (e.g., "20230218-2332").
        output_dir (str): Directory to save visualizations.

    Returns:
        results (list): A list of dictionaries containing ESD results for each station.
    """
    results = []
    sac_files = read(f'{data_path}/{event_date}/AC.MDSA5..GL*.SAC')

    # Divide the Stream into two groups
    station_traces_groups = [sac_files[:3], sac_files[3:]]  # Assume the first 3 and next 3 are different stations

    for i, station_traces in enumerate(station_traces_groups, start=1):
        try:
            print(f"Processing Station {i}...")

            # Process station data
            waveforms = []
            times = None
            for trace in station_traces:
                amplitudes = trace.data*100
                npts = trace.stats.npts
                start_time = trace.stats.starttime
                delta = trace.stats.delta
                if times is None:
                    times = np.array([start_time + i * delta for i in range(npts)])
                waveforms.append(amplitudes)
            waveforms = np.array(waveforms)
            resultant_waveform = np.sqrt(np.sum(waveforms ** 2, axis=0))

            # Calculate ESD
            valid_indices = resultant_waveform >= 0.01
            valid_time = times[valid_indices]
            valid_amplitudes = resultant_waveform[valid_indices]
            if len(valid_time) == 0:
                raise ValueError("No data points above the amplitude threshold (0.01 g).")
            energy = valid_amplitudes ** 2
            cumulative_energy = np.cumsum(energy)
            total_energy = cumulative_energy[-1]
            cumulative_energy_normalized = cumulative_energy / total_energy
            start_index = np.searchsorted(cumulative_energy_normalized, 0.05)
            end_index = np.searchsorted(cumulative_energy_normalized, 0.95)
            start_time = valid_time[start_index]
            end_time = valid_time[end_index]
            duration = end_time - start_time

            # Save visualization
            time_dt = [t.datetime for t in times]
            valid_time_dt = [t.datetime for t in valid_time]
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            components = ["N", "E", "Z"]
            for j, (ax, waveform, component) in enumerate(zip(axes[:3], waveforms, components)):
                ax.plot(time_dt, waveform, label=f"{component} Component")
                ax.set_ylabel("Amplitude (g)")
                ax.legend(loc="upper right")
                ax.grid()
                if j == 0:
                    ax.set_title(f"Seismic Waveforms and Resultant Amplitude: Station {i}")
            axes[3].plot(time_dt, resultant_waveform, label="Resultant Waveform", color="blue", alpha=0.4)
            axes[3].set_ylabel("Amplitude (g)")
            axes[3].grid()
            ax_energy = axes[3].twinx()
            ax_energy.plot(
                valid_time_dt,
                cumulative_energy_normalized,
                label="Cumulative Energy",
                color="black"
            )
            ax_energy.axvline(
                x=start_time.datetime,
                color="green",
                linestyle="--",
                label=f"5% Energy ({start_time.isoformat()})"
            )
            ax_energy.axvline(
                x=end_time.datetime,
                color="red",
                linestyle="--",
                label=f"95% Energy ({end_time.isoformat()})"
            )
            ax_energy.set_ylabel("Normalized Energy")
            ax_energy.legend(loc="upper left")
            ax_energy.grid()
            axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            plt.xticks(rotation=45)
            plt.tight_layout()
            if output_path:
                plt.savefig(f"{output_path}/{event_date}/ESD{i}_plot.png", dpi=300)
                print(f"Saved plot for Station {i} to {output_path}/{event_date}")
            else:
                print(f'Processed')
            plt.show()
            # plt.close(fig)

            # Append results
            results.append({
                "station": f"Station {i}",
                "duration": duration,
                "start_time": start_time,
                "end_time": end_time,
                # "plot_path": save_path
            })
        except Exception as e:
            print(f"Error processing Station {i}: {e}")

    return results