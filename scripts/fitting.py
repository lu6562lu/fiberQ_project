import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

# ===============================
# === Omega-Square Fitting ===
# ===============================

def plot_velocity_spectra(
    signal_freq_list, 
    signal_amp_list, 
    noise_freq_list, 
    noise_amp_list, 
    st_strain, 
    wave='s',
    f_range=(0, 20), 
    initial_guess=[0.5, 1.5, 0.03], 
    use_default_fc=False,  
    default_fc=1.5,
    save_path=None,
    save_name='omega_fit',
    plot=True
):
    """
    Plot and fit velocity spectra using the ω-square source model.

    Parameters:
    - signal_freq_list: List/array, frequency data for signal spectra.
    - signal_amp_list: List/array, amplitude data for signal spectra.
    - noise_freq_list: List/array, frequency data for noise spectra.
    - noise_amp_list: List/array, amplitude data for noise spectra.
    - st_strain: List of objects with station info for labeling.
    - wave: String, 's' or 'p'.
    - f_range: Tuple, frequency range for filtering data.
    - initial_guess: List, initial guesses for [Omega_0, f_c, t_star].
    - use_default_fc: Bool, whether to fix f_c to default_fc (default: False).
    - default_fc: Float, default value for f_c if use_default_fc=True.
    - save_path: String, optional path to save the output plot.
    - save_name: String, optional name for the saved plot (default: 'omega_fit').
    - plot: Bool, whether to plot and/or save the figure.

    Returns:
    - results: List of dictionaries with fitted parameters for each station.
    - fitting: List of arrays, fitted log-amplitude spectra for each station.
    - f_filtered: Array, filtered frequency array (for the last station).
    """

    def velocity_spectra(f, Omega_0, f_c, t_star):
        return np.log10(2 * np.pi * f * Omega_0 * (f_c**2 / (f_c**2 + f**2)) * np.exp(-np.pi * f * t_star))

    results = []
    fitting = []

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, sharey=True, constrained_layout=True)

    f = signal_freq_list
    for i in range(2):
        mask = (f >= f_range[0]) & (f <= f_range[1])  
        f_filtered = f[mask]
        sig_filtered = signal_amp_list[i][mask]

        if use_default_fc:
            def residuals(params):
                Omega_0, t_star = params
                return velocity_spectra(f_filtered, Omega_0, default_fc, t_star) - sig_filtered
            initial_fit_params = [initial_guess[0], initial_guess[2]]
            bounds = ([0, 0], [np.inf, np.inf])
        else:
            def residuals(params):
                return velocity_spectra(f_filtered, *params) - sig_filtered
            initial_fit_params = initial_guess
            bounds = ([0, 0.5, 0], [np.inf, 20, np.inf])

        result = least_squares(residuals, initial_fit_params, loss='huber', f_scale=0.1, bounds=bounds)

        Omega_0_fit = result.x[0]
        if use_default_fc:
            f_c_fit = default_fc
            t_star_fit = result.x[1]
        else:
            f_c_fit = result.x[1]
            t_star_fit = result.x[2]

        A_fit = velocity_spectra(f_filtered, Omega_0_fit, f_c_fit, t_star_fit)
        
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
        
        if plot:
            ax = axes[i]
            ax.plot(signal_freq_list, 10**signal_amp_list[i], 'k', label=f'Signal')
            ax.plot(noise_freq_list, 10**noise_amp_list[i], 'gray', linestyle='--', label='Noise')
            ax.plot(f_filtered, 10**A_fit, 'r-', label='Fitting')
            ax.axvline(x=f_c_fit, color='y', linestyle='--', label='$f_c$', alpha=0.5)
            if wave == 's':
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
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(0.5,150)
            ax.set_title(fr'$\omega^2$ Model Fit of #{str(st_strain[i].stats.station)[0:]}')
            ax.legend(loc='lower left')

    if plot:
        axes[0].set_ylabel('Amplitude [Log]')
        axes[1].set_xlabel('Frequency (Hz)')
        if save_path:
            fig.savefig(f'{save_path}/{save_name}.png', dpi=300)
            print(f"Plot saved to {save_path}/{save_name}.png")
        plt.show()
        plt.close(fig)
        
    return results, fitting, f_filtered

# ===============================
# === Amplitude Ratio Fitting ===
# ===============================

def fit_and_plot_amplitude_ratio(
    signal_amp_list, 
    signal_freq_list, 
    delta_f,
    wave='s',
    f_range=(3, 25), 
    min_points=10, 
    save_path=None, 
    save_name='amp_ratio',
    plot=True
):
    """
    Performs amplitude ratio calculation, linear fitting, variance estimation, and plots the results.

    Parameters:
    - signal_amp_list: List of arrays, amplitude data for signal spectra.
    - signal_freq_list: List of arrays, frequency data for signal spectra.
    - delta_f: Float, frequency step size for FFT.
    - wave: String, 's' for shear wave, 'p' for primary wave.
    - f_range: Tuple, frequency range for filtering data (default: (0.5, 25) Hz).
    - min_points: Int, minimum number of points after filtering.
    - save_path: String, path to save the output plot.
    - save_name: String, optional name for the saved plot (default: 'amp_ratio').
    - plot: Bool, whether to plot and/or save the figure.

    Returns:
    - coefficients: Tuple, linear fitting coefficients (slope and intercept).
    - variance: Tuple, variances of the slope and intercept.
    - tps_info: Dict, calculated Δt_p* or Δt_s* value and error.
    """
    f = signal_freq_list[0] if isinstance(signal_freq_list, list) else signal_freq_list

    try:
        div_amp = signal_amp_list[0] / signal_amp_list[1]
    except ValueError as e:
        print(str(e))
        return None, None, None

    if np.any(div_amp <= 0):
        raise ValueError("Error: div_amp contains non-positive values, cannot compute log.")

    mask = (f >= f_range[0]) & (f <= f_range[1])  
    filtered_frequency = f[mask]
    filtered_amplitude = np.log(div_amp)[mask]

    if len(filtered_frequency) < min_points:
        interp_func = interp1d(filtered_frequency, filtered_amplitude, kind='linear', fill_value="extrapolate")
        new_x = np.linspace(filtered_frequency[0], f_range[1], min_points)
        filtered_frequency = new_x
        filtered_amplitude = interp_func(new_x)

    coefficients, covariance_matrix = np.polyfit(filtered_frequency, filtered_amplitude, 1, cov=True)
    slope, intercept = coefficients
    slope_var, intercept_var = np.diag(covariance_matrix)

    slope_std = np.sqrt(slope_var)
    intercept_std = np.sqrt(intercept_var)
    ci_slope = 1.96 * slope_std
    ci_intercept = 1.96 * intercept_std

    tps = -slope / np.pi
    tps_std = ci_slope / np.pi
    tps_ci = 1.96 * tps_std

    fitted_values = np.polyval(coefficients, filtered_frequency)
    lnR_std = np.sqrt((filtered_frequency ** 2) * slope_var + intercept_var + 2 * filtered_frequency * covariance_matrix[0, 1])
    fitted_upper = fitted_values + 1.96 * lnR_std
    fitted_lower = fitted_values - 1.96 * lnR_std

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(f, np.log(div_amp), 'k-', label='Amplitude Ratio')
        plt.plot(filtered_frequency, fitted_values, 'r--', label='Fitted Line')
        plt.fill_between(filtered_frequency, fitted_lower, fitted_upper, color='red', alpha=0.2, label='95% Confidence Interval')
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('ln(R)', fontsize=12)
        plt.xlim(f_range[0]-1, f_range[1]+1)
        y_min, y_max = min(filtered_amplitude), max(filtered_amplitude)
        margin = (y_max - y_min) * 0.3
        plt.ylim(y_min - margin, y_max + margin)
        label_text = f"$lnR(f) = {intercept:.3f} + {slope:.3f}f$\n"
        label_text += f"$\\Delta t_s^*$ = {tps:.4f} ± {tps_ci:.4f}" if wave == 's' else f"$\\Delta t_p^*$ = {tps:.4f} ± {tps_ci:.4f}"
        plt.text(f_range[1] * 0.95, y_min - margin, label_text,
                 fontsize=12, ha='right', va='bottom')
        plt.legend(loc='upper right')
        if save_path:
            plt.savefig(f'{save_path}/{save_name}.png', dpi=300)
        plt.close()

    print(f"斜率 a: {slope:.3f} ± {ci_slope:.3f}")
    print(f"截距 b: {intercept:.3f} ± {ci_intercept:.3f}")
    out = f"Δt_p*: {tps:.4f} ± {tps_ci:.4f}" if wave == 'p' else f"Δt_s*: {tps:.4f} ± {tps_ci:.4f}"
    print(out)
    tps_info = {"ts": tps, "ts_error": tps_ci}

    return coefficients, (slope_var, intercept_var), tps_info