import numpy as np
import matplotlib.pyplot as plt

def amplitude_ratio_original(
    signal_amp_list, signal_freq_list, noise_amp_list, noise_freq_list,
    station1, station2, ds, v, wave_type, fc_avg, delta_f, save_path,
    plot=True
):
    """
    第一種 Q 計算（原始法），可選擇是否儲存繪圖。
    回傳 Q、誤差與圖檔路徑。
    """
    fit_range = (fc_avg, 20)
    mask = (signal_freq_list[0] >= fit_range[0]) & (signal_freq_list[0] <= fit_range[1])
    fit_freq = signal_freq_list[0][mask]
    fit_amp_0 = signal_amp_list[0][mask]
    fit_amp_1 = signal_amp_list[1][mask]
    lnR_a = np.log(fit_amp_0 / fit_amp_1)
    coeffs_a, cov_a = np.polyfit(fit_freq, lnR_a, 1, cov=True)
    slope_a, intercept_a = coeffs_a
    lnR_fit_a = np.polyval(coeffs_a, fit_freq)
    lnR_std_a = np.sqrt((fit_freq**2)*cov_a[0,0] + cov_a[1,1] + 2*fit_freq*cov_a[0,1])
    lnR_upper_a = lnR_fit_a + 1.96 * lnR_std_a
    lnR_lower_a = lnR_fit_a - 1.96 * lnR_std_a
    tps_ori = -slope_a / np.pi
    ts_error_ori = 1.96 * (np.sqrt(cov_a[0,0]) / np.pi)

    Q_value_ori = ds / (v * tps_ori)
    Q_error_ori = abs((ds / (v * (tps_ori - ts_error_ori))) - Q_value_ori)
    Q_error_neg_ori = abs((ds / (v * (tps_ori + ts_error_ori))) - Q_value_ori)

    original_full_png = None
    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(6, 8))
        # 上圖：Signal, Noise, Fit of both stations
        axs[0].plot(signal_freq_list[0], np.log10(signal_amp_list[0]), color='black', linewidth=1.2, alpha=0.4, label=f'Signal (Upper #{station1})')
        axs[0].plot(noise_freq_list[0], np.log10(noise_amp_list[0]), color='black', linestyle='--', alpha=0.4, label=f'Noise (Upper #{station1})')
        axs[0].plot(fit_freq, np.log10(fit_amp_0), color='black', linewidth=2.0, alpha=0.8)

        axs[0].plot(signal_freq_list[1], np.log10(signal_amp_list[1]), color='orange', linewidth=1.2, alpha=0.4, label=f'Signal (Lower #{station2})')
        axs[0].plot(noise_freq_list[1], np.log10(noise_amp_list[1]), color='orange', linestyle='--', alpha=0.4, label=f'Noise (Lower #{station2})')
        axs[0].plot(fit_freq, np.log10(fit_amp_1), color='orange', linewidth=2.0, alpha=0.8)

        axs[0].set_title(f"Original Amplitude Spectra — #{station1} & #{station2}", fontsize=11)
        axs[0].set_ylabel("Amplitude (log10)", fontsize=11)
        axs[0].set_xscale("log")
        axs[0].legend(fontsize=9)
        axs[0].grid(True, which="both", linestyle="--", alpha=0.3)

        axs[1].plot(fit_freq, lnR_a, color='k', label='ln(R)', linewidth=1)
        axs[1].plot(fit_freq, lnR_fit_a, 'r--', label='Fitted')
        axs[1].fill_between(fit_freq, lnR_lower_a, lnR_upper_a, color='red', alpha=0.2, label='95% CI')

        axs[1].set_xlabel("Frequency (Hz)", fontsize=11)
        axs[1].set_ylabel("ln(R)", fontsize=11)
        axs[1].legend(fontsize=9)
        axs[1].grid(True, linestyle="--", alpha=0.3)

        textstr = (
            f"$lnR(f) = {intercept_a:.3f} + {slope_a:.3f}f$\n"
            f"$\\Delta t_{{{wave_type}}}^* = {tps_ori:.4f} \\pm {ts_error_ori:.4f}$"
        )
        axs[1].text(
            0.05, 0.03,
            textstr,
            transform=axs[1].transAxes,
            fontsize=12,
            ha='left',
            va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)
        )
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)
        original_full_png = f"{save_path}/original_full.png"
        plt.savefig(original_full_png, dpi=300)
        plt.close(fig)

    return {
        "Q_value_ori": Q_value_ori,
        "Q_error_ori_pos": Q_error_ori,
        "Q_error_ori_neg": Q_error_neg_ori,
        "ts_ori": tps_ori,
        "ts_error_ori": ts_error_ori,
        "original_full_png": original_full_png,
    }



def amplitude_ratio_resampled(
    signal_amp_resampled, signal_freq_resampled,
    signal_amp_list, signal_freq_list,
    noise_amp_list, noise_freq_list,
    station1, station2, ds, v, wave_type, fc_avg, save_path,
    plot=True
):
    """
    第二種 Q 計算（重取樣法），可選擇是否儲存繪圖。
    回傳 Q、誤差與圖檔路徑。
    """
    fit_range = (fc_avg, 20)
    mask = (signal_freq_resampled >= fit_range[0]) & (signal_freq_resampled <= fit_range[1])
    fit_freq = signal_freq_resampled[mask]
    fit_amp_0 = signal_amp_resampled[0][mask]
    fit_amp_1 = signal_amp_resampled[1][mask]

    amp_ratio = 10**fit_amp_0 / 10**fit_amp_1
    lnR = np.log(amp_ratio)

    coefficients, covariance_matrix = np.polyfit(fit_freq, lnR, 1, cov=True)
    slope, intercept = coefficients
    slope_var, intercept_var = np.diag(covariance_matrix)
    slope_std = np.sqrt(slope_var)
    intercept_std = np.sqrt(intercept_var)
    ci_slope = 1.96 * slope_std
    ci_intercept = 1.96 * intercept_std

    fitted_lnR = np.polyval(coefficients, fit_freq)
    lnR_std = np.sqrt((fit_freq ** 2) * slope_var + intercept_var + 2 * fit_freq * covariance_matrix[0, 1])
    fitted_upper = fitted_lnR + 1.96 * lnR_std
    fitted_lower = fitted_lnR - 1.96 * lnR_std

    tps = -slope / np.pi
    tps_ci = 1.96 * (slope_std / np.pi)

    Q_value_re = ds / (v * tps)
    Q_error_re = abs((ds / (v * (tps - tps_ci))) - Q_value_re)
    Q_error_neg_re = abs((ds / (v * (tps + tps_ci))) - Q_value_re)

    resample_ratio_png = None
    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(6, 8))
        # 上圖：Resampled 和 Original Amplitude Spectra
        axs[0].scatter(signal_freq_resampled, signal_amp_resampled[0], color='k', s=5, label=f'Resampled (Upper #{station1})')
        axs[0].plot(fit_freq, fit_amp_0, 'k', linewidth=2)
        axs[0].plot(signal_freq_list[0], np.log10(signal_amp_list[0]), 'k', alpha=0.4, label=f'Original (Upper #{station1})')
        axs[0].plot(noise_freq_list[0], np.log10(noise_amp_list[0]), 'k', linestyle='--', alpha=0.2, label=f'Noise (Upper #{station1})')
        axs[0].scatter(signal_freq_resampled, signal_amp_resampled[1], color='orange', s=5, label=f'Resampled (Lower #{station2})')
        axs[0].plot(fit_freq, fit_amp_1, 'orange', linewidth=2)
        axs[0].plot(signal_freq_list[1], np.log10(signal_amp_list[1]), 'orange', alpha=0.4, label=f'Original (Lower #{station2})')
        axs[0].plot(noise_freq_list[1], np.log10(noise_amp_list[1]), 'orange', linestyle='--', alpha=0.2, label=f'Noise (Lower #{station2})')
        axs[0].axvline(fit_range[0], color='gray', linestyle='--', alpha=0.5, label=f'fc={fit_range[0]:.2f}')
        axs[0].set_title(f"Resampled Amplitude Spectra — #{station1} & #{station2}", fontsize=11)
        axs[0].set_ylabel("Amplitude [Log]")
        axs[0].set_xscale("log")
        axs[0].legend()
        axs[0].grid(True, which="both", linestyle="--", alpha=0.3)

        # 下圖：lnR和fit
        axs[1].scatter(fit_freq, lnR, s=5, color='k', label="Amplitude Ratio")
        axs[1].plot(fit_freq, fitted_lnR, 'r--', label="Fitted Line")
        axs[1].fill_between(fit_freq, fitted_lower, fitted_upper, color='red', alpha=0.2, label="95% Confidence Interval")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("ln(R)")
        axs[1].legend()
        axs[1].grid(True, linestyle="--", alpha=0.3)
        textstr = (
            f"$lnR(f) = {intercept:.3f} + {slope:.3f}f$\n"
            f"$\\Delta t_{{{wave_type}}}^* = {tps:.4f} \\pm {tps_ci:.4f}$"
        )
        axs[1].text(0.05, 0.03, textstr, transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
        plt.tight_layout()
        resample_ratio_png = f"{save_path}/resample_ratio.png"
        plt.savefig(resample_ratio_png, dpi=300)
        plt.close(fig)

    return {
        "Q_value_re": Q_value_re,
        "Q_error_re_pos": Q_error_re,
        "Q_error_re_neg": Q_error_neg_re,
        "ts_re": tps,
        "ts_error_re": tps_ci,
        "resample_ratio_png": resample_ratio_png,
    }



def amplitude_ratio_omega_square(
    signal_amp_resampled, signal_freq_resampled,
    noise_amp_resampled, noise_freq_resampled,
    omega_results_resample, fitting, f_filtered,
    station1, station2, ds, v, wave_type, fc_avg, save_path,
    plot=True
):
    """
    第三種 Q 計算（Omega-square擬合法），可選擇是否儲存繪圖。
    回傳 Q、誤差與圖檔路徑。
    """
    omega_full_png = None
    fit_range = (fc_avg, 20)
    mask = (f_filtered >= fit_range[0]) & (f_filtered <= fit_range[1])
    f = f_filtered[mask]

    # --- 繪圖 ---
    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(6, 8))
        plt.subplots_adjust(hspace=0.1)

        for i in range(2):
            amp_obs = signal_amp_resampled[i]
            amp_noise = noise_amp_resampled[i]

            # 擬合資料
            station_id = omega_results_resample[i]["station"]
            fc = omega_results_resample[i]["f_c"]
            tstar = omega_results_resample[i]["t_star"]

            signal_color = 'k' if i == 0 else 'orange'
            noise_color = 'gray' if i == 0 else 'orange'
            label_suffix = f"(Upper #{station1})" if i == 0 else f"(Lower #{station2})"

            axs[0].plot(signal_freq_resampled, 10 ** amp_obs, color=signal_color, label=f"Signal {label_suffix}")
            axs[0].plot(noise_freq_resampled, 10 ** amp_noise, color=noise_color, linestyle='--', label=f"Noise {label_suffix}")
            axs[0].plot(f_filtered, 10 ** fitting[i], 'r', linewidth=2)

        axs[0].axvline(fc_avg, color='gold', linestyle='--', alpha=0.6, label=rf'$f_c$={fc_avg:.2f}')
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].set_xlim(0.5, 150)
        axs[0].set_ylabel("Amplitude [log10]")
        axs[0].set_title(rf"$\omega^2$ Fitting Amplitude Spectra — #{station1} & #{station2}", fontsize=11)
        axs[0].legend(fontsize=9)
        axs[0].grid(True, which="both", linestyle='--', alpha=0.3)

    # --- lnR擬合 ---
    lnR = np.log(10 ** fitting[0][mask] / 10 ** fitting[1][mask])
    coeffs, cov = np.polyfit(f, lnR, 1, cov=True)
    slope, intercept = coeffs
    lnR_fit = np.polyval(coeffs, f)
    lnR_std = np.sqrt((f ** 2) * cov[0, 0] + cov[1, 1] + 2 * f * cov[0, 1])
    lnR_upper = lnR_fit + 1.96 * lnR_std
    lnR_lower = lnR_fit - 1.96 * lnR_std
    tps = -slope / np.pi
    tps_ci = 1.96 * (np.sqrt(cov[0, 0]) / np.pi)

    if plot:
        axs[1].plot(f, lnR, 'k', label='ln(R)')
        axs[1].plot(f, lnR_fit, 'r--', label='Fitted')
        axs[1].fill_between(f, lnR_lower, lnR_upper, color='red', alpha=0.2, label='95% CI')
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("ln(R)")
        axs[1].legend()
        axs[1].grid(True, linestyle='--', alpha=0.3)

        textstr = (
            rf"$\ln R(f) = {slope:.3f}f + {intercept:.3f}$" + "\n" +
            rf"$\Delta t_{{{wave_type}}}^* = {tps:.4f} \pm {tps_ci:.4f}$"
        )
        axs[1].text(0.05, 0.03, textstr,
                    transform=axs[1].transAxes,
                    fontsize=12, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
        omega_full_png = f"{save_path}/omega_full_combined.png"
        plt.savefig(omega_full_png, dpi=300)
        plt.close(fig)

    ts = tps
    ts_error = tps_ci
    Q_value = ds / (v * ts)
    Q_error = abs((ds / (v * (ts - ts_error))) - Q_value)
    Q_error_neg = abs((ds / (v * (ts + ts_error))) - Q_value)

    return {
        "Q_value": Q_value,
        "Q_error_pos": Q_error,
        "Q_error_neg": Q_error_neg,
        "ts": ts,
        "ts_error": ts_error,
        "omega_full_png": omega_full_png,
    }