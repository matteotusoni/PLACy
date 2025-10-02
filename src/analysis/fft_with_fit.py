import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import f
import matplotlib.pyplot as plt
import os
import json


def fit_pvalue(y_true, y_pred, num_params):
    residuals = y_true - y_pred
    ssr = np.sum((y_pred - np.mean(y_true)) ** 2)
    sse = np.sum(residuals ** 2)
    df_model = num_params - 1
    df_resid = len(y_true) - num_params
    if df_resid <= 0:
        return np.nan, np.nan

    f_stat = (ssr / df_model) / (sse / df_resid)
    p_value = 1 - f.cdf(f_stat, df_model, df_resid)
    return p_value


def fft_with_fit(
    process: np.ndarray, window_length: int, stride: int,
    remove_outliers: bool = False, f_min: float = 1e-5, 
    alpha: float = 0., plot=False, p_value_test = False
) -> np.array:

    _, length = process.shape

    results = list()
    p_values = list()

    if p_value_test:
        path_logs = 'fit_logs'
        os.makedirs(path_logs, exist_ok=True) 
    if plot:
        path_plot = 'plots/fit'
        os.makedirs(path_plot, exist_ok=True)


    for i_signal, signal in enumerate(process):
        
        λ = list()
        a = list()
        p_value = list()

        for i in range(((length - window_length) // stride) + 1):
            signal_window = signal[stride * i: stride * i + window_length]

            fft = np.fft.fft(signal_window)
            xf = np.fft.fftfreq(window_length, d=1)[:window_length // 2]

            mask = xf > f_min
            freqs_filtered = xf[mask]
            power_spectrum = np.abs(fft[:window_length // 2][mask]) ** 2

            log_freqs = np.log10(freqs_filtered)
            log_power = np.log10(power_spectrum)

            def linear_fit(x, a, b):
                return a + b * x

            weights = 1 / (log_freqs ** alpha)

            try:
                popt, _ = curve_fit(linear_fit, log_freqs, log_power, p0=[1, -1], sigma=weights, absolute_sigma=True)
                lambda_fit = -popt[1]
                a_fit = popt[0]

                log_power_pred = linear_fit(log_freqs, *popt)

                p_val = fit_pvalue(log_power, log_power_pred, num_params=2)

                if remove_outliers:
                    residuals = log_power - log_power_pred
                    q1, q3 = np.percentile(residuals, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    mask_no_outliers = (residuals >= lower_bound) & (residuals <= upper_bound)

                    freqs_filtered = freqs_filtered[mask_no_outliers]
                    power_spectrum = power_spectrum[mask_no_outliers]
                    log_freqs = log_freqs[mask_no_outliers]
                    log_power = log_power[mask_no_outliers]

                    popt, _ = curve_fit(linear_fit, log_freqs, log_power, p0=[1, -1], sigma=1 / (log_freqs ** alpha), absolute_sigma=True)
                    lambda_fit = -popt[1]
                    a_fit = popt[0]  
                    log_power_pred = linear_fit(log_freqs, *popt)
                
                if plot:
                    os.makedirs(path_plot, exist_ok=True)
                    plt.figure(figsize=(14, 7))
                    plt.scatter(log_freqs, log_power)
                    plt.plot(
                        np.linspace(log_freqs[0], log_freqs[-1], 100), 
                        linear_fit(np.linspace(log_freqs[0], log_freqs[-1], 100), *popt), 
                        color='red', label=f"Fit: λ={lambda_fit:.2f}, a={a_fit:.2f}"
                    )
                    plt.title("Frequency Analysis with Power-Law Fit")
                    plt.xlabel("Frequency (1/dt)")
                    plt.ylabel("Power Spectrum")
                    plt.legend()
                    plt.grid(True, which="both", linestyle="--")
                    plt.savefig(f'{path_plot}/{i_signal}_window={i}.png')
                    plt.close()

                λ.append(lambda_fit)
                a.append(a_fit)
                p_value.append(p_val)

            except Exception as e:
                results.append({"lambda": np.nan, "fit_A": np.nan})

        results.append(λ)
        results.append(a)

        if p_value_test:
            p_values.append(p_value)
            to_save = dict(
                p_values=p_values,
            )
            path_logs = 'fit_logs'
            with open(f'{path_logs}/Pvalues_window_{window_length}_len_{len(process.T)}_stride_{stride}.json', 'w') as f:
                json.dump(to_save, f)

    return np.asarray(results)
