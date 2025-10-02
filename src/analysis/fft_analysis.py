import numpy as np
import matplotlib.pyplot as plt
import os


def fft_analysis(
    process: np.ndarray, window_length: int, stride: int,
    f_min: float = 0, plot=False,
) -> np.array:

    _, length = process.shape

    results = list()
    
    for i_signal, signal in enumerate(process):
        
        power = list()

        for i in range(((length - window_length) // stride) + 1):
            signal_window = signal[stride * i: stride * i + window_length]

            fft = np.fft.fft(signal_window)
            xf = np.fft.fftfreq(window_length, d=1)[:window_length // 2]

            mask = xf > f_min
            freqs_filtered = xf[mask]
            power_spectrum = np.abs(fft[:window_length // 2][mask]) ** 2

            if plot:
                path = f'plots/'
                os.makedirs(path, exist_ok=True)
                plt.figure(figsize=(14, 7))
                plt.loglog(
                    freqs_filtered, power_spectrum, '.', 
                    label=f"{i_signal}"
                )
                plt.title("Frequency Analysis of Timeseries with Power-Law Fit")
                plt.xlabel("Frequency (1/dt) ")
                plt.ylabel("Power Spectrum")
                plt.legend()
                plt.grid(True, which="both", linestyle="--")
                plt.savefig(f'plots/{i_signal}_window={i}.pdf')
                plt.close()

            power.append(power_spectrum)
        results.append(power)

    return np.asarray(results)