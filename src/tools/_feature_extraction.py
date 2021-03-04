"""This script contains all the function used to compute features."""
import numpy as np
import spectrum
from numba import njit
from scipy import fft, stats


def spectral_centroid(signal, samplerate: float):
    """Return the spectral centroid measure for the 'signal'.

    The spectral centroid can be seen as the center of mass
    of the spectrum.

    See: https://en.wikipedia.org/wiki/Spectral_centroid

    Args:
        signal: a one dimension array which contains the sampled signal.
        samplerate: a float which gives the sampling rate of the signal.

    Returns:
        The spectral centroid measure, which is a scalar value
    """
    # Magnitudes of positive frequencies (used as weights)
    magnitudes = np.abs(np.fft.rfft(signal))
    # Number of samples
    length = signal.shape[-1]
    # Returns the weighted mean of the frequencies
    return np.sum(
        magnitudes * np.abs(
            np.fft.fftfreq(
                length, 1.0 / samplerate,
            )[:length // 2 + 1],
        ),
        axis=-1,
    ) / np.sum(
        magnitudes,
        axis=-1,
    )


def spectral_flatness(signal):
    """Return the spectral flatness of the signal.

    Also known as the Wiener entropy,
    see 'https://en.wikipedia.org/wiki/Spectral_flatness'.

    It is a good indicator in audio processing to differenciate noise
    from actual signal.

    Args:
        signal: a one dimension array which contains the sampled signal.

    Returns:
        The spectral flatness, which is a scalar value.
    """
    x_f = np.array(fft.fft(signal, axis=-1))
    magnitude = abs(x_f[:, :x_f.shape[-1] // 2])

    return stats.mstats.gmean(magnitude, axis=-1) / np.mean(magnitude, axis=-1)


# Copied from pyeegModif <-- (to review)
def original_hjorth_parameters(
        signal,
        sampling_frequency: float = 1,
        first_diff=None):
    r"""Compute the Hjorth parameters following
    the flow graph (Fig. 2) provided
    in the Hjorth's paper (doi: 10.1016/0013-4694(70)90143-4).

    Args:
        signal: the considered signal as an 1D NumPy array.
        sampling_frequency: the sampling frequency of the signal as a float.
        first_diff: the first differentiation of the signal array.

    Returns:
        An inline tupple which contains the computed Hjorth parameters
        (activity, mobility and complexity).

    Note (according to Hjorth's paper):
        The activity can be seen as
        the variance or the mean power of the signal.
        The mobility can be seen as
        the mean frequency of the signal.
        The complexity is giving a measure of
        excessive detail with reference a sine wave.
    """
    if first_diff is None:
        first_diff = np.diff(signal)
    second_diff = np.diff(first_diff)

    # Time interval in second
    time_interval = signal.shape[-1] / sampling_frequency
    activity_times_interval = (
        signal ** 2).sum(axis=-1) / sampling_frequency
    activity = activity_times_interval / time_interval

    mobility_squared = (
        (first_diff ** 2).sum(axis=-1) / sampling_frequency
    ) / activity_times_interval
    mobility = np.sqrt(mobility_squared)

    partial_complexity_squared = (
        (second_diff ** 2).sum(axis=-1) / sampling_frequency
    ) / mobility_squared
    complexity = np.sqrt(partial_complexity_squared) / mobility

    return activity, mobility, complexity


# D~=log(N)/(log(N)+log(N/(N+1.4*Nd-Nd))
def petrosian_fractal_dimension(
    signal,
    first_diff=None,
    method: str = 'average',
):
    r"""Compute the Petrosian fractal dimension using
    one of the following method:

    'average' method - the EEG sample was assigned to be 1,
    if it was above the signal average value, and 0 otherwise.

    modified zone method ('modified_zone') - it was assigned to be 1 if it was
    out of bounds of average plus or minus standard deviation,
    and 0 otherwise.

    'differential' method - the sample was given value 1
    if the difference between two consecutive samples
    is positive, and 0 if it is negative.

    Args:
        signal: the input signal.
        first_diff: the first differential of the signal.
        method: the method used to compute the Petrosian fractal dimension.

    Returns:
        The value of the Petrosian fractal dimension.
    """
    n = signal.shape[-1]
    if method == 'average':
        mean = signal.mean(axis=-1)
        try:  # Makes the function works with 1D and 2D arrays
            v = np.where(signal < mean, -1, 1)
        except NameError:
            v = np.vstack(
                [
                    np.where(sig < mea, -1, 1)
                    for sig, mea in zip(signal, mean)
                ],
            )

        nd = np.count_nonzero((v[:, 1:] * v[:, :-1]) == -1, axis=-1)
        return np.log(n) / (np.log(n) + np.log(n / (n + 0.4 * nd)))

    elif method == 'modified_zone':
        mean = signal.mean(axis=-1)
        standard_deviation = signal.std(axis=-1)
        # v = np.where(np.logical_and(
        #     signal < (mean + standard_deviation),
        #     signal > (mean - standard_deviation)), -1, 1)
        v = np.vstack([np.where(
            np.logical_and(
                sig < (mea + std),
                sig > (mea - std)), -1, 1)
            for sig, mea, std
            in zip(signal, mean, standard_deviation)])

        nd = np.count_nonzero((v[:, 1:] * v[:, :-1]) == -1, axis=-1)
        return np.log(n) / (np.log(n) + np.log(n / (n + 0.4 * nd)))

    elif method == 'differential':
        if first_diff is None:
            v = np.diff(signal)
        else:
            v = first_diff
        v = np.where(v < 0, -1, 1)
        nd = np.count_nonzero((v[:, 1:] * v[:, :-1]) == -1, axis=-1)
        return np.log(n) / (2 * np.log(n) - np.log(n + 0.4 * nd))


# D~=log(N)/(log(N)+log(N/(N+1.4*Nd-Nd))
def paul_fractal_dimension(
    signal,
    first_diff=None,
    method: str = 'average',
):
    r"""Compute the PyEEG erroneous Petrosian fractal dimension using
    one of the following method:

    'average' method - the EEG sample was assigned to be 1,
    if it was above the signal average value, and 0 otherwise.

    modified zone method ('modified_zone') - it was assigned to be 1 if it was
    out of bounds of average plus or minus standard deviation,
    and 0 otherwise.

    'differential' method - the sample was given value 1
    if the difference between two consecutive samples
    is positive, and 0 if it is negative.

    Args:
        signal: the input signal.
        first_diff: the first differential of the signal.
        method: the method used to compute the Petrosian fractal dimension.

    Returns:
        The value of the Petrosian fractal dimension.
    """
    n = signal.shape[-1]
    if method == 'average':
        mean = signal.mean(axis=-1)
        try:  # Makes the function works with 1D and 2D arrays
            v = np.where(signal < mean, -1, 1)
        except NameError:
            v = np.vstack(
                [
                    np.where(sig < mea, -1, 1)
                    for sig, mea in zip(signal, mean)
                ],
            )

        nd = np.count_nonzero((v[:, 1:] * v[:, :-1]) == -1, axis=-1)
        return np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * nd))

    elif method == 'modified_zone':
        mean = signal.mean(axis=-1)
        standard_deviation = signal.std(axis=-1)
        v = np.vstack([np.where(
            np.logical_and(
                sig < (mea + std),
                sig > (mea - std)), -1, 1)
            for sig, mea, std
            in zip(signal, mean, standard_deviation)])

        nd = np.count_nonzero((v[:, 1:] * v[:, :-1]) == -1, axis=-1)
        return np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * nd))

    elif method == 'differential':
        if first_diff is None:
            v = np.diff(signal)
        else:
            v = first_diff
        v = np.where(v < 0, -1, 1)
        nd = np.count_nonzero((v[:, 1:] * v[:, :-1]) == -1, axis=-1)
        return np.log10(n) / (np.log10(n) + np.log10(n / n + 0.4 * nd))


def higuchi_fractal_dimension(signal, k_max: int):
    r"""Compute Higuchi Fractal Dimension of a time series.

    Improved version of the Pyrem module.

    Args:
        signal: 1D or 2D array containing the signal to process.
        k_max: the value of the 'k_max' used for the computation.

    Returns:
        The Higuchi's fractal dimension of the given signal.
    """
    l_higuchi = []
    n = signal.shape[-1]

    # TODO this could be used to pregenerate k and m idxs ... but memory pblem?
    # km_idxs = np.triu_indices(k_max - 1)
    # km_idxs = k_max - np.flipud(np.column_stack(km_idxs)) -1
    # km_idxs[:,1] -= 1

    # k = 1, 2, 3, ..., k_max (k_max depends on n)
    for k in range(1, k_max + 1):
        lk = 0

        # m = 1, 2, 3, ..., k
        for m in range(1, k + 1):
            # We pregenerate all idxs
            # i = 1, 2, ..., int((n-m)/k) --> idxs = i
            idxs = np.arange(1, int(np.floor((n - m) / k) + 1), dtype=np.int32)

            # Sum of L_m(k)
            lk += (
                np.sum(
                    np.abs(
                        signal[
                            :, (m + idxs * k) - 1
                        ] - signal[
                            :, (m + k * (idxs - 1)) - 1
                        ],
                    ),
                    axis=-1,
                ) * (n - 1) / (int((n - m) / k) * k)
            ) / k

        l_higuchi.append(np.log(lk / (m + 1)))

    return np.linalg.lstsq(
        list(
            map(
                lambda x: [np.log(1.0 / x), 1],
                np.arange(1, k_max + 1),
            ),
        ),
        l_higuchi,
        rcond=None,
    )[0][0]


# Imported from the old pyEEG lib (modified)
def bin_power(signal: list, band: list, fs: float):
    """Compute power in each frequency bin specified by band from FFT result of
    X. By default, X is a real signal.

    Note:
        A real signal can be synthesized, thus not real.

    Args:
        band:   boundary frequencies (in Hz) of bins.
                They can be unequal bins, e.g.
                [0.5, 4, 7, 12, 30] which are delta, theta, alpha
                and beta respectively.
                You can also use range() function of Python
                to generate equal bins and
                pass the generated list to this function.
                Each element of band is a physical frequency
                and shall not exceed the Nyquist frequency,
                i.e., half of sampling frequency.

        signal: a 1-D real time series.

        fs:     the sampling rate in physical frequency

    Returns:
        A tuple with the spectral power in each frequency bin
        and the spectral power in each frequency bin normalized
        by total power in ALL frequency bins.

    # noqa: DAR101
    """
    amplitudes = abs(np.fft.fft(signal))

    if len(amplitudes.shape) > 1:
        power = np.zeros((len(band) - 1, amplitudes.shape[0]))
    else:
        power = np.zeros(len(band) - 1)

    band = np.array(band) / fs * signal.shape[-1]

    for index, (low_freq, up_freq) in enumerate(zip(band[:-1], band[1:])):
        power[index] = amplitudes[
            :, round(low_freq): round(up_freq)].mean(axis=-1)

    return power, power / power.sum(axis=0)
# No more functions copied from pyeegModif


def find_nearest_index(array, item):
    """Find the index with the closest value compared to the item."""
    return (np.abs(array - item)).argmin()


find_nearest_index == njit()(find_nearest_index)


def welch_band_power(signal: list, fs: float):
    """Compute the band power using the Welch periodogram.

    Args:
        signal: the signal for which to compute the band power.
        fs: the sample rate of the signal.
    Returns:
        A tuple with the signal power and power ratio over different bands.
    """
    # Classical FFT
    yf = np.fft.fft(signal)
    N = signal.shape[-1]
    xf = np.linspace(0.0, fs / 2, N // 2)
    welch_powers = abs(yf[0:N // 2]) ** 2 / (N * fs)

    # Bands power
    # Delta (0.5–4 Hz)
    welch_delta_05hz_4hz_power = np.sum(
        welch_powers[:, find_nearest_index(xf, 0.5):find_nearest_index(xf, 4)],
        axis=-1,
    )

    # Theta (4–8 Hz)
    welch_theta_4hz_8hz_power = np.sum(
        welch_powers[:, find_nearest_index(xf, 4):find_nearest_index(xf, 8)],
        axis=-1,
    )

    # Alpha (8–12 Hz)
    welch_alpha_8hz_12hz_power = np.sum(
        welch_powers[:, find_nearest_index(xf, 8):find_nearest_index(xf, 12)],
        axis=-1,
    )

    # Beta (12–30 Hz)
    welch_beta_12hz_30hz_power = np.sum(
        welch_powers[:, find_nearest_index(xf, 12):find_nearest_index(xf, 30)],
        axis=-1,
    )

    # Gamma (30–100 Hz)
    welch_gamma_30hz_100hz_power = np.sum(
        welch_powers[:, find_nearest_index(xf, 30):find_nearest_index(xf, 100)],
        axis=-1,
    )

    # 2Hz4hz
    welch_epilepsy_2hz_4hz_power = np.sum(
        welch_powers[:, find_nearest_index(xf, 2):find_nearest_index(xf, 4)],
        axis=-1,
    )

    # 1Hz5hz
    welch_epilepsy_1hz_5hz_power = np.sum(
        welch_powers[:, find_nearest_index(xf, 1):find_nearest_index(xf, 5)],
        axis=-1,
    )

    # 0Hz6hz
    welch_epilepsy_0hz_6hz_power = np.sum(
        welch_powers[:, find_nearest_index(xf, 0):find_nearest_index(xf, 6)],
        axis=-1,
    )

    # Total spectrum power
    total_power = np.sum(welch_powers, axis=-1)

    return (
        welch_delta_05hz_4hz_power,
        welch_theta_4hz_8hz_power,
        welch_alpha_8hz_12hz_power,
        welch_beta_12hz_30hz_power,
        welch_gamma_30hz_100hz_power,
        welch_epilepsy_2hz_4hz_power,
        welch_epilepsy_1hz_5hz_power,
        welch_epilepsy_0hz_6hz_power,
        total_power,
        welch_delta_05hz_4hz_power / total_power,
        welch_theta_4hz_8hz_power / total_power,
        welch_alpha_8hz_12hz_power / total_power,
        welch_beta_12hz_30hz_power / total_power,
        welch_gamma_30hz_100hz_power / total_power,
        welch_epilepsy_2hz_4hz_power / total_power,
        welch_epilepsy_1hz_5hz_power / total_power,
        welch_epilepsy_0hz_6hz_power / total_power,
    )


def multitaper_band_power(signal: list, fs: float):
    """Compute the band power using the multitaper method.

    Args:
        signal: the signal for which to compute the band power.
        fs: the sample rate of the signal.
    Returns:
        A tuple with the signal power and power ratio over different bands.
    """
    # Classical FFT
    N = signal.shape[-1]
    xf = np.linspace(0.0, fs / 2, N // 2)

    # The multitapered method
    NW = 2.5
    k = 4
    [tapers, eigen] = spectrum.dpss(N, NW, k)

    # print(signal.shape, signal[0].shape)
    Sk = []
    for sig in signal:
        Sk_complex, weights, _ = spectrum.pmtm(
            sig,
            e=eigen,
            v=tapers,
            NFFT=N,
            show=False,
        )

        sk = np.mean(abs(Sk_complex) ** 2 * np.transpose(weights), axis=0) / fs
        Sk.append(sk)

    Sk = np.vstack(Sk)

    # Bands power
    # Delta (0.5–4 Hz)
    multitaper_delta_05hz_4hz_power = np.sum(
        Sk[:, find_nearest_index(xf, 0.5):find_nearest_index(xf, 4)],
        axis=-1,
    )

    # Theta (4–8 Hz)
    multitaper_theta_4hz_8hz_power = np.sum(
        Sk[:, find_nearest_index(xf, 4):find_nearest_index(xf, 8)],
        axis=-1,
    )

    # Alpha (8–12 Hz)
    multitaper_alpha_8hz_12hz_power = np.sum(
        Sk[:, find_nearest_index(xf, 8):find_nearest_index(xf, 12)],
        axis=-1,
    )

    # Beta (12–30 Hz)
    multitaper_beta_12hz_30hz_power = np.sum(
        Sk[:, find_nearest_index(xf, 12):find_nearest_index(xf, 30)],
        axis=-1,
    )

    # Gamma (30–100 Hz)
    multitaper_gamma_30hz_100hz_power = np.sum(
        Sk[:, find_nearest_index(xf, 30):find_nearest_index(xf, 100)],
        axis=-1,
    )

    # 2Hz4hz
    multitaper_epilepsy_2hz_4hz_power = np.sum(
        Sk[:, find_nearest_index(xf, 2):find_nearest_index(xf, 4)],
        axis=-1,
    )

    # 1Hz5hz
    multitaper_epilepsy_1hz_5hz_power = np.sum(
        Sk[:, find_nearest_index(xf, 1):find_nearest_index(xf, 5)],
        axis=-1,
    )

    # 0Hz6hz
    multitaper_epilepsy_0hz_6hz_power = np.sum(
        Sk[:, find_nearest_index(xf, 0):find_nearest_index(xf, 6)],
        axis=-1,
    )

    # Total spectrum power
    total_power = np.sum(Sk[:, 0:N // 2], axis=-1)

    return (
        multitaper_delta_05hz_4hz_power,
        multitaper_theta_4hz_8hz_power,
        multitaper_alpha_8hz_12hz_power,
        multitaper_beta_12hz_30hz_power,
        multitaper_gamma_30hz_100hz_power,
        multitaper_epilepsy_2hz_4hz_power,
        multitaper_epilepsy_1hz_5hz_power,
        multitaper_epilepsy_0hz_6hz_power,
        total_power,
        multitaper_delta_05hz_4hz_power / total_power,
        multitaper_theta_4hz_8hz_power / total_power,
        multitaper_alpha_8hz_12hz_power / total_power,
        multitaper_beta_12hz_30hz_power / total_power,
        multitaper_gamma_30hz_100hz_power / total_power,
        multitaper_epilepsy_2hz_4hz_power / total_power,
        multitaper_epilepsy_1hz_5hz_power / total_power,
        multitaper_epilepsy_0hz_6hz_power / total_power,
    )


def line_length(
        signal,
        sampling_frequency: float = 1,
):
    """Compute the line length of a signal.

    Args:
        signal: the signal on which to compute the line length.
        sampling_frequency: the sampling frequency of the signal.

    Returns:
        The line length of the signal.
    """
    squared_width = (1 / sampling_frequency) ** 2
    return np.sqrt(
        np.diff(signal) ** 2 + squared_width,
    ).sum(axis=-1)
