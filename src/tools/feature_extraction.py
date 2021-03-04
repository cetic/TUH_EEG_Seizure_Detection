"""Module which contains most of the tools to extract
the features and the metadata of the TUHSZ dataset.

Authors:
    Paul Vanabelle,
    Pierre De Handschutter,
    Vincent Stragier

Logs:
    03/11/2020 (Vincent Stragier)
    - makes features related functions
    working on a full segment
        - MIN
        - MAX
        - MEAN
        - VARIANCE
        - KURTOSIS
        - SKEWNESS
        - INTER_QUARTILE_RANGE
        - SPECTRAL_CENTROID
        - SPECTRAL_FLATNESS

    26/10/2020 (Vincent Stragier)
    - move the tools folder from root
    to source
    - add the tusz_analyzer script as
    a module in the tools folder
    - comply to PEP8
    04/10/2020 (Vincent Stragier)
    - add original_hjorth_parameters
    (corrected version of the Hjorth function)
    01/10/2020 (Vincent Stragier)
    - add extraction method for other metadata
    and calibration periods
    30/09/2020 (Vincent Stragier)
    - add extract_metadata_edf
    29/09/2020 (Vincent Stragier)
    - add channels_segmentation
    - remove unused functions
    28/09/2020 (Vincent Stragier)
    - add extract_channels_signal_from_file
    (to directly extract the channels
    according to the .lbl files)
    26/09/2020 (Vincent Stragier)
    - cry, a lot
    - remove some function (used locally
    to parse lines)
    - add the function extract_files_list
    (to extract the list of files in a path
    using an extension filter)
    - ensure that the extract_files_list
    return a sorted list
    25/09/2020 (Vincent Stragier)
    - convert the Python notebook
    to a Python script
    - create this module

# noqa: D205, D400, RST301
"""
# noqa: D205, DAR003
import os
import re

import numpy as np
import pyedflib
from scipy import stats
from skimage.util.shape import view_as_windows
from sklearn.metrics import confusion_matrix

try:
    import tusz_analyzer as ta
    import _feature_extraction as _fe

# Default to relative import
except ImportError:
    import tools.tusz_analyzer as ta
    import tools._feature_extraction as _fe

# Path to the metadata
METADATA_PATH = os.path.join(
    os.path.dirname(__file__),
    'metadata.pickle.xz',
)


TEST_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'test',
    'data',
)


MAX_THREAD = int(1.5 * os.cpu_count())


# Functions
def file_len(filename: str):
    """Return the number of lines in a file.

    Args:
        filename: the filename (with or without its path).

    Returns:
        _lines_count: the number of lines in the file.
    """
    with open(filename) as f:  # Open the file
        for _lines_count, _ in enumerate(f, start=1):
            continue
        return _lines_count


def extract_files_list(path: str, extension_filter: str = 'tse'):
    """Return a list of files without its extension (filter).

    The extension will not appear in the files list.

    Args:
        path: the base path to walk through.
        extension_filter: the extension that will serve
        to filter the files list.

    Returns:
        A list of files without extension (if extension_filter is not '').
    """
    extension_filter = '.{0}'.format(extension_filter)
    files_list = set()

    file_ = [path]

    if extension_filter != '':
        for folder, _, files in os.walk(path):
            for file_ in files:
                if file_.endswith(extension_filter):
                    files_list.add(
                        os.path.join(
                            folder,
                            file_,
                        ).split(extension_filter)[0],
                    )

    else:
        for folder, _, files in os.walk(path):
            for file_ in files:
                files_list.add(
                    os.path.join(
                        folder,
                        file_,
                    ).split(extension_filter)[0],
                )

    return sorted(list(files_list))


def binary_smoothing(x, window_width, threshold):
    """Smoothe a binary vector 'x' to prevent outliers.

    Args:
        x: the vector to smooth.
        window_width: the size of the window to look at
        and decide if it's necessary to change the vector value
        for a given element
        threshold: the minimal proportion of points of a given class
        in the neighborhood of a point needed to change the vector's value.
    Returns:
        A filtered vector.
    """
    x_length = x.shape[-1]
    window_width = min(window_width, x_length)

    vecteur_bis = np.copy(x)
    x_copy = np.copy(x)

    for i in range(x_length):
        count = np.unique(x_copy[min(
            max(i - window_width // 2, 0),
            x_length - window_width,
        ): max(
            window_width,
            min(i + window_width // 2 + 1, x_length),
        )], return_counts=True)

        val = count[1]

        if (np.max(val) / np.sum(val)) > threshold:
            vecteur_bis[i] = count[0][np.argmax(count[1])]

    return vecteur_bis


# Do not forget to update this list when adding
# new features in extract_features().
# Upper case only.
ALL_FEATURES = ['MIN',
                'MEAN',
                'MAX',
                'VARIANCE',
                'KURTOSIS',
                'SKEWNESS',
                'INTER_QUARTILE_RANGE',
                'HIGUCHI_FRACTAL_DIMENSION',
                'FIRST_DIFFERENTIAL',
                'HJORTH_ACTIVITY',
                'HJORTH_MOBILITY',
                'HJORTH_COMPLEXITY',
                'PETROSIAN_FRACTAL_DIMENSION',
                'PAUL_FRACTAL_DIMENSION',
                'LINE_LENGTH',
                'POWER_DELTA_WAVE',
                'POWER_RATIO_DELTA_WAVE',
                'POWER_THETA_WAVE',
                'POWER_RATIO_THETA_WAVE',
                'POWER_ALPHA_WAVE',
                'POWER_RATIO_ALPHA_WAVE',
                'POWER_BETA_WAVE',
                'POWER_RATIO_BETA_WAVE',
                'POWER_GAMMA_WAVE',
                'POWER_RATIO_GAMMA_WAVE',
                'WELCH_DELTA_05HZ_4HZ_POWER',
                'WELCH_THETA_4HZ_8HZ_POWER',
                'WELCH_ALPHA_8HZ_12HZ_POWER',
                'WELCH_BETA_12HZ_30HZ_POWER',
                'WELCH_GAMMA_30HZ_100HZ_POWER',
                'WELCH_EPILEPSY_2HZ_4HZ_POWER',
                'WELCH_EPILEPSY_1HZ_5HZ_POWER',
                'WELCH_EPILEPSY_0HZ_6HZ_POWER',
                'WELCH_TOTAL_POWER',
                'WELCH_DELTA_05HZ_4HZ_POWER_RATIO',
                'WELCH_THETA_4HZ_8HZ_POWER_RATIO',
                'WELCH_ALPHA_8HZ_12HZ_POWER_RATIO',
                'WELCH_BETA_12HZ_30HZ_POWER_RATIO',
                'WELCH_GAMMA_30HZ_100HZ_POWER_RATIO',
                'WELCH_EPILEPSY_2HZ_4HZ_POWER_RATIO',
                'WELCH_EPILEPSY_1HZ_5HZ_POWER_RATIO',
                'WELCH_EPILEPSY_0HZ_6HZ_POWER_RATIO',
                'MULTITAPER_DELTA_05HZ_4HZ_POWER',
                'MULTITAPER_THETA_4HZ_8HZ_POWER',
                'MULTITAPER_ALPHA_8HZ_12HZ_POWER',
                'MULTITAPER_BETA_12HZ_30HZ_POWER',
                'MULTITAPER_GAMMA_30HZ_100HZ_POWER',
                'MULTITAPER_EPILEPSY_2HZ_4HZ_POWER',
                'MULTITAPER_EPILEPSY_1HZ_5HZ_POWER',
                'MULTITAPER_EPILEPSY_0HZ_6HZ_POWER',
                'MULTITAPER_TOTAL_POWER',
                'MULTITAPER_DELTA_05HZ_4HZ_POWER_RATIO',
                'MULTITAPER_THETA_4HZ_8HZ_POWER_RATIO',
                'MULTITAPER_ALPHA_8HZ_12HZ_POWER_RATIO',
                'MULTITAPER_BETA_12HZ_30HZ_POWER_RATIO',
                'MULTITAPER_GAMMA_30HZ_100HZ_POWER_RATIO',
                'MULTITAPER_EPILEPSY_2HZ_4HZ_POWER_RATIO',
                'MULTITAPER_EPILEPSY_1HZ_5HZ_POWER_RATIO',
                'MULTITAPER_EPILEPSY_0HZ_6HZ_POWER_RATIO',
                'SPECTRAL_CENTROID',
                'SPECTRAL_FLATNESS',
                ]


# TODO:
# - add HRV
# - add EMD (IMF)
# - add random variable
# - ...


def extract_features(signal, sampling_rate, features_list: list, **kwargs):
    """Extract the signal features.

    Extract the features to 'features_dict',
    based on 'features_list'.

    Note:
        The list of existing features are stored
        in the `ALL_FEATURES` list.

    Args:
        signal: the signal sample from wich the features will be extracted
        sampling_rate: the sampling frequency (a.k.a. sampling rate)
            of the sample
        features_list: the list of features to extract from the signal
        **kwargs: additionnal parameters
            k_max (default = 6): the Higuchi's fractal dimension
            parameter `k_max`.

            band_values (default = [0.5, 4, 7, 12, 30, 100]):
            list of power bands (here, Delta, Theta, Alpha and Gamma).

            petrosian_method (default = 'differential'):
            the method used to compute the petrosian dimension.

            paul_method (default = 'differential'):
            the method used to compute the petrosian dimension.

    Returns:
        features_dict: a dictionary with the extracted features
    """
    # Only the features in the feature_list will be added.
    features_dict = {}

    # list(set(features_list)): # Avoid duplicate list
    for feature in features_list:
        # ################### Temporal parameters ####################
        if feature == 'MIN':
            features_dict[feature] = signal.min(axis=-1)

        elif feature == 'MEAN':
            features_dict[feature] = signal.mean(axis=-1)

        elif feature == 'MAX':
            features_dict[feature] = signal.max(axis=-1)

        elif feature == 'VARIANCE':
            features_dict[feature] = signal.var(axis=-1)

        elif feature in [
            'FIRST_DIFFERENTIAL',
            'HJORTH_ACTIVITY',
            'HJORTH_MOBILITY',
            'HJORTH_COMPLEXITY',
            'PETROSIAN_FRACTAL_DIMENSION',
            'PAUL_FRACTAL_DIMENSION',
        ]:

            try:  # Avoid to recompute the variable
                signal_first_differential  # noqa: F821

            except NameError:
                # first derivative (vector)
                signal_first_differential = np.diff(signal)

            # Not really a feature
            if feature == 'FIRST_DIFFERENTIAL':
                features_dict[feature] = signal_first_differential

            elif feature in [
                'HJORTH_ACTIVITY',
                'HJORTH_MOBILITY',
                'HJORTH_COMPLEXITY',
            ]:
                try:  # Avoid to recompute the variables
                    signal_hjorth_parameters  # noqa: F821
                except NameError:
                    signal_hjorth_parameters = _fe.original_hjorth_parameters(
                        signal=signal,
                        sampling_frequency=sampling_rate,
                        first_diff=signal_first_differential,
                    )

                if feature == 'HJORTH_ACTIVITY':
                    features_dict[feature] = signal_hjorth_parameters[0]

                elif feature == 'HJORTH_MOBILITY':
                    features_dict[feature] = signal_hjorth_parameters[1]

                else:  # 'HJORTH_COMPLEXITY'
                    features_dict[feature] = signal_hjorth_parameters[2]

            elif feature == 'PETROSIAN_FRACTAL_DIMENSION':
                petrosian_method = kwargs.get(
                    'petrosian_method',
                    'differential',
                )

                features_dict[feature] = _fe.petrosian_fractal_dimension(
                    signal=signal,
                    first_diff=signal_first_differential,
                    method=petrosian_method,
                )

            elif feature == 'PAUL_FRACTAL_DIMENSION':
                paul_method = kwargs.get(
                    'paul_method',
                    'differential',
                )

                features_dict[feature] = _fe.paul_fractal_dimension(
                    signal=signal,
                    first_diff=signal_first_differential,
                    method=paul_method,
                )

        elif feature == 'KURTOSIS':
            features_dict[feature] = stats.kurtosis(signal, axis=-1)

        elif feature == 'SKEWNESS':
            features_dict[feature] = stats.skew(signal, axis=-1)

        elif feature == 'INTER_QUARTILE_RANGE':
            features_dict[feature] = stats.iqr(signal, axis=-1)

        elif feature == 'HIGUCHI_FRACTAL_DIMENSION':
            k_max = int(kwargs.get('k_max', 15))
            # Compute Hjorth Fractal Dimension of a time series X,
            # where k_max is an HFD parameter
            # <-- to check (from pyEEG)
            features_dict[feature] = _fe.higuchi_fractal_dimension(
                signal,
                k_max=k_max,
            )

        elif feature == 'LINE_LENGTH':
            features_dict[feature] = _fe.line_length(
                signal,
                sampling_frequency=sampling_rate,
            )

        # ################### Frequencial features ####################
        elif feature in [
            'POWER_DELTA_WAVE',
            'POWER_RATIO_DELTA_WAVE',
            'POWER_THETA_WAVE',
            'POWER_RATIO_THETA_WAVE',
            'POWER_ALPHA_WAVE',
            'POWER_RATIO_ALPHA_WAVE',
            'POWER_BETA_WAVE',
            'POWER_RATIO_BETA_WAVE',
            'POWER_GAMMA_WAVE',
            'POWER_RATIO_GAMMA_WAVE',
        ]:

            try:  # Avoid to recompute the values
                signal_power        # noqa: F821
                signal_power_ratio  # noqa: F821

            except NameError:
                band_values = kwargs.get(
                    'band_values',
                    [0.5, 4, 7, 12, 30, 100],
                )

                # <-- to check (from pyEEG)
                signal_power, signal_power_ratio = _fe.bin_power(
                    signal,
                    band=band_values,
                    fs=sampling_rate,
                )

            if feature == 'POWER_DELTA_WAVE':
                features_dict[feature] = signal_power[0]
            elif feature == 'POWER_RATIO_DELTA_WAVE':
                features_dict[feature] = signal_power_ratio[0]

            elif feature == 'POWER_THETA_WAVE':
                features_dict[feature] = signal_power[1]
            elif feature == 'POWER_RATIO_THETA_WAVE':
                features_dict[feature] = signal_power_ratio[1]

            elif feature == 'POWER_ALPHA_WAVE':
                features_dict[feature] = signal_power[2]
            elif feature == 'POWER_RATIO_ALPHA_WAVE':
                features_dict[feature] = signal_power_ratio[2]

            elif feature == 'POWER_BETA_WAVE':
                features_dict[feature] = signal_power[3]
            elif feature == 'POWER_RATIO_BETA_WAVE':
                features_dict[feature] = signal_power_ratio[3]

            elif feature == 'POWER_GAMMA_WAVE':
                features_dict[feature] = signal_power[4]
            else:  # elif feature == 'POWER_RATIO_GAMMA_WAVE':
                features_dict[feature] = signal_power_ratio[4]

        elif feature in (
            'WELCH_DELTA_05HZ_4HZ_POWER',
            'WELCH_THETA_4HZ_8HZ_POWER',
            'WELCH_ALPHA_8HZ_12HZ_POWER',
            'WELCH_BETA_12HZ_30HZ_POWER',
            'WELCH_GAMMA_30HZ_100HZ_POWER',
            'WELCH_EPILEPSY_2HZ_4HZ_POWER',
            'WELCH_EPILEPSY_1HZ_5HZ_POWER',
            'WELCH_EPILEPSY_0HZ_6HZ_POWER',
            'WELCH_TOTAL_POWER',
            'WELCH_DELTA_05HZ_4HZ_POWER_RATIO',
            'WELCH_THETA_4HZ_8HZ_POWER_RATIO',
            'WELCH_ALPHA_8HZ_12HZ_POWER_RATIO',
            'WELCH_BETA_12HZ_30HZ_POWER_RATIO',
            'WELCH_GAMMA_30HZ_100HZ_POWER_RATIO',
            'WELCH_EPILEPSY_2HZ_4HZ_POWER_RATIO',
            'WELCH_EPILEPSY_1HZ_5HZ_POWER_RATIO',
            'WELCH_EPILEPSY_0HZ_6HZ_POWER_RATIO',
        ):
            try:  # Avoid to recompute the values
                welch_powers  # noqa: F821

            except NameError:
                welch_powers = _fe.welch_band_power(signal, sampling_rate)

            if feature == 'WELCH_DELTA_05HZ_4HZ_POWER':
                features_dict[feature] = welch_powers[0]

            elif feature == 'WELCH_THETA_4HZ_8HZ_POWER':
                features_dict[feature] = welch_powers[1]

            elif feature == 'WELCH_ALPHA_8HZ_12HZ_POWER':
                features_dict[feature] = welch_powers[2]

            elif feature == 'WELCH_BETA_12HZ_30HZ_POWER':
                features_dict[feature] = welch_powers[3]

            elif feature == 'WELCH_GAMMA_30HZ_100HZ_POWER':
                features_dict[feature] = welch_powers[4]

            elif feature == 'WELCH_EPILEPSY_2HZ_4HZ_POWER':
                features_dict[feature] = welch_powers[5]

            elif feature == 'WELCH_EPILEPSY_1HZ_5HZ_POWER':
                features_dict[feature] = welch_powers[6]

            elif feature == 'WELCH_EPILEPSY_0HZ_6HZ_POWER':
                features_dict[feature] = welch_powers[7]

            elif feature == 'WELCH_TOTAL_POWER':
                features_dict[feature] = welch_powers[8]

            elif feature == 'WELCH_DELTA_05HZ_4HZ_POWER_RATIO':
                features_dict[feature] = welch_powers[9]

            elif feature == 'WELCH_THETA_4HZ_8HZ_POWER_RATIO':
                features_dict[feature] = welch_powers[10]

            elif feature == 'WELCH_ALPHA_8HZ_12HZ_POWER_RATIO':
                features_dict[feature] = welch_powers[11]

            elif feature == 'WELCH_BETA_12HZ_30HZ_POWER_RATIO':
                features_dict[feature] = welch_powers[12]

            elif feature == 'WELCH_GAMMA_30HZ_100HZ_POWER_RATIO':
                features_dict[feature] = welch_powers[13]

            elif feature == 'WELCH_EPILEPSY_2HZ_4HZ_POWER_RATIO':
                features_dict[feature] = welch_powers[14]

            elif feature == 'WELCH_EPILEPSY_1HZ_5HZ_POWER_RATIO':
                features_dict[feature] = welch_powers[15]

            elif feature == 'WELCH_EPILEPSY_0HZ_6HZ_POWER_RATIO':
                features_dict[feature] = welch_powers[16]

        elif feature in (
            'MULTITAPER_DELTA_05HZ_4HZ_POWER',
            'MULTITAPER_THETA_4HZ_8HZ_POWER',
            'MULTITAPER_ALPHA_8HZ_12HZ_POWER',
            'MULTITAPER_BETA_12HZ_30HZ_POWER',
            'MULTITAPER_GAMMA_30HZ_100HZ_POWER',
            'MULTITAPER_EPILEPSY_2HZ_4HZ_POWER',
            'MULTITAPER_EPILEPSY_1HZ_5HZ_POWER',
            'MULTITAPER_EPILEPSY_0HZ_6HZ_POWER',
            'MULTITAPER_TOTAL_POWER',
            'MULTITAPER_DELTA_05HZ_4HZ_POWER_RATIO',
            'MULTITAPER_THETA_4HZ_8HZ_POWER_RATIO',
            'MULTITAPER_ALPHA_8HZ_12HZ_POWER_RATIO',
            'MULTITAPER_BETA_12HZ_30HZ_POWER_RATIO',
            'MULTITAPER_GAMMA_30HZ_100HZ_POWER_RATIO',
            'MULTITAPER_EPILEPSY_2HZ_4HZ_POWER_RATIO',
            'MULTITAPER_EPILEPSY_1HZ_5HZ_POWER_RATIO',
            'MULTITAPER_EPILEPSY_0HZ_6HZ_POWER_RATIO',
        ):
            try:  # Avoid to recompute the values
                multitaper_powers  # noqa: F821

            except NameError:
                multitaper_powers = _fe.multitaper_band_power(
                    signal,
                    sampling_rate,
                )

            if feature == 'MULTITAPER_DELTA_05HZ_4HZ_POWER':
                features_dict[feature] = multitaper_powers[0]

            elif feature == 'MULTITAPER_THETA_4HZ_8HZ_POWER':
                features_dict[feature] = multitaper_powers[1]

            elif feature == 'MULTITAPER_ALPHA_8HZ_12HZ_POWER':
                features_dict[feature] = multitaper_powers[2]

            elif feature == 'MULTITAPER_BETA_12HZ_30HZ_POWER':
                features_dict[feature] = multitaper_powers[3]

            elif feature == 'MULTITAPER_GAMMA_30HZ_100HZ_POWER':
                features_dict[feature] = multitaper_powers[4]

            elif feature == 'MULTITAPER_EPILEPSY_2HZ_4HZ_POWER':
                features_dict[feature] = multitaper_powers[5]

            elif feature == 'MULTITAPER_EPILEPSY_1HZ_5HZ_POWER':
                features_dict[feature] = multitaper_powers[6]

            elif feature == 'MULTITAPER_EPILEPSY_0HZ_6HZ_POWER':
                features_dict[feature] = multitaper_powers[7]

            elif feature == 'MULTITAPER_TOTAL_POWER':
                features_dict[feature] = multitaper_powers[8]

            elif feature == 'MULTITAPER_DELTA_05HZ_4HZ_POWER_RATIO':
                features_dict[feature] = multitaper_powers[9]

            elif feature == 'MULTITAPER_THETA_4HZ_8HZ_POWER_RATIO':
                features_dict[feature] = multitaper_powers[10]

            elif feature == 'MULTITAPER_ALPHA_8HZ_12HZ_POWER_RATIO':
                features_dict[feature] = multitaper_powers[11]

            elif feature == 'MULTITAPER_BETA_12HZ_30HZ_POWER_RATIO':
                features_dict[feature] = multitaper_powers[12]

            elif feature == 'MULTITAPER_GAMMA_30HZ_100HZ_POWER_RATIO':
                features_dict[feature] = multitaper_powers[13]

            elif feature == 'MULTITAPER_EPILEPSY_2HZ_4HZ_POWER_RATIO':
                features_dict[feature] = multitaper_powers[14]

            elif feature == 'MULTITAPER_EPILEPSY_1HZ_5HZ_POWER_RATIO':
                features_dict[feature] = multitaper_powers[15]

            elif feature == 'MULTITAPER_EPILEPSY_0HZ_6HZ_POWER_RATIO':
                features_dict[feature] = multitaper_powers[16]

        elif feature == 'SPECTRAL_CENTROID':
            features_dict[feature] = _fe.spectral_centroid(signal, sampling_rate)

        elif feature == 'SPECTRAL_FLATNESS':
            features_dict[feature] = _fe.spectral_flatness(signal)

        else:
            print('Unknown feature: {0}'.format(feature))

    return features_dict


def filters_dataset_files(filepath: str):
    """Filter function allowing to exclude uncomplete recordings.

    Args:
        filepath: the path to the filename of the recording
        without extension

    Returns:
        True if all the ``filepath``.[edf|lbl|lbl_bi|tse|tse_bi],
        False otherwise
    """
    edf = os.path.exists(filepath + '.edf')
    lbl = os.path.exists(filepath + '.lbl')
    lbl_bi = os.path.exists(filepath + '.lbl_bi')
    tse = os.path.exists(filepath + '.tse')
    tse_bi = os.path.exists(filepath + '.tse_bi')

    if not (edf and lbl and lbl_bi and tse and tse_bi):
        msg = 'One or more files missing for the recording: "{0}"'.format(
            filepath,
        )

        print((len(msg) + 2) * '#')
        print('', msg)
        print('\t".edf"    exists:', 'True' if edf else 'False')
        print('\t".lbl"    exists:', 'True' if lbl else 'False')
        print('\t".lbl_bi" exists:', 'True' if lbl_bi else 'False')
        print('\t".tse"    exists:', 'True' if tse else 'False')
        print('\t".tse_bi" exists:', 'True' if tse_bi else 'False')
        print('Removed from the list.')
        print((len(msg) + 2) * '#')
        return False
    return True


def extract_channels_signal_from_file(
    filepath: str,
    recordings_metadata_list: list,
):
    """Extract the channels data from the .edf and .lbl files
    to compose the signals of each channel following the specified montage

    Args:
        filepath: the path to the files (without extension), .lbl
        and .edf file will be used
        recordings_metadata_list: the list of the recordings metadata

    Returns:
        Vertically stacked numpy structures.
    """
    file_index = ta.get_index(
        filepath=os.path.basename(filepath),
        list_of_recording_metadata=recordings_metadata_list,
    )

    recording_metadata = recordings_metadata_list[file_index]

    # Extract calibration stop time if any
    # The signal will be trimed accordingly
    try:
        start_after_calibration = recording_metadata[
            'calibration'
        ]['stop']

    except Exception:
        start_after_calibration = 0

    # Determine if an EKG or ECG channel is present (from the metadata)
    ekg_present = [
        {
            'label': header['label'],
            'index': index,
            'sample_rate': header['sample_rate'],
        } for index, header in enumerate(recording_metadata['headers'])
        if 'EKG' in header['label'] or 'ECG' in header['label']
    ]

    with pyedflib.EdfReader(filepath + '.edf') as edf_file:
        # Link the EEG channel number to its label
        eeg_channel_by_label = {}
        eeg_labels = edf_file.getSignalLabels()
        for i, signal_label in enumerate(eeg_labels):
            if 'EEG' in signal_label:
                eeg_channel_by_label[signal_label] = i

        signal_1 = []  # np.array([], dtype=np.float32)
        signal_2 = []  # np.array([], dtype=np.float32)
        sampling_frequency = float()

        montages = recording_metadata['annotations_lbl']['montages']

        # Extract the annotations from the metadata
        events = ta.labels_to_events(recording_metadata)
        events_per_montages = {
            montage: [{
                'start': event['start'],
                'stop': event['stop'],
                'event': event['event']}
                for event in events
                if event['montage'] == montage]
            for montage in montages}

        # Build the signal on each differential montage
        for montage in montages:
            labels = montage.split('-')
            events = events_per_montages[montage]

            for eeg_label in eeg_channel_by_label.keys():
                if labels[0] in eeg_label:
                    signal_1 = np.float32(edf_file.readSignal(
                        eeg_channel_by_label[eeg_label]),
                    )

                    sampling_frequency = edf_file.samplefrequency(
                        eeg_channel_by_label[eeg_label],
                    )

                elif labels[1] in eeg_label:
                    signal_2 = np.float32(edf_file.readSignal(
                        eeg_channel_by_label[eeg_label]),
                    )

            # Build the targets vector
            targets = [[
                round(event['start'] * sampling_frequency),
                round(event['stop'] * sampling_frequency),
                True if event['event'] in ta.EPILEPTIC_SEIZURE_LABELS
                else False]
                for event in events]

            targets = np.concatenate([
                np.ones(target[1] - target[0])
                if target[2] else np.zeros(target[1] - target[0])
                for target in targets])

            # Compute the signal per channel according to the montage
            try:
                channels.append(np.array(           # noqa: F821
                    ((signal_1 - signal_2)[scope],  # noqa: F821
                     targets[scope],                # noqa: F821
                     sampling_frequency,
                     montage),
                    dtype=dt),                      # noqa: F821
                )

            except Exception:
                # https://jakevdp.github.io/PythonDataScienceHandbook/02.09-structured-data-numpy.html
                # dt = [('signal', 'float32', (len(signal_1))),
                # ('sampling_frequency', np.int16),
                # ('montage', np.str)]
                scope = slice(round(
                    sampling_frequency * start_after_calibration), None)

                dt = [('signal', 'float32', (len(signal_1[scope]))),
                      ('targets', 'uint8', (len(signal_1[scope]))),
                      ('sampling_frequency', np.int16),
                      ('montage', 'S100')]

                channels = [np.array(
                    ((signal_1 - signal_2)[scope],
                     targets[scope],
                     sampling_frequency,
                     montage),
                    dtype=dt),
                ]

        # Add EKG/ECG and global targets vector
        targets = np.vstack(channels)['targets']
        targets = targets.reshape(
            targets.shape[0],
            targets.shape[2])
        targets = targets.max(axis=0)

        # Add EKG/ECG signals
        for ekg_signal in ekg_present:
            if round(sampling_frequency) == ekg_signal['sample_rate']:
                channels.append(np.array((
                    np.array(edf_file.readSignal(
                        ekg_signal['index']), dtype='float32')[scope],
                    targets,
                    sampling_frequency,
                    ekg_signal['label']),
                    dtype=dt),
                )

    return np.vstack(channels)


def channels_segmentation(signals_structure,
                          window_in_seconds: float = 3,
                          step_in_seconds: float = 1):
    """Segment the EEG signals using a sliding window
    moving according to a fixed step.

    Args:
        signals_structure: a data structure created
            in the 'extract_channels_signal_from_file'
        window_in_seconds: width of the sliding window in seconds
            (the sampling rate is extracted from the signal structure)
        step_in_seconds: length of the step in seconds
            (the sampling rate is extracted from the signal structure)

    Returns:
        A tuple that contains an array with the sliding windows
        (signals and targets),
        the sampling rate and the vertical list which contains
        the montage of each channel.

    Note:
        Not optimisable with Numba
        ("The use of yield in a closure is unsupported.")
    """
    sampling_frequency = signals_structure['sampling_frequency'][0][0]
    # Assemption: sampling_frequency is constant
    step_in_samples = int(step_in_seconds * sampling_frequency)
    # Assemption: sampling_frequency is constant
    window_in_samples = int(window_in_seconds * sampling_frequency)
    number_of_channels = signals_structure['signal'].shape[0]

    # https://github.com/scikit-image/scikit-image/blob/master/skimage/util/shape.py
    return (
        view_as_windows(
            signals_structure['signal'].reshape(
                number_of_channels,
                signals_structure['signal'].shape[-1]),
            (number_of_channels, window_in_samples),
            step_in_samples,
        )[0],
        view_as_windows(
            signals_structure['targets'].reshape(
                number_of_channels,
                signals_structure['targets'].shape[-1]),
            (number_of_channels, window_in_samples),
            step_in_samples,
        )[0],
        sampling_frequency,
        signals_structure['montage'])

# ################### #
# Metadata extraction #
# ################### #


def extract_metadata_edf(filepath: str):
    r"""Extract most of the metadata of an EDF file using pyedflib.

    Args:
        filepath: the path to the EDF file without the .edf extension.

    Returns:
        Return a dictionary with the meta data:
            {   'admincode': '',
                'birthdate': '',
                'duration [s]': 309,
                'equipment': '',
                'filepath': 'd:/Vincent/
                                Documents/
                                GitHub/
                                Epilepsy_Seizure_Detection/
                                src\\test\\data\\00000258\\
                                s003_2003_07_22\\
                                00000258_s003_t005',

                'gender': '',
                'headers': [   {   'digital_max': 32767,
                                'digital_min': -32767,
                                'dimension': 'uV',
                                'label': 'EEG FP1-REF',
                                'physical_max': 29483.12,
                                'physical_min': -29483.1,
                                'prefilter': 'HP:-1.000 Hz LP:-2.0 Hz N:0.0',
                                'sample_rate': 400,
                                'transducer': 'EEG'},...
                            ],
                'patient_additional': '',
                'patientcode': '',
                'patientname': '',
                'recording_additional': '',
                'startdate': '2003-08-12T00:00:00',
                'technician': ''}
    """
    try:  # Open file
        metadata = {}
        edf_file = pyedflib.EdfReader(filepath + '.edf')

    except OSError as err:  # If fail, return an empty dict.
        print('\n\t OS error: {0}'.format(err))
        return metadata

    else:
        """ For the file
        "technician": self.getTechnician(),
        "recording_additional": self.getRecordingAdditional(),
        "patientname": self.getPatientName(),
        "patient_additional": self.getPatientAdditional(),
        "patientcode": self.getPatientCode(),
        "equipment": self.getEquipment(),
        "admincode": self.getAdmincode(),
        "gender": self.getGender(),
        "startdate": self.getStartdatetime(),
        "birthdate": self.getBirthdate()
        """
        metadata = edf_file.getHeader()

        """ For each channel
        'label': self.getLabel(chn),
        'dimension': self.getphysicalDimension(chn),
        'sample_rate': self.getSampleFrequency(chn),
        'physical_max':self.getPhysicalMaximum(chn),
        'physical_min': self.getPhysicalMinimum(chn),
        'digital_max': self.getDigitalMaximum(chn),
        'digital_min': self.getDigitalMinimum(chn),
        'prefilter':self.getPrefilter(chn),
        'transducer': self.getTransducer(chn)
        """
        metadata.update({'headers': edf_file.getSignalHeaders()})
        metadata.update({'duration [s]': edf_file.getFileDuration()})
        # Convert datetime object to a string (ISO8601)
        # https://stackoverflow.com/q/34044820/10949679
        metadata['startdate'] = metadata['startdate'].strftime(
            '%Y-%m-%dT%H:%M:%S%z',
        )

        metadata.update({'filepath': filepath})
        edf_file._close()
        del edf_file

    return metadata


def extract_calibration_periods(filename: str):
    """Extract the start time and stop time of the calibration
    for the concerned files from the .xlsx file
    (here: '[...]/_DOCS/seizures_v36r.xlsx')

    Args:
        filename: a string containing the full path of the file
        with its extension

    Returns:
        calibration: a dictionary for which the keys are the filename
        (without path nor extension) and that contains a dictonnary
        for which the keys are 'start' and 'stop' and store respectively
        the start time and the stop time in seconds.

    Usage:
        calibration['00004671_s008_t000']['start']
    """
    import pandas  # Will be imported only once

    # Open the Excel file
    xl = pandas.ExcelFile(filename)
    # Extract the data from the 'Calibration' sheet
    data = xl.parse(xl.sheet_names[3])
    # Retrieves the columns name
    col = data.columns

    # Extract the information from the sheet
    filespaths = list(
        data[col[0]][1:].dropna(),
    ) + list(
        data[col[4]][1:].dropna(),
    ) + list(
        data[col[8]][1:].dropna(),
    )

    starts = list(
        data[col[1]][2:].dropna(),
    ) + list(
        data[col[5]][2:].dropna(),
    ) + list(
        data[col[9]][2:].dropna(),
    )

    stops = list(
        data[col[2]][2:].dropna(),
    ) + list(
        data[col[6]][2:].dropna(),
    ) + list(
        data[col[10]][2:].dropna(),
    )

    # Build a dictionary with the previously extracted informations
    calibration = {}
    for i, filepath in enumerate(filespaths):
        calibration[
            os.path.basename(
                filepath[filepath.find('.') + 1:filepath.rfind('.')],
            )
        ] = {'start': starts[i], 'stop': stops[i]}

    return calibration


def extract_annotations_lbl(filename: str):
    """Extract the EEG montages, level(s) information,
    labels from the '.lbl' and '.lbl_bi'

    Args:
        filename: a string containing the full path of the file
        with its extension.

    Returns:
        A dictionary: {'version': version,
                       'montages': montages,
                       'number_of_levels': number_of_levels,
                       'level': level,
                       'symbols': symbols,
                       'labels': labels}

        where:
            - ``version`` is the .lbl/.lbl_bi version
              (which describe the structure of the file).
            - ``montages`` is the list of each EEG channels montage.
            - ``number_of_levels`` is the number of levels
              used to annotate the .edf file.
            - ``level`` is a list that contains the number
              of sublevels per level.
            - ``symbols`` is a list which contains for each level
              a list of symbols used for the annotations.
            - ``labels`` is a list of label struture which contains
              the following informations, where:
                - ``level`` is the annotation level used
                  (links the symbols with the probabilities).
                - ``sublevel`` is the sublevel used.
                - ``start`` is the starting time of the event in seconfs.
                - ``stop`` is the stoping time of the event in seconds.
                - ``montage`` is the observed montage.
                - ``probabilities`` is a list of probabilities that the events
                  corresponding to the symbols of the level are occuring.
    """
    with open(filename) as f:
        montages = []
        level = [None]
        symbols = [None]
        labels = []
        version = ''

        for line in f:
            if 'montage' in line:  # Extract the montages
                montages.append(
                    line[line.find(',') + 1: line.find(':')].replace(' ', ''),
                )

            elif 'number_of_levels' in line:
                # Extract the number of levels in the annotation
                number_of_levels = int(line[line.rfind('=') + 1:])
                level = [None] * number_of_levels
                symbols = [None] * number_of_levels

            elif 'level[' in line:
                # Extract the number of sublevels by level
                level[int(line[line.find('[') + 1: line.find(']')])
                      ] = int(line[line.rfind('=') + 1:])

            elif 'symbols[' in line:
                # Extract the symbols used by level
                # (sublevels inheritate of their parent symbols)
                symbols[int(
                    line[line.find('[') + 1: line.find(']')],
                )] = re.findall(
                    "[']([a-zA-Z]+|[(][a-zA-Z]+[)])[']",
                    line[line.rfind('{') + 1: line.rfind('}')],
                )

            elif 'label' in line:  # Extract the annotations
                temp = line[line.rfind('{') + 1: line.rfind('}')]
                level_sublevel_start_stop_montage = temp[
                    : temp.find('[')
                ].replace(' ', '').strip(',').split(',')

                probabilities = temp[
                    temp.rfind('[') + 1: temp.rfind(']')
                ].replace(' ', '').split(',')

                labels.append(
                    {
                        'level': int(
                            level_sublevel_start_stop_montage[0],
                        ),
                        'sublevel': int(
                            level_sublevel_start_stop_montage[1],
                        ),
                        'start': float(
                            level_sublevel_start_stop_montage[2],
                        ),
                        'stop': float(
                            level_sublevel_start_stop_montage[3],
                        ),
                        'montage': int(
                            level_sublevel_start_stop_montage[4],
                        ),
                        'probabilities': [
                            float(probability) for probability in probabilities
                        ],
                    },
                )

            elif 'version' in line:
                version = line[line.rfind('=') + 1:].replace(' ', '')

        return {'version': version,
                'montages': montages,
                'number_of_levels': number_of_levels,
                'level': level,
                'symbols': symbols,
                'labels': labels,
                }


def extract_annotations_tse(filename: str):
    """Extract the EEG annotations
    from the .tse files or .tse_bi files.

    Args:
        filename: a string containing
        the full path of the file
        with its extension.

    Returns:
        annotations: is a list of dictionaries,
        in which are stored the start,
        the end (stop), the event
        and its probability.
    """
    with open(filename) as f:
        annotations = []
        for line in f:  # Skip the 'version' line
            if 'version' in line:
                break

        for line in f:  # Extract the annotations
            start_stop_event_probability = line.split()
            if len(start_stop_event_probability) == 4:
                annotations.append(
                    {
                        'start': float(start_stop_event_probability[0]),
                        'stop': float(start_stop_event_probability[1]),
                        'event': start_stop_event_probability[2],
                        'probability': float(start_stop_event_probability[3]),
                    },
                )

    return annotations


def extract_metadata_all(filepath: str, calibration=None):
    """Extract all the metadata except
    the calibration period (which can
    be added as a parameter or after
    hand in the returned dictionary)

    Args:
        filepath: a string containing the full path
            of the file without its extension.
        calibration: None or a list with the start
            and end time of the calibration.

    Returns:
        metadata: a dictionary with
        all the extracted metadata.

    # noqa: D205, D400
    """
    metadata = extract_metadata_edf(filepath)
    metadata.update(
        {
            'annotations_tse': extract_annotations_tse(filepath + '.tse'),
        },
    )

    metadata.update(
        {
            'annotations_lbl': extract_annotations_lbl(filepath + '.lbl'),
        },
    )

    metadata.update({'calibration': calibration})

    return metadata


def object_to_raw_numpy(data: dict):
    """Convert a serialisable Python object to a numpy array.

    Args:
        data: any serialisable Python object.

    Returns:
        Return a numpy array of bytes.
    """
    import lzma
    import json

    return np.array(lzma.compress(json.dumps(data).encode('utf-16')))


def raw_numpy_to_object(numpy_bytes_array):
    """Convert numpy array to a Python object.

    Args:
        numpy_bytes_array: a numpy array of bytes.

    Returns:
        Return a Python object.
    """
    import lzma
    import json

    return json.loads(
        lzma.decompress(numpy_bytes_array.tobytes()).decode('utf-16'),
    )

# ######################### #
# Metadata extraction (end) #
# ######################### #


# https://stackoverflow.com/questions/31324218/
# scikit-learn-how-to-obtain-true-positive
# -true-negative-false-positive-and-fal
def compute_metrics(y_pred, y_true):
    """Compute the metrics of interest.

    Based on https://en.wikipedia.org/wiki/Receiver_operating_characteristic.

    Args:
        y_pred: an array like binary predictions.
        y_true: an array like binary true values.

    Returns:
        A dictionary containing the metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    all_conditions = tn + fp + fn + tp

    # Sensitivity
    tpr = tp / (tp + fn)

    # Specificity
    tnr = tn / (tn + fp)

    # Precision
    ppv = tp / (tp + fp)

    # Negative predictive value
    npv = tn / (tn + fn)

    # Miss rate
    fnr = fn / (fn + tp)

    # Fall out
    fpr = fp / (fp + tn)

    # False discovery rate
    fdr = fp / (fp + tp)

    # False omission rate
    false_omission_rate = fn / (fn + tn)

    # Prevalence threshold
    pt = np.sqrt(tpr * (1 - tnr) + tnr - 1) / (tpr + tnr - 1)

    # Thread score
    ts = tp / (tp + fn + fp)

    # Accuracy
    acc = (tp + tn) / all_conditions

    # Balanced accuracy
    ba = (tpr + tnr) / 2

    # F1 score
    # (harmonic mean of precision and sensitivity)
    f1 = 2 * tp / (2 * tp + fp + fn)

    # Matthews correlation coefficient
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt(
        (tp + fp) * (tp + fn) * (tp + fp) * (tn + fn))

    # Fowlkes–Mallows index
    fm = np.sqrt(ppv * tpr)

    # Bookmaker informedness
    bm = tpr + tnr - 1

    # Markedness
    mk = ppv + npv - 1

    return {'condition_positive': tp + fn,
            'condition_negative': tn + fp,
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'sensitivity': tpr,
            'specificity': tnr,
            'precision': ppv,
            'negative_predictive_value': npv,
            'miss_rate': fnr,
            'fall_out': fpr,
            'false_discovery_rate': fdr,
            'false_omission_rate': false_omission_rate,
            'prevalence_threshold': pt,
            'thread_score': ts,
            'accuracy': acc,
            'balanced_accuracy': ba,
            'f1_score': f1,
            'matthews_correlation_coefficient': mcc,
            'fowlkes_mallows_index': fm,
            'bookmaker informedness': bm,
            'markedness': mk,
            }


def generate_line(filepath: str, filename: str, metrics: dict):
    """Convert a metrics dictionary, recording name and location to a line.

    Make them compatible with a Pandas dataframe.

    Args:
        filepath: the path to the recording.
        filename: the name of the recording.
        metrics: a dictionary of the metrics.

    Returns:
        A list with the filepath, filename and metrics.
    """
    line = [filepath, filename]
    metrics_names = [
        'condition_positive',
        'condition_negative',
        'true_positive',
        'true_negative',
        'false_positive',
        'false_negative',
        'sensitivity',
        'specificity',
        'precision',
        'accuracy',
        'f1_score',
        'negative_predictive_value',
        'miss_rate',
        'fall_out',
        'false_discovery_rate',
        'false_omission_rate',
        'prevalence_threshold',
        'thread_score',
        'balanced_accuracy',
        'matthews_correlation_coefficient',
        'fowlkes_mallows_index',
        'bookmaker informedness',
        'markedness']

    line.extend(
        (metrics[metrics_name] for metrics_name in metrics_names),
    )

    return line
