#! python3
r"""This script extracts the features of the dataset.

It uses the previously extacted dataset from the HDF5 file.

Author:
    - Vincent Stragier

Logs:
    - 2020/11/04
        - Create this script

C:\Users\vince\AppData\Local\Programs\Python\Python38\Scripts\ipython.exe -i
.\src\dataset_h5_to_feature_h5.py .\dataset.h5
srun --partition debug -n 30 --mem 60G --pty /bin/bash
py src/dataset_h5_to_feature_h5.py dataset.h5 -y --features MIN
"""
import multiprocessing as mp
import multiprocessing.pool
import os
from functools import partial

import h5py
import numpy as np
import tqdm

# Local application imports
import tools.feature_extraction as fe


# https://stackoverflow.com/a/53180921/10949679
class NoDaemonProcess(mp.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)


def get_metadata(filename: str, extract_meta: bool = False):
    """Get the list of recordings stored in the HDF5, and the metadata.

    Args:
        filename: the path to the HDF5
            file with its extension.

    Returns:
        A tuple with (recordings, metadata, metadata_dict) if
        extract_meta is True, otherwise only return the list of
        recordings.
    """
    with h5py.File(
            filename,
            mode='r',
            libver='latest',
            swmr=True) as hdf5:

        recordings = list(hdf5['dataset'].keys())

        if extract_meta:
            metadata_dict =\
                fe.raw_numpy_to_object(
                    np.array(hdf5['metadata_dict']),
                )

            metadata =\
                fe.raw_numpy_to_object(
                    np.array(hdf5['metadata']),
                )

            return metadata_dict, metadata, recordings

        return recordings


def pad(signal, length: int, n_pad: int):
    """Pads the front and the end of the signal for a length
    `length` * `n_pad`, with pad of length `length`.
    Even pad are a reversed array of the signal
    head or tail on a length corresponding to the `length` value.

    Args:
        signal: the signal to pad.
        length: the length of the front and end pads.
        n_pad: number of pad to add to the front and to the end of the signal.

    Returns:
        The padded signal in a Numpy array.
    """
    if len(signal.shape) == 2:
        for _ in range(n_pad):
            signal = np.pad(
                signal,
                (
                    (0, 0),
                    (length, length),
                ),
                'symmetric',
            )

    elif len(signal.shape) == 1:
        for _ in range(n_pad):
            signal = np.pad(
                signal,
                (length, length),
                'symmetric',
            )

    return signal


def process_features(recording_name: str,
                     #  metadata: dict,
                     features: list,
                     targets_features: list,
                     filename_hdf5: str,
                     padding: float,
                     window: float,
                     step: float,
                     n_child_processes: int,
                     overwrite: bool):
    """Extracts the features of each recording and saves them.

    Args:
        recording_name: the name of the recording (filename of the EDF file).
        metadata: the metadata of all the dataset.
        features: a list of the features to extract.
        targets: a list of the features to extract from the targets.
        filename_hdf5: the hdf5 file to use to read the signals from
                       and to save the features and segement related targets.
        padding: the among of second of padding to add.
        window: the width of the sliding window in second.
        step: the length of the step between two consecutive window in second.
        n_child_processes: the number of child processes to use in the pool.

    Returns:
        The recording name or a tupple with the recording name
        and the error message if any.
    """
    attributes = {
        'padding': padding,
        'window': window,
        'step': step,
    }

    # Protect the HDF5 file IO
    with process_features.read_write_lock:
        # Retrieves signals
        with h5py.File(
                filename_hdf5,
                mode='r+',
                libver='latest',
                swmr=True) as hdf5:
            signals = np.array(
                hdf5['dataset/' + recording_name],
            )

            # Extract the montages and the sampling frequencies.
            montages = signals['montage'].transpose()[0]
            fs = set(
                signals['sampling_frequency'].transpose()[0],
            )

            attributes.update(
                {
                    'signal_count': len(montages),
                },
            )

            try:
                assert len(fs) == 1
                fs = fs.pop()

            except AssertionError:
                return (recording_name, 'Error: multiple sampling frequencies'
                                        ' were found, which is not handled'
                                        ' by the padding function.')

            if not overwrite:
                # Check which features have to be updated
                if 'features' in list(hdf5.keys()):
                    if recording_name in list(hdf5['features'].keys()):
                        # Read the attributes of each feature vector
                        # and check with the features and parameters
                        # (windows and step)
                        infile = list(
                            hdf5['features/' + recording_name].keys(),
                        )

                        for feature in infile:
                            key = 'features/' + recording_name + '/' + feature
                            attrs = dict(hdf5[key].attrs)
                            if attrs == attributes:
                                features = [
                                    f for f in features if f != feature
                                ]

                            # Check which features have to be updated.

                if 'targets' in list(hdf5.keys()):
                    if recording_name in list(hdf5['targets'].keys()):
                        # Read the attributes of each targets' features' vector
                        # and check with the features and parameters
                        # (windows and step)
                        infile = list(hdf5['targets/' + recording_name].keys())

                        for target in infile:
                            key = 'targets/' + recording_name + '/' + target
                            attrs = dict(hdf5[key].attrs)

                            if attrs == attributes:
                                targets_features = [
                                    f for f in targets_features
                                    if f != target
                                ]

                            # Check which targets' features have to be updated.

                # Else, we leave this check and
                # begin to compute everything.

    if len(features) > 0 or len(targets_features) > 0 or overwrite:
        # Extract all the feature to a NumPy array with a specific structure.
        n_montages = signals['signal'].shape[0]
        n_samples = signals['signal'].shape[-1]

        if padding:
            padded_signals = pad(
                signal=signals['signal'].reshape(
                    (
                        n_montages,
                        n_samples,
                    ),
                ),
                length=int(fs * step),
                n_pad=int(padding / step),
            )

            padded_targets = pad(
                signal=signals['targets'].reshape(
                    (
                        n_montages,
                        n_samples,
                    ),
                ),
                length=int(fs * step),
                n_pad=int(padding / step),
            )

            dt = signals.dtype
            # Put the dtype of the structure in a list
            dt = [
                tuple([name]) + field[0].subdtype
                if field[0].subdtype is not None else (name, field[0])
                for name, field in dict(dt.fields).items()
            ]

            # Change the size in the dtype
            n_samples = padded_signals.shape[-1]
            dt = [
                tuple([tup[0], tup[1], (n_samples,)])
                if tup[0] == 'signal' or tup[0] == 'targets'
                else tup for tup in dt
            ]

            # Convert the list of dtype to a dtype object
            dt = np.dtype(dt)

            # Put the padded signals and targets in the new object
            signals = np.array(signals, dtype=dt)
            signals['signal'] = padded_signals.reshape(
                (
                    n_montages,
                    1,
                    n_samples,
                ),
            )

            signals['targets'] = padded_targets.reshape(
                (
                    n_montages,
                    1,
                    n_samples,
                ),
            )

        # Segmentation
        (
            segments,
            targets,
            sampling_frequency,
            montages,
        ) = fe.channels_segmentation(
            signals_structure=signals,
            window_in_seconds=window,
            step_in_seconds=step,
        )

        # Preload the extraction function
        partial_extraction_worker = partial(
            fe.extract_features,
            sampling_rate=sampling_frequency,
            features_list=features,
        )

        # Extract the features
        chunksize = max(1, round(n_child_processes / 2))
        features_list = []
        with mp.Pool(n_child_processes) as pool:
            for returned_features in pool.imap(
                    partial_extraction_worker,
                    segments,
                    chunksize=chunksize,
            ):
                features_list.append(returned_features)

        # Preload the features before writing them in the HDF5 file
        dt = np.dtype(
            [
                (
                    'features',
                    'float64',
                    (len(features_list), ),
                ),
                (
                    'montage',
                    'S100',
                ),
            ],
        )

        features_loading_dict = {}
        for feature in features:
            temp_feature = np.vstack(
                [
                    extf[feature] for extf in features_list
                ],
            ).transpose()

            features_loading_dict[feature] = np.vstack(
                [
                    np.array(
                        (
                            temp_feature[index],
                            montage[0].decode(),
                        ),
                        dtype=dt,
                    ) for index, montage in enumerate(montages)
                ],
            )[:, 0]

        # Preload the extraction function
        partial_extraction_worker = partial(
            fe.extract_features,
            sampling_rate=sampling_frequency,
            features_list=targets_features,
        )

        # Extract the targets' features
        targets_features_list = []
        with mp.Pool(n_child_processes) as pool:
            for returned_features in pool.imap(
                    partial_extraction_worker,
                    targets,
                    chunksize=chunksize,
            ):
                targets_features_list.append(returned_features)

        # Preload the targets' features before writing them in the HDF5 file
        dt = np.dtype(
            [
                (
                    'features',
                    'float64',
                    (len(targets_features_list), ),
                ),
                (
                    'montage',
                    'S100',
                ),
            ],
        )

        targets_features_loading_dict = {}
        for feature in targets_features:
            temp_feature = np.vstack(
                [
                    extf[feature] for extf
                    in targets_features_list
                ],
            ).transpose()

            targets_features_loading_dict[feature] = np.vstack(
                [
                    np.array(
                        (
                            temp_feature[index],
                            'ALL',
                        ),
                        dtype=dt,
                    ) if 'EKG' in montage[0].decode() or 'ECG'
                    in montage[0].decode()
                    else np.array(
                        (
                            temp_feature[index],
                            montage[0].decode(),
                        ),
                        dtype=dt,
                    ) for index, montage in enumerate(montages)
                ],
            )[:, 0]

        # Write the extracted features
        # and the targets to the HDF5 file.
        with process_features.read_write_lock:
            with h5py.File(
                    filename_hdf5,
                    mode='r+',
                    libver='latest',
                    swmr=True) as hdf5:

                for feature in features:
                    key = 'features/{0}/{1}'.format(
                        recording_name,
                        feature,
                    )

                    try:
                        del hdf5[key]
                    except Exception:
                        pass

                    hdf5.create_dataset(
                        key,
                        data=features_loading_dict[feature],
                    )

                    hdf5[key].attrs.clear()
                    hdf5[key].attrs.update(attributes)

                for feature in targets_features:
                    key = 'targets/{0}/{1}'.format(
                        recording_name,
                        feature,
                    )

                    try:
                        del hdf5[key]

                    except Exception:
                        pass

                    hdf5.create_dataset(
                        key,
                        data=targets_features_loading_dict[feature],
                    )

                    hdf5[key].attrs.clear()
                    hdf5[key].attrs.update(attributes)

    return recording_name


# Static members of feature process
process_features.read_write_lock = mp.Lock()

if __name__ == '__main__':
    # Create the script arguments parser
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=True)

    parser.add_argument(
        'hdf5',
        type=str,
        help='Path to the dataset (HDF5 file), the features'
             ' will be saved there.',
    )

    parser.add_argument(
        '-f',
        '--features',
        nargs='+',
        type=str,
        help='List of features to extract ("a", "A" or "all",'
             ' "ALL" will lead to extract all the features,'
             ' as well as not providing this parameter).'
             ' Not case sensitive.',
    )

    parser.add_argument(
        '-t',
        '--targets',
        nargs='+',
        type=str,
        help='List of features to extract ("a", "A" or "all",'
             ' "ALL" will lead to extract all the features'
             ' from the targets). Not providing this parameter,'
             ' will lead to extract the MIN, MEAN and MAX of the'
             ' targets.'
             ' Not case sensitive.',
    )

    parser.add_argument(
        '-w',
        '--window',
        type=float,
        help='The width of the windows in second (default=4).',
        default=4,
    )

    parser.add_argument(
        '-s',
        '--step',
        type=float,
        help='The number of seconds whose the window is'
             ' sliding on the signals after each step (default=1).',
        default=1)

    parser.add_argument(
        '-p',
        '--padding',
        type=float,
        help='Number of seconds of signals to repeat before'
             ' and after the signals before computing the features'
             ' (default=3).',
        default=None,
    )

    n_cpu = os.cpu_count()
    parser.add_argument(
        '-j',
        '-J',
        '--job',
        nargs='+',
        type=int,
        help='Maximum number of jobs if only one argument. If two'
             ' arguments, the first is the number of parent processes'
             ' and the second, the number of childs (default={0}).'.format(
                 n_cpu),
        default=[n_cpu],
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-y',
        '--yes',
        help='If set, the program will start directly.',
        action='store_true',
        default=False,
    )

    group.add_argument(
        '-n',
        '--no',
        help='If set, the program will exit directly.',
        action='store_true',
        default=False,
    )

    parser.add_argument(
        '-o',
        '--overwrite',
        help='Recompute all the features, even previously computed ones.',
        action='store_true',
        default=False,
    )

    args = parser.parse_args()

    # Jobs allocation
    n_jobs = args.job
    len_n_jobs = len(n_jobs)

    try:
        assert len_n_jobs > 0 and len_n_jobs < 3
    except AssertionError:
        print('You should provide 0, 1 or 2 arguments to the "--job" option.'
              ' Got {0}: {1}.'.format(len_n_jobs, n_jobs))
        exit(2)

    try:
        for j in n_jobs:
            assert j > 0
    except AssertionError:
        print('You should provide positive, non zero'
              ' value to the "--job" option.'
              ' Got {0}.'.format(j))
        exit(3)

    if len_n_jobs == 1:
        n_cpu = n_jobs[0]
        n_parent_processes = int(max(1, np.floor((n_cpu) ** 0.5)))
        n_child_processes = int(
            max(
                1,
                np.floor((n_cpu - n_parent_processes) / n_parent_processes),
            ),
        )

    elif len_n_jobs == 2:
        n_parent_processes = n_jobs[0]
        n_child_processes = n_jobs[1]

    # Extract HFD5's filename
    filename_hdf5 = args.hdf5

    # Check that those arguments are positives
    # and non zero for 'step' and 'window'.
    step = args.step
    window = args.window
    padding = args.padding

    try:
        assert step > 0
    except AssertionError:
        print('"--step" should be positive, non zero.')
        exit(4)

    try:
        assert window > 0
    except AssertionError:
        print('"--window" should be positive, non zero.')
        exit(5)

    # Since we are using a sliding window, we have to pad the signal
    # in order to increase the sensitivity at the begining
    # and at the end of the signal.
    if padding is None:
        padding = window - step

    try:
        assert padding >= 0
    except AssertionError:
        print('"--padding" should be positive.')
        exit(6)

    # Check that the window's width
    # is divisible by the step length.
    try:
        assert (args.window / args.step) % 1.0 == 0.0
    except AssertionError:
        print('The window width should be divisible'
              r' by the step\'s length. Got window={0} s'
              ' and step={1} s.'.format(args.window, args.step))
        exit(7)

    # Generate or check the list of features.
    features = args.features

    # Take all the features.
    if (features is None or 'a'
            in features or 'A'
            in features or 'all'
            in features or 'ALL'
            in features):

        features = fe.ALL_FEATURES
        # Removes non univariate features.
        features.remove('FIRST_DIFFERENTIAL')

    else:
        features = list(set(features))
        features = [
            f.upper() for f in features if f.upper() in fe.ALL_FEATURES
        ]

    try:
        assert len(features) > 0
    except AssertionError:
        print('No valid features have been provided.'
              '\nThe available features are:\n\t{0}'.format(
                  '\n\t'.join(fe.ALL_FEATURES)))
        exit(8)

    # Generate or check the list of features.
    targets = args.targets

    # If no targets' feature have been provided,
    # extract ('MIN', 'MEAN' and 'MAX')
    if targets is None:
        targets = ['MIN', 'MEAN', 'MAX']

    # Take all the features.
    elif ('a' in targets or 'A'
            in targets or 'all'
            in targets or 'ALL'
            in targets):

        targets = fe.ALL_FEATURES
        # Removes non univariate features.
        targets.remove('FIRST_DIFFERENTIAL')

    else:
        targets = list(set(targets))
        targets = [
            f.upper() for f in targets if f.upper() in fe.ALL_FEATURES
        ]

    try:
        assert len(targets) > 0
    except AssertionError:
        print('No valid targets features have been provided.'
              '\nThe available features are:\n\t{0}'.format(
                  '\n\t'.join(fe.ALL_FEATURES)))
        exit(9)

    # Print the used paramters
    print('\nThe features extraction will be executed'
          ' with the following parmaters:\n')
    print('\tHDF5 file:      {0}'.format(filename_hdf5))
    print('\tWindow width:   {0} s'.format(window))
    print('\tStep length:    {0} s'.format(step))
    print('\tPadding length: {0} s'.format(padding))
    print('\n\tUsed features:')
    print('\t\t', '\n\t\t'.join(features), sep='', end='\n')

    print('\n\tUsed features for the targets:')
    print('\t\t', '\n\t\t'.join(targets), sep='', end='\n\n')

    print('The jobs will be mapped as follow:')
    print('\tNumber of parent(s):           {0}'.format(n_parent_processes))
    print('\tNumber of child(s) per parent: {0}'.format(n_child_processes))
    if args.overwrite:
        print('\nThe process will overwrite all the previous computations.')
    else:
        print('\nThe process will update only the necessary features'
              ' and targets.')
    print()

    if not args.no:
        if not args.yes:
            try:
                while True:
                    conti = input('Do you want to run the program (yes/no)? ')
                    if conti in ['y', 'yes', 'Y', 'YES']:
                        break

                    elif conti in ['n', 'no', 'N', 'NO']:
                        exit()

            except KeyboardInterrupt:
                print(
                    '\nThe user requested the end of the program'
                    ' (KeyboardInterrupt).',
                )

                exit()
    else:
        exit()

    print('Starts the job(s).')

    # Extract the list of recordings from the HDF5 file
    recordings = get_metadata(
        filename=filename_hdf5,
    )

    partial_extract_features = partial(
        process_features,
        features=features,
        targets_features=targets,
        filename_hdf5=filename_hdf5,
        padding=padding,
        window=window,
        step=step,
        n_child_processes=n_child_processes,
        overwrite=args.overwrite,
    )

    # Warning, we are silencing error here.
    # https://numpy.org/doc/stable/reference/generated/numpy.seterr.html
    with np.errstate(divide='warn', invalid='warn'):
        chunksize = max(1, round(n_parent_processes / 2))
        with NestablePool(processes=n_parent_processes) as pool:
            for _ in tqdm.tqdm(
                pool.imap(
                    func=partial_extract_features,
                    iterable=recordings,
                    chunksize=chunksize,
                ),
                total=len(recordings),
            ):
                continue

    print('Done.')
