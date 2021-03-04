"""Extract the metadata of the dataset.

Authors: Vincent Stragier

Description:
    - List all the files of the dataset to find the EEG recording.
    - Extract all the metadata of the dataset and store them
    in a pickle and a JSON file.
    - Provides functions to load the extracted metadata
    in a dictionary

Logs:
    27/10/2020 (Vincent Stragier)
    - Comply to PEP8 and makes this script only
    excecutable as a module
    (py -m src.tools.tuh_sz_extract_metadata)
    01/10/2020 (Vincent Stragier)
    - create this script
"""
import datetime
import lzma
import multiprocessing as mp
import os
import pickle
from functools import partial

import numpy as np
import tqdm

import feature_extraction as fe

# Maximal number of thread to use
MAX_THREAD = int(1.5 * os.cpu_count())


def save_pickle(filename: str, variable):
    """Save a variable in a binary file.

    Args:
        filename: file name (with or without path or extension).
        variable: the variable to save as a binary.

    Returns:
        Returns nothing.
    """
    with open(filename, 'wb') as f:
        pickle.dump(variable, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_pickle_lzma(filename: str, variable):
    """Save a variable in a lzma compressed binary file.

    Args:
        filename: file name (with or without path or extension).
        variable: the variable to save as a binary.

    Returns:
        Returns nothing.
    """
    with lzma.open(filename, 'wb') as f:
        pickle.dump(variable, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle_lzma(filename: str):
    """Load a variable from a lzma compressed binary file.

    Args:
        filename: file name.

    Returns:
        Returns the variable.
    """
    with lzma.open(filename, 'rb') as f:
        return pickle.load(f)


def load_pickle(filename: str):
    """Load a variable from a binary file.

    Args:
        filename: file name.

    Returns:
        Returns the variable.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def extract_metadata_all_worker(filepath,
                                calibration_dict: dict,
                                calibrated_files: list):
    """Extract all the metadata of the recording linked to filepath.

    Args:
        filepath: the path to the recording related files without extension.
        calibration_dict: a dict which contains the calibration information
        (start and stop time) of the calibrated files.
        calibrated_files: the list of the calibrated files.

    Returns:
        A dictonnary with all the metadata.
    """
    if os.path.basename(filepath) in calibrated_files:
        metadata = fe.extract_metadata_all(
            filepath, calibration_dict[os.path.basename(filepath)])

    else:
        metadata = fe.extract_metadata_all(filepath)
    return metadata


def seconds_to_human_readble_time(seconds):
    """Converts a number of seconds to a readable string
    (days, hours, minutes, seconds, etc.)

    Args:
        seconds: the number of seconds to convert.

    Returns:
        A string with the human readable among of seconds.
    """
    return str(datetime.timedelta(seconds=seconds))


def labels_to_events(recording_meta):
    """From the recording converts the labels to a more explicite form.

    Args:
        recording_meta: a dictionary which contains the metadata
        of the recording.

    Returns:
        A list of dictionaries, which is structured as followed.
        {'start': l['start'],
         'stop': l['stop'],
         'montage': montages[l['montage']],
         'event':symbols[np.argmax(l['probabilities'])]}

        where:
            - ``l['start']`` is the start time of the event
            - ``l['stop']`` is the stop time of the event
            - ``montages[l['montage']]`` is the montage
            on which the event is occuring
            - ``symbols[np.argmax(l['probabilities'])]`` is
            the label of the most probable event
    """
    symbols = recording_meta['annotations_lbl']['symbols'][0]
    montages = recording_meta['annotations_lbl']['montages']
    # For each label extract start, stop, montage, symbol
    labels = list()  # List of dictionaries
    for label in recording_meta['annotations_lbl']['labels']:
        labels.append(
            {
                'start': label['start'],
                'stop': label['stop'],
                'montage': montages[label['montage']],
                'event': symbols[np.argmax(label['probabilities'])],
            },
        )

    return labels


def focal_starting_points(
        recording_meta: dict,
        seizure_types: list = ['fnsz', 'cpsz', 'spsz']):
    """Return a list of dictionaries which contain the focal events
    with their starting points.

    Args:
        recording_meta: a dictionary which contains the metadata
        of the recording.

    Returns:
        A list of dictionaries whom contain the information
        about the starting point of a focal seizure of any kind.
        The dictionary looks like:
            {'start': start_time,
             'event': event_tse['event'],
             'montages': montages}

            where:
                - ``start_time`` is the starting time of the event
                - ``event_tse['event']`` is the kind
                of event according to the tse file
                - ``montages`` is a list of montages
                on which the event did start from
    """
    # Convert the labels to more workable events
    events_list_lbl = labels_to_events(recording_meta)
    # Only keep focal seizure related events
    focal_events_list_lbl = [
        e for e in events_list_lbl
        if e['event'] in seizure_types
    ]

    events_list_tse = recording_meta['annotations_tse']
    # Only keep focal seizure related events
    focal_events_list_tse = [
        e for e in events_list_tse
        if e['event'] in seizure_types
    ]

    events = []
    for event_tse in focal_events_list_tse:
        start_time = event_tse['start']
        montages = [
            event_lbl['montage'] for event_lbl
            in focal_events_list_lbl
            if event_lbl['start'] == start_time
        ]

        events.append(
            {
                'start': start_time,
                'event': event_tse['event'],
                'montages': montages,
            },
        )

    return events


def main(path: str, path_calibration: str, metadata_save_path: str):
    """
    -List all the files of the dataset to find the EEG recording.
    -Extract all the metadata of the dataset
    and store them in a pickle and a JSON file.
    """
    # List the recording paths
    files_list = fe.extract_files_list(
        path=path,
        extension_filter='tse',
    ) + fe.extract_files_list(
        path=path,
        extension_filter='lbl',
    ) + fe.extract_files_list(
        path=path,
        extension_filter='edf',
    )

    print('Number of used files (found):', len(files_list))
    print('Number of recordings (found):', len(list(set(files_list))))

    # Remove incomplete recording from the dataset
    print('Filter the filelist to remove incomplete recordings.')
    filtered_paths = sorted(
        list(
            filter(
                fe.filters_dataset_files,
                list(set(files_list)),
            ),
        ),
    )

    print('Number of complete recordings:', len(filtered_paths))

    # Extract calibration periods
    calibration = fe.extract_calibration_periods(path_calibration)
    calibrated_files = list(calibration.keys())

    # Create a partial version of the 'extract_metadata_all_worker()'
    # to fix constant parameters
    partial_extract_metadata_all_worker = partial(
        extract_metadata_all_worker,
        calibration_dict=calibration,
        calibrated_files=calibrated_files,
    )

    # Start pool, inspired from
    # https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
    print('Start pool using {0} workers.'.format(MAX_THREAD))
    result_list_tqdm = []

    with mp.Pool(processes=MAX_THREAD) as pool:
        for result in tqdm.tqdm(
            pool.imap(
                func=partial_extract_metadata_all_worker,
                iterable=filtered_paths,
            ),
            total=len(filtered_paths),
        ):

            result_list_tqdm.append(result)

    print('The metadata collection is finished.')
    print('Save variable in a file.')
    save_pickle_lzma(metadata_save_path, result_list_tqdm)
    print('Variable saved.')


if __name__ == '__main__':
    # Create the script arguments parser
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'dataset_path',
        type=str,
        help='path to the dataset',
    )

    parser.add_argument(
        'calibration',
        type=str,
        help='path to the Excel calibration file',
    )

    parser.add_argument(
        'metadata_file',
        type=str,
        help='path to the metadata file to create',
    )

    args = parser.parse_args()
    main(
        path=args.dataset_path,
        path_calibration=args.calibration,
        metadata_save_path=args.metadata_file,
    )
