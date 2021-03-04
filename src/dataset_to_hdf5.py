"""This script converts the dataset to a HDF5 file
in order to easy futher processing.
Note that the metadata have already been extracted
to an lzma pickle file and will be added to the
resulting HDF5 file.

Author:
    - Vincent Stragier

Logs:
    - 2020/11/02
        - Create this script
"""
import os
from functools import partial

import h5py
import tqdm

import tools.feature_extraction as fe


def signals_extraction_worker(
        filepath: str,
        metadata: list,
        hdf5_path: str,
        hdf5_dataset=None):
    """Extract the signals of the edf files and save them in a HDF5 dataset.

    Args:
        filepath: the path the .edf file without its extension.
        metadata: dictionaries containing the metadata of each recording.
        hdf5_path: the top path to use in the HDF5 file.
        hdf5_dataset: a HDF5 dataset object.
    """
    # Extract the signals from the .EDF file
    # Use the metadata labels to generate the targets
    signals = fe.extract_channels_signal_from_file(
        filepath,
        metadata,
    )

    # Generate the path of the HDF5 path
    hdf5_path = os.path.join(
        hdf5_path,
        os.path.basename(path),
    ).replace('\\', '/')

    # Save signals in the HDF5 file
    hdf5_dataset.create_dataset(
        hdf5_path,
        data=signals,
    )


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
        'meta_path',
        type=str,
        help='path to the metadata',
    )

    parser.add_argument(
        'hd5_path',
        type=str,
        help='path to the HDF5 file',
    )

    args = parser.parse_args()

    # Inspect the dataset path
    print('Inspects dataset path')

    # Could be optimised,
    # but is used only once here
    files_list = fe.extract_files_list(
        path=args.dataset_path,
        extension_filter='tse',
    ) + fe.extract_files_list(
        path=args.dataset_path,
        extension_filter='lbl',
    ) + fe.extract_files_list(
        path=args.dataset_path,
        extension_filter='edf',
    )

    # Remove incomplete recording from the dataset
    print('Filters the filelist to remove incomplete recordings.')
    filtered_paths = sorted(
        list(
            filter(
                fe.filters_dataset_files,
                list(
                    set(
                        files_list,
                    ),
                ),
            ),
        ),
    )

    print(
        '{0} partial recording(s) were found, '
        'which represent {1} full recording(s).'
        .format(
            len(
                list(
                    set(
                        files_list,
                    ),
                ),
            ),
            len(
                filtered_paths,
            ),
        ),
    )

    # Load the metadata
    print('Loads metadata.')
    metadata = fe.ta.load_pickle_lzma(args.meta_path)
    print('Metadata are loaded.')
    print('\tTransforms the list of dictionaries '
          'in a dictionary of dictionaries.')

    metadata_dict = {
        os.path.basename(meta['filepath']):
        meta for meta in metadata
    }

    print('Converts the metadata (list and dict) '
          'to numpy arrays of bytes.')

    compressed_metadata = fe.object_to_raw_numpy(metadata)

    compressed_metadata_dict = fe.object_to_raw_numpy(metadata_dict)

    # Create HDF5 file
    with h5py.File(args.hd5_path, mode='w') as hdf5:
        print('HDF5 file has been created or opened '
              '(it will be loaded with the dataset now).')
        hdf5.attrs['dataset_description'] =\
            """This file contains the TUSZ dataset version 1.5.2.
The montages and useful signal in the dataset collection.
The metadata in the metadata collection."""

        print('\tSave the metadata to the HDF5 file.')
        hdf5.create_dataset('metadata', data=compressed_metadata)
        hdf5.create_dataset('metadata_dict', data=compressed_metadata_dict)

        print('Start extracting all the signals.')
        partial_signals_extraction_worker = partial(
            signals_extraction_worker,
            metadata=metadata,
            hdf5_path='dataset',
            hdf5_dataset=hdf5,
        )

        # Using a for loop here since the HDF5
        # is not (easily) thread safe
        for path in tqdm.tqdm(filtered_paths):
            partial_signals_extraction_worker(path)

        print('Done.')
