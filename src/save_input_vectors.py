"""This script imports the needed features in memory
(after choising the wanted features) and generates
the input vector for XGBoost.

Author:
    - Vincent Stragier

Logs:
    - (2020/11/10)
        - Create this script

"""
import datetime
import hashlib
import os
import sys

import git
import h5py
import numpy as np
import tqdm

import tools.feature_extraction as fe

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
ONE_HOT_VECTORS_PATH = os.path.abspath(
    os.path.join(SCRIPT_PATH, '..', 'input_vectors'),
)

gnsz_and_bckg = {'absz', 'tcsz', 'gnsz', 'tnsz', 'mysz', 'bckg'}


def generate_hot_vector(
    h5_file,
    subset: list,
    features: list,
    montages: list,
    single_channel: bool = True,
    keep_padding: bool = False,
):
    """Generate the one hot vectors.

    Args:
        h5_file: HDF5 file.
        subset: a list of recordings names.
        features: a list of the features included in the one hot vector.
        montages: a list with the montage to include.
        single_channel: if True, agregate the montages back to back.
        keep_padding: if True, keep the padding if any.

    Returns:
        A tupple with the x and y vectors.
    """
    x = []
    y = []

    if single_channel:
        for filename in tqdm.tqdm(subset[:]):
            if keep_padding:
                n_pad = 0

            else:
                key = 'features/{0}/{1}'.format(filename, features[0])
                attributes = dict(h5_file[key].attrs)
                n_pad = round(attributes['padding'] / attributes['step'])

            input_vector_temp = []
            for feature in features:
                # Get the indexing of the montage for the feature
                montage_index_dict = {
                    mntg.decode(): i
                    for i, mntg in enumerate(
                        h5_file['features/' + filename + '/' + feature][
                            'montage'
                        ],
                    )
                }

                # Bufferize the values for all the montages
                # (for one specific feature)
                feature_buffer = h5_file[
                    'features/' + filename + '/' + feature
                ][
                    'features'
                ]

                f_stack = np.hstack(
                    [
                        feature_buffer[montage_index_dict[mntg]][n_pad:-n_pad]
                        if n_pad >= 1
                        else feature_buffer[montage_index_dict[mntg]]
                        for mntg in montages
                    ],
                )

                input_vector_temp.append(f_stack)

            # For each recordings compose the hot vector
            # (for the features 'x_dev' and the targets 'y_dev')
            x.append(np.vstack(input_vector_temp).transpose())

            t_stack = [
                (
                    h5_file['targets/' + filename + '/MEAN']['features'][
                        montage_index_dict[mntg]
                    ] >= args.threshold
                ) * 1 for mntg in montages
            ]

            if n_pad >= 1:
                y.append(np.hstack(np.array(t_stack)[:, n_pad:-n_pad]))
            else:
                y.append(np.hstack(np.array(t_stack)))
        return np.vstack(x), np.hstack(y)

    else:  # All the channel
        for filename in tqdm.tqdm(subset[:]):
            if keep_padding:
                n_pad = 0

            else:
                key = 'features/{0}/{1}'.format(filename, features[0])
                attributes = dict(h5_file[key].attrs)
                n_pad = round(attributes['padding'] / attributes['step'])

            input_vector_temp = []
            for feature in features:
                # Get the indexing of the montage for the feature
                montage_index_dict = {
                    mntg.decode(): i
                    for i, mntg in enumerate(
                        h5_file['features/' + filename + '/' + feature][
                            'montage'
                        ],
                    )
                }

                # Bufferize the values for all the montages
                # (for one specific feature)
                feature_buffer = h5_file[
                    'features/' + filename + '/' + feature
                ][
                    'features'
                ]

                f_stack = np.vstack(
                    [
                        feature_buffer[montage_index_dict[mntg]]
                        for mntg in montages
                    ],
                ).transpose()

                input_vector_temp.append(f_stack)

            # For each recordings compose the hot vector
            # (for the features 'x_dev' and the targets 'y_dev')
            if n_pad >= 1:
                x.append(np.hstack(input_vector_temp)[n_pad:-n_pad])
            else:
                x.append(np.hstack(input_vector_temp))

            t_stack = [
                (
                    h5_file['targets/' + filename + '/MEAN']['features'][
                        montage_index_dict[mntg]
                    ] >= args.threshold
                ) * 1 for mntg in montages
            ]

            if n_pad >= 1:
                y.append(np.max(t_stack, axis=0)[n_pad:-n_pad])
            else:
                y.append(np.max(t_stack, axis=0))

        return np.vstack(x), np.hstack(y)


def main(args, prefix: str):
    """Execute the main instructions.

    Args:
        args: the parsed arguments.
        prefix: an unique prefix generated at runtime.
    """
    with h5py.File(args.hdf5, 'r') as hdf5:
        # Load metadata from HDF5
        meta = fe.raw_numpy_to_object(np.array(hdf5['metadata']))

        meta_dict = fe.raw_numpy_to_object(np.array(hdf5['metadata_dict']))

        # Separate dev set and train set
        dev_set = [m for m in meta if 'dev' in m['filepath']]
        train_set = [m for m in meta if 'train' in m['filepath']]

        # Try to import the list of recording from a provided file
        # No check here, can fail later
        try:
            with open(args.dev_list, 'r') as dev_list:
                dev_set_ = list(dev_list)

        except (TypeError, OSError):
            if args.dev_list is not None:
                print('An invalid list (file) has been passed for the dev list.')

            print('Using the default dev list.')
            # Take only the recordings using the AR montage
            dev_set_ = [
                m for m in dev_set
                if '_ar/' in m['filepath']
            ]

            # AR and GNSZ only
            ar_gnsz_only = []
            for patient in dev_set_:
                patient_lbl = {
                    lbl['event'] for lbl in patient['annotations_tse']
                }

                if (patient_lbl & gnsz_and_bckg) == patient_lbl:
                    ar_gnsz_only.append(patient)

            dev_set_ = [os.path.basename(m['filepath']) for m in ar_gnsz_only]

        # Try to import the list of recording from a provided file
        # No check here, can fail later
        try:
            with open(args.train_list, 'r') as train_list:
                train_set_ = list(train_list)

        except (TypeError, OSError):
            if args.train_list is not None:
                print('An invalid list (file) has been passed for the train list.')

            print('Using the default train list.')
            train_set_ = [
                m for m in train_set
                if '_ar/' in m['filepath']
            ]

            # AR and GNSZ only
            ar_gnsz_only = []
            for patient in train_set_:
                patient_lbl = {
                    lbl['event'] for lbl in patient['annotations_tse']
                }

                if (patient_lbl & gnsz_and_bckg) == patient_lbl:
                    ar_gnsz_only.append(patient)

            train_set_ = [os.path.basename(m['filepath']) for m in ar_gnsz_only]

        # List the montages to use
        montages_ = meta_dict[os.path.basename(dev_set_[0])][
            'annotations_lbl']['montages']

        if isinstance(args.montages, list):
            montages = [m for m in args.montages if m in montages_]
            try:
                assert len(montages) == len(args.montages)
            except AssertionError:
                error_message = 'All the montages passed are not'
                ' in list of available montages:'
                '\n{0}'.format(
                    '\n'.join(montages_),
                )

                raise AssertionError(error_message)
        else:
            montages = montages_

        if args.all:
            columns = [
                '{0} ({1})'.format(f, m)
                for m in montages
                for f in args.features]
        else:
            columns = ['{0}'.format(f) for f in args.features]

        np.save('{0}/{1}_columns.npy'.format(args.path, prefix), columns)

        # Generate the hot vectors
        print('Convert features and targets for the hot vector (train).')
        x_train, y_train = generate_hot_vector(
            h5_file=hdf5,
            subset=train_set_,
            features=args.features,
            montages=montages,
            single_channel=not args.all,
            keep_padding=args.keep_padding,
        )
        print('Shape of x_train: {0}'.format(x_train.shape))
        print('Size of x_train: {0} bytes.'.format(sys.getsizeof(x_train)))
        print('Shape of y_train: {0}'.format(y_train.shape))
        print('Size of y_train: {0} bytes.'.format(sys.getsizeof(y_train)))

        # Imbalance ratio.
        # Allows to compensate the imbalance between the classes.
        imbalance_ratio = len(y_train) / np.sum(y_train)
        print('Train "imbalance ratio": {0}'.format(imbalance_ratio))

        np.save('{0}/{1}_x_train.npy'.format(args.path, prefix), x_train)
        np.save('{0}/{1}_y_train.npy'.format(args.path, prefix), y_train)
        del x_train
        del y_train

        print('Convert features and targets for the hot vector (dev).')
        x_dev, y_dev = generate_hot_vector(
            h5_file=hdf5,
            subset=dev_set_,
            features=args.features,
            montages=montages,
            single_channel=not args.all,
            keep_padding=args.keep_padding,
        )
        print('Shape of x_dev: {0}'.format(x_dev.shape))
        print('Size of x_dev: {0} bytes'.format(sys.getsizeof(x_dev)))
        print('Shape of y_dev: {0}'.format(y_dev.shape))
        print('Size of y_dev: {0} bytes'.format(sys.getsizeof(y_dev)))

        np.save('{0}/{1}_x_dev.npy'.format(args.path, prefix), x_dev)
        np.save('{0}/{1}_y_dev.npy'.format(args.path, prefix), y_dev)


if __name__ == '__main__':
    # Create the script arguments parser
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=True)

    all_features = fe.ALL_FEATURES
    # Removes non univariate features.
    all_features.remove('FIRST_DIFFERENTIAL')
    str_all_features = ', '.join(all_features)

    parser.add_argument(
        'hdf5',
        type=str,
        help='Path to the dataset (HDF5 file), the features'
        ' will be sourced there.',
    )

    parser.add_argument(
        '-f',
        '--features',
        nargs='+',
        type=str,
        help=(
            'List of features to convert to the hot vector ("a", "A"'
            ' or "all", "ALL" will lead to convert all the features,'
            ' as well as not providing this parameter).'
            ' Not case sensitive: ' + str_all_features
        ),
    )

    parser.add_argument(
        '-t',
        '--target',
        type=str,
        help='Targets feature to use as labels.',
        default='MEAN',
    )

    parser.add_argument(
        '-m',
        '--montages',
        nargs='+',
        type=str,
        help='List of montages to use.',
    )

    parser.add_argument(
        '-th',
        '--threshold',
        type=float,
        help='Threshold to use on the target segment.',
        default=0.8,
    )

    parser.add_argument(
        '-tl',
        '--train_list',
        type=str,
        help='List of the files used for the training.',
        default=None,
    )

    parser.add_argument(
        '-dl',
        '--dev_list',
        type=str,
        help='List of the files used for the testing (using dev).',
        default=None,
    )

    parser.add_argument(
        '-p',
        '--path',
        type=str,
        help='Saving path where the ".npy"'
        ' will be saved (default="{0}").'.format(
            ONE_HOT_VECTORS_PATH,
        ),
        default=ONE_HOT_VECTORS_PATH,
    )

    # By default, we remove the padding
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-rp',
        '--remove-padding',
        help='If set, the program will remove the padding if any',
        action='store_true',
        default=False,
    )

    group.add_argument(
        '-kp',
        '--keep-padding',
        help='If set, the program will keep the padding if any.',
        action='store_true',
        default=False,
    )

    # By default, only all the channels are agregated to one channel.
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-a',
        '--all',
        help='If set, each column is a specific montage and feature.',
        action='store_true',
        default=False,
    )

    group.add_argument(
        '-s',
        '--single',
        help='If set, each column is a specific feature.',
        action='store_true',
        default=False,
    )

    # By default ask to the user if a want to proceed.
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

    args = parser.parse_args()

    try:
        assert args.threshold <= 1
        assert args.threshold >= 0
    except AssertionError:
        error_message = 'The threshold should be between 0 and 1 included.'
        raise AssertionError(error_message)

    # Generate or check the list of features.
    features = args.features

    # Take all the features.
    if (features is None or 'a'
            in features or 'A'
            in features or 'all'
            in features or 'ALL'
            in features):

        features = all_features

    else:
        features = list(set(features))
        features = [
            f.upper() for f in features if f.upper() in fe.ALL_FEATURES
        ]

    try:
        assert len(features) > 0
    except AssertionError:
        error_message = 'No valid features have been provided.'
        '\nThe available features are:\n\t{0}'.format(
            '\n\t'.join(fe.ALL_FEATURES),
        )

        raise AssertionError(error_message)

    args.features = features

    # Get repo information
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
    except Exception:
        sha = 'none'

    date = datetime.datetime.now()
    prefix = '{0}_-_{1}_-_{2}_-_target:{3}_githash:{4}'.format(
        date.strftime('%Y-%m-%d_%Hh%Mm%Ss-%f'),
        'no_padding' if (
            args.remove_padding or not args.keep_padding
        ) else 'padding',
        'single_channel' if (args.single or not args.all) else 'all_channels',
        args.target,
        sha,
    )

    short_prefix_hash = hashlib.md5(prefix.encode('UTF-16')).hexdigest()

    short_prefix = '{0}_{1}'.format(
        date.strftime('%Y-%m-%d_%Hh%Mm'),
        short_prefix_hash,
    )

    info_filename = os.path.join(
        args.path, '{0}.info'.format(short_prefix),
    )

    print('The following arguments have been parsed:')
    for key, value in vars(args).items():
        print('{0}: {1}'.format(key, value))

    if not args.no:
        print()
        if not args.yes:
            try:
                while True:
                    conti = input('Do you want to run the program (yes/no)? ')
                    if conti in ('y', 'yes', 'Y', 'YES'):
                        break

                    elif conti in ('n', 'no', 'N', 'NO'):
                        exit()

            except KeyboardInterrupt:
                print(
                    '\nThe user requested the end of the program'
                    ' (KeyboardInterrupt).',
                )

                exit()
    else:
        exit()

    print('Create the information file ({0})'.format(info_filename))
    with open(info_filename, 'w+') as info_file:
        info_file.write('[path]\n')
        info_file.write('{0}\n'.format(ONE_HOT_VECTORS_PATH))

        info_file.write('[prefix]\n')
        info_file.write('{0}\n'.format(prefix))

        info_file.write('[short prefix hash]\n')
        info_file.write('{0}\n'.format(short_prefix_hash))

        info_file.write('[short prefix]\n')
        info_file.write('{0}\n'.format(short_prefix))

        info_file.write('[sys.argv]\n')
        info_file.write('{0}\n'.format(str(sys.argv)))

        info_file.write('[parsed arguments]\n')
        for key, value in vars(args).items():
            info_file.write('{0}: {1}\n'.format(key, value))

    main(args=args, prefix=short_prefix)
