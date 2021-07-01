"""Compute the metrics."""
import os
from functools import partial

import h5py
import numpy as np
import pandas as pd
import tqdm
import xgboost as xgb
from skimage.util.shape import view_as_windows

import tools.feature_extraction as fe
import tools.files_filters as ff

report_columns = [
    'Filepath',
    'Filename',
    'Condition positive',
    'Condition negative',
    'True positive',
    'True negative',
    'False positive',
    'False negative',
    'Sensitivity',
    'Specificity',
    'Precision',
    'Accuracy',
    'F1 score',
    'Negative predictive value',
    'Miss rate',
    'Fall-rate',
    'False discovery rate',
    'False omission rate',
    'Prevalence threshold',
    'Threat score',
    'Balanced accuracy',
    'Matthews correlation coefficient',
    'Fowlkes–Mallows index',
    'Bookmaker informedness',
    'Markedness',
]


def auto_adjust(writer, sheet_name: str, dataframe, offset: int = 2):
    """Adjust roughly the columns width of an Excel sheet.

    Args:
        writer: the xlsx writer.
        sheet_name: the name of the sheet on which
            to adjust the width of the columns.
        dataframe: the data used to fill the sheet
            (used to find the maximal width)
         offset: a value to offset.

    https://stackoverflow.com/a/40535454
    """
    worksheet = writer.sheets[sheet_name]

    for idx, col in enumerate(dataframe):  # loop through all columns
        series = dataframe[col]
        max_len = max(
            (
                series.astype(str).map(len).max(),  # len of largest item
                len(str(series.name)),  # len of column name/header
            ),
        ) + offset  # adding a little extra space

        worksheet.set_column(idx, idx, max_len)  # set column width


def concatenate_labels_columns(array_0, array_1):
    """Concatenate the labels comparison.

    Args:
        array_0: the first array.
        array_1: the second array.

    Returns:
        The merged arrays.
    """
    n_row_array_0 = array_0.shape[0]
    n_row_array_1 = array_1.shape[0]

    if n_row_array_0 > n_row_array_1:
        # Change array_1 shape and merge.
        return np.hstack(
            (
                array_0,
                np.pad(
                    array_1, ((0, n_row_array_0-n_row_array_1), (0, 0)),
                    'constant',
                    constant_values=np.nan,
                ),
            ),
        )

    elif n_row_array_0 < n_row_array_1:
        # Change array_0 shape and merge
        return np.hstack(
            (
                np.pad(
                    array_0, ((0, n_row_array_1-n_row_array_0), (0, 0)),
                    'constant',
                    constant_values=np.nan,
                ),
                array_1,
            ),
        )

    else:
        # Merge
        return np.hstack((array_0, array_1))


def main(
    args,
    h5_file,
    model_file,
    xlsx_file,
    model_mode,
    features,
    montages,
    threshold,
    smoothing_window: float = 0,
    smoothing_ratio: float = 0,
):
    """Proceed to execute the main instructions.

    Args:
        args: the arguments passed to the script.
        h5_file: the path to the dataset.
        model_file: the path to the model.
        xlsx_file: the path to the xlsx file.
        model_mode: if True, the mode is 'all' otherwise 'single'.
        threshold: used to determine if a segment is a seizure or not.
        smoothing_window: the width of the smoothing window.
        smoothing_ratio: the ratio used for the smoothing.
    """
    # Determine is it is necessary to apply smoothing (post processing)
    smoothing = not (smoothing_ratio == 0 or smoothing_window == 0)

    if not model_mode:
        raise NotImplementedError(
            'Only the "old" all mode model metrics can be computed for now.',
        )

    # Load the model
    bst = xgb.Booster(model_file=model_file)

    # Load x_dev and y_dev for prediction.
    print('Load the data to run the predictions on.')
    with h5py.File(h5_file, 'r') as h5:
        print('In hdf5 file...')

        # Load metadata from h5
        meta = fe.raw_numpy_to_object(np.array(h5['metadata']))

        meta_dict = fe.raw_numpy_to_object(np.array(h5['metadata_dict']))

        # Separate dev set and train set
        dev_set = (m for m in meta if 'dev' in m['filepath'])
        train_set = (m for m in meta if 'train' in m['filepath'])

        # Filter the dev and train set
        try:
            with open(args.train_list, 'r') as train_list:
                train_set_filtered = [
                    file_.splitline()[0] for file_ in list(train_list)
                ]

        except (TypeError, OSError):
            if args.train_list is not None:
                print(
                    'An invalid list (file) has been passed for the dev list.',
                )

            if not len(args.train_filters):
                print('Using the default dev list.')
                train_set_filtered = list(train_set)

            else:
                train_set_filtered = list(
                    ff.metalist_to_filelist(
                        ff.filter_eval(
                            args.train_filters,
                            train_set,
                        ),
                    ),
                )

        try:
            with open(args.dev_list, 'r') as dev_list:
                dev_set_filtered = [
                    file_.splitline()[0] for file_ in list(dev_list)
                ]

        except (TypeError, OSError):
            if args.dev_list is not None:
                print(
                    'An invalid list (file) has been passed for the dev list.',
                )

            if not len(args.dev_filters):
                print('Using the default dev list.')
                dev_set_filtered = list(dev_set)

            else:
                dev_set_filtered = list(
                    ff.metalist_to_filelist(
                        ff.filter_eval(
                            args.dev_filters,
                            dev_set,
                        ),
                    ),
                )

        # Debug
        # dev_set_filtered = dev_set_filtered[:5]
        # train_set_filtered = train_set_filtered[:5]

        # List the features to use
        # All if not defined in the input vectors info file
        if not len(features):
            features = fe.ALL_FEATURES

            try:
                features.remove('FIRST_DIFFERENTIAL')

            except Exception:
                pass

        # List the montages to use
        # All if not defined in the input vectors info file
        if not len(montages):
            montages = meta_dict[os.path.basename(
                dev_set_filtered[0])]['annotations_lbl']['montages']

        # columns = [
        #     '{0} ({1})'.format(f, m) for m in montages for f in features
        # ]

        # x_train features in input vector
        x_train, y_train, x_dev, y_dev = [], [], [], []

        # Generate the metrics
        print('Convert features and targets for the input vector (train).')
        global_y_true = []
        global_y_pred = []
        all_metrics = []
        train_metrics_lines = []
        labels_comparison_train = np.array([])

        # Go through the training files
        for filename in tqdm.tqdm(train_set_filtered):
            input_vector_temp = []
            attrs = dict(h5['features/' + filename + '/' + features[0]].attrs)
            step = attrs['step']
            # Number of pads to remove
            n_added_padding = int(attrs['padding'] / step)
            window_width = int(attrs['window'] / step)
            n_pad = window_width - 1

            for feature in features:
                # Get the indexing of the montage for the feature
                montage_index_dict = {
                    mntg.decode(): i for i, mntg in enumerate(
                        h5['features/' + filename + '/' + feature]['montage'],
                    )
                }

                # Buffer size the values for all the montages
                # (for one specific feature)
                feature_buffer = h5[
                    'features/' + filename + '/' + feature
                ]['features']

                f_stack = np.vstack(
                    [
                        feature_buffer[montage_index_dict[mntg]]
                        for mntg in montages
                    ],
                ).transpose()

                input_vector_temp.append(f_stack)

            # For each recordings compose the input vector
            # (for the features 'x_train' and the targets 'y_train')
            y_train = h5['dataset/' + filename]['targets']
            if n_added_padding:
                x_train = np.hstack(
                    input_vector_temp,
                )[n_added_padding:-n_added_padding]
            else:
                x_train = np.hstack(input_vector_temp)

            sampling_frequency = h5['dataset/' + filename][
                'sampling_frequency'
            ][0][0]

            # Assumption: sampling_frequency is constant
            step_in_samples = int(step * sampling_frequency)

            number_of_channels = h5['dataset/' + filename][
                'signal'
            ].shape[0]

            segments = view_as_windows(
                y_train.reshape(number_of_channels, y_train.shape[-1]),
                (number_of_channels, step_in_samples),
                step_in_samples,
            )[0]

            # Preload the extraction function
            partial_extract = partial(
                fe.extract_features,
                sampling_rate=sampling_frequency,
                features_list=['MEAN'],
            )

            y_train = np.array(
                [m['MEAN'] for m in map(partial_extract, segments)],
            ).max(axis=-1)

            y_pred = bst.predict(xgb.DMatrix(x_train))

            new_shape = list(y_pred.shape)
            new_shape[-1] = new_shape[-1] + n_pad
            y_pred.resize(tuple(new_shape))

            # Dewindow the y_pred.
            y_dewin = np.copy(y_pred)
            for index in range(n_pad):
                y_dewin += np.roll(y_pred, index + 1)

            for index in range(int(max(0, (n_pad - 1)))):
                y_dewin[index + 1] /= (index + 2)
                y_dewin[- (index + 2)] /= (index + 2)

            if n_pad:
                y_dewin[n_pad:-n_pad] /= (n_pad + 1)

            train_metrics_lines.append(
                fe.generate_line(
                    meta_dict[filename]['filepath'],
                    filename,
                    fe.compute_metrics(
                        fe.binary_smoothing(
                            (y_dewin > 0.5) * 1,
                            smoothing_window,
                            smoothing_ratio,
                        ) if smoothing else (y_dewin > 0.5) * 1,
                        (y_train > threshold) * 1,
                    ),
                ),
            )

            global_y_true.append(y_train)
            global_y_pred.append(y_dewin)

            if smoothing:
                labels_comparison = np.vstack(
                    (
                        (y_train > threshold) * 1,
                        (y_dewin > 0.5) * 1,
                        y_dewin,
                        fe.binary_smoothing(
                            (y_dewin > 0.5) * 1,
                            smoothing_window,
                            smoothing_ratio,
                        ),
                    ),
                ).T
            else:
                labels_comparison = np.vstack(
                    (
                        (y_train > threshold) * 1,
                        (y_dewin > 0.5) * 1,
                        y_dewin,
                    ),
                ).T

            try:
                labels_comparison_train = concatenate_labels_columns(
                    labels_comparison_train, labels_comparison,
                )

            except ValueError:
                labels_comparison_train = labels_comparison

        global_y_true = np.hstack(global_y_true)
        global_y_pred = np.hstack(global_y_pred)

        all_metrics.append(
            fe.generate_line(
                'train/',
                '*',
                fe.compute_metrics(
                    fe.binary_smoothing(
                        (global_y_pred > 0.5) * 1,
                        smoothing_window,
                        smoothing_ratio,
                    ) if smoothing else (global_y_pred > 0.5) * 1,
                    (global_y_true > threshold) * 1,
                ),
            ),
        )

        print('Convert features and targets for the input vector (dev).')
        global_y_true = []
        global_y_pred = []
        dev_metrics_lines = []
        labels_comparison_dev = np.array([])

        for filename in tqdm.tqdm(dev_set_filtered):
            input_vector_temp = []
            attrs = dict(h5['features/' + filename + '/' + features[0]].attrs)
            step = attrs['step']
            # Number of pads to remove
            n_added_padding = int(attrs['padding'] / step)
            window_width = int(attrs['window'] / step)
            n_pad = window_width - 1

            for feature in features:
                # Get the indexing of the montage for the feature
                montage_index_dict = {
                    mntg.decode(): i for i, mntg in enumerate(
                        h5['features/' + filename + '/' + feature]['montage'],
                    )
                }

                # Buffer size the values for all the montages
                # (for one specific feature)
                feature_buffer = h5[
                    'features/' + filename + '/' + feature
                ]['features']

                f_stack = np.vstack(
                    [
                        feature_buffer[montage_index_dict[mntg]]
                        for mntg in montages
                    ],
                ).transpose()

                input_vector_temp.append(f_stack)

            # For each recordings compose the input vector
            # (for the features 'x_train' and the targets 'y_train')
            y_dev = h5['dataset/' + filename]['targets']
            if n_added_padding:
                x_dev = np.hstack(
                    input_vector_temp,
                )[n_added_padding:-n_added_padding]
            else:
                x_dev = np.hstack(input_vector_temp)

            sampling_frequency = h5['dataset/' + filename][
                'sampling_frequency'
            ][0][0]

            # Assumption: sampling_frequency is constant
            step_in_samples = int(step * sampling_frequency)

            number_of_channels = h5['dataset/' + filename][
                'signal'
            ].shape[0]

            segments = view_as_windows(
                y_dev.reshape(number_of_channels, y_dev.shape[-1]),
                (number_of_channels, step_in_samples),
                step_in_samples,
            )[0]

            # Preload the extraction function
            partial_extract = partial(
                fe.extract_features,
                sampling_rate=sampling_frequency,
                features_list=['MEAN'],
            )

            y_dev = np.array(
                [
                    m['MEAN'] for m in map(partial_extract, segments)
                ],
            ).max(axis=-1)

            # Predict
            y_pred = bst.predict(xgb.DMatrix(x_dev))

            new_shape = list(y_pred.shape)
            new_shape[-1] = new_shape[-1] + n_pad
            y_pred.resize(tuple(new_shape))

            # Dewindow the y_pred.
            y_dewin = np.copy(y_pred)
            for index in range(n_pad):
                y_dewin += np.roll(y_pred, index + 1)

            for index in range(int(max(0, (n_pad - 1)))):
                y_dewin[index + 1] /= (index + 2)
                y_dewin[- (index + 2)] /= (index + 2)

            y_dewin[n_pad:-n_pad] /= (n_pad + 1)

            dev_metrics_lines.append(
                fe.generate_line(
                    meta_dict[filename]['filepath'],
                    filename,
                    fe.compute_metrics(
                        fe.binary_smoothing(
                            (y_dewin > 0.5) * 1,
                            smoothing_window,
                            smoothing_ratio,
                        ) if smoothing else (y_dewin > 0.5) * 1,
                        (y_dev > threshold) * 1,
                    ),
                ),
            )

            global_y_true.append(y_dev)
            global_y_pred.append(y_dewin)

            if smoothing:
                labels_comparison = np.vstack(
                    (
                        (y_dev > threshold) * 1,
                        (y_dewin > 0.5) * 1,
                        y_dewin,
                        fe.binary_smoothing(
                            (y_dewin > 0.5) * 1,
                            smoothing_window,
                            smoothing_ratio,
                        ),
                    ),
                ).T
            else:
                labels_comparison = np.vstack(
                    (
                        (y_dev > threshold) * 1,
                        (y_dewin > 0.5) * 1,
                        y_dewin,
                    ),
                ).T

            try:
                labels_comparison_dev = concatenate_labels_columns(
                    labels_comparison_dev, labels_comparison,
                )

            except ValueError:
                labels_comparison_dev = labels_comparison

        global_y_true = np.hstack(global_y_true)
        global_y_pred = np.hstack(global_y_pred)

        all_metrics.append(
            fe.generate_line(
                'dev/', '*', fe.compute_metrics(
                    fe.binary_smoothing(
                        (global_y_pred > 0.5) * 1,
                        smoothing_window,
                        smoothing_ratio,
                    ) if smoothing else (global_y_pred > 0.5) * 1,
                    (global_y_true > threshold) * 1,
                ),
            ),
        )

        df_dev = pd.DataFrame(dev_metrics_lines, columns=report_columns)
        df_train = pd.DataFrame(train_metrics_lines, columns=report_columns)
        df_all = pd.DataFrame(all_metrics, columns=report_columns)

        # Build dev set labels comparison sheet
        if smoothing:
            header = pd.MultiIndex.from_product(
                [
                    dev_set_filtered,
                    [
                        'Truth',
                        'Predicted binary',
                        'Predicted',
                        'Smoothed prediction',
                    ],
                ],
                names=['loc', 'Time interval [s]'],
            )
        else:
            header = pd.MultiIndex.from_product(
                [
                    dev_set_filtered,
                    ['Truth', 'Predicted binary', 'Predicted'],
                ],
                names=['loc', 'Time interval [s]'],
            )

        indexes = (
            '[{0}, {1}['.format(i * step, (i + 1) * step)
            for i in range(labels_comparison_dev.shape[0])
        )

        df_labels_dev = pd.DataFrame(
            labels_comparison_dev,
            index=list(indexes),
            columns=header,
        )

        # Build train set labels comparison sheet
        if smoothing:
            header = pd.MultiIndex.from_product(
                [
                    train_set_filtered,
                    [
                        'Truth',
                        'Predicted binary',
                        'Predicted',
                        'Smoothed prediction',
                    ],
                ],
                names=['loc', 'Time interval [s]'],
            )
        else:
            header = pd.MultiIndex.from_product(
                [
                    train_set_filtered,
                    ['Truth', 'Predicted binary', 'Predicted'],
                ],
                names=['loc', 'Time interval [s]'],
            )

        indexes = (
            '[{0}, {1}['.format(i * step, (i + 1) * step)
            for i in range(labels_comparison_train.shape[0])
        )

        df_labels_train = pd.DataFrame(
            labels_comparison_train,
            index=list(indexes),
            columns=header,
        )

        with pd.ExcelWriter(  # pylint: disable=abstract-class-instantiated
            xlsx_file,
            date_format='YYYY-MM-DD',
            datetime_format='YYYY-MM-DD HH:MM:SS',
            engine='xlsxwriter',
        ) as writer:

            df_dev.to_excel(writer, sheet_name='dev', index=False)
            df_train.to_excel(writer, sheet_name='train', index=False)
            df_all.to_excel(writer, sheet_name='all', index=False)
            df_labels_dev.to_excel(writer, sheet_name='labels_dev')
            df_labels_train.to_excel(writer, sheet_name='labels_train')

            auto_adjust(writer, sheet_name='dev', dataframe=df_dev)
            auto_adjust(writer, sheet_name='train', dataframe=df_train)
            auto_adjust(writer, sheet_name='all', dataframe=df_all)
            auto_adjust(
                writer, sheet_name='labels_dev', dataframe=df_labels_dev,
            )
            auto_adjust(
                writer, sheet_name='labels_train', dataframe=df_labels_train,
            )

            writer.sheets['labels_dev'].set_row(
                2, None, None, {'hidden': True},
            )
            writer.sheets['labels_train'].set_row(
                2, None, None, {'hidden': True},
            )


if __name__ == '__main__':
    # Create the script arguments parser
    import argparse
    import re

    parser = argparse.ArgumentParser(allow_abbrev=True)

    parser.add_argument(
        'info_file',
        type=str,
        help='Information about the model.',
    )

    parser.add_argument(
        '--xlsx_file',
        type=str,
        help='Excel file (result of the metrics).',
        default=None,
    )

    parser.add_argument(
        '-sw',
        '--smoothing_window',
        type=int,
        help=(
            'Width of the window on which to apply the post processing '
            '(smoothing). Set to zero by default.'
        ),
        default=0,
    )

    parser.add_argument(
        '-sr',
        '--smoothing_ratio',
        type=float,
        help=(
            'Ratio to use when applying the post processing '
            '(smoothing). Set to zero by default.'
        ),
        default=0,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-tl',
        '--train_list',
        type=str,
        help='List of the files used for the training (from file).',
        default=None,
    )

    group.add_argument(
        '-tf',
        '--train_filters',
        type=str,
        nargs='+',
        help='Filters the training set.',
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-dl',
        '--dev_list',
        type=str,
        help='List of the files used for the testing (dev, from file).',
        default=None,
    )

    group.add_argument(
        '-df',
        '--dev_filters',
        type=str,
        nargs='+',
        help='Filters the dataset, for the dev set.',
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

    # Extract input vectors path and filename
    with open(args.info_file, 'r') as f:
        for line in f:
            if '[input vectors path]' in line:
                break

        for line in f:
            if line != '\n':
                input_vector_path = line.replace('\n', '').replace('\r', '')
                break

        for line in f:
            if line.startswith('input_vectors_info:'):
                input_vector_info = os.path.join(
                    input_vector_path,
                    os.path.basename(
                        re.findall(r'([^\s]+)\n', line.replace('\r', ''))[-1],
                    ),
                )
                break

    # Extract useful parameters
    with open(input_vector_info, 'r') as f:
        for line in f:
            if line.startswith('features:'):
                features = re.findall(r'\'([^\s]+)\'', line)

            if line.startswith('montages:'):
                montages = re.findall(r'\'([^\s]+)\'', line)

            if line.startswith('threshold:'):
                threshold = float(
                    re.findall(r'([^\s]+)\n', line.replace('\r', ''))[-1],
                )

            if line.startswith('hdf5:'):
                hdf5 = re.findall(
                    r'([^\s]+)\n',
                    line.replace('\r', ''),
                )[-1]

            if line.startswith('all:'):
                mode_is_all = re.findall(
                    r'([^\s]+)\n',
                    line.replace('\r', ''),
                )[-1] == 'True'
                break

    main(
        args=args,
        h5_file=hdf5,
        model_file=args.info_file.replace('.info', '.model'),
        xlsx_file=args.xlsx_file,
        model_mode=mode_is_all,
        features=features,
        montages=montages,
        threshold=threshold,
        smoothing_window=args.smoothing_window,
        smoothing_ratio=args.smoothing_ratio,
    )
