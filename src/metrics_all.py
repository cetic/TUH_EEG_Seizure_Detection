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


def main():
    """Proceed to execute the main instructions."""
    base = '/home_nfs/stragierv/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection/'
    h5_file = '/home_nfs/stragierv/dataset.h5'
    model_file = 'models/2021-02-15_12h13m_'
    model_file += '(2021-02-15_11h34m_8ebe024395ade47ba5ecf9a128e76a58).model'
    model_file = base + model_file
    xlsx_file = 'reports/metrics/AR_GNSZ-no_smoothing.xlsx'
    xlsx_file = base + xlsx_file
    # one_hot_vectors/2021-02-15_11h34m_8ebe024395ade47ba5ecf9a128e76a58.info

    smoothing = False
    smoothing_windows = 8
    smoothing_ratio = 8/12

    gnsz_and_bckg = {'absz', 'tcsz', 'gnsz', 'tnsz', 'mysz', 'bckg'}

    # Load the model
    bst = xgb.Booster(model_file=model_file)

    # Load x_dev and y_dev for prediction.
    print('Load the data to run the predictions on.')

    threshold = 0.8

    with h5py.File(h5_file, 'r') as h5:
        print('In hdf5 file.')
        # Load metadata from h5
        meta = fe.raw_numpy_to_object(np.array(h5['metadata']))

        meta_dict = fe.raw_numpy_to_object(np.array(h5['metadata_dict']))

        # Separate dev set and train set
        dev_set = [m for m in meta if 'dev' in m['filepath']]
        train_set = [m for m in meta if 'train' in m['filepath']]

        # Take only the recordings using the AR montage
        # dev_set_ar = [
        #     os.path.basename(m['filepath'])
        #     for m in dev_set if '_ar/' in m['filepath']
        # ]
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

        dev_set_ar = [os.path.basename(m['filepath']) for m in ar_gnsz_only]

        # train_set_ar = [
        #     os.path.basename(m['filepath'])
        #     for m in train_set if '_ar/' in m['filepath']
        # ]
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

        train_set_ar = [os.path.basename(m['filepath']) for m in ar_gnsz_only]

        # List the features to use
        features = fe.ALL_FEATURES
        features.remove('FIRST_DIFFERENTIAL')

        # List the montages to use
        montages = meta_dict[os.path.basename(
            dev_set_ar[0])]['annotations_lbl']['montages']

        columns = [
            '{0} ({1})'.format(f, m) for m in montages for f in features
        ]

        # x_train features in hot vector
        x_train, y_train, x_dev, y_dev = [], [], [], []

        # Generate the metrics
        print('Convert features and targets for the hot vector (train).')
        global_y_true = []
        global_y_pred = []
        all_metrics = []
        train_metrics_lines = []

        for filename in tqdm.tqdm(train_set_ar):
            one_hot_vector_temp = []
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

                # Bufferize the values for all the montages
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

                one_hot_vector_temp.append(f_stack)

            # For each recordings compose the hot vector
            # (for the features 'x_train' and the targets 'y_train')
            y_train = h5['dataset/' + filename]['targets']
            if n_added_padding:
                x_train = np.hstack(
                    one_hot_vector_temp,
                )[n_added_padding:-n_added_padding]
            else:
                x_train = np.hstack(one_hot_vector_temp)

            sampling_frequency = h5['dataset/' + filename][
                'sampling_frequency'
            ][0][0]

            # Assemption: sampling_frequency is constant
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
                [
                    m['MEAN'] for m in map(partial_extract, segments)
                ],
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
                            smoothing_windows,
                            smoothing_ratio,
                        ) if smoothing else (y_dewin > 0.5) * 1,
                        (y_train > threshold) * 1,
                    ),
                ),
            )

            global_y_true.append(y_train)
            global_y_pred.append(y_dewin)

        global_y_true = np.hstack(global_y_true)
        global_y_pred = np.hstack(global_y_pred)

        all_metrics.append(
            fe.generate_line(
                'train/',
                '*',
                fe.compute_metrics(
                    fe.binary_smoothing(
                        (global_y_pred > 0.5) * 1,
                        smoothing_windows,
                        smoothing_ratio,
                    ) if smoothing else (global_y_pred > 0.5) * 1,
                    (global_y_true > threshold) * 1,
                ),
            ),
        )

        print('Convert features and targets for the hot vector (dev).')
        global_y_true = []
        global_y_pred = []
        dev_metrics_lines = []

        for filename in tqdm.tqdm(dev_set_ar):
            one_hot_vector_temp = []
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

                # Bufferize the values for all the montages
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

                one_hot_vector_temp.append(f_stack)

            # For each recordings compose the hot vector
            # (for the features 'x_train' and the targets 'y_train')
            y_dev = h5['dataset/' + filename]['targets']
            if n_added_padding:
                x_dev = np.hstack(
                    one_hot_vector_temp,
                )[n_added_padding:-n_added_padding]
            else:
                x_dev = np.hstack(one_hot_vector_temp)

            sampling_frequency = h5['dataset/' + filename][
                'sampling_frequency'
            ][0][0]

            # Assemption: sampling_frequency is constant
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
                            smoothing_windows,
                            smoothing_ratio,
                        ) if smoothing else (y_dewin > 0.5) * 1,
                        (y_dev > threshold) * 1,
                    ),
                ),
            )

            global_y_true.append(y_dev)
            global_y_pred.append(y_dewin)

        global_y_true = np.hstack(global_y_true)
        global_y_pred = np.hstack(global_y_pred)

        all_metrics.append(
            fe.generate_line(
                'dev/', '*', fe.compute_metrics(
                    fe.binary_smoothing(
                        (global_y_pred > 0.5) * 1,
                        smoothing_windows,
                        smoothing_ratio,
                    ) if smoothing else (global_y_pred > 0.5) * 1,
                    (global_y_true > threshold) * 1,
                ),
            ),
        )

        columns = [
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

        df_dev = pd.DataFrame(dev_metrics_lines, columns=columns)
        df_train = pd.DataFrame(train_metrics_lines, columns=columns)
        df_all = pd.DataFrame(all_metrics, columns=columns)

        with pd.ExcelWriter(  # pylint: disable=abstract-class-instantiated
            xlsx_file,
            date_format='YYYY-MM-DD',
            datetime_format='YYYY-MM-DD HH:MM:SS',
        ) as writer:

            df_dev.to_excel(writer, sheet_name='dev')
            df_train.to_excel(writer, sheet_name='train')
            df_all.to_excel(writer, sheet_name='all')


if __name__ == '__main__':
    main()
