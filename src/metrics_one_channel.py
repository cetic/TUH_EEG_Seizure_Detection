import os
from functools import partial

import h5py
import numpy as np
import pandas as pd
import tqdm
import xgboost as xgb
from skimage.util.shape import view_as_windows

import tools.feature_extraction as fe

if __name__ == "__main__":
    h5_file = "/home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/TUH_v1_5_2.hdf5"
    model_file = "/home_nfs/stragierv/TUH_SZ_v1.5.2"
    model_file += "/Epilepsy_Seizure_Detection/models"
    model_file += "/single_chanels_initials.model"
    xlsx_file = "/home_nfs/stragierv/TUH_SZ_v1.5.2"
    xlsx_file += "/Epilepsy_Seizure_Detection/reports"
    xlsx_file += "/metrics"
    xlsx_file += "/metrics_single_channel_model_with_sliding_windows_rev.xlsx"
    feature_name = "/home_nfs/stragierv/TUH_SZ_v1.5.2"
    feature_name += "/Epilepsy_Seizure_Detection/one_hot_vectors/"
    feature_name += "columns_single_channel.npy"
    # Load the model
    bst = xgb.Booster(model_file=model_file)

    # Load X_dev and Y_dev for prediction.
    print("Load the data to run the predictions on.")

    threshold = 0.8

    with h5py.File(h5_file, "r") as h5:
        print("In hdf5 file.")
        # Load metadata from h5
        meta = fe.raw_numpy_to_object(
            np.array(h5["metadata"]),
        )

        meta_dict = fe.raw_numpy_to_object(
            np.array(h5["metadata_dict"]),
        )

        # Separate dev set and train set
        dev_set = [m for m in meta if "dev" in m["filepath"]]
        train_set = [m for m in meta if "train" in m["filepath"]]

        # Take only the recordings using the AR montage
        dev_set_ar = [
            os.path.basename(m["filepath"])
            for m in dev_set if "_ar/" in m["filepath"]
        ]

        train_set_ar = [
            os.path.basename(m["filepath"])
            for m in train_set if "_ar/" in m["filepath"]
        ]

        # List the features to use
        features = np.load(feature_name)
        # features = fe.ALL_FEATURES
        # features.remove("FIRST_DIFFERENTIAL")

        # List the montages to use
        montages = meta_dict[os.path.basename(
            dev_set_ar[0],
        )]["annotations_lbl"]["montages"]

        columns = [
            "{0} ({1})".format(f, m) for m in montages for f in features
        ]

        # X_train features in hot vector
        X_train, Y_train, X_dev, Y_dev = [], [], [], []

        # Generate the metrics
        print("Convert features and targets for the hot vector (train).")
        global_Y_true = []
        global_Y_pred = []
        all_metrics = []
        train_metrics_lines = []

        for filename in tqdm.tqdm(train_set_ar[:]):
            one_hot_vector_temp = []
            attrs = dict(h5["features/" + filename + "/" + features[0]].attrs)
            step = attrs['step']
            # Number of pads to remove
            n_pad = int(attrs['padding'] / step)

            for feature in features:
                # Key
                key = "features/{0}/{1}".format(filename, feature)

                # Get the indexing of the montage for the feature
                montage_index_dict = {
                    mntg.decode(): i for i, mntg
                    in enumerate(h5[key]["montage"])
                }

                # Bufferize the values for all the montages
                # (for one specific feature)
                feature_buffer = h5[key]["features"]

                f_stack = np.hstack(
                    [
                        feature_buffer[montage_index_dict[mntg]][n_pad:-n_pad]
                        for mntg in montages
                    ],
                )

                one_hot_vector_temp.append(f_stack)

            # For each recordings compose the hot vector
            # (for the features "X_train" and the targets "Y_train")
            Y_train = np.array(
                [
                    t for t, m
                    in zip(
                        np.concatenate(
                            h5["dataset/" + filename]["targets"],
                        ),
                        np.concatenate(
                            h5["dataset/" + filename]["montage"],
                        ),
                    ) if m.decode() in montages
                ],
            )

            X_train = np.vstack(one_hot_vector_temp).transpose()

            sampling_frequency = h5["dataset/" + filename][
                'sampling_frequency'
            ][0][0]

            # Assemption: sampling_frequency is constant
            step_in_samples = int(step * sampling_frequency)

            number_of_channels = len(montages)

            segments = view_as_windows(
                Y_train.reshape(
                    number_of_channels,
                    Y_train.shape[-1],
                ),
                (number_of_channels, step_in_samples),
                step_in_samples,
            )[0]

            # Preload the extraction function
            partial_extract = partial(
                fe.extract_features,
                sampling_rate=sampling_frequency,
                features_list=["MEAN"],
            )

            Y_train = np.hstack(
                [
                    m['MEAN'] for m in map(
                        partial_extract,
                        segments,
                    )
                ],
            )

            y_pred = bst.predict(xgb.DMatrix(X_train))

            # Dewindow y_pred
            # Each channel has to be dewindowned
            y_pred_copy = np.copy(y_pred)

            new_shape = list(y_pred.shape)
            n_sample_per_channel = int(
                new_shape[-1] / number_of_channels,
            )

            new_shape[-1] = new_shape[-1] + n_pad * number_of_channels
            n_sample_per_channel_dewin = int(
                new_shape[-1] / number_of_channels,
            )

            y_pred.resize(tuple(new_shape))

            for montage_index in range(number_of_channels):
                low_index = montage_index * n_sample_per_channel
                high_index = (montage_index + 1) * n_sample_per_channel
                low_index_dewin = montage_index * n_sample_per_channel_dewin
                high_index_dewin = (
                    montage_index + 1
                ) * n_sample_per_channel_dewin

                # Dewindow the y_pred.
                y_pred_ch = np.copy(y_pred_copy[low_index:high_index])
                new_shape = list(y_pred_ch.shape)
                new_shape[-1] = new_shape[-1] + n_pad
                y_pred_ch.resize(tuple(new_shape))
                y_dewin = np.copy(y_pred_ch)

                for index in range(n_pad):
                    y_dewin += np.roll(y_pred_ch, index + 1)

                for index in range(int(max(0, (n_pad - 1)))):
                    y_dewin[index + 1] /= (index + 2)
                    y_dewin[- (index + 2)] /= (index + 2)

                y_dewin[n_pad:-n_pad] /= (n_pad + 1)
                y_pred[low_index_dewin:high_index_dewin] = y_dewin

            y_pred = y_pred.reshape(
                number_of_channels,
                n_sample_per_channel_dewin,
            ).max(axis=0)

            Y_train = Y_train.reshape(
                number_of_channels,
                n_sample_per_channel_dewin,
            ).max(axis=0)

            train_metrics_lines.append(
                fe.generate_line(
                    meta_dict[filename]['filepath'],
                    filename,
                    fe.compute_metrics(
                        (y_pred > 0.5) * 1,
                        (Y_train > threshold) * 1,
                    ),
                ),
            )

            global_Y_true.append(Y_train)
            global_Y_pred.append(y_pred)

        global_Y_true = np.hstack(global_Y_true)
        global_Y_pred = np.hstack(global_Y_pred)

        all_metrics.append(
            fe.generate_line(
                "train/",
                "*",
                fe.compute_metrics(
                    (global_Y_pred > 0.5) * 1,
                    (global_Y_true > threshold) * 1,
                ),
            ),
        )

        print("Convert features and targets for the hot vector (dev).")
        global_Y_true = []
        global_Y_pred = []
        dev_metrics_lines = []

        for filename in tqdm.tqdm(dev_set_ar):
            one_hot_vector_temp = []
            attrs = dict(h5["features/" + filename + "/" + features[0]].attrs)
            step = attrs['step']
            # Number of pads to remove
            n_pad = int(attrs['padding'] / step)

            for feature in features:
                # Get the indexing of the montage for the feature
                montage_index_dict =\
                    {mntg.decode(): i for i, mntg
                     in enumerate(h5[
                         "features/" + filename + "/" + feature]["montage"])}

                # Bufferize the values for all the montages
                # (for one specific feature)
                feature_buffer = h5[
                    "features/" + filename + "/" + feature
                ]["features"]

                f_stack = np.hstack(
                    [
                        feature_buffer[montage_index_dict[mntg]][n_pad:-n_pad]
                        for mntg in montages
                    ],
                ).transpose()

                one_hot_vector_temp.append(f_stack)

            # For each recordings compose the hot vector
            # (for the features "X_dev" and the targets "Y_dev")
            Y_dev = np.array(
                [
                    t for t, m in zip(
                        np.concatenate(h5["dataset/" + filename]["targets"]),
                        np.concatenate(h5["dataset/" + filename]["montage"]),
                    ) if m.decode() in montages
                ],
            )

            X_dev = np.vstack(one_hot_vector_temp).transpose()

            sampling_frequency = h5["dataset/" + filename][
                'sampling_frequency'
            ][0][0]

            # Assemption: sampling_frequency is constant
            step_in_samples = int(step * sampling_frequency)

            number_of_channels = len(montages)

            segments = view_as_windows(
                Y_dev.reshape(
                    number_of_channels,
                    Y_dev.shape[-1]),
                (number_of_channels, step_in_samples),
                step_in_samples,
            )[0]

            # Preload the extraction function
            partial_extract = partial(
                fe.extract_features,
                sampling_rate=sampling_frequency,
                features_list=["MEAN"],
            )

            Y_dev = np.hstack(
                [m['MEAN'] for m in map(partial_extract, segments)],
            )

            # Predict
            y_pred = bst.predict(xgb.DMatrix(X_dev))

            # Dewindow y_pred
            # Each channel has to be dewindowned
            y_pred_copy = np.copy(y_pred)

            new_shape = list(y_pred.shape)
            n_sample_per_channel = int(new_shape[-1] / number_of_channels)
            new_shape[-1] = new_shape[-1] + n_pad * number_of_channels
            n_sample_per_channel_dewin = int(
                new_shape[-1] / number_of_channels,
            )
            y_pred.resize(tuple(new_shape))

            for montage_index in range(number_of_channels):
                low_index = montage_index * n_sample_per_channel
                high_index = (montage_index + 1) * n_sample_per_channel
                low_index_dewin = montage_index * n_sample_per_channel_dewin
                high_index_dewin = (
                    montage_index + 1
                ) * n_sample_per_channel_dewin

                # Dewindow the y_pred.
                y_pred_ch = np.copy(y_pred_copy[low_index:high_index])
                new_shape = list(y_pred_ch.shape)
                new_shape[-1] = new_shape[-1] + n_pad
                y_pred_ch.resize(tuple(new_shape))
                y_dewin = np.copy(y_pred_ch)

                for index in range(n_pad):
                    y_dewin += np.roll(y_pred_ch, index + 1)

                for index in range(int(max(0, (n_pad - 1)))):
                    y_dewin[index + 1] /= (index + 2)
                    y_dewin[- (index + 2)] /= (index + 2)

                y_dewin[n_pad:-n_pad] /= (n_pad + 1)

                y_pred[low_index_dewin:high_index_dewin] = y_dewin

            y_pred = y_pred.reshape(
                number_of_channels,
                n_sample_per_channel_dewin,
            ).max(axis=0)

            Y_dev = Y_dev.reshape(
                number_of_channels,
                n_sample_per_channel_dewin,
            ).max(axis=0)

            dev_metrics_lines.append(
                fe.generate_line(
                    meta_dict[filename]['filepath'],
                    filename,
                    fe.compute_metrics(
                        (y_pred > 0.5) * 1,
                        (Y_dev > threshold) * 1,
                    ),
                ),
            )

            global_Y_true.append(Y_dev)
            global_Y_pred.append(y_pred)

        global_Y_true = np.hstack(global_Y_true)
        global_Y_pred = np.hstack(global_Y_pred)

        all_metrics.append(
            fe.generate_line(
                "dev/",
                "*",
                fe.compute_metrics(
                    (global_Y_pred > 0.5) * 1,
                    (global_Y_true > threshold) * 1,
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
            'Negative predictive value',
            'Miss rate',
            'Fall-rate',
            'False discovery rate',
            'False omission rate',
            'Prevalence threshold',
            'Threat score',
            'Accuracy',
            'Balanced accuracy',
            'F1 score',
            'Matthews correlation coefficient',
            'Fowlkes–Mallows index',
            'Bookmaker informedness',
            'Markedness']

        df_dev = pd.DataFrame(
            dev_metrics_lines,
            columns=columns,
        )

        df_train = pd.DataFrame(
            train_metrics_lines,
            columns=columns,
        )

        df_all = pd.DataFrame(
            all_metrics,
            columns=columns,
        )

        with pd.ExcelWriter(  # pylint: disable=abstract-class-instantiated
            xlsx_file,
            date_format='YYYY-MM-DD',
            datetime_format='YYYY-MM-DD HH:MM:SS',
        ) as writer:

            df_dev.to_excel(writer, sheet_name='dev')
            df_train.to_excel(writer, sheet_name='train')
            df_all.to_excel(writer, sheet_name='all')
