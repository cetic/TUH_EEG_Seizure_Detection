import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode

import tuh_sz_extract_metadata as mt

EPILEPTIC_SEIZURE_LABELS = ['fnsz',  # Focal nonspecific seizure.
                            'gnsz',  # Generalized seizure.
                            'spsz',  # Simple partial seizure.
                            'cpsz',  # Complex partial seizure.
                            'absz',  # Absence seizure.
                            'tnsz',  # Tonic seizure.
                            'cnsz',  # Clonic seizure.
                            'tcsz',  # Tonic-clonic seizure.
                            'atsz',  # Atonic seizure.
                            'mysz']  # Myoclonic seizure.

#                                  Focal nonspecific seizure.
EPILEPTIC_SEIZURE_LABELS_DICT = {'fnsz': "focal nonspecific",
                                 # Generalized seizure.
                                 'gnsz': "generalised",
                                 # Simple partial seizure.
                                 'spsz': "simple partial",
                                 # Complex partial seizure.
                                 'cpsz': "complex partial",
                                 # Absence seizure.
                                 'absz': "absence",
                                 # Tonic seizure.
                                 'tnsz': "tonic",
                                 # Clonic seizure.
                                 'cnsz': "clonic",
                                 # Tonic-clonic seizure.
                                 'tcsz': "tonic-clonic",
                                 # Atonic seizure.
                                 'atsz': "atonic",
                                 # Myoclonic seizure.
                                 'mysz': "myoclonic"}

REPORT = False
HISTOGRAMS = False
FNSZ_ORIGIN = False
PIE = True


if __name__ == "__main__":
    meta = mt.load_pickle_lzma(
        os.path.join(
            os.path.dirname(__file__),
            'metadata.pickle',
        ),
    )

    dev_seizures_durations_by_type = {}
    train_seizures_durations_by_type = {}

    # Dev set metadata
    # Extract dev metadata subset
    dev = [e for e in meta if 'dev' in e['filepath']]

    if REPORT:
        # Number of recordings
        dev_number_of_recording = len(dev)
        print("Dev set, number of recordings:", dev_number_of_recording)

        # Total duration of the recordings
        dev_total_duration = sum([e['duration [s]'] for e in dev])
        print(
            "Dev set, total duration:",
            mt.seconds_to_human_readble_time(seconds=dev_total_duration),
        )

        # Total duration of the calibration
        dev_total_calibration_duration = sum(
            [
                e['calibration']['stop'] - e['calibration']['start']
                for e in dev if e['calibration'] is not None
            ],
        )

        print(
            "Dev set, total calibration time:",
            mt.seconds_to_human_readble_time(
                seconds=dev_total_calibration_duration,
            ),
        )

        # Usage portion of the dev set:
        dev_usable_duration = dev_total_duration - sum(
            [
                e['calibration']['stop'] for e
                in dev if e['calibration'] is not None
            ],
        )

        print(
            "Dev set, usable duration:",
            mt.seconds_to_human_readble_time(
                seconds=dev_usable_duration,
            ),
            "\n",
        )

        for seizure_type in EPILEPTIC_SEIZURE_LABELS:
            seizure_type_number = [
                True for element in dev if len(
                    [
                        e['event'] for e in element['annotations_tse']
                        if e['event'] == seizure_type
                    ],
                ) != 0
            ].count(True)

            print(
                "Number of recordings presenting '{0}' as seizure type:"
                "                     ".format(
                    seizure_type,
                ),
                seizure_type_number,
            )

            if seizure_type_number > 0:
                seizure_type_durations_per_event = np.hstack(
                    [
                        [
                            e['stop'] - e['start']
                            for e in element['annotations_tse']
                            if e['event'] == seizure_type
                        ] for element in dev
                    ],
                )

                dev_seizures_durations_by_type[seizure_type] =\
                    seizure_type_durations_per_event

                print(
                    "Number of '{0}':                                   "
                    .format(seizure_type),
                    len(list(seizure_type_durations_per_event)),
                )

                print(
                    "Mean duration [s] of '{0}':                        "
                    .format(seizure_type),
                    seizure_type_durations_per_event.mean(),
                )

                print(
                    "Mode duration [s] of '{0}':                        "
                    .format(seizure_type),
                    mode(seizure_type_durations_per_event)[0][0],
                )

                print(
                    "Median duration [s] of '{0}':                      "
                    .format(seizure_type),
                    np.median(seizure_type_durations_per_event),
                )

                print(
                    "Min duration [s] of '{0}':                         "
                    .format(seizure_type),
                    seizure_type_durations_per_event.min(),
                )

                print(
                    "Max duration [s] of '{0}':                         "
                    .format(seizure_type),
                    seizure_type_durations_per_event.max(),
                )

                print(
                    "Standard deviation of the duration in [s] of '{0}':"
                    .format(seizure_type),
                    seizure_type_durations_per_event.std(),
                )

            print()

        print("\n\n")

    # Train set metadata
    train = [e for e in meta if 'train' in e['filepath']]

    if REPORT:
        # Number or recordings
        train_number_of_recording = len(train)
        print("Train set, number of recordings:", train_number_of_recording)

        # Total duration of the recordings
        train_total_duration = sum([e['duration [s]'] for e in train])
        print(
            "Train set total duration:",
            mt.seconds_to_human_readble_time(seconds=train_total_duration),
        )

        # Total duration of the calibration in the recordings
        train_total_calibration_duration = sum(
            [
                e['calibration']['stop'] - e['calibration']['start']
                for e in train if e['calibration'] is not None
            ],
        )

        print(
            "Train set, total calibration time:",
            mt.seconds_to_human_readble_time(
                seconds=train_total_calibration_duration,
            ),
        )

        # Usage portion of the train set:
        train_usable_duration = train_total_duration - sum(
            [
                e['calibration']['stop'] for e in train
                if e['calibration'] is not None
            ],
        )

        print(
            "Train set, usable duration:",
            mt.seconds_to_human_readble_time(seconds=train_usable_duration),
            "\n",
        )

        for seizure_type in EPILEPTIC_SEIZURE_LABELS:
            seizure_type_number = [
                True for element in train if len(
                    [
                        e['event'] for e in element['annotations_tse']
                        if e['event'] == seizure_type
                    ],
                ) != 0
            ].count(True)

            print(
                "Number of recordings presenting '{0}' as seizure type:"
                "                       ".format(seizure_type),
                seizure_type_number,
            )

            if seizure_type_number > 0:
                seizure_type_durations_per_event = np.hstack(
                    [
                        [
                            e['stop'] - e['start'] for e
                            in element['annotations_tse']
                            if e['event'] == seizure_type
                        ] for element in train
                    ],
                )

                train_seizures_durations_by_type[seizure_type] =\
                    seizure_type_durations_per_event

                print(
                    "Number of '{0}' seizures in the train set:"
                    "                                   "
                    .format(seizure_type),
                    len(list(seizure_type_durations_per_event)),
                )

                print(
                    "Mean duration [s] of '{0}':                        "
                    .format(seizure_type),
                    seizure_type_durations_per_event.mean(),
                )

                print(
                    "Mode duration [s] of '{0}':                        "
                    .format(seizure_type),
                    mode(seizure_type_durations_per_event)[0][0],
                )

                print(
                    "Median duration [s] of '{0}':                      "
                    .format(seizure_type),
                    np.median(seizure_type_durations_per_event),
                )

                print(
                    "Min duration [s] of '{0}':                         "
                    .format(seizure_type),
                    seizure_type_durations_per_event.min(),
                )

                print(
                    "Max duration [s] of '{0}':                         "
                    .format(seizure_type),
                    seizure_type_durations_per_event.max(),
                )

                print(
                    "Standard deviation of the duration in [s] of '{0}':"
                    .format(seizure_type),
                    seizure_type_durations_per_event.std(),
                )

            print()

        plt.figure(
            "Seizures duration histograms by types"
            " (on top, dev set data, on the bottom, train set data):",
        )

        nr = 2
        labels_in_the_dataset = sorted(
            list(
                set(
                    list(
                        dev_seizures_durations_by_type.keys(),
                    ) + list(
                        train_seizures_durations_by_type.keys(),
                    ),
                ),
            ),
        )  # [1:4]

        nc = len(labels_in_the_dataset)

        for i, seizure_type in enumerate(labels_in_the_dataset, start=1):
            if seizure_type in dev_seizures_durations_by_type.keys():
                plt.subplot(nr, nc, i)
                _ = plt.hist(
                    dev_seizures_durations_by_type[seizure_type],
                    bins='auto',
                )

                plt.title(seizure_type)

            if seizure_type in train_seizures_durations_by_type.keys():
                plt.subplot(nr, nc, i + nc)
                _ = plt.hist(
                    train_seizures_durations_by_type[seizure_type],
                    bins='auto',
                )

                plt.title(seizure_type)

        print("Dev abnormal values 'gnsz'")
        d = [
            [
                e['stop'] - e['start']
                for e in element['annotations_tse']
                if e['event'] == 'gnsz'
            ] for element in dev
        ]

        for i1, e1 in enumerate(d):
            for i2, e2 in enumerate(e1):
                if e2 > 200:
                    print(dev[i1]['filepath'])
                    print(i1, e2)

        print("Train abnormal values 'gnsz'")
        d = [
            [
                e['stop'] - e['start']
                for e in element['annotations_tse']
                if e['event'] == 'gnsz'
            ] for element in train
        ]

        for i1, e1 in enumerate(d):
            for i2, e2 in enumerate(e1):
                if e2 > 200:
                    print(train[i1]['filepath'])
                    print(i1, e2)

    if HISTOGRAMS:
        for seizure_type in EPILEPTIC_SEIZURE_LABELS:
            dev_seizure_type_number = [
                True for element in dev if len(
                    [
                        e['event'] for e in element['annotations_tse']
                        if e['event'] == seizure_type
                    ],
                ) != 0
            ].count(True)

            train_seizure_type_number = [
                True for element in train if len(
                    [
                        e['event'] for e in element['annotations_tse']
                        if e['event'] == seizure_type
                    ],
                ) != 0
            ].count(True)

            if dev_seizure_type_number > 0:
                seizure_type_durations_per_event = np.hstack(
                    [
                        [
                            e['stop'] - e['start'] for e
                            in element['annotations_tse']
                            if e['event'] == seizure_type
                        ] for element in dev
                    ],
                )

                dev_seizures_durations_by_type[seizure_type] =\
                    seizure_type_durations_per_event

            if train_seizure_type_number > 0:
                seizure_type_durations_per_event = np.hstack(
                    [
                        [
                            e['stop'] - e['start'] for e
                            in element['annotations_tse']
                            if e['event'] == seizure_type
                        ] for element in train
                    ],
                )

                train_seizures_durations_by_type[seizure_type] =\
                    seizure_type_durations_per_event

        labels_in_the_dataset = sorted(
            list(
                set(
                    list(
                        dev_seizures_durations_by_type.keys(),
                    ) + list(
                        train_seizures_durations_by_type.keys(),
                    ),
                ),
            ),
        )

        # Used later to remove extrema
        dev_seizures_durations_by_type_filtered = {}
        train_seizures_durations_by_type_filtered = {}

        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages('Histograms.pdf')
        for seizure_type in labels_in_the_dataset:
            plt.figure(
                "Histograms for {0} seizures ('{1}')".format(
                    EPILEPTIC_SEIZURE_LABELS_DICT[seizure_type],
                    seizure_type,
                ),
                figsize=(16/2.54*2, 9/2.54*2),
            )

            plt.suptitle(
                "Histograms for {0} seizures ('{1}')".format(
                    EPILEPTIC_SEIZURE_LABELS_DICT[seizure_type],
                    seizure_type,
                ),
                fontsize=15,
            )

            data = dev_seizures_durations_by_type[seizure_type]
            mu = data.mean()
            median = np.median(data)
            sigma = data.std()
            minimum = data.min()
            maximum = data.max()
            plt.subplot(2, 2, 1)
            counts, bins, patches = plt.hist(data, bins='auto')
            i = np.argmax(counts)
            hist_mode = (bins[i] + bins[i + 1]) / 2

            plt.ylabel(r'Count per bin')
            plt.legend(
                [
                    r'$\mu={0:.4f}$, $\sigma={1:.4f}$,{7}min$={2:.4f}$,'
                    r' max$={6:.4f}$,{7}median$={4:.4f}$, mode$={5:.4f}$,'
                    r'{7}Number of seizures: {3}'.format(
                        mu,
                        sigma,
                        minimum,
                        len(data),
                        median,
                        hist_mode,
                        maximum,
                        "\n",
                    ),
                ],
            )

            plt.xlabel(r'Time in seconds')
            plt.title(r'Dev set, unfiltered data')

            lower_bound = mu - 3 * sigma
            upper_bound = mu + 3 * sigma
            condition = data[data >= lower_bound] <= upper_bound
            dev_seizures_durations_by_type_filtered[seizure_type] =\
                data[condition]
            data = dev_seizures_durations_by_type_filtered[seizure_type]
            mu = data.mean()
            median = np.median(data)
            sigma = data.std()
            minimum = data.min()
            maximum = data.max()
            # hist_mode = mode(np.round(data, decimals=0))[0][0]
            plt.subplot(2, 2, 2)
            counts, bins, patches = plt.hist(data, bins='auto')
            i = np.argmax(counts)
            hist_mode = (bins[i] + bins[i + 1])/2

            # add a 'best fit' line
            # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
            # -0.5 * (1 / sigma * (bins - mu))**2))
            # plt.plot(bins, y, '--')
            plt.ylabel(r'Count per bin')
            plt.legend(
                [
                    r'$\mu={0:.4f}$, $\sigma={1:.4f}$,{7}min$={2:.4f}$,'
                    r' max$={6:.4f}$,{7}median$={4:.4f}$, mode$={5:.4f}$,'
                    r'{7}Number of seizures: {3}'.format(
                        mu,
                        sigma,
                        minimum,
                        len(data),
                        median,
                        hist_mode,
                        maximum,
                        "\n",
                    ),
                ],
            )

            plt.xlabel(r'Time in seconds')
            plt.title(
                r'Dev set, filtered data'
                r' ($\mu - 3 \sigma \leq X \leq \mu + 3 \sigma$)',
            )

            data = train_seizures_durations_by_type[seizure_type]
            mu = data.mean()
            median = np.median(data)
            sigma = data.std()
            minimum = data.min()
            maximum = data.max()
            # hist_mode = mode(np.round(data, decimals=0))[0][0]
            plt.subplot(2, 2, 3)
            counts, bins, patches = plt.hist(data, bins='auto')
            i = np.argmax(counts)
            hist_mode = (bins[i] + bins[i + 1])/2

            # add a 'best fit' line
            # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
            # -0.5 * (1 / sigma * (bins - mu))**2))
            # plt.plot(bins, y, '--')
            plt.ylabel(r'Count per bin')
            plt.legend(
                [
                    r'$\mu={0:.4f}$, $\sigma={1:.4f}$,'
                    r'{7}min$={2:.4f}$, max$={6:.4f}$,'
                    r'{7}median$={4:.4f}$, mode$={5:.4f}$,'
                    r'{7}Number of seizures: {3}'.format(
                        mu,
                        sigma,
                        minimum,
                        len(data),
                        median,
                        hist_mode,
                        maximum,
                        "\n",
                    ),
                ],
            )

            plt.xlabel(r'Time in seconds')
            plt.title(r'Train set, unfiltered data')

            lower_bound = mu - 3 * sigma
            upper_bound = mu + 3 * sigma
            condition = data[data >= lower_bound] <= upper_bound
            train_seizures_durations_by_type_filtered[seizure_type] =\
                data[condition]
            data = train_seizures_durations_by_type_filtered[seizure_type]
            mu = data.mean()
            median = np.median(data)
            sigma = data.std()
            minimum = data.min()
            maximum = data.max()
            # hist_mode = mode(np.round(data, decimals=0))[0][0]
            plt.subplot(2, 2, 4)
            counts, bins, patches = plt.hist(data, bins='auto')
            i = np.argmax(counts)
            hist_mode = (bins[i] + bins[i + 1])/2

            # add a 'best fit' line
            # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
            # -0.5 * (1 / sigma * (bins - mu))**2))
            # plt.plot(bins, y, '--')
            plt.ylabel(r'Count per bin')
            plt.legend(
                [
                    r'$\mu={0:.4f}$, $\sigma={1:.4f}$,'
                    r'{7}min$={2:.4f}$, max$={6:.4f}$,'
                    r'{7}median$={4:.4f}$, mode$={5:.4f}$,'
                    r'{7}Number of seizures: {3}'.format(
                        mu,
                        sigma,
                        minimum,
                        len(data),
                        median,
                        hist_mode,
                        maximum,
                        "\n",
                    ),
                ],
            )

            plt.xlabel(r'Time in seconds')
            plt.title(
                r'Train set, filtered data'
                r' ($\mu - 3 \sigma \leq X \leq \mu + 3 \sigma$)',
            )

            # tight_layout docs: [left, bottom, right, top]
            # in normalized (0, 1) figure
            plt.tight_layout(
                rect=[0, 0.03, 1, 0.95],
            )

            plt.show(block=False)
            pp.savefig()
        pp.close()
        input("Press any key to exit histogram generation")

    if FNSZ_ORIGIN:
        seizure_types = ['fnsz', 'cpsz', 'spsz']
        # Filters the fnsz only
        dev_filter_f = [
            True if len(
                [
                    e['event'] for e
                    in element['annotations_tse']
                    if e['event'] in seizure_types
                ],
            ) != 0 else False for element in dev
        ]

        train_filter_f = [
            True if len(
                [
                    e['event'] for e
                    in element['annotations_tse']
                    if e['event'] in seizure_types
                ],
            ) != 0 else False for element in train
        ]

        # Contains all the recordings that contains
        # at least one focal seizure of any kind
        dev_f = np.array(dev)[dev_filter_f]
        train_f = np.array(train)[train_filter_f]

        # seizure_types.append()

        # List of number of seizure per recordings containing seizure
        # l = [len([e['event'] for e in element['annotations_tse']
        # if e['event'] in EPILEPTIC_SEIZURE_LABELS])
        # for element in meta if len([e['event']
        # for e in element['annotations_tse']
        # if e['event'] in EPILEPTIC_SEIZURE_LABELS]) != 0]
        # sum(l) = 3050 seizures for all the recording

        # set               recordings with seizure  number of seizure events
        # number of ['fnsz', 'bckg', 'gnsz'] events sequence
        # meta (dev+train)  1149                     3050          20
        # dev               280                      673           4
        # train             869                      2377          6

        for recording in dev_f[5:]:
            """
            for event_tse in focal_events_list_tse:
                for event_lbl in focal_events_list_lbl:
                    if event_lbl['start'] == event_tse['start']:
                        print(event_lbl['montage'])
                print()
            """
            # Remove background event
            # (better to keep only the labels of interest)
            # [e for e in events_list if e['event'] in seizure_types]
            # fnsz = [e for e in events_list if e['event'] == 'spsz']
            # for element in sorted(fnsz, key=lambda e:e['start']):
            print(mt.focal_starting_points(recording))

            break

        # Intersection focal seizure/generalised seizure
        # dev_f_and_gnsz = [e for e in dev_f if len(
        # [i for i in e['annotations_tse'] if i['event'] == 'gnsz'])>0]
        # sum(["".join([e['event'] for e in r['annotations_tse']]).count(
        # 'fnszbckggnsz') for r in dev_f_and_gnsz])

        # Recordings with seizures
        # [True for element in meta if len([e['event'] for e
        #  in element['annotations_tse'] if e['event']
        #  in EPILEPTIC_SEIZURE_LABELS]) != 0].count(True)
        # Out[1]: 1149

        # [True for element in dev if len([e['event'] for e
        #  in element['annotations_tse'] if e['event']
        #  in EPILEPTIC_SEIZURE_LABELS]) != 0].count(True)
        # Out[2]: 280

        # [True for element in train if len([e['event'] for e
        #  in element['annotations_tse'] if e['event']
        #  in EPILEPTIC_SEIZURE_LABELS]) != 0].count(True)
        # Out[3]: 869

        # e['annotations_lbl']['level'][0]
        # df[0]['annotations_lbl']['labels'][20]['montage']

        # The hierachical annotation is only on one level and one sublevel
        # for i, e in enumerate(meta): # Do not show anything
        #   if e['annotations_lbl']['level'][0] != 1:
        #       print(e['annotations_lbl']['level'][0], end=' ')

        # def labels_to_events(e):
        #   symbols = e['annotations_lbl']['symbols'][0]
        #   montages = e['annotations_lbl']['montages']
        #   # for each label extract start, stop, montage, symbol
        #   labels = list() # list of dictionaries
        #   for l in e['annotations_lbl']['labels']:
        #       labels.append({'start': l['start'],
        #                      'stop': l['stop'],
        #                      'montage': montages[l['montage']],
        #                      'event':symbols[np.argmax(l['probabilities'])]})
        #   return labels

        # Remove background event
        # (better to keep only the labels of interest)
        # [e for e in l if e['event'] != 'bckg']
        # fnsz = [e for e in l if e['event'] == 'fnsz']
        # sorted(fnsz, key=lambda e:e['start'])

        # In [66]: for e in df:
        # ...:     if 'gnsz' in [i['event'] for i in e['annotations_tse']]:
        # ...:         print([i['event'] for i in e['annotations_tse']])
        # ...:
        # ['bckg', 'fnsz', 'bckg', 'fnsz', 'bckg', 'fnsz', 'bckg', 'fnsz',
        #  'bckg', 'gnsz', 'bckg', 'fnsz', 'bckg', 'gnsz', 'bckg']
        # ['bckg', 'gnsz', 'bckg', 'fnsz', 'bckg', 'fnsz', 'bckg',
        #  'fnsz', 'bckg']
        # ['bckg', 'fnsz', 'bckg', 'fnsz', 'bckg', 'gnsz', 'bckg']
        # ['bckg', 'gnsz', 'bckg', 'spsz', 'bckg', 'gnsz', 'bckg',
        #  'fnsz', 'bckg']
        # ['bckg', 'spsz', 'bckg', 'gnsz', 'bckg', 'fnsz', 'bckg',
        #  'gnsz', 'bckg']
        # ['bckg', 'gnsz', 'bckg', 'gnsz', 'bckg', 'gnsz', 'bckg', 'gnsz',
        #  'bckg', 'gnsz', 'bckg', 'gnsz', 'bckg', 'gnsz', 'bckg', 'gnsz',
        #  'bckg', 'fnsz', 'bckg']

        # for e in df:
        #   if 'gnsz' in [i['event'] for i in e['annotations_tse']]:
        #       print([i['event'] for i in e['annotations_tse']],
        #             e['filepath'][:e['filepath'].rfind('/')])

        # ['bckg', 'fnsz', 'bckg', 'fnsz', 'bckg', 'fnsz', 'bckg', 'fnsz',
        #  'bckg', 'gnsz', 'bckg', 'fnsz', 'bckg', 'gnsz', 'bckg']
        #  /home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/edf/dev/01_tcp_ar/065/00006546/s018_2012_01_30
        # ['bckg', 'gnsz', 'bckg', 'fnsz', 'bckg', 'fnsz', 'bckg', 'fnsz',
        #  'bckg']
        # /home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/edf/dev/01_tcp_ar/065/00006546/s020_2012_01_30
        # ['bckg', 'fnsz', 'bckg', 'fnsz', 'bckg', 'gnsz', 'bckg']
        # /home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/edf/dev/01_tcp_ar/065/00006546/s022_2012_02_23
        # ['bckg', 'gnsz', 'bckg', 'spsz', 'bckg', 'gnsz', 'bckg',
        #  'fnsz', 'bckg']
        # /home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/edf/dev/01_tcp_ar/065/00006546/s024_2012_02_25
        # ['bckg', 'spsz', 'bckg', 'gnsz', 'bckg', 'fnsz', 'bckg',
        #  'gnsz', 'bckg']
        # /home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/edf/dev/01_tcp_ar/065/00006546/s024_2012_02_25
        # ['bckg', 'gnsz', 'bckg', 'gnsz', 'bckg', 'gnsz', 'bckg',
        #  'gnsz', 'bckg', 'gnsz', 'bckg', 'gnsz', 'bckg', 'gnsz',
        #  'bckg', 'gnsz', 'bckg', 'fnsz', 'bckg']
        # /home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/edf/dev/01_tcp_ar/085/00008512/s007_2012_07_03
        """
        dev_seizure_type_number = [True for element in dev if len(
            [e['event'] for e in element['annotations_tse']
            if e['event'] == seizure_type]) != 0].count(True)
        train_seizure_type_number = [True for element in train if len(
            [e['event'] for e in element['annotations_tse']
             if e['event'] == seizure_type]) != 0].count(True)

        if dev_seizure_type_number > 0:
            seizure_type_durations_per_event = np.hstack(
                [[e['stop'] - e['start'] for e
                  in element['annotations_tse']
                  if e['event'] == seizure_type] for element in dev])
            dev_seizures_durations_by_type[seizure_type] =\
                seizure_type_durations_per_event

        if train_seizure_type_number > 0:
            seizure_type_durations_per_event = np.hstack(
                [[e['stop'] - e['start'] for e
                  in element['annotations_tse']
                  if e['event'] == seizure_type]
                 for element in train])
            train_seizures_durations_by_type[seizure_type] =\
                seizure_type_durations_per_event
        """
    if PIE:
        dev_seizures_number_by_type = dict()
        train_seizures_number_by_type = dict()

        for seizure_type in EPILEPTIC_SEIZURE_LABELS:
            dev_seizure_type_number = [
                True for element in dev if len(
                    [
                        e['event'] for e in element['annotations_tse']
                        if e['event'] == seizure_type
                    ],
                ) != 0
            ].count(True)

            train_seizure_type_number = [
                True for element in train if len(
                    [
                        e['event'] for e in element['annotations_tse']
                        if e['event'] == seizure_type
                    ],
                ) != 0
            ].count(True)

            if dev_seizure_type_number > 0:
                seizure_type_durations_per_event = np.hstack(
                    [
                        [
                            e['stop'] - e['start'] for e
                            in element['annotations_tse']
                            if e['event'] == seizure_type
                        ] for element in dev
                    ],
                )

                dev_seizures_number_by_type[seizure_type] = len(
                    seizure_type_durations_per_event,
                )

            if train_seizure_type_number > 0:
                seizure_type_durations_per_event = np.hstack(
                    [
                        [
                            e['stop'] - e['start'] for e
                            in element['annotations_tse']
                            if e['event'] == seizure_type
                        ] for element in train
                    ],
                )

                train_seizures_number_by_type[seizure_type] = len(
                    seizure_type_durations_per_event,
                )

        events_counts_dev = dev_seizures_number_by_type
        events_counts_train = train_seizures_number_by_type

        print(dev_seizures_number_by_type.keys())
        print(train_seizures_number_by_type.keys())

        fig, ax = plt.subplots(
            figsize=(16, 9),
            subplot_kw=dict(aspect="equal"))

        # labels = list(events_counts_dev.keys())
        labels = [
            'FNSZ',
            'CPSZ',
            'SPSZ',
            'GNSZ',
            'TNSZ',
            'TCSZ',
            'ABSZ',
            'MYSZ',
        ]

        data = [events_counts_dev[event.lower()] for event in labels]

        # See https://matplotlib.org/3.3.2/tutorials/colors/colormaps.html
        ax.set_prop_cycle(
            'color',
            plt.cm.get_cmap('Paired')(np.linspace(0, 1, len(data) + 1)),
        )

        def func(pct, allvals):
            # absolute = int(pct/100.*np.sum(allvals))
            return "{:.1f}%".format(pct)

        wedges, texts, autotexts = ax.pie(
            data,
            autopct=lambda pct: func(pct, data),
            textprops=dict(color="black"),
            pctdistance=0.85,
            startangle=90,
            counterclock=False,
            rotatelabels=True,
        )

        ax.legend(
            wedges,
            [
                element[0] +
                element[1] +
                str(element[2]) +
                element[3]
                for element in zip(
                    labels,
                    [' ('] * len(data),
                    data, [')'] * len(data),
                )
            ],
            title="Seizure type",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))

        plt.setp(autotexts, size=12, weight="bold")

        # Create a circle for the center of the plot
        plt.gcf().gca().add_artist(plt.Circle((0, 0), 0.7, color='white'))

        ax.set_title("Seizures by type (event wise) in dev")
        plt.tight_layout()
        plt.show(block=False)

        fig, ax = plt.subplots(
            figsize=(16, 9),
            subplot_kw=dict(aspect="equal"))

        # labels = list(events_counts_train.keys())
        labels = [
            'FNSZ',
            'CPSZ',
            'SPSZ',
            'GNSZ',
            'TNSZ',
            'TCSZ',
            'ABSZ',
            'MYSZ',
        ]

        data = [events_counts_train[event.lower()] for event in labels]

        # See https://matplotlib.org/3.3.2/tutorials/colors/colormaps.html
        ax.set_prop_cycle(
            'color',
            plt.cm.get_cmap('Paired')(np.linspace(0, 1, len(data) + 1)),
        )

        wedges, texts, autotexts = ax.pie(
            data,
            autopct=lambda pct: func(pct, data),
            textprops=dict(color="black"),
            pctdistance=1.1,
            startangle=90,
            counterclock=False,
            rotatelabels=True,
        )

        percents = data/np.sum(data) * 100
        ax.legend(
            wedges,
            [
                element[0] +
                element[1] +
                str(element[2]) +
                element[3] +
                "{:.1f}%".format(
                    element[4],
                ) for element in zip(
                    labels,
                    [' ('] * len(data),
                    data, [') - '] * len(data),
                    percents,
                )
            ],
            title="Seizure type",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
        )

        plt.setp(autotexts, size=12, weight="bold")

        # Create a circle for the center of the plot
        plt.gcf().gca().add_artist(plt.Circle((0, 0), 0.7, color='white'))

        ax.set_title("Seizures by type (event wise) in train")
        plt.tight_layout()
        plt.show()
