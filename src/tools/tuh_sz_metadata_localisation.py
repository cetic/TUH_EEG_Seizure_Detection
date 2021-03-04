import datetime
import os

import matplotlib.pyplot as plt
import numpy as np

import tuh_sz_extract_metadata as mt


def seconds_to_human_readble_time(seconds):
    """Converts a number of seconds to a readable string.

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
            - ``symbols[np.argmax(l['probabilities'])]``
              is the label of the most probable event

    # noqa: RST301
    """
    symbols = recording_meta['annotations_lbl']['symbols'][0]
    montages = recording_meta['annotations_lbl']['montages']
    # for each label extract start, stop, montage, symbol
    labels = []  # list of dictionaries
    for label in recording_meta['annotations_lbl']['labels']:
        labels.append({'start': label['start'],
                       'stop': label['stop'],
                       'montage': montages[label['montage']],
                       'event': symbols[np.argmax(label['probabilities'])]})
    return labels


def focal_starting_points(
        recording_meta: dict,
        seizure_types: list = None):
    """Return a list of dictionaries which contain the focal events.

    Here the interest is only on the starting points.

    Args:
        recording_meta: a dictionary which contains the metadata
        of the recording.
        seizure_types: a list with the events of interest

    Returns:
        A list of dictionaries whom contain the information
        about the starting point of a focal seizure of any kind.
        The dictionary looks like:
            {'start': start_time,
             'event': event_tse['event'],
             'montages': montages}

            where:
                - ``start_time`` is the starting time of the event
                - ``event_tse['event']`` is the kind of event
                  according to the tse file
                - ``montages`` is a list of montages
                  on which the event did start from

    # noqa: RST301
    """
    if seizure_types is None:
        seizure_types = ['fnsz', 'cpsz', 'spsz']

    # Convert the labels to more workable events
    events_list_lbl = labels_to_events(recording_meta)
    # Only keep focal seizure related events
    focal_events_list_lbl = [
        e for e in events_list_lbl if e['event'] in seizure_types
    ]

    events_list_tse = recording_meta['annotations_tse']
    # Only keep focal seizure related events
    focal_events_list_tse = [
        e for e in events_list_tse if e['event'] in seizure_types
    ]

    events = []
    for event_tse in focal_events_list_tse:
        start_time = event_tse['start']
        montages = [
            event_lbl['montage']
            for event_lbl
            in focal_events_list_lbl
            if event_lbl['start'] == start_time
        ]

        events.append({'start': start_time,
                       'event': event_tse['event'],
                       'montages': montages,
                       'filepath': recording_meta['filepath'],
                       },
                      )

    return events


EPILEPTIC_SEIZURE_LABELS = ['fnsz',  # Focal nonspecific seizure.
                            'gnsz',  # Generalized seizure.
                            'spsz',  # Simple partial seizure.
                            'cpsz',  # Complex partial seizure.
                            'absz',  # Absence seizure.
                            'tnsz',  # Tonic seizure.
                            'cnsz',  # Clonic seizure.
                            'tcsz',  # Tonic-clonic seizure.
                            'atsz',  # Atonic seizure.
                            'mysz',  # Myoclonic seizure.
                            ]

#                                 Focal nonspecific seizure.
EPILEPTIC_SEIZURE_LABELS_DICT = {'fnsz': 'focal nonspecific',
                                 # Generalized seizure.
                                 'gnsz': 'generalised',
                                 # Simple partial seizure.
                                 'spsz': 'simple partial',
                                 # Complex partial seizure.
                                 'cpsz': 'complex partial',
                                 # Absence seizure.
                                 'absz': 'absence',
                                 # Tonic seizure.
                                 'tnsz': 'tonic',
                                 # Clonic seizure. (not present)
                                 'cnsz': 'clonic',
                                 # Tonic-clonic seizure.
                                 'tcsz': 'tonic-clonic',
                                 # Atonic seizure. (not present)
                                 'atsz': 'atonic',
                                 # Myoclonic seizure.
                                 'mysz': 'myoclonic',
                                 }


if __name__ == '__main__':
    meta = mt.load_pickle(
        os.path.join(
            os.path.dirname(__file__),
            'metadata.pickle',
        ),
    )

    dev_seizures_durations_by_type = {}
    train_seizures_durations_by_type = {}

    # Dev set metadata (Extract dev metadata subset)
    dev = [e for e in meta if 'dev' in e['filepath']]

    # Train set metadata
    train = [e for e in meta if 'train' in e['filepath']]

    SEIZURE_TYPES_FOCAL = ['fnsz', 'cpsz', 'spsz']
    SEIZURE_TYPES_GENERALISED = ['gnsz', 'absz', 'tnsz', 'tcsz', 'mysz']

    # Filters the fnsz only
    dev_filter_focal = [
        True if len(
            [
                e['event'] for e in element['annotations_tse']
                if e['event'] in SEIZURE_TYPES_FOCAL
            ],
        ) != 0 else False for element in dev
    ]

    train_filter_focal = [
        True if len(
            [
                e['event'] for e in element['annotations_tse']
                if e['event'] in SEIZURE_TYPES_FOCAL
            ],
        ) != 0 else False for element in train
    ]

    # Contains all the recordings that contains
    # at least one focal seizure of any kind
    dev_focal = np.array(dev)[dev_filter_focal]
    train_focal = np.array(train)[train_filter_focal]

    # List of number of seizure per recordings containing seizure
    # l = [
    #     len([
    #             e['event'] for e in element['annotations_tse']
    #             if e['event'] in EPILEPTIC_SEIZURE_LABELS
    #         ]) for element in meta if len([e['event']
    #     for e in element['annotations_tse']
    #     if e['event'] in EPILEPTIC_SEIZURE_LABELS]) != 0]

    # sum(l) = 3050 seizures for all the recording

    # set               recordings with seizure  number of seizure events
    # number of ['fnsz', 'bckg', 'gnsz'] events sequence
    # meta (dev+train)  1149                     3050  20
    # dev               280                      673   4
    # train             869                      2377  16

    dev_all_fsp = []
    dev_uniques_fsp = set()
    dev_lines = []
    for recording in dev_focal[:]:
        fsp = focal_starting_points(recording)

        for event in fsp:
            dev_lines.append(
                [
                    event['filepath'][len(
                        '/home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/'):],
                    event['event'],
                    event['start'],
                    str(sorted(event['montages'])),
                ],
            )

            dev_all_fsp.append(sorted(event['montages']))
            dev_uniques_fsp.add('_'.join(sorted(event['montages'])))

            # print(
            #     event['filepath'][len(
            #         '/home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/edf/dev/'
            #         '03_tcp_ar_a/065/00006546/s011_2011_02_16/'):
            #     ],
            #     event['event'],
            #     sorted(event['montages']),
            # )

    # print(dev_lines)

    print('Unique elements   :', len(dev_uniques_fsp))
    print('Number of elements:', len(dev_all_fsp))

    train_all_fsp = []
    train_uniques_fsp = set()
    train_lines = []
    for recording in train_focal[:]:
        fsp = focal_starting_points(recording)

        for event in fsp:
            train_all_fsp.append(sorted(event['montages']))
            train_uniques_fsp.add('_'.join(sorted(event['montages'])))
            train_lines.append(
                [
                    event['filepath'][len(
                        '/home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/'):
                    ],
                    event['event'],
                    event['start'],
                    str(sorted(event['montages'])),
                ],
            )

    # print(
    #     {
    #         montage: list(np.concatenate(dev_all_fsp)).count(montage)
    #         for montage in set(list(np.concatenate(dev_all_fsp)))
    #     },
    # )

    # print(
    #     {
    #         montage: list(np.concatenate(train_all_fsp)).count(montage)
    #         for montage in set(list(np.concatenate(train_all_fsp)))
    #     },
    # )

    dev_montage_importance = {
        montage: list(np.concatenate(dev_all_fsp)).count(montage)
        for montage in {[np.concatenate(dev_all_fsp)]}
    }

    train_montage_importance = {
        montage: list(np.concatenate(train_all_fsp)).count(montage)
        for montage in {[np.concatenate(train_all_fsp)]}
    }

    print(len(train_uniques_fsp))
    print(len(train_all_fsp))

    # print(
    #     sorted(
    #         dev_montage_importance.items(),
    #         key=lambda x: x[1], reverse=True,
    #     ),
    # )

    # print(
    #     [
    #         [key, int(value)] for (key, value)
    #         in sorted(
    #             train_montage_importance.items(),
    #             key=lambda x: x[1], reverse=True)
    #     ],
    # )

    train_labels_data = np.array(
        [
            [key, int(value)] for (key, value)
            in sorted(
                train_montage_importance.items(),
                key=lambda x: x[1], reverse=True)
        ],
    ).transpose()

    # print(np.array(train_labels_data[1], dtype=np.int16))
    plt.figure()

    x = np.arange(len(train_labels_data[0]))  # the label locations
    width = 0.35  # the width of the bars
    # print(x)

    plt.bar(
        x=x,
        height=np.array(train_labels_data[1], dtype=np.int16),
        color='#575656ff',
    )

    plt.gca().set_xticks(x)

    plt.gca().set_xticklabels(
        train_labels_data[0],
        rotation=45,
        rotation_mode='anchor',
        ha='right',
    )

    # plt.hist(
    #     x=np.concatenate(dev_all_fsp),
    #     bins=22,
    #     rwidth=None,
    #     align='mid',
    # )

    plt.show(block=False)

    dev_labels_data = np.array(
        [
            [key, int(value)] for (key, value)
            in sorted(
                dev_montage_importance.items(),
                key=lambda x: x[1], reverse=True)
        ],
    ).transpose()

    plt.figure()
    x = np.arange(len(dev_labels_data[0]))  # the label locations
    width = 0.35  # the width of the bars

    plt.bar(
        x=x,
        height=np.array(dev_labels_data[1], dtype=np.int16),
        color='#6eb055ff',
    )

    plt.gca().set_xticks(x)

    plt.gca().set_xticklabels(
        dev_labels_data[0],
        rotation=45,
        rotation_mode='anchor',
        ha='right',
    )

    # plt.hist(
    #     x=np.concatenate(dev_all_fsp),
    #     bins=22,
    #     rwidth=None,
    #     align='mid',
    # )

    plt.show(block=False)

    # np.array(
    #     dev_labels_data[1],
    #     dtype=np.int16) / np.array(
    #     dev_labels_data[1],
    #     dtype=np.int16
    # ).max() * 100

    dev_height = np.array(
        [
            dev_montage_importance[key] for key in train_labels_data[0]
        ],
        dtype=np.int16,
    )

    dev_height = dev_height / dev_height.max() * 100

    train_height = np.array(
        train_labels_data[1],
        dtype=np.int16) / np.array(
        train_labels_data[1],
        dtype=np.int16,
    ).max() * 100

    plt.figure()
    x = np.arange(len(dev_labels_data[0]))  # the label locations
    width = 0.35  # the width of the bars
    plt.bar(
        x=x - width / 2,
        height=dev_height,
        width=width,
        label='Dev set',
        color='#6eb055ff',
    )

    plt.bar(
        x=x + width / 2,
        height=train_height,
        width=width,
        label='Train set',
        color='#575656ff',
    )

    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(
        train_labels_data[0],
        rotation=45,
        rotation_mode='anchor',
        ha='right',
    )

    # plt.hist(
    #     x=np.concatenate(dev_all_fsp),
    #     bins=22,
    #     rwidth=None,
    #     align='mid',
    # )

    plt.gca().legend()
    plt.show(block=False)

    # np.array(
    #     dev_labels_data[1],
    #     dtype=np.int16) / np.array(
    #     dev_labels_data[1],
    #     dtype=np.int16
    # ).max() * 100

    dev_height = np.array(
        [
            dev_montage_importance[key] for key in train_labels_data[0]
        ],
        dtype=np.int16,
    )
    dev_height = dev_height / dev_height.sum() * 100
    # print(dev_height)
    train_height = np.array(
        train_labels_data[1],
        dtype=np.int16) / np.array(
            train_labels_data[1],
            dtype=np.int16,
    ).sum() * 100

    plt.figure()
    x = np.arange(len(dev_labels_data[0]))  # the label locations
    width = 0.35  # the width of the bars
    plt.bar(
        x=x - width / 2,
        height=dev_height,
        width=width,
        label='Dev set',
        color='#6eb055ff',
    )

    plt.bar(
        x=x + width / 2,
        height=train_height,
        width=width,
        label='Train set',
        color='#575656ff',
    )

    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(
        train_labels_data[0],
        rotation=45,
        rotation_mode='anchor',
        ha='right',
    )

    # plt.hist(
    #     x=np.concatenate(dev_all_fsp),
    #     bins=22,
    #     rwidth=None,
    #     align='mid',
    # )

    plt.gca().legend()
    plt.show(block=False)

    channels = sorted([
        set(np.concatenate([
            montage.split('-')
            for montage
            in train_labels_data[0]
        ]))],
    )

    dev_montage_importance_per_channel = {}

    for k, v in dev_montage_importance.items():
        montage_channels = k.split('-')
        dev_montage_importance_per_channel[
            montage_channels[0]
        ] = dev_montage_importance_per_channel.get(
            montage_channels[0],
            0,
        ) + v

        dev_montage_importance_per_channel[
            montage_channels[1]
        ] = dev_montage_importance_per_channel.get(
            montage_channels[1],
            0,
        ) + v

    print(dev_montage_importance_per_channel)

    train_montage_importance_per_channel = {}

    for k, v in train_montage_importance.items():
        montage_channels = k.split('-')
        train_montage_importance_per_channel[
            montage_channels[0]
        ] = train_montage_importance_per_channel.get(
            montage_channels[0],
            0,
        ) + v

        train_montage_importance_per_channel[
            montage_channels[1]
        ] = train_montage_importance_per_channel.get(
            montage_channels[1],
            0,
        ) + v

    print(train_montage_importance_per_channel)
    # print(dev_montage_importance)
    # print(train_montage_importance)
    # input()

    # # The averaged data
    # data_evoked = data.mean(0)

    # # The number of epochs that were averaged
    # nave = data.shape[0]

    # # A comment to describe to evoked (usually the condition name)
    # comment = 'Smiley faces'

    # # Create the Evoked object
    # evoked_array = mne.EvokedArray(
    #     data_evoked,
    #     info,
    #     tmin,
    #     comment=comment,
    #     nave=nave)

    # print(evoked_array)
    # _ = evoked_array.plot(time_unit='s')

    # df_dev = pd.DataFrame(dev_lines,
    #                     #   index=['row 1', 'row 2'],
    #                       columns=['Filepath',
    #                                'Seizure Type',
    #                                'Start time',
    #                                'Concerned montages'])

    # df_train = pd.DataFrame(train_lines,
    #                         # index=['row 1', 'row 2'],
    #                         columns=['Filepath',
    #                                  'Seizure Type',
    #                                  'Start time',
    #                                  'Concerned montages'])

    # with pd.ExcelWriter('focal_starting_points.xlsx',
    #                     date_format='YYYY-MM-DD',
    #                     datetime_format='YYYY-MM-DD HH:MM:SS') as writer:

    #     df_dev.to_excel(writer, sheet_name='dev')
    #     df_train.to_excel(writer, sheet_name='train')

    # {count: [len(fsp) for fsp in dev_all_fsp].count(count)
    #  for count in set([len(fsp) for fsp in dev_all_fsp])}

    # {count: [len(fsp) for fsp in [fst.split('_') for fst
    #  in uniques_fst]].count(count) for count
    #  in set([len(fsp) for fsp in [fst.split('_')
    #  for fst in uniques_f: st]])}

    # Intersection focal seizure/generalised seizure
    # dev_f_and_gnsz = [
    #     e for e in dev_f
    #     if len([
    #         i for i in e['annotations_tse']
    #         if i['event'] == 'gnsz']) > 0]

    # sum([''.join([e['event'] for e
    #      in r['annotations_tse']]).count('fnszbckggnsz')
    #      for r in dev_f_and_gnsz])

    # Recordings with seizures
    # [True for element in meta
    # if len([
    # e['event'] for e in element['annotations_tse']
    # if e['event'] in EPILEPTIC_SEIZURE_LABELS]) != 0].count(True)
    # Out[1]: 1149

    # [True for element in dev
    # if len([
    # e['event'] for e in element['annotations_tse']
    # if e['event'] in EPILEPTIC_SEIZURE_LABELS]) != 0].count(True)
    # Out[2]: 280

    # [True for element in train
    #  if len([
    #  e['event'] for e in element['annotations_tse']
    #  if e['event'] in EPILEPTIC_SEIZURE_LABELS]) != 0].count(True)
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

    # Remove background event (better to keep only the labels of interest)
    # [e for e in l if e['event'] != 'bckg']
    # fnsz = [e for e in l if e['event'] == 'fnsz']
    # sorted(fnsz, key=lambda e:e['start'])

    # In [66]: for e in df:
    # ...:     if 'gnsz' in [i['event'] for i in e['annotations_tse']]:
    # ...:         print([i['event'] for i in e['annotations_tse']])

    # ['bckg', 'fnsz', 'bckg', 'fnsz', 'bckg', 'fnsz', 'bckg',
    #  'fnsz', 'bckg', 'gnsz', 'bckg', 'fnsz', 'bckg', 'gnsz',
    #  'bckg']
    # ['bckg', 'gnsz', 'bckg', 'fnsz', 'bckg', 'fnsz', 'bckg',
    #  'fnsz', 'bckg']
    # ['bckg', 'fnsz', 'bckg', 'fnsz', 'bckg', 'gnsz', 'bckg']
    # ['bckg', 'gnsz', 'bckg', 'spsz', 'bckg', 'gnsz', 'bckg',
    #  'fnsz', 'bckg']
    # ['bckg', 'spsz', 'bckg', 'gnsz', 'bckg', 'fnsz', 'bckg',
    #  'gnsz', 'bckg']
    # ['bckg', 'gnsz', 'bckg', 'gnsz', 'bckg', 'gnsz', 'bckg',
    #  'gnsz', 'bckg', 'gnsz', 'bckg', 'gnsz', 'bckg', 'gnsz',
    #  'bckg', 'gnsz', 'bckg', 'fnsz', 'bckg']

    # for e in df:
    #   if 'gnsz' in [i['event'] for i in e['annotations_tse']]:
    #       print([
    #           i['event']
    #           for i
    #           in e['annotations_tse']
    #     ], e['filepath'][:e['filepath'].rfind('/')])

    # ['bckg', 'fnsz', 'bckg', 'fnsz', 'bckg', 'fnsz', 'bckg',
    #  'fnsz', 'bckg', 'gnsz', 'bckg', 'fnsz', 'bckg', 'gnsz',
    #  'bckg']
    # /home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/edf/dev/
    # 01_tcp_ar/065/00006546/s018_2012_01_30
    # ['bckg', 'gnsz', 'bckg', 'fnsz', 'bckg', 'fnsz', 'bckg',
    #  'fnsz', 'bckg']
    # /home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/edf/dev/
    # 01_tcp_ar/065/00006546/s020_2012_01_30
    # ['bckg', 'fnsz', 'bckg', 'fnsz', 'bckg', 'gnsz', 'bckg']
    # /home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/edf/dev/
    # 01_tcp_ar/065/00006546/s022_2012_02_23
    # ['bckg', 'gnsz', 'bckg', 'spsz', 'bckg', 'gnsz', 'bckg',
    #  'fnsz', 'bckg']
    # /home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/edf/dev/
    # 01_tcp_ar/065/00006546/s024_2012_02_25
    # ['bckg', 'spsz', 'bckg', 'gnsz', 'bckg', 'fnsz', 'bckg',
    #  'gnsz', 'bckg']
    # /home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/edf/dev/
    # 01_tcp_ar/065/00006546/s024_2012_02_25
    # ['bckg', 'gnsz', 'bckg', 'gnsz', 'bckg', 'gnsz', 'bckg',
    #  'gnsz', 'bckg', 'gnsz', 'bckg', 'gnsz', 'bckg', 'gnsz',
    #  'bckg', 'gnsz', 'bckg', 'fnsz', 'bckg']
    # /home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/edf/dev/01_tcp_ar/085/
    # 00008512/s007_2012_07_03
