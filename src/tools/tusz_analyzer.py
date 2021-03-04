"""This module aims to provide an easy way to add figures
to LaTeX imported from Python with the help of PythonTeX
for the TUSZ dataset metadata analysis
Author: Vincent Stragier

Logs:
    - 20/10/2020
        - Make the script comply PEP8 (pycodestyle)
    - 14/10/2020
        - create this module add a test function
"""
import datetime
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

# Path to the dataset
PATH_TO_EDF = '/home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/'


# The script and the montages are in the same folder
TUSZ_1020 = os.path.join(os.path.dirname(__file__), 'TUSZ_1020.elc')


PATH_TO_SCRIPT = os.path.dirname(__file__)


USED_CHANNELS = [
    'Fp1',
    'Fp2',
    'F7',
    'F3',
    'F4',
    'F8',
    'C3',
    'Cz',
    'C4',
    'P3',
    'P4',
    'O1',
    'O2',
    'T3',
    'T5',
    'T4',
    'T6',
    'A1',
    'A2',
]


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


SEIZURE_TYPES_FOCAL = ['fnsz', 'cpsz', 'spsz']


SEIZURE_TYPES_GENERALISED = ['gnsz', 'absz', 'tnsz', 'tcsz', 'mysz']


def get_index(filepath: str, list_of_recording_metadata: list):
    """Retrieves the index of the recording in the filepath,
    using the list of recordings metadata.

    Parameters:
    -----------
        filepath: a string which contains the filepath to the recording.
        list_of_recording_metadata: list of the dictonaries
        which contain the metadata of each recording.
    Returns:
    --------
        The index of the recording or None if not found.
    """
    for index, element in enumerate(list_of_recording_metadata):
        if filepath in element['filepath']:
            return index

    raise("No matching index have been found,"
          "the filename is out of the scope of this metadata list.")


def save_topo_propagation(
    filepath: str,
    list_of_recording_metadata: list,
    path: str = PATH_TO_SCRIPT,
):
    """Generates all the seizure propagation plots.

    Args:
        filepath: a string which contains the filename
        of the recording without extension.
        list_of_recording_metadata: a list of dictionaries
        which contain the metadata of each recording.
        path: a string which contains the path to a folder
        which will be used to save the video.
    """
    index = get_index(
        filepath=filepath,
        list_of_recording_metadata=list_of_recording_metadata)

    recording_metadata = list_of_recording_metadata[index]

    ten_twenty_montage = _mgh_or_standard(fname=TUSZ_1020, head_size=0.095)
    fake_info = mne.create_info(
        ch_names=ten_twenty_montage.ch_names,
        sfreq=250.,
        ch_types='eeg',
    )

    recording_filepath = os.path.basename(recording_metadata['filepath'])
    recording_filepath = os.path.join(path, recording_filepath)
    propagation_dictionaries = extract_recording_propagation(
        recording_metadata)[1]

    for seizure_index in range(len(propagation_dictionaries)):
        propagation = propagation_dictionaries[
            seizure_index]['propagation']

        times = np.array(
            propagation_dictionaries[seizure_index]['times'])

        for index, montages in enumerate(propagation):
            list(np.concatenate([
                channels.split('-')
                for channels in [montage for montage in montages]]))

        data = np.array([np.array([
            pattern.count(key.upper()) for key in USED_CHANNELS])/max([
                pattern.count(key.upper()) * 1e6 for key in USED_CHANNELS])
                for pattern in [list(np.concatenate([
                    channels.split('-') for channels in [
                        montage for montage in montages]]))
                        for montages in propagation]]).transpose()

        # Repeat last frame
        data = np.hstack((data, data[:, [-1]]))

        fake_evoked = mne.EvokedArray(data, fake_info)
        fake_evoked.set_montage(ten_twenty_montage)
        fake_evoked.times = times

        fps = 1

        # plt.figure(figsize=(16, 9))
        # plt.title('%s, (%s)' % (recording_filepath, seizure_index))
        fig, anim = fake_evoked.animate_topomap(
            times=times, ch_type=None,
            frame_rate=fps, time_unit='s',
            blit=False, show=False,
            butterfly=False)

        ax = fig.add_subplot(111)
        ax.get_yticks()
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_frame_on(False)
        ax.set_title(
            '%s (%s)\n%s, %s' % (
                os.path.basename(recording_filepath),
                seizure_index,
                propagation_dictionaries[
                    seizure_index][
                        r'event (tse, {lbl})'][0],
                str(
                    propagation_dictionaries[
                        seizure_index][
                            r'event (tse, {lbl})'][1])),
            fontsize=35)

        # plt.subplots_adjust(left=0.16, bottom=0.19, top=0.82)

        fig.set_size_inches(16, 9, forward=True)

        # plt.gca().annotate(
        #     '                %s (%s)\n                %s, %s' % (
        #         recording_filepath,
        #         seizure_index,
        #         propagation_dictionaries[
        #             seizure_index][
        #                 r'event (tse, {lbl})'][0],
        #         str(
        #             propagation_dictionaries[
        #                 seizure_index][
        #                     r'event (tse, {lbl})'][1])),
        #         (0, 0.85))

        # # tight_layout docs: [left, bottom, right, top]
        # # in normalized (0, 1) figure
        # fig.tight_layout(
        #     rect=[0, 0.03, 1, 0.9])

        anim.save('%s_%s.mp4' % (
            recording_filepath, seizure_index),
            dpi=300,
            writer='ffmpeg')

        plt.close(fig)


def latex_env(environment: str, content: str, options: str = ""):
    """As shown on PythonTeX Wiki this is an helper function
    which is used to fill a LaTeX environment and its content.
    """
    return r"\begin{%s}%s%s%s%s\end{%s}" % (
        environment, options, "\n", content, "\n", environment)


def load_pickle_lzma(filename: str):
    """Load a variable from a lzma compressed binary file.
    Parameters:
    -----------
        filename: file name.
    Returns:
    --------
        Returns the variable.
    """
    import lzma
    import pickle

    with lzma.open(filename, 'rb') as f:
        return pickle.load(f)


def ensure_path(path: str):
    if not os.path.isdir(path):
        os.makedirs(path)


def seconds_to_human_readble_time(seconds):
    """Converts a number of seconds to a readable string
    (days, hours, minutes, seconds, etc.)
    Parameters:
    -----------
        seconds: the number of seconds to convert.
    Returns:
    --------
        A string with the human readable among of seconds.
    """
    return str(datetime.timedelta(seconds=seconds))


def labels_to_events(recording_meta):
    """From the recording converts the labels to a more explicite form.
    Parameters:
    -----------
        recording_meta: a dictionary which contains the metadata
        of the recording.
    Returns:
    --------
        A list of dictionaries, which is structured as followed.
        {'start': l['start'],
         'stop': l['stop'],
         'montage': montages[l['montage']],
         'event': symbols[np.argmax(l['probabilities'])]}
        where:
            - ``l['start']`` is the start time of the event
            - ``l['stop']`` is the stop time of the event
            - ``montages[l['montage']]`` is the montage
            on which the event is occuring
            - ``symbols[np.argmax(l['probabilities'])]``
            is the label of the most probable event
    """
    symbols = recording_meta['annotations_lbl']['symbols'][0]
    montages = recording_meta['annotations_lbl']['montages']

    # For each label extract start, stop, montage, symbol
    labels = list()  # list of dictionaries
    for label in recording_meta['annotations_lbl']['labels']:
        labels.append(
            {'start': label['start'],
             'stop': label['stop'],
             'montage': montages[label['montage']],
             'event': symbols[np.argmax(label['probabilities'])]})

    return labels


def focal_starting_points(recording_meta: dict,
                          seizure_types: list = ['fnsz', 'cpsz', 'spsz']):
    """Return a list of dictionaries which contain the focal events
    with their starting points.
    Parameters:
    -----------
        recording_meta: a dictionary which contains the metadata
        of the recording.
        seizure_types: a list with the events of interest
    Returns:
    --------
        A list of dictionaries whom contain the information
        about the starting point of a focal seizure of any kind.
        The dictionary looks like:
            {'start': start_time,
             'event': event_tse['event'],
             'montages': montages}
            where:
                - ``start_time`` is the starting time
                of the event.
                - ``event_tse['event']`` is the kind
                of event according to the tse file.
                - ``montages`` is a list
                of montages on which the event did start from.
    """
    # Convert the labels to more workable events
    events_list_lbl = labels_to_events(recording_meta)
    # Only keep focal seizure related events
    focal_events_list_lbl = [
        e for e in events_list_lbl if e['event'] in seizure_types]
    events_list_tse = recording_meta['annotations_tse']
    # Only keep focal seizure related events
    focal_events_list_tse = [
        e for e in events_list_tse if e['event'] in seizure_types]

    events = []
    for event_tse in focal_events_list_tse:
        start_time = event_tse['start']
        montages = [
            event_lbl['montage']
            for event_lbl in focal_events_list_lbl
            if event_lbl['start'] == start_time]

        events.append(
            {'start': start_time,
             'event': event_tse['event'],
             'montages': montages,
             'filepath': recording_meta['filepath']})

    return events


def extract_recording_propagation(
        recording_meta: dict,
        only_start_and_number: bool = False):
    """Extract the propagation of the seizure from the lbl labels compared
    to the tse labels
    Parameters:
    -----------
        recording_meta: a dictionary which contains
        all the recording metadata
    Returns:
    --------
        excel_lines: a list which contains the filepath,
        the seizure type(s), a time vector and the propagation pattern.
    """
    non_sz_lbl = [event for event in labels_to_events(
        recording_meta) if event['event'] != 'bckg']

    non_sz_tse = [event for event in recording_meta['annotations_tse']
                  if event['event'] != 'bckg']

    filepath = recording_meta['filepath'][len(PATH_TO_EDF):]
    excel_lines = list()
    propagation_dictionaries = list()
    for event in non_sz_tse:
        tse_event = event['event']
        # print("\ntse:", event)
        times = set()
        lbl_event = set()
        for lbl in non_sz_lbl:
            if lbl['stop'] <= event['stop'] and lbl['start'] >= event['start']:
                # print("lbl:", lbl)
                times.add(lbl['start'])
                times.add(lbl['stop'])
                lbl_event.add(lbl['event'])
        times = sorted(list(times))
        # print(tse_event, lbl_event)
        # print(times)

        propagation = list()
        for start, stop in zip(times[:-1], times[1:]):
            # print("\ninterval", start, '-', stop)
            montages = list()
            for lbl in non_sz_lbl:
                if lbl['stop'] >= stop and lbl['start'] <= start:
                    montages.append(lbl['montage'])

            propagation.append(sorted(list(set(montages))))

        if only_start_and_number is False:
            excel_lines.append([filepath, r"%s, %s" % (
                tse_event, str(lbl_event)), str(times), str(propagation)])
        else:
            excel_lines.append([filepath, r"%s, %s" % (
                tse_event, str(lbl_event)),
                str(times[0]),
                str(propagation[0]),
                len(propagation[0])])

        propagation_dictionaries.append(
            {'filepath': filepath,
             r'event (tse, {lbl})': (tse_event, lbl_event),
             'times': times,
             'propagation': propagation})

    return excel_lines, propagation_dictionaries


def make_a_pie(data, labels, n_colors, no_text_on_pie=False):
    """Make a pie figure for the given labels and data
    """
    # fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(aspect="equal"))
    # See https://matplotlib.org/3.3.2/tutorials/colors/colormaps.html
    plt.gca().set_prop_cycle('color', plt.cm.get_cmap(
        "Paired")(np.linspace(0, 1, n_colors + 1)))

    percents = data/np.sum(data) * 100

    mask = [
        False if value < 4 and (percents[
            (index + 1) % len(percents)] < 4 or percents[
                (index - 1) % len(percents)] < 4)
        else True for index, value in enumerate(percents, start=0)]

    def func(pct, allvals, mask):
        func.counter += 1
        # absolute = int(pct/100.*np.sum(allvals))
        if mask[func.counter - 1]:
            return "{:.1f}\\%".format(pct)
        else:
            return ""
    func.counter = 0

    if no_text_on_pie:
        wedges, _, autotexts = plt.pie(
            data,
            autopct=lambda pct: func(pct, data, mask),
            textprops=dict(color="black"),
            pctdistance=0.85,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.3))

        plt.legend(
            wedges,
            [element[0] + element[1] + str(
                element[2]) + element[3]
             for element
             in zip(labels, [' ('] * len(data), data, [')'] * len(data))],
            title="Seizure types",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))

    else:
        wedges, _, autotexts = plt.pie(
            data,
            autopct=lambda pct: func(pct, data, mask),
            textprops=dict(color="black"),
            pctdistance=1.1,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.3))  # Make a donut

        plt.legend(
            wedges,
            [element[0] + element[1] + str(
                element[2]) + element[3] + "{:.1f}\\%".format(
                    element[4])
             for element
             in zip(labels, [' ('] * len(data), data, [') - '] * len(data),
             percents)],
            title="Seizure types",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=12, weight="bold")


def save_plot_pies_sz_per_type(filename: str,
                               data_dev,
                               labels_dev,
                               data_train,
                               labels_train):
    """Save the plotted pies
    """
    plt.figure(figsize=(16, 9))
    plt.rc('text', usetex=True)
    plt.rcParams["font.family"] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = "Calibri"

    number_of_colors = max(len(data_dev), len(data_train)) + 1
    plt.subplot(1, 2, 1)
    plt.title(
        r"\begin{center}\Huge Seizures by"
        r" type (event wise) v1.2.1\end{center}")

    make_a_pie(data_dev, labels_dev, number_of_colors)

    plt.subplot(1, 2, 2)
    plt.title(
        r"\begin{center}\Huge Seizures by"
        r" type (event wise) v1.5.2\end{center}")

    make_a_pie(data_train, labels_train, number_of_colors)

    plt.tight_layout()
    plt.savefig(filename, format="PDF", transparent=True)
    plt.close()


def save_plot_pies_sz_duration_per_type(filename: str,
                                        data_dev,
                                        labels_dev,
                                        data_train,
                                        labels_train):
    """Save the plotted pies
    """
    plt.figure(figsize=(16, 9))
    plt.rc('text', usetex=True)
    plt.rcParams["font.family"] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = "Calibri"

    number_of_colors = max(len(data_dev), len(data_train)) + 1
    plt.subplot(1, 2, 1)
    plt.title(
        r"\begin{center}\Huge Seizures duration"
        r"(in seconds) by type v1.2.1\end{center}")

    make_a_pie(data_dev, labels_dev, number_of_colors)

    plt.subplot(1, 2, 2)
    plt.title(
        r"\begin{center}\Huge Seizures duration"
        r"(in seconds) by type v1.5.2\end{center}")

    make_a_pie(data_train, labels_train, number_of_colors)

    plt.tight_layout()
    plt.savefig(filename, format="PDF", transparent=True)
    plt.close()


def latex_pies_sz_per_type(figures_path,
                           figure_filename,
                           data_dev,
                           labels_dev,
                           data_train,
                           labels_train):
    """Make the figure pdf for the pie chart of seizures
    and return a string with the LaTeX command to use
    """
    ensure_path(figures_path)
    save_plot_pies_sz_per_type(
        os.path.join(figures_path, figure_filename),
        data_dev,
        labels_dev,
        data_train,
        labels_train)

    return latex_env(
        environment="center",
        content=r"\includegraphics[width=0.9\textwidth]{%s}" % (
            os.path.join(
                os.getcwd(),
                figures_path,
                figure_filename).replace('\\', '/')))


def latex_pies_sz_duration_per_type(figures_path,
                                    figure_filename,
                                    data_dev,
                                    labels_dev,
                                    data_train,
                                    labels_train):
    """Make the figure pdf for the pie chart of seizure durations
    and return a string with the LaTeX command to use
    """
    ensure_path(figures_path)
    save_plot_pies_sz_duration_per_type(
        os.path.join(figures_path, figure_filename),
        data_dev,
        labels_dev,
        data_train,
        labels_train)

    return latex_env(
        environment="center",
        content=r"\includegraphics[width=0.9\textwidth]{%s}" % (
            os.path.join(
                os.getcwd(),
                figures_path,
                figure_filename).replace('\\', '/')))


def durations_histo(filename: str, v1_2_1, v1_5_2):
    """Generate all the figures for the histograms.
    Returns a dictionary of dict with dict containing the full filename.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    ensure_path(os.path.dirname(filename))
    filenames = dict()
    filenames['dev'] = dict()
    filenames['train'] = dict()
    labels_in_the_datasets = set(list(
        v1_2_1['events_counts_dev'].keys()) + list(
            v1_2_1['events_counts_train'].keys()) + list(
                v1_5_2['events_counts_dev'].keys()) + list(
                    v1_5_2['events_counts_train'].keys()))

    labels_in_the_datasets = sorted(
        list(set([label.lower() for label in labels_in_the_datasets])))

    dev_seizures_durations_by_type_v1_2_1 = {
        seizure_type: np.array(
            [event['duration'] for event in v1_2_1['events_dev']
             if event['event'] == seizure_type])
        for seizure_type in list(v1_2_1['events_counts_dev'].keys())}

    train_seizures_durations_by_type_v1_2_1 = {
        seizure_type: np.array(
            [event['duration'] for event in v1_2_1['events_train']
             if event['event'] == seizure_type])
        for seizure_type in list(v1_2_1['events_counts_train'].keys())}

    dev_seizures_durations_by_type_v1_5_2 = v1_5_2[
        'dev_seizures_durations_by_type']

    train_seizures_durations_by_type_v1_5_2 = v1_5_2[
        'train_seizures_durations_by_type']

    with PdfPages(filename + ".pdf") as pdf:
        d = pdf.infodict()
        d['Title'] = 'Seizures duration histograms'
        d['Author'] = 'Vincent Stragier'
        d['Subject'] = 'Compilation of all the duration histograms'
        d['Keywords'] = 'seizures epilepsy histogram TUSZ EEG'
        d['CreationDate'] = datetime.datetime(2020, 10, 21)
        d['ModDate'] = datetime.datetime.today()

        for seizure_type in labels_in_the_datasets:
            plt.figure(
                "Histograms for {0} seizures ('{1}') - dev set".format(
                    EPILEPTIC_SEIZURE_LABELS_DICT[
                        seizure_type.lower()], seizure_type),
                figsize=(16/2.54*2, 9/2.54*2))

            pdf.attach_note(
                "Histograms for {0} seizures ('{1}') - dev set".format(
                    EPILEPTIC_SEIZURE_LABELS_DICT[
                        seizure_type.lower()],
                    seizure_type))

            plt.rc('text', usetex=True)
            plt.rcParams["font.family"] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = "Calibri"

            plt.suptitle(
                "\\Huge Histograms for {0} seizures ('{1}') - dev set".format(
                    EPILEPTIC_SEIZURE_LABELS_DICT[
                        seizure_type.lower()],
                    seizure_type),
                fontsize=15)

            if seizure_type.upper()\
                    in dev_seizures_durations_by_type_v1_2_1.keys():
                data = dev_seizures_durations_by_type_v1_2_1[
                    seizure_type.upper()]

                mu = data.mean()
                median = np.median(data)
                sigma = data.std()
                minimum = data.min()
                maximum = data.max()
                plt.subplot(2, 2, 1)
                counts, bins, _ = plt.hist(
                    data, bins=data.size, rwidth=0.8, color='#6eb055ff')
                i = np.argmax(counts)
                hist_mode = (bins[i] + bins[i + 1])/2

                plt.ylabel(r'Count per bin')
                plt.legend(
                    [r'$\mu={0:.4f}$, $\sigma={1:.4f}$,{7}min$\,={2:.4f}$,'
                     r' max$\,={6:.4f}$,{7}median$\,={4:.4f}$,'
                     r' mode$\,={5:.4f}$,{7}Number of seizures: {3}'.format(
                         mu,
                         sigma,
                         minimum,
                         len(data),
                         median,
                         hist_mode,
                         maximum, "\n")])

                plt.xlabel(r'Time in seconds')
                plt.title(r'Dev set v1.2.1')

                plt.subplot(2, 2, 3)
                counts, bins, _ = plt.hist(
                    data, bins=data.size, rwidth=0.8, color='#6eb055ff')
                plt.xlim(0, mu)
                i = np.argmax(counts)
                hist_mode = (bins[i] + bins[i + 1])/2
                plt.ylabel(r'Count per bin')
                plt.legend(
                    [r'$\mu={0:.4f}$, $\sigma={1:.4f}$,{7}min$\,={2:.4f}$,'
                     r' max$\,={6:.4f}$,{7}median$\,={4:.4f}$,'
                     r' mode$\,={5:.4f}$,{7}Number of seizures: {3}'.format(
                         mu,
                         sigma,
                         minimum,
                         len(data),
                         median,
                         hist_mode,
                         maximum,
                         "\n")])

                plt.xlabel(r'Time in seconds')
                plt.title(r'Dev set v1.2.1 [0, %.2f]' % mu)

            if seizure_type in dev_seizures_durations_by_type_v1_5_2.keys():
                data = dev_seizures_durations_by_type_v1_5_2[seizure_type]
                mu = data.mean()
                median = np.median(data)
                sigma = data.std()
                minimum = data.min()
                maximum = data.max()
                plt.subplot(2, 2, 2)
                counts, bins, _ = plt.hist(
                    data, bins=data.size, rwidth=0.8, color='#575656ff')
                i = np.argmax(counts)
                hist_mode = (bins[i] + bins[i + 1])/2

                plt.ylabel(r'Count per bin')
                plt.legend(
                    [r'$\mu={0:.4f}$, $\sigma={1:.4f}$,{7}min$\,={2:.4f}$,'
                     r' max$\,={6:.4f}$,{7}median$\,={4:.4f}$,'
                     r' mode$\,={5:.4f}$,{7}Number of seizures: {3}'.format(
                         mu,
                         sigma,
                         minimum,
                         len(data),
                         median,
                         hist_mode,
                         maximum,
                         "\n")])

                plt.xlabel(r'Time in seconds')
                plt.title(r'Dev set v1.5.2')

                plt.subplot(2, 2, 4)
                counts, bins, _ = plt.hist(
                    data, bins=data.size, rwidth=0.8, color='#575656ff')
                plt.xlim(0, mu)
                i = np.argmax(counts)
                hist_mode = (bins[i] + bins[i + 1])/2
                plt.ylabel(r'Count per bin')
                plt.legend(
                    [r'$\mu={0:.4f}$, $\sigma={1:.4f}$,{7}min$\,={2:.4f}$,'
                     r' max$\,={6:.4f}$,{7}median$\,={4:.4f}$,'
                     r' mode$\,={5:.4f}$,{7}Number of seizures: {3}'.format(
                         mu,
                         sigma,
                         minimum,
                         len(data),
                         median,
                         hist_mode,
                         maximum,
                         "\n")])

                plt.xlabel(r'Time in seconds')
                plt.title(r'Dev set v1.5.2 [0, %.2f]' % mu)

            # tight_layout docs: [left, bottom, right, top]
            # in normalized (0, 1) figure
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(
                "_".join([filename, seizure_type, 'dev.pdf']),
                format="PDF",
                transparent=True)

            filenames['dev'][seizure_type] = "_".join(
                [filename, seizure_type, 'dev.pdf']).replace('\\', '/')

            pdf.savefig(transparent=True)
            plt.close()

            plt.figure(
                "Histograms for {0} seizures ('{1}') - train set".format(
                    EPILEPTIC_SEIZURE_LABELS_DICT[
                        seizure_type.lower()],
                    seizure_type),
                figsize=(16/2.54*2, 9/2.54*2))

            plt.rc('text', usetex=True)
            plt.rcParams["font.family"] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = "Calibri"

            plt.suptitle(
                "\\Huge Histograms for {0}"
                " seizures ('{1}') - train set".format(
                    EPILEPTIC_SEIZURE_LABELS_DICT[
                        seizure_type.lower()],
                    seizure_type),
                fontsize=15)

            if seizure_type.upper()\
                    in train_seizures_durations_by_type_v1_2_1.keys():
                data = train_seizures_durations_by_type_v1_2_1[
                    seizure_type.upper()]

                mu = data.mean()
                median = np.median(data)
                sigma = data.std()
                minimum = data.min()
                maximum = data.max()
                plt.subplot(2, 2, 1)
                counts, bins, _ = plt.hist(
                    data, bins=data.size, rwidth=0.8, color='#6eb055ff')
                i = np.argmax(counts)
                hist_mode = (bins[i] + bins[i + 1])/2

                plt.ylabel(r'Count per bin')
                plt.legend(
                    [r'$\mu={0:.4f}$, $\sigma={1:.4f}$,{7}min$\,={2:.4f}$,'
                     r' max$\,={6:.4f}$,{7}median$\,={4:.4f}$,'
                     r' mode$\,={5:.4f}$,{7}Number of seizures: {3}'.format(
                         mu,
                         sigma,
                         minimum,
                         len(data),
                         median,
                         hist_mode,
                         maximum,
                         "\n")])

                plt.xlabel(r'Time in seconds')
                plt.title(r'Train set v1.2.1')

                plt.subplot(2, 2, 3)
                counts, bins, _ = plt.hist(
                    data, bins=data.size, rwidth=0.8, color='#6eb055ff')
                plt.xlim(0, mu)
                i = np.argmax(counts)
                hist_mode = (bins[i] + bins[i + 1])/2
                plt.ylabel(r'Count per bin')
                plt.legend(
                    [r'$\mu={0:.4f}$, $\sigma={1:.4f}$,{7}min$\,={2:.4f}$,'
                     r' max$\,={6:.4f}$,{7}median$\,={4:.4f}$,'
                     r' mode$\,={5:.4f}$,{7}Number of seizures: {3}'.format(
                         mu,
                         sigma,
                         minimum,
                         len(data),
                         median,
                         hist_mode,
                         maximum,
                         "\n")])

                plt.xlabel(r'Time in seconds')
                plt.title(r'Train set v1.2.1 [0, %.2f]' % mu)

            if seizure_type in train_seizures_durations_by_type_v1_5_2.keys():
                data = train_seizures_durations_by_type_v1_5_2[seizure_type]
                mu = data.mean()
                median = np.median(data)
                sigma = data.std()
                minimum = data.min()
                maximum = data.max()
                plt.subplot(2, 2, 2)
                counts, bins, _ = plt.hist(
                    data, bins=data.size, rwidth=0.8, color='#575656ff')
                i = np.argmax(counts)
                hist_mode = (bins[i] + bins[i + 1])/2

                plt.ylabel(r'Count per bin')
                plt.legend(
                    [r'$\mu={0:.4f}$, $\sigma={1:.4f}$,{7}min$\,={2:.4f}$,'
                     r' max$\,={6:.4f}$,{7}median$\,={4:.4f}$,'
                     r' mode$\,={5:.4f}$,{7}Number of seizures: {3}'.format(
                         mu,
                         sigma,
                         minimum,
                         len(data),
                         median,
                         hist_mode,
                         maximum,
                         "\n")])

                plt.xlabel(r'Time in seconds')
                plt.title(r'Train set v1.5.2')

                plt.subplot(2, 2, 4)
                counts, bins, _ = plt.hist(
                    data, bins=data.size, rwidth=0.8, color='#575656ff')
                plt.xlim(0, mu)
                i = np.argmax(counts)
                hist_mode = (bins[i] + bins[i + 1])/2
                plt.ylabel(r'Count per bin')
                plt.legend(
                    [r'$\mu={0:.4f}$, $\sigma={1:.4f}$,{7}min$\,={2:.4f}$,'
                     r' max$\,={6:.4f}$,{7}median$\,={4:.4f}$,'
                     r' mode$\,={5:.4f}$,{7}Number of seizures: {3}'.format(
                         mu,
                         sigma,
                         minimum,
                         len(data),
                         median,
                         hist_mode,
                         maximum,
                         "\n")])

                plt.xlabel(r'Time in seconds')
                plt.title(r'Train set v1.5.2 [0, %.2f]' % mu)

            # tight_layout docs: [left, bottom, right, top]
            # in normalized (0, 1) figure
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig("_".join(
                [filename, seizure_type, 'train.pdf']),
                format="PDF",
                transparent=True)

            filenames['train'][seizure_type] = "_".join(
                [filename, seizure_type, 'train.pdf']).replace('\\', '/')
            pdf.savefig(transparent=True)
            plt.close()
        return filenames


def durations_histo_v1_5_2(filename: str, v1_5_2):
    """Generate all the figures for the histograms
    Returns a dictionary of dict with dict containing the full filename
    """
    ensure_path(os.path.dirname(filename))
    filenames = dict()
    filenames['dev'] = dict()
    filenames['train'] = dict()
    labels_in_the_datasets = set(list(
        v1_5_2['events_counts_dev'].keys()) + list(
            v1_5_2['events_counts_train'].keys()))

    labels_in_the_datasets = sorted(
        list(set([label.lower() for label in labels_in_the_datasets])))

    dev_seizures_durations_by_type_v1_5_2 = v1_5_2[
        'dev_seizures_durations_by_type']

    train_seizures_durations_by_type_v1_5_2 = v1_5_2[
        'train_seizures_durations_by_type']

    for seizure_type in labels_in_the_datasets:
        plt.figure(
            "Histograms for {0} seizures ('{1}') - dev set".format(
                EPILEPTIC_SEIZURE_LABELS_DICT[
                    seizure_type.lower()],
                seizure_type),
            figsize=(16/2.54*2, 9/2.54*2))

        plt.rc('text', usetex=True)
        plt.rcParams["font.family"] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = "Calibri"
        plt.suptitle(
            "\\Huge Histograms for {0}"
            " seizures ('{1}') - dev set".format(
                EPILEPTIC_SEIZURE_LABELS_DICT[
                    seizure_type.lower()],
                seizure_type),
            fontsize=15)

        if seizure_type\
                in dev_seizures_durations_by_type_v1_5_2.keys():
            data = dev_seizures_durations_by_type_v1_5_2[
                seizure_type]

            mu = data.mean()
            median = np.median(data)
            sigma = data.std()
            minimum = data.min()
            maximum = data.max()
            plt.subplot(2, 1, 1)
            counts, bins, _ = plt.hist(
                data, bins=data.size, rwidth=0.8, color='#575656ff')
            i = np.argmax(counts)
            hist_mode = (bins[i] + bins[i + 1])/2

            plt.ylabel(r'Count per bin')
            plt.legend(
                [r'$\mu={0:.4f}$, $\sigma={1:.4f}$,{7}min$\,={2:.4f}$,'
                 r' max$\,={6:.4f}$,{7}median$\,={4:.4f}$,'
                 r' mode$\,={5:.4f}$,{7}Number of seizures: {3}'.format(
                     mu,
                     sigma,
                     minimum,
                     len(data),
                     median,
                     hist_mode,
                     maximum,
                     "\n")])

            plt.xlabel(r'Time in seconds')
            plt.title(r'Dev set v1.5.2')

            plt.subplot(2, 1, 2)
            counts, bins, _ = plt.hist(
                data, bins=data.size, rwidth=0.8, color='#575656ff')
            plt.xlim(0, mu)
            i = np.argmax(counts)
            hist_mode = (bins[i] + bins[i + 1])/2
            plt.ylabel(r'Count per bin')
            plt.legend(
                [r'$\mu={0:.4f}$, $\sigma={1:.4f}$,{7}min$\,={2:.4f}$,'
                 r' max$\,={6:.4f}$,{7}median$\,={4:.4f}$,'
                 r' mode$\,={5:.4f}$,{7}Number of seizures: {3}'.format(
                     mu,
                     sigma,
                     minimum,
                     len(data),
                     median,
                     hist_mode,
                     maximum,
                     "\n")])

            plt.xlabel(r'Time in seconds')
            plt.title(r'Dev set v1.5.2 [0, %.2f]' % mu)

        # tight_layout docs: [left, bottom, right, top]
        # in normalized (0, 1) figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(
            "_".join([filename, seizure_type, 'dev.pdf']),
            format="PDF",
            transparent=True)

        filenames['dev'][seizure_type] = "_".join(
            [filename, seizure_type, 'dev.pdf']).replace('\\', '/')

        plt.figure(
            "Histograms for {0}"
            " seizures ('{1}') - train set".format(
                EPILEPTIC_SEIZURE_LABELS_DICT[
                    seizure_type.lower()],
                seizure_type),
            figsize=(16/2.54*2, 9/2.54*2))

        plt.rc('text', usetex=True)
        plt.rcParams["font.family"] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = "Calibri"

        plt.suptitle(
            "\\Huge Histograms for {0}"
            " seizures ('{1}') - train set".format(
                EPILEPTIC_SEIZURE_LABELS_DICT[
                    seizure_type.lower()],
                seizure_type),
            fontsize=15)

        if seizure_type\
                in train_seizures_durations_by_type_v1_5_2.keys():
            data = train_seizures_durations_by_type_v1_5_2[
                seizure_type]

            mu = data.mean()
            median = np.median(data)
            sigma = data.std()
            minimum = data.min()
            maximum = data.max()
            plt.subplot(2, 1, 1)
            counts, bins, _ = plt.hist(
                data, bins=data.size, rwidth=0.8, color='#575656ff')
            i = np.argmax(counts)
            hist_mode = (bins[i] + bins[i + 1])/2

            plt.ylabel(r'Count per bin')
            plt.legend(
                [r'$\mu={0:.4f}$, $\sigma={1:.4f}$,{7}min$\,={2:.4f}$,'
                 r' max$\,={6:.4f}$,{7}median$\,={4:.4f}$,'
                 r' mode$\,={5:.4f}$,{7}Number of seizures: {3}'.format(
                     mu,
                     sigma,
                     minimum,
                     len(data),
                     median,
                     hist_mode,
                     maximum,
                     "\n")])

            plt.xlabel(r'Time in seconds')
            plt.title(r'Train set v1.5.2')

            plt.subplot(2, 1, 2)
            counts, bins, _ = plt.hist(
                data, bins=data.size, rwidth=0.8, color='#575656ff')
            plt.xlim(0, mu)
            i = np.argmax(counts)
            hist_mode = (bins[i] + bins[i + 1])/2
            plt.ylabel(r'Count per bin')
            plt.legend(
                [r'$\mu={0:.4f}$, $\sigma={1:.4f}$,{7}min$\,={2:.4f}$,'
                 r' max$\,={6:.4f}$,{7}median$\,={4:.4f}$,'
                 r' mode$\,={5:.4f}$,{7}Number of seizures: {3}'.format(
                     mu,
                     sigma,
                     minimum,
                     len(data),
                     median,
                     hist_mode,
                     maximum,
                     "\n")])

            plt.xlabel(r'Time in seconds')
            plt.title(r'Train set v1.5.2 [0, %.2f]' % mu)

        # tight_layout docs: [left, bottom, right, top]
        # in normalized (0, 1) figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(
            "_".join([filename, seizure_type, 'train.pdf']),
            format="PDF",
            transparent=True)

        filenames['train'][seizure_type] = "_".join(
            [filename, seizure_type, 'train.pdf']).replace('\\', '/')

    return filenames


def save_barh_montages_focal_starting_points(filename: str, v1_5_2: dict):
    """Save the plot in bar graph the number or starting focal seizures
    which start on 1 to 22 montages.
    """
    # dev_montages_len = {
    #     len(montages) for montages in v1_5_2['dev_all_fsp']}
    # train_montages_len = {
    #     len(montages) for montages in v1_5_2['train_all_fsp']}

    # dev_len_by_montages = [
    #     len(montages) for montages in v1_5_2['dev_all_fsp']]
    # train_len_by_montages = [
    #     len(montages) for montages in v1_5_2['train_all_fsp']]

    # dev_occurence_by_number_of_montages = {
    #     str(count): [
    #         len(montages) for montages in v1_5_2[
    #             'dev_all_fsp']].count(count) for count in {
    #                 len(montages) for montages in v1_5_2['dev_all_fsp']}}

    # train_occurence_by_number_of_montages = {
    #     str(count): [
    #         len(montages) for montages in v1_5_2[
    #             'train_all_fsp']].count(count) for count in {
    #                 len(montages) for montages in v1_5_2['train_all_fsp']}}

    dev_occurence_by_number_of_montages = {
        str(count): [
            len(montages) for montages in v1_5_2[
                'dev_all_fsp']].count(count) for count in range(1, 23)}

    train_occurence_by_number_of_montages = {
        str(count): [
            len(montages) for montages in v1_5_2[
                'train_all_fsp']].count(count) for count in range(1, 23)}

    labels = sorted(
        list(train_occurence_by_number_of_montages.keys()),
        key=lambda x: int(x))

    y = np.arange(len(labels)) + 1  # the label locations

    dev_data = np.array([dev_occurence_by_number_of_montages[key]
                         for key in labels], dtype=np.int16)
    train_data = np.array([train_occurence_by_number_of_montages[key]
                           for key in labels], dtype=np.int16)

    plt.figure(figsize=(16, 9))
    plt.rc('text', usetex=True)
    plt.rcParams["font.family"] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = "Calibri"

    plt.subplot(1, 2, 2)
    plt.title(r"\begin{center}\Huge Dev set\end{center}")
    plt.barh(y=y, width=dev_data, height=0.7, color='#6eb055ff')
    plt.gca().set_yticks(y)

    plt.subplot(1, 2, 1)
    plt.title(r"\begin{center}\Huge Train set\end{center}")
    plt.barh(y=y, width=train_data, height=0.7, color='#575656ff')
    plt.gca().set_yticks(y)

    plt.tight_layout()
    plt.savefig(filename, format="PDF", transparent=True)
    plt.close()


def latex_barh_montages_focal_starting_points(figures_path,
                                              figure_filename,
                                              v1_5_2: dict):
    """Make the plot in bar graph the number or starting focal seizures
    which start on 1 to 22 montages.
    """
    ensure_path(figures_path)
    save_barh_montages_focal_starting_points(
        os.path.join(figures_path, figure_filename),
        v1_5_2=v1_5_2)

    return latex_env(
        environment="center",
        content=r"\includegraphics[width=0.8\textwidth]{%s}" % (
            os.path.join(
                os.getcwd(),
                figures_path,
                figure_filename).replace('\\', '/')))


def save_focal_starting_points(filename: str, kind: str, v1_5_2: dict):
    dev_labels_data = np.array([[key, int(value)] for (key, value) in sorted(
        v1_5_2['dev_montage_importance'].items(),
        key=lambda x: x[1], reverse=True)]).transpose()

    train_labels_data = np.array([[key, int(value)] for (key, value) in sorted(
        v1_5_2['train_montage_importance'].items(),
        key=lambda x: x[1], reverse=True)]).transpose()

    def save_dev_focal_starting_points(filename: str, v1_5_2: dict):
        plt.figure(figsize=(16, 9))
        plt.rc('text', usetex=True)
        plt.rcParams["font.family"] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = "Calibri"
        x = np.arange(len(dev_labels_data[0]))  # the label locations
        plt.bar(x=x, height=np.array(
            dev_labels_data[1], dtype=np.int16), width=0.7, color='#6eb055ff')
        plt.gca().set_xticks(x)
        plt.gca().set_xticklabels(
            dev_labels_data[0],
            rotation=45,
            rotation_mode="anchor",
            ha="right")

        plt.tight_layout()
        plt.savefig(filename, format="PDF", transparent=True)
        plt.close()

    def save_train_focal_starting_points(filename: str, v1_5_2: dict):
        plt.figure(figsize=(16, 9))
        plt.rc('text', usetex=True)
        plt.rcParams["font.family"] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = "Calibri"
        x = np.arange(len(train_labels_data[0]))  # the label locations
        plt.bar(x=x, height=np.array(
            train_labels_data[1],
            dtype=np.int16),
            width=0.7,
            color='#575656ff')

        plt.gca().set_xticks(x)
        plt.gca().set_xticklabels(
            train_labels_data[0],
            rotation=45,
            rotation_mode="anchor",
            ha="right")

        plt.tight_layout()
        plt.savefig(filename, format="PDF", transparent=True)
        plt.close()

    def save_dev_train_focal_starting_points_max(filename: str,
                                                 v1_5_2: dict):
        dev_height = np.array([v1_5_2['dev_montage_importance'][key]
                               for key in train_labels_data[0]],
                              dtype=np.int16)
        dev_height = dev_height/dev_height.max() * 100
        train_height = np.array(train_labels_data[1], dtype=np.int16)/np.array(
            train_labels_data[1], dtype=np.int16).max() * 100

        plt.figure(figsize=(16, 9))
        plt.rc('text', usetex=True)
        plt.rcParams["font.family"] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = "Calibri"
        x = np.arange(len(dev_labels_data[0]))  # the label locations
        width = 0.35  # the width of the bars
        plt.bar(x=x + width/2, height=dev_height, width=width,
                label="Dev set", color='#6eb055ff')

        plt.bar(x=x - width/2, height=train_height, width=width,
                label="Train set", color='#575656ff')

        plt.gca().set_xticks(x)
        plt.gca().set_xticklabels(
            train_labels_data[0],
            rotation=45,
            rotation_mode="anchor",
            ha="right")

        plt.gca().legend()
        plt.tight_layout()
        plt.savefig(filename, format="PDF", transparent=True)
        plt.close()

    def save_dev_train_focal_starting_points_sum(filename: str, v1_5_2: dict):
        dev_height = np.array([v1_5_2['dev_montage_importance'][key]
                               for key in train_labels_data[0]],
                              dtype=np.int16)

        dev_height = dev_height/dev_height.sum() * 100
        train_height = np.array(train_labels_data[1], dtype=np.int16)/np.array(
            train_labels_data[1], dtype=np.int16).sum() * 100

        plt.figure(figsize=(16, 9))
        plt.rc('text', usetex=True)
        plt.rcParams["font.family"] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = "Calibri"
        x = np.arange(len(dev_labels_data[0]))  # the label locations
        width = 0.35  # the width of the bars
        plt.bar(x=x + width/2, height=dev_height, width=width,
                label="Dev set", color='#6eb055ff')

        plt.bar(x=x - width/2, height=train_height, width=width,
                label="Train set", color='#575656ff')

        plt.gca().set_xticks(x)
        plt.gca().set_xticklabels(
            train_labels_data[0],
            rotation=45,
            rotation_mode="anchor",
            ha="right")

        plt.gca().legend()
        plt.tight_layout()
        plt.savefig(filename, format="PDF", transparent=True)
        plt.close()

    if kind == 'dev':
        return save_dev_focal_starting_points(
            filename=filename, v1_5_2=v1_5_2)

    elif kind == 'train':
        return save_train_focal_starting_points(
            filename=filename, v1_5_2=v1_5_2)

    elif kind == 'max':
        return save_dev_train_focal_starting_points_max(
            filename=filename, v1_5_2=v1_5_2)

    elif kind == 'sum':
        return save_dev_train_focal_starting_points_sum(
            filename=filename, v1_5_2=v1_5_2)


def latex_montage_focal_starting_points(figures_path,
                                        figure_filename,
                                        v1_5_2: dict,
                                        kind: str):
    """Make the figure pdf for the focal starting poins histograms.
    """
    ensure_path(figures_path)
    save_focal_starting_points(
        os.path.join(figures_path, figure_filename),
        kind=kind,
        v1_5_2=v1_5_2)

    return latex_env(
        environment="center",
        content=r"\includegraphics[width=0.8\textwidth]{%s}" % (
            os.path.join(
                os.getcwd(),
                figures_path,
                figure_filename).replace('\\', '/')))


def save_topo_focal_starting_points(
    filename: str,
    dev_montage_importance_per_channel: dict,
    train_montage_importance_per_channel: dict,
):

    ten_twenty_montage = _mgh_or_standard(fname=TUSZ_1020, head_size=0.095)
    # n_channels = len(ten_twenty_montage.ch_names)

    fake_info = mne.create_info(
        ch_names=ten_twenty_montage.ch_names,
        sfreq=250.,
        ch_types='eeg',
    )

    plt.figure(figsize=(16, 9))
    plt.rc('text', usetex=True)
    plt.rcParams["font.family"] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = "Calibri"
    plt.subplot(1, 2, 2)
    data = [[dev_montage_importance_per_channel[key.upper()]]
            for key in USED_CHANNELS]
    fake_evoked = mne.EvokedArray(data, fake_info)
    fake_evoked.set_montage(ten_twenty_montage)

    mne.viz.plot_topomap(
        fake_evoked.data[:, 0],
        fake_evoked.info,
        axes=plt.gca(),
        show=False,
    )

    plt.gca().set_title(r"\begin{center}\Huge Dev set\end{center}")

    plt.subplot(1, 2, 1)
    data = [
        [
            train_montage_importance_per_channel[key.upper()]
        ] for key in USED_CHANNELS
    ]

    fake_evoked = mne.EvokedArray(data, fake_info)
    fake_evoked.set_montage(ten_twenty_montage)

    mne.viz.plot_topomap(
        fake_evoked.data[:, 0],
        fake_evoked.info,
        axes=plt.gca(),
        show=False,
    )

    plt.gca().set_title(r"\begin{center}\Huge Train set\end{center}")
    plt.tight_layout()
    plt.savefig(filename, format="PDF", transparent=True)
    plt.close()


def latex_topo_focal_starting_points(
        figures_path,
        figure_filename,
        dev_montage_importance_per_channel: dict,
        train_montage_importance_per_channel: dict):
    """Make the figure pdf for the topo plots
    """
    ensure_path(figures_path)
    save_topo_focal_starting_points(
        os.path.join(figures_path, figure_filename),
        dev_montage_importance_per_channel,
        train_montage_importance_per_channel)

    return latex_env(
        environment="center",
        content=r"\includegraphics[width=0.85\textwidth]{%s}" % (
            os.path.join(
                os.getcwd(),
                figures_path,
                figure_filename).replace('\\', '/')))


def _check_dupes_odict(ch_names, pos):
    """Warn if there are duplicates, then turn to ordered dict."""
    ch_names = list(ch_names)
    dups = OrderedDict((ch_name, ch_names.count(ch_name))
                       for ch_name in ch_names)

    dups = OrderedDict((ch_name, count) for ch_name, count in dups.items()
                       if count > 1)
    n = len(dups)
    if n:
        dups = ', '.join(
            f'{ch_name} ({count})' for ch_name, count in dups.items())

        mne.utils.warn(f'Duplicate channel position{mne.utils._pl(n)} '
                       f'found, the last will be '
                       f'used for {dups}')

    return OrderedDict(zip(ch_names, pos))


def _mgh_or_standard(fname, head_size):
    """Used to extract the montage information from the custom montage file"""
    fid_names = ('Nz', 'LPA', 'RPA')

    ch_names_, pos = [], []
    with open(fname) as fid:
        # Ignore units as we will scale later using the norms anyway
        for line in fid:
            if 'Positions\n' in line:
                break
        pos = []
        for line in fid:
            if 'Labels\n' in line:
                break
            pos.append(list(map(float, line.split())))
        for line in fid:
            if not line or not set(line) - {' '}:
                break
            ch_names_.append(line.strip(' ').strip('\n'))

    pos = np.array(pos)
    ch_pos = _check_dupes_odict(ch_names_, pos)
    nasion, lpa, rpa = [ch_pos.pop(n) for n in fid_names]
    scale = head_size / np.median(np.linalg.norm(pos, axis=1))
    for value in ch_pos.values():
        value *= scale
    nasion *= scale
    lpa *= scale
    rpa *= scale

    return mne.channels.montage.make_dig_montage(
        ch_pos=ch_pos,
        coord_frame='unknown',
        nasion=nasion,
        lpa=lpa,
        rpa=rpa)


def extract_data_v1_2_1(filename):
    """Extract and compute the necessary metadata
    for the plots
    """
    info = pd.ExcelFile(filename)  # Open the Excel file
    v1_2_1 = dict()
    v1_2_1['dev'] = info.parse(info.sheet_names[1])
    v1_2_1['train'] = info.parse(info.sheet_names[2])
    present_sz_dev = set(v1_2_1['dev']['Seizure Type'][1:2860].dropna())
    present_sz_dev.remove(679)
    v1_2_1['present_sz_dev'] = present_sz_dev
    v1_2_1['present_sz_train'] = set(
        v1_2_1['train']['Seizure Type'][1:2860].dropna())

    v1_2_1['events_dev'] = [{
        'start': event[0],
        'stop': event[1],
        'event': event[2],
        'duration': event[1] - event[0]}
        for event in np.array(
            v1_2_1['dev'][
                ['Seizure Time',
                 'Unnamed: 13',
                 'Seizure Type']][1:1422].dropna())]

    v1_2_1['events_train'] = [{
        'start': event[0],
        'stop': event[1],
        'event': event[2],
        'duration': event[1] - event[0]}
        for event in np.array(
            v1_2_1['train'][
                ['Seizure Time',
                 'Unnamed: 13',
                 'Seizure Type']][1:2860].dropna())]

    v1_2_1['events_counts_dev'] = {seizure_type: [
        element['event'] for element in v1_2_1['events_dev']].count(
            seizure_type) for seizure_type in v1_2_1['present_sz_dev']}

    v1_2_1['events_counts_train'] = {seizure_type: [
        element['event'] for element in v1_2_1['events_train']].count(
            seizure_type) for seizure_type in v1_2_1['present_sz_train']}

    v1_2_1['events_total_duration_train'] = {seizure_type: sum([
        element['duration'] for element in v1_2_1['events_train']
        if element['event'] == seizure_type])
        for seizure_type in v1_2_1['present_sz_train']}

    v1_2_1['events_total_duration_dev'] = {seizure_type: sum([
        element['duration'] for element in v1_2_1['events_dev']
        if element['event'] == seizure_type])
        for seizure_type in v1_2_1['present_sz_dev']}

    v1_2_1['labels_train'] = ['FNSZ', 'CPSZ', 'SPSZ',
                              'GNSZ', 'TCSZ', 'ABSZ',
                              'TNSZ']

    v1_2_1['data_train'] = [v1_2_1['events_counts_train'][event]
                            for event in v1_2_1['labels_train']]
    v1_2_1['data_duration_train'] = [round(
        v1_2_1['events_total_duration_train'][event], 4)
        for event in v1_2_1['labels_train']]

    v1_2_1['labels_dev'] = ['FNSZ', 'CPSZ', 'SPSZ',
                            'GNSZ', 'TNSZ', 'TCSZ',
                            'ABSZ', 'MYSZ']

    v1_2_1['data_dev'] = [v1_2_1['events_counts_dev'][event]
                          for event in v1_2_1['labels_dev']]

    v1_2_1['data_duration_dev'] = [round(
        v1_2_1['events_total_duration_dev'][event], 4)
        for event in v1_2_1['labels_dev']]

    return v1_2_1


def extract_data_v1_5_2(filename):
    """Extract and compute the necessary metadata for the plots
    """
    v1_5_2 = dict()
    v1_5_2['meta'] = meta = load_pickle_lzma(filename)
    # Extract dev metadata subset
    v1_5_2['dev'] = dev = [e for e in meta if 'dev' in e['filepath']]
    # Extract dev metadata subset
    v1_5_2['train'] = train = [e for e in meta if 'train' in e['filepath']]
    dev_seizures_number_by_type = dict()
    train_seizures_number_by_type = dict()
    v1_5_2['events_total_duration_dev'] = dict()
    v1_5_2['events_total_duration_train'] = dict()

    v1_5_2['dev_seizures_durations_by_type'] = dict()
    v1_5_2['train_seizures_durations_by_type'] = dict()

    for seizure_type in EPILEPTIC_SEIZURE_LABELS:
        dev_seizure_type_number = [True for element in dev if len([
            e['event'] for e in element['annotations_tse']
            if e['event'] == seizure_type]) != 0].count(True)

        train_seizure_type_number = [True for element in train if len([
            e['event'] for e in element['annotations_tse']
            if e['event'] == seizure_type]) != 0].count(True)

        if dev_seizure_type_number > 0:
            seizure_type_durations_per_event = np.hstack(
                [[e['stop'] - e['start']
                  for e in element['annotations_tse']
                  if e['event'] == seizure_type] for element in dev])

            v1_5_2['dev_seizures_durations_by_type'][
                seizure_type] = seizure_type_durations_per_event

            v1_5_2['events_total_duration_dev'][seizure_type] = sum(
                seizure_type_durations_per_event)

            dev_seizures_number_by_type[seizure_type] = len(
                seizure_type_durations_per_event)

        if train_seizure_type_number > 0:
            seizure_type_durations_per_event = np.hstack([[
                e['stop'] - e['start']
                for e in element['annotations_tse']
                if e['event'] == seizure_type] for element in train])

            v1_5_2['train_seizures_durations_by_type'][
                seizure_type] = seizure_type_durations_per_event

            v1_5_2['events_total_duration_train'][seizure_type] = sum(
                seizure_type_durations_per_event)
            train_seizures_number_by_type[seizure_type] = len(
                seizure_type_durations_per_event)

    # Filters the fnsz only
    v1_5_2['dev_filter_focal'] = [
        True if len([
            e['event'] for e in element['annotations_tse']
            if e['event'] in SEIZURE_TYPES_FOCAL]) != 0
        else False for element in v1_5_2['dev']]

    v1_5_2['train_filter_focal'] = [
        True if len([
            e['event'] for e in element['annotations_tse']
            if e['event'] in SEIZURE_TYPES_FOCAL]) != 0
        else False for element in v1_5_2['train']]

    v1_5_2['dev_focal'] = np.array(dev)[v1_5_2['dev_filter_focal']]
    v1_5_2['train_focal'] = np.array(train)[v1_5_2['train_filter_focal']]

    dev_all_fsp = list()
    dev_uniques_fsp = set()
    dev_lines = list()
    for recording in v1_5_2['dev_focal']:
        fsp = focal_starting_points(recording)

        for event in fsp:
            dev_lines.append(
                [event['filepath'][len(PATH_TO_EDF):],
                 event['event'], event['start'],
                 str(sorted(event['montages']))])

            dev_all_fsp.append(sorted(event['montages']))
            dev_uniques_fsp.add("_".join(sorted(event['montages'])))

    # Focal starting points
    v1_5_2['dev_uniques_fsp'] = dev_uniques_fsp
    v1_5_2['dev_all_fsp'] = dev_all_fsp
    v1_5_2['dev_lines'] = dev_lines

    train_all_fsp = list()
    train_uniques_fsp = set()
    train_lines = list()
    for recording in v1_5_2['train_focal']:
        fsp = focal_starting_points(recording)

        for event in fsp:
            train_all_fsp.append(sorted(event['montages']))
            train_uniques_fsp.add("_".join(sorted(event['montages'])))
            train_lines.append([
                event['filepath'][
                    len('/home_nfs/stragierv/TUH_SZ_v1.5.2/TUH/'):],
                event['event'],
                event['start'],
                str(sorted(event['montages']))])

    # Focal starting points
    v1_5_2['train_uniques_fsp'] = train_uniques_fsp
    v1_5_2['train_all_fsp'] = train_all_fsp
    v1_5_2['train_lines'] = train_lines

    # Number or event per montage
    v1_5_2['dev_montage_importance'] = {
        montage: list(np.concatenate(dev_all_fsp)).count(montage)
        for montage in set(list(np.concatenate(dev_all_fsp)))}

    v1_5_2['train_montage_importance'] = {
        montage: list(np.concatenate(train_all_fsp)).count(montage)
        for montage in set(list(np.concatenate(train_all_fsp)))}

    v1_5_2['dev_labels_data'] = np.array(
        [[key, int(value)] for (key, value) in sorted(
            v1_5_2['dev_montage_importance'].items(),
            key=lambda x: x[1],
            reverse=True)]).transpose()

    v1_5_2['train_labels_data'] = np.array(
        [[key, int(value)] for (key, value) in sorted(
            v1_5_2['train_montage_importance'].items(),
            key=lambda x: x[1],
            reverse=True)]).transpose()

    # channels = sorted(
    #     list(
    #         set(
    #             np.concatenate(
    #                 [montage.split('-')
    #                 for montage in v1_5_2[
    #                     'train_labels_data'][0]]))))

    dev_montage_importance_per_channel = dict()

    for k, v in v1_5_2['dev_montage_importance'].items():
        montage_channels = k.split('-')
        dev_montage_importance_per_channel[
            montage_channels[0]] = dev_montage_importance_per_channel.get(
            montage_channels[0], 0) + v

        dev_montage_importance_per_channel[
            montage_channels[1]] = dev_montage_importance_per_channel.get(
            montage_channels[1], 0) + v

    v1_5_2['dev_montage_importance_per_channel'] =\
        dev_montage_importance_per_channel

    train_montage_importance_per_channel = dict()

    for k, v in v1_5_2['train_montage_importance'].items():
        montage_channels = k.split('-')
        train_montage_importance_per_channel[
            montage_channels[0]] = train_montage_importance_per_channel.get(
            montage_channels[0], 0) + v

        train_montage_importance_per_channel[
            montage_channels[1]] = train_montage_importance_per_channel.get(
            montage_channels[1], 0) + v

    v1_5_2['train_montage_importance_per_channel'] =\
        train_montage_importance_per_channel

    # Contains all the recordings that contains
    # at least one focal seizure of any kind
    v1_5_2['dev_focal'] = np.array(dev)[v1_5_2['dev_filter_focal']]
    v1_5_2['train_focal'] = np.array(train)[v1_5_2['train_filter_focal']]

    v1_5_2['events_counts_dev'] = dev_seizures_number_by_type
    v1_5_2['events_counts_train'] = train_seizures_number_by_type

    v1_5_2['labels_train'] = ['FNSZ', 'CPSZ', 'SPSZ',
                              'GNSZ', 'TNSZ', 'TCSZ',
                              'ABSZ', 'MYSZ']

    v1_5_2['data_train'] = [v1_5_2['events_counts_train'][event.lower()]
                            for event in v1_5_2['labels_train']]

    v1_5_2['data_duration_train'] = [round(
        v1_5_2['events_total_duration_train'][event.lower()], 4)
        for event in v1_5_2['labels_train']]

    v1_5_2['labels_dev'] = ['FNSZ', 'CPSZ', 'SPSZ',
                            'GNSZ', 'TNSZ', 'TCSZ',
                            'ABSZ', 'MYSZ']

    v1_5_2['data_dev'] = [v1_5_2['events_counts_dev'][event.lower()]
                          for event in v1_5_2['labels_dev']]

    v1_5_2['data_duration_dev'] = [round(
        v1_5_2['events_total_duration_dev'][event.lower()], 4)
        for event in v1_5_2['labels_dev']]
    return v1_5_2


if __name__ == "__main__":
    # print(
    #     latex_env(
    #         environment="center",
    #         content=r"\includegraphics[width=0.85\textwidth]{myplot.pdf}"))

    # The script is executed in the build folder
    filename_v1_2_1 = os.path.join(os.path.dirname(
        __file__), "_SEIZURES_v29r.xlsx")
    filename_v1_5_2 = os.path.join(os.path.dirname(
        __file__), "metadata.pickle.xz")

    v1_2_1 = extract_data_v1_2_1(filename_v1_2_1)
    v1_5_2 = extract_data_v1_5_2(filename_v1_5_2)

    figure_filename = "pie_sz_v1.2.1.pdf"
    figures_path = "figures"

    print("The script is here:", os.path.dirname(__file__))
    print("You are working here:", os.getcwd())

    # Extract number of activated montages
    dev_lines = np.concatenate(
        list(filter(
            None,
            [extract_recording_propagation(
                recording_meta, True)[0]
                for recording_meta in v1_5_2['dev']],)))

    train_lines = np.concatenate(
        list(filter(
            None,
            [extract_recording_propagation(
                recording_meta, True)[0]
                for recording_meta in v1_5_2['train']])))

    df_dev = pd.DataFrame(
        dev_lines,
        columns=[
            'Filepath',
            r'Seizure Type (tse), {Seizure Types (lbl)}',
            'Time',
            'Activated montages',
            'Number of montages'])

    df_train = pd.DataFrame(
        train_lines,
        columns=[
            'Filepath',
            r'Seizure Type (tse), {Seizure Types (lbl)}',
            'Time',
            'Activated montages',
            'Number of montages'])

    with pd.ExcelWriter(  # pylint: disable=abstract-class-instantiated
            'number_of_activated_montages.xlsx',
            date_format='YYYY-MM-DD',
            datetime_format='YYYY-MM-DD HH:MM:SS',
            engine="xlsxwriter",
            options={'strings_to_numbers': True}) as writer:

        df_dev.to_excel(writer, sheet_name='dev')
        df_train.to_excel(writer, sheet_name='train')

    # # Count the number of recordings with an ECG channel
    # ekg_ecg = [len(
    #     [header['label']
    #      for header in recording_meta[
    #         'headers']
    #         if 'EKG' in header['label']
    #         or 'ECG' in header['label']])
    #         for recording_meta in v1_5_2['meta']]

    # len(ekg_ecg)
    # # 5612

    # set(ekg_ecg)
    # # {0, 1}

    # ekg_ecg.count(1)
    # # 5090

    # save_topo_propagation(
    #     filepath="00006546_s025_t003",
    #     list_of_recording_metadata=v1_5_2['dev'])

    # save_topo_propagation(
    #     filepath="00007128_s002_t002",
    #     list_of_recording_metadata = v1_5_2['train'])

    # print(
    #     latex_barh_montages_focal_starting_points(
    #         figures_path=figures_path,
    #         figure_filename="barh_montages_focal_starting_points.pdf",
    #         v1_5_2=v1_5_2))

    # dev_montage_importance_per_channel = {
    #     'F7': 148,
    #     'T3': 259,
    #     'FP2': 100,
    #     'F4': 62,
    #     'F3': 113,
    #     'C3': 277,
    #     'P3': 117,
    #     'O1': 130,
    #     'CZ': 158,
    #     'FP1': 151,
    #     'T5': 120,
    #     'C4': 203,
    #     'P4': 90,
    #     'T6': 122,
    #     'O2': 122,
    #     'F8': 118,
    #     'T4': 242,
    #     'A2': 41,
    #     'A1': 39}

    # train_montage_importance_per_channel = {
    #     'F7': 1338,
    #     'T3': 2482,
    #     'FP2': 668,
    #     'F4': 713,
    #     'F3': 1076,
    #     'C3': 2708,
    #     'P3': 1405,
    #     'O1': 1479,
    #     'CZ': 1248,
    #     'FP1': 1196,
    #     'T5': 1436,
    #     'C4': 2114,
    #     'P4': 1211,
    #     'T6': 1169,
    #     'O2': 1344,
    #     'F8': 829,
    #     'T4': 1912,
    #     'A2': 249,
    #     'A1': 365}

    # print(
    #     latex_topo_focal_starting_points(
    #         figures_path,
    #         "topo_focal_starting_points_v1.5.2.pdf",
    #         v1_5_2['dev_montage_importance_per_channel'],
    #         v1_5_2['train_montage_importance_per_channel']))

    # histo = durations_histo(
    #     os.path.join(
    #         os.getcwd(),
    #         figures_path,
    #         'histo_v1_2_1-v1_5_2'),
    #     v1_2_1,
    #     v1_5_2)

    # print(
    #     latex_env(
    #         environment="center",
    #         content=r"\includegraphics[width=0.9\textwidth]{%s}" % (
    #             histo['dev']['cpsz'])))

    # print(
    #     latex_pies_sz_per_type(
    #         figures_path,
    #         "pie_sz_dev_v1.2.1-v1.5.2.pdf",
    #         v1_2_1['data_dev'],
    #         v1_2_1['labels_dev'],
    #         v1_5_2['data_dev'],
    #         v1_5_2['labels_dev']))

    # print(
    #     latex_pies_sz_per_type(
    #         figures_path,
    #         "pie_sz_train_v1.2.1-v1.5.2.pdf",
    #         v1_2_1['data_train'],
    #         v1_2_1['labels_train'],
    #         v1_5_2['data_train'],
    #         v1_5_2['labels_train']))

    # print(
    #     latex_pies_sz_duration_per_type(
    #         figures_path,
    #         "pie_sz_duration_dev_v1.2.1-v1.5.2.pdf",
    #         v1_2_1['data_duration_dev'],
    #         v1_2_1['labels_dev'],
    #         v1_2_1['data_duration_train'],
    #         v1_2_1['labels_train']))

    # print(
    #     latex_pies_sz_duration_per_type(
    #         figures_path,
    #         "pie_sz_duration_dev_v1.2.1-v1.5.2.pdf",
    #         v1_2_1['data_duration_dev'],
    #         v1_2_1['labels_dev'],
    #         v1_5_2['data_duration_dev'],
    #         v1_5_2['labels_dev']))

    # print(
    #     latex_pies_sz_duration_per_type(
    #         figures_path,
    #         "pie_sz_duration_train_v1.2.1-v1.5.2.pdf",
    #         v1_2_1['data_duration_train'],
    #         v1_2_1['labels_train'],
    #         v1_5_2['data_duration_train'],
    #         v1_5_2['labels_train']))

    # dev_lines = np.concatenate(
    #     list(
    #         filter(
    #             None,
    #             [extract_recording_propagation(
    #                 recording_meta)[0]
    #                 for recording_meta in v1_5_2['dev']])))

    # train_lines = np.concatenate(
    #     list(
    #         filter(
    #             None,
    #             [extract_recording_propagation(
    #                 recording_meta)[0]
    #                 for recording_meta in v1_5_2['train']])))

    # df_dev = pd.DataFrame(
    #     dev_lines,
    #     # index=[
    #     #     'row 1',
    #     #     'row 2'],
    #     columns=[
    #         'Filepath',
    #         r'Seizure Type (tse), {Seizure Types (lbl)}',
    #         'Time vector',
    #         'Propagation'])

    # df_train = pd.DataFrame(
    #     train_lines,
    #     # index=[
    #     #     'row 1',
    #     #     'row 2'],
    #     columns=[
    #         'Filepath',
    #         r'Seizure Type (tse), {Seizure Types (lbl)}',
    #         'Time vector',
    #         'Propagation'])

    # with pd.ExcelWriter(
    #     'propagation.xlsx',
    #     date_format='YYYY-MM-DD',
    #     datetime_format='YYYY-MM-DD HH:MM:SS')
    #     as writer:

    #     df_dev.to_excel(writer, sheet_name='dev')
    #     df_train.to_excel(writer, sheet_name='train')

    ###############

    # ten_twenty_montage = _mgh_or_standard(
    #     fname=TUSZ_1020,
    #     head_size=0.095)

    # fake_info = mne.create_info(
    #     ch_names=ten_twenty_montage.ch_names,
    #     sfreq=250.,
    #     ch_types='eeg')

    # fake_evoked = mne.EvokedArray(data, fake_info)
    # fake_evoked.set_montage(ten_twenty_montage)

    # plt.figure(figsize=(16, 9))
    # plt.rc('text', usetex=True)
    # plt.rcParams["font.family"] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = "Calibri"
    # plt.subplot(1, 2, 2)

    # data = [[
    #     dev_montage_importance_per_channel[
    #         key.upper()]] for key in USED_CHANNELS]

    # fake_evoked = mne.EvokedArray(data, fake_info)
    # fake_evoked.set_montage(ten_twenty_montage)
    # mne.viz.plot_topomap(
    #     fake_evoked.data[:, 0],
    #     fake_evoked.info,
    #     axes=plt.gca(),
    #     show=False)

    # plt.gca().set_title(
    #     r"\begin{center}\Huge Dev set\end{center}")

    # plt.subplot(1, 2, 1)
    # data = [[
    #     train_montage_importance_per_channel[
    #         key.upper()]]
    #     for key in USED_CHANNELS]

    # fake_evoked = mne.EvokedArray(data, fake_info)
    # fake_evoked.set_montage(ten_twenty_montage)
    # mne.viz.plot_topomap(
    #     fake_evoked.data[:, 0],
    #     fake_evoked.info,
    #     axes=plt.gca(),
    #     show=False)

    # plt.gca().set_title(
    #     r"\begin{center}\Huge Train set\end{center}")
    # plt.tight_layout()
    # plt.savefig(filename, format="PDF", transparent=True)
    # plt.close()

    # # Find file meta from filename
    # for index, element in enumerate(v1_5_2['train']):
    #     if filepath in element['filepath']:
    #         print(index)

    # seizure_index = 2
    # recording_file = os.path.basename(
    #     v1_5_2['train'][4207]['filepath'])
    # propagation = extract_recording_propagation(
    #     v1_5_2['train'][4207])[1][seizure_index]['propagation']
    # times = np.array(extract_recording_propagation(
    #     v1_5_2['train'][4207])[1][seizure_index]['times'])
    # for index, montages in enumerate(propagation):
    #     list(
    #         np.concatenate(
    #             [channels.split(
    #                 '-')
    #                 for channels in [montage for montage in montages]]))

    # data = np.array(
    #     [np.array(
    #         [pattern.count(
    #             key.upper()) for key in USED_CHANNELS])/max(
    #                 [pattern.count(
    #                     key.upper()) for key in USED_CHANNELS])
    #         for pattern in [list(
    #             np.concatenate(
    #                 [channels.split(
    #                     '-') for channels in [
    #                         montage for montage in montages]]))
    #                         for montages in propagation]]).transpose()

    # data = np.hstack((data, data[:, [-1]])) # Repeat last frame

    # ten_twenty_montage = _mgh_or_standard(fname=TUSZ_1020, head_size=0.095)
    # fake_info = mne.create_info(
    #                         ch_names=ten_twenty_montage.ch_names,
    #                         sfreq=250.,
    #                         ch_types='eeg')
    #
    # fake_evoked = mne.EvokedArray(data, fake_info)
    # fake_evoked.set_montage(ten_twenty_montage)
    # fake_evoked.times = times

    # filelist = list()
    # for i in range(data.shape[1]):
    #     plt.figure(figsize=(16, 9))
    #     mne.viz.plot_topomap(
    #         fake_evoked.data[:, i],
    #         fake_evoked.info,
    #         axes=plt.gca(),
    #         show=False)

    #     # plt.show(block=False)
    #     plt.gca().set_xlabel(str(propagation[i]))
    #     plt.savefig(
    #         '%s_%s.png' % (
    #             recording_file,
    #             str(i)), format="PNG", dpi=600)
    #     filelist.append(
    #         '%s_%s.png' % (
    #             recording_file, str(i)))

    # vid_path = '%s_%s.mp4' % (recording_file, seizure_index)
    # dilatation = 1 # 10
    # fps = 2

    # fig, anim = fake_evoked.animate_topomap(
    #     times=times,
    #     ch_type=None,
    #     frame_rate=fps,
    #     time_unit='s',
    #     blit=False,
    #     show=True)
    # anim.save(vid_path, dpi=300) #  writer='ffmpeg',
    # import skvideo.io
    # import imageio

    # outputdata = np.random.random(size=(5, 480, 680, 3)) * 255
    # outputdata = outputdata.astype(np.uint8)

    # writer = skvideo.io.FFmpegWriter("outputvideo.mp4")
    # for i in xrange(5):
    #         writer.writeFrame(outputdata[i, :, :, :])
    # writer.close()

    # with imageio.get_writer(vid_path, mode='I', fps=fps) as writer:
    # with skvideo.io.FFmpegWriter("outputvideo.mp4") as writer:
    #     for repeat_frame, frame_path in zip(
    #         (times[1:]-times[:-1]) * fps *dilatation, filelist):
    #         frame = np.array(frame, imageio.imread(frame_path))
    #         print(repeat_frame)
    #         for _ in range(round(repeat_frame)):
    #             writer.writeFrame(frame)
    # optimize(gif_path)
