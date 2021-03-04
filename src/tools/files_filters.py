"""This script contains the filters used to filters the dataset recordings."""
import os

GSZ_AND_BCKG = {'absz', 'tcsz', 'gnsz', 'tnsz', 'mysz', 'bckg'}
FSZ_AND_BCKG = {'cpsz', 'spsz', 'fnsz', 'bckg'}


def filter_eval(list_of_filters: list, list_of_recordings_metadata: list):
    dataset = list_of_filters[0].lower()
    list_of_filters = (f.lower() for f in list_of_filters[1:])

    filtered_list = list_of_recordings_metadata
    if dataset == 'tuh':
        for filter_ in list_of_filters:
            if filter_ == 'gsz':  # Generalised and background
                filtered_list = TUH_filters.gsz(filtered_list)

            elif filter_ == 'fsz':  # Focalised and background
                filtered_list = TUH_filters.fsz(filtered_list)

            elif filter_ == 'ar':  # AR montage only
                filtered_list = TUH_filters.ar(filtered_list)

            else:
                raise NotImplementedError('\'{0}\' is not implemented yet.'.format(filter_))

    elif dataset == 'stluc':
        raise NotImplementedError('filters for the Saint-Luc dataset are not implemented yet.')

    else:
        raise NotImplementedError('unkwnow dataset.')

    return filtered_list


def metalist_to_filelist(metalist: list):
    """Convert the list of metadata to a list of filename.

    Args:
        metalist: a list of recording metadata.

    Returns:
        A list of filepaths.
    """
    return (os.path.basename(m['filepath']) for m in metalist)


class TUH_filters:
    """Provide filters for the TUH SZ dataset recordings."""

    @staticmethod
    def ar(metalist: list):
        """Return only the AR recordings.

        Args:
            metalist: a list of recording metadata.

        Returns:
            Return a generator of recording metadata of metadata.
        """
        return (
            m for m in metalist
            if '_ar/' in m['filepath']
        )

    @staticmethod
    def gsz(metalist: list):
        """Return only the generalised seizures recordings.

        Args:
            metalist: a list of recording metadata.

        Returns:
            A list of metadata.
        """
        gsz = []
        for patient in metalist:
            # Extract patient's labels
            patient_lbl = {
                lbl['event'] for lbl in patient['annotations_tse']
            }

            if (patient_lbl & GSZ_AND_BCKG) == patient_lbl:
                gsz.append(patient)

        return gsz

    @staticmethod
    def fsz(metalist: list):
        """Return only the focalised seizures recordings.

        Args:
            metalist: a list of recording metadata.

        Returns:
            Return a list of recordings metadata.
        """
        fsz = []
        for patient in metalist:
            # Extract patient's labels
            patient_lbl = {
                lbl['event'] for lbl in patient['annotations_tse']
            }

            if (patient_lbl & FSZ_AND_BCKG) == patient_lbl:
                fsz.append(patient)

        return fsz
