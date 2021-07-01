"""This script contains the filters used to filters the dataset recordings."""
import os

GSZ_AND_BCKG = {'absz', 'tcsz', 'gnsz', 'tnsz', 'mysz', 'bckg'}
FSZ_AND_BCKG = {'cpsz', 'spsz', 'fnsz', 'bckg'}


def filter_eval(list_of_filters: list, list_of_recordings_metadata: list):
    """Filter a list of recordings according to a list of metadata.

    Args:
        list_of_filters: is the list of filter to apply on the list of data.
        list_of_recordings_metadata: is a list of recordings metadata.

    Returns:
        A list of a generator for the filtered list.

    Raises:
        NotImplementedError: for all the non implemented filters.

    """
    dataset = list_of_filters[0].lower()
    list_of_filters = (f.lower() for f in list_of_filters[1:])

    filtered_list = list_of_recordings_metadata
    if dataset == 'tuh':
        for filter_ in list_of_filters:
            if filter_ == 'gsz':  # Generalized and background
                filtered_list = TUHFilters.gsz(filtered_list)

            elif filter_ == 'fsz':  # Focalised and background
                filtered_list = TUHFilters.fsz(filtered_list)

            elif filter_ == 'ar':  # AR montage only
                filtered_list = TUHFilters.ar(filtered_list)

            elif filter_ == 'le':  # LE montage only
                filtered_list = TUHFilters.le(filtered_list)

            elif filter_ == 'ar_le':  # AR/LE montage only
                filtered_list = TUHFilters.ar_le(filtered_list)

            else:
                raise NotImplementedError(
                    '"{0}" is not implemented yet.'.format(filter_),
                )

    elif dataset == 'stluc':
        raise NotImplementedError(
            'filters for the Saint-Luc dataset are not implemented yet.',
        )

    else:
        raise NotImplementedError('Unknown dataset.')

    return filtered_list


def metalist_to_filelist(metalist: list):
    """Convert the list of metadata to a list of filename.

    Args:
        metalist: a list of recording metadata.

    Returns:
        A list of filepaths.
    """
    return (os.path.basename(m['filepath']) for m in metalist)


class TUHFilters:
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
    def le(metalist: list):
        """Return only the LE recordings.

        Args:
            metalist: a list of recording metadata.

        Returns:
            Return a generator of recording metadata of metadata.
        """
        return (
            m for m in metalist
            if '_le/' in m['filepath']
        )

    @staticmethod
    def ar_le(metalist: list):
        """Return only the AR/LE recordings.

        Args:
            metalist: a list of recording metadata.

        Returns:
            Return a generator of recording metadata of metadata.
        """
        return (
            m for m in metalist
            if '_ar/' in m['filepath'] or '_le/' in m['filepath']
        )

    @staticmethod
    def gsz(metalist: list):
        """Return only the generalized seizures recordings.

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
