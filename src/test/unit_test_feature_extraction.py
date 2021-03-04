"""
Authors: Vincent Stragier

Logs:
26/09/2020 (Vincent Stragier)
- implement test cases for the feature_extraction module
"""
import os
import sys
import unittest

import numpy as np

import feature_extraction as fe

# Add the path of the script to the module
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class TestFeatureExtraction(unittest.TestCase):

    def test_exploit_edf(self):
        file_path = fe.extract_files_list(
            path=os.path.join(
                os.path.dirname(__file__),
                'data'),
            extension_filter='tse')[0] + '.edf'

        # Produce an OS error
        print('"exploit_edf()" will produce an "error" message')
        _, _, exit_code, _, _, _, _ = fe.exploit_edf(
            filename=file_path + '.error',
        )

        print('"exploit_edf()" did produce the "error" message')
        self.assertEqual(exit_code, 2)

        # I don't know a clean way to break the line.
        eeg_signal_labels, eeg_signal, exit_code, sampling_rate, tot_dur, nb_ch, nb_eeg_ch = fe.exploit_edf(filename=file_path)  # noqa: E501

        # Extract the information from a .edf file
        self.assertEqual(len(eeg_signal_labels), 32)
        self.assertEqual(np.array(eeg_signal).shape, (32, 8000))
        self.assertEqual(exit_code, 0)
        self.assertEqual(sampling_rate, 400)
        self.assertEqual(tot_dur, 20)
        self.assertEqual(nb_ch, 32)
        self.assertEqual(nb_eeg_ch, 32)

    def test_file_len(self):
        import tempfile

        filename = 'temp.txt'
        n_line = int(6)  # Number of lines

        with tempfile.TemporaryDirectory() as tmpdirname:
            with open(os.path.join(tmpdirname, filename), mode='w+') as f:
                for i in range(1, n_line + 1):
                    f.write('line %s\n' % (i))

            self.assertEqual(
                fe.file_len(
                    os.path.join(
                        tmpdirname,
                        filename,
                    ),
                ),
                n_line,
            )

    def test_extract_files_list(self):
        """# noqa: RST301

        test
            │   unit_test_feature_extraction.py
            │
            └───data
                └───00000258
                    ├───s002_2003_07_21
                    │       00000258_s002.txt
                    │       00000258_s002_t000.edf
                    │       00000258_s002_t000.lbl
                    │       00000258_s002_t000.tse
                    │       00000258_s002_t002.edf
                    │       00000258_s002_t002.lbl
                    │       00000258_s002_t002.tse
                    │
                    └───s003_2003_07_22
                            00000258_s003.txt
                            00000258_s003_t000.edf
                            00000258_s003_t000.lbl
                            00000258_s003_t000.tse
                            00000258_s003_t001.edf
                            00000258_s003_t001.lbl
                            00000258_s003_t001.tse
                            00000258_s003_t002.edf
                            00000258_s003_t002.lbl
                            00000258_s003_t002.tse
                            00000258_s003_t003.edf
                            00000258_s003_t003.lbl
                            00000258_s003_t003.tse
                            00000258_s003_t004.edf
                            00000258_s003_t004.lbl
                            00000258_s003_t004.tse
                            00000258_s003_t005.edf
                            00000258_s003_t005.lbl
                            00000258_s003_t005.tse
        """
        # print('\n'.join(fe.extract_files_list(
        #     path=os.path.join(
        #         os.path.dirname(__file__),
        #         'data',
        #     ),
        #     extension_filter = 'tse')))
        self.assertEqual(len(fe.extract_files_list(
            path=os.path.join(
                os.path.dirname(__file__),
                'data',
            ),
            extension_filter='tse')), 8)

        self.assertEqual(len(fe.extract_files_list(
            path=os.path.join(
                os.path.dirname(__file__),
                'data',
            ),
            extension_filter='pdf')), 0)

    def test_extract_info_from_annotation(self):
        file_path = fe.extract_files_list(
            path=os.path.join(
                os.path.dirname(__file__),
                'data',
            ),
            extension_filter='tse',
        )[0]

        eeg_signal_labels, eeg_signal, _, _, _, _, _ = fe.exploit_edf(
            filename=file_path + '.edf',
        )

        # Extract the information from a .lbl file
        channels = fe.extract_info_from_annotation(
            filename=file_path + '.lbl',
            signal_base=eeg_signal,
            label_names=eeg_signal_labels,
        )

        self.assertEqual(channels.shape, (8000, 22))

    def test_extract_features(self):
        file_path = fe.extract_files_list(
            path=os.path.join(
                os.path.dirname(__file__),
                'data',
            ),
            extension_filter='tse',
        )[0]

        eeg_signal_labels, eeg_signal, _, _, _, _, _ = fe.exploit_edf(
            filename=file_path + '.edf',
        )

        # Extract the information from a .lbl file
        channels = fe.extract_info_from_annotation(
            filename=file_path + '.lbl',
            signal_base=eeg_signal,
            label_names=eeg_signal_labels,
        )

        del channels

    """
    def test_isupper_smth(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

        self.assertEqual('foo'.upper(), 'FOO')

        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)
    """


if __name__ == '__main__':
    # print(help(fe.bin_power))
    unittest.main()
