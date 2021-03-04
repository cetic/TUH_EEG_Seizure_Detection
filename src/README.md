# Sources (src)

This folder contains the source files of this project.

## Requirements

Checks the `requirements.txt` file at the root of this repository.

`Python 3.8.6rc1` has been used during the development of this script (look to the main `README.md` to get the installation steps).

The dependencies can be installed using `pip`:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade requirements.txt
```

## Usage

All the next script have an built in help.

1. Download the dataset [2 hours]

   ```bash
   stragierv@datamaster:~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection$ \
   python3 ./src/tuh_sz_download.py\
   https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/v1.5.2/ \
   -u nedc -p nedc_resources --path ~/path/to/dataset/
   ```

2. After downloading the dataset, the metadata must be extracted to a `.pickle.xz` file [6 minutes]

   ```bash
   # Start interactive session (could be done in a sbatch file)
   stragierv@datamaster:~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection$ \
   srun --partition=debug -c 16 --mem=30G --pty bash

   # Start the metadata extraction
   stragierv@datamaster:~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection$ \
   python3 src/tools/tuh_sz_extract_metadata.py \
   ~/path/to/dataset/ \
   ~/path/to/dataset/_DOCS/seizures_v36r.xlsx \
   ~/path/to/metadata.pickle.xz
   ```

3. When the metadata are extracted, the useful signals and target can be extracted to a `.h5` file [2 hours]

   ```bash
   # Start interactive session (could be done in a sbatch file)
   stragierv@datamaster:~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection$ \
   srun --partition=debug -c 16 --mem=30G --pty bash

   # Start the signal extraction
   stragierv@datamaster:~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection$ \
   python3 src/dataset_to_hdf5.py ~/TUH_SZ_v1.5.2/TUH/ \
   ~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection/src/tools/metadata.pickle.xz \
   ~/path/to/dataset.h5
   ```

4. From the `.h5` the features can be extracted to the same file [2 hours using `multiprocessing`]

   ```bash
   # Start interactive session (could be done in a sbatch file)
   stragierv@datamaster:~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection$ \
   srun --partition=debug -c 16 --mem=30G --pty bash

   # Start the signal features extraction
   stragierv@datamaster:~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection$ \
   python3 src/dataset_h5_to_feature_h5.py ~/path/to/dataset.h5 -y \
   --features ALL --targets MIN MEAN MAX --window 4 --step 1 --padding 0
   ```

5. Before training any model, the one hot vectors have to be created to 4 separated `.npy` file (`X_train`, `Y_train`, `X_dev` and `Y_dev`) [15 minutes]

   ```bash
   # Start interactive session (could be done in a sbatch file)
   stragierv@datamaster:~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection$ \
   srun --partition=debug -c 16 --mem=30G --pty bash

   # Start generating the hot vectors
   stragierv@datamaster:~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection$ \
   python3 src/save_input_vectors.py ~/dataset.h5 -th 0.8 -rp -a -y
   ```

   By default it outputs the hot vectors file in the `input_vectors` directory as well as a `.info` file.

6. Here we can train a XGBoost model using the desired input vectors [< 10 minutes on GPU]

   ```bash
   # Start interactive session (could be done in a sbatch file)
   stragierv@datamaster:~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection$ \
   srun --partition=gpu -N 1 -c 16 --job-name=xgb --mem=60G --gres="gpu:3" --pty bash

   # Start training the model
   stragierv@datamaster:~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection$ \
   python3 src/train_xgb.py \
   input_vectors/2020-12-04_21:29_\
   792135c8a820f347b477a36b061f961c.info -y
   ```

7. Then rough metrics can be computed using the `.h5` file, the `.npy` files and the `.model` file [20 minutes]

   ```bash
   # Start interactive session (could be done in a sbatch file)
   stragierv@datamaster:~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection$ \
   srun --partition=debug -c 16 --mem=30G --pty bash

   # Start training the model
   stragierv@datamaster:~/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection$ \
   python3 src/metrics.py \
   /home_nfs/stragierv/TUH_SZ_v1.5.2/Epilepsy_Seizure_Detection/\
   models/2021-02-23_02h20m_\
   (2021-02-23_02h01m_5ead98ad3c4c54869bee137756c5d14c).info \
   -sw 12 \
   -sr 0.666666666 \
   -df tuh ar gsz \
   -tf tuh ar gsz \
   --xlsx_file AR_GSZ_SMOOTHING_8_12.xlsx
   ```
