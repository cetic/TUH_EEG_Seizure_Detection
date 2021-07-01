"""Train the XGBoost model.

This script imports the needed features in memory
(after choosing the wanted features) and train the model.

Author:
    - Vincent Stragier

Logs:
    - (2020/11/10)
        - Create this script

srun --partition=gpu --job-name=xgb -N 1 -n 16 --mem=120G -t 1:00:00
--gres="gpu:3" --mail-type=ALL
--mail-user=vincent.stragier@student.umons.ac.be --pty bash
"""
import datetime
import json
import os
import sys

import git
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(
    os.path.join(SCRIPT_PATH, '..', 'models'),
)


def main(
    args,
    path: str,
    prefix: str,
    files_prefix: str,
    model_path: str,
    date_str: str,
):
    """Run the main process.

    Args:
        args: the parsed arguments.
        path: the path to the input data.
        prefix: the prefix of the input data.
        files_prefix: the prefix of the output files.
        model_path: the path where to store the model,
            the training config file and the importance of the features.
        date_str: the current date.
    """
    input_col = '{0}/{1}_columns.npy'.format(path, prefix)
    input_x_train = '{0}/{1}_x_train.npy'.format(path, prefix)
    input_y_train = '{0}/{1}_y_train.npy'.format(path, prefix)
    input_x_dev = '{0}/{1}_x_dev.npy'.format(path, prefix)
    input_y_dev = '{0}/{1}_y_dev.npy'.format(path, prefix)
    model_file = '{0}/{1}.model'.format(model_path, files_prefix)
    config_file = '{0}/{1}.config'.format(model_path, files_prefix)
    importance_file = '{0}/{1}.xlsx'.format(
        model_path, files_prefix,
    )

    if args.configuration is None:
        config = {
            'max_depth': 3,
            'n_estimators': 400,
            'min_child_weight': 1,
            'tree_method': 'gpu_hist',
            'learning_rate': 0.07,
            'objective': 'binary:logistic',
        }

    else:
        with open(file=args.configuration, mode='r') as json_file:
            config = json.load(json_file)

    print('Start training...')
    columns = np.load(input_col)
    x_train = np.load(input_x_train)
    y_train = np.load(input_y_train)

    if args.split_train:
        x_train, x_dev, y_train, y_dev = train_test_split(
            x_train,
            y_train,
            test_size=args.test_size,
            stratify=y_train,
        )

    else:
        x_dev = np.load(input_x_dev)
        y_dev = np.load(input_y_dev)

    # Imbalance ratio.
    # Allows to compensate the imbalance between the classes.
    imbalance_ratio = len(y_train) / np.sum(y_train)
    print('Imbalance:', imbalance_ratio)
    config.update({'scale_pos_weight': imbalance_ratio})

    # Train the classifier
    # 'rmse' for root mean squared error.
    # 'mae' for mean absolute error.
    # 'logloss' for binary logarithmic loss
    # and 'mlogloss' for multi-class log loss (cross entropy).
    # 'error' for classification error.
    # 'auc' for area under ROC curve.
    with open(file=config_file, mode='w+') as cfg_file:
        json.dump(config, cfg_file)

    clf = xgb.XGBClassifier(**config)

    clf.fit(
        x_train, y_train,
        eval_set=[(x_dev, y_dev)],
        eval_metric=['auc'],
        early_stopping_rounds=400,
        verbose=True,
    )

    bst = clf.get_booster()

    bst.save_model(model_file)
    imp = clf.feature_importances_

    for feature_name, importance in zip(imp, columns):
        print(
            'Feature name: {0}, importance: {1}'.format(
                importance,
                feature_name,
            ),
        )

    # Output the features importance in an Excel file
    df_importances = pd.DataFrame(
        np.hstack(
            (
                np.array(
                    sorted(
                        zip(columns, imp, imp/np.max(imp)),
                        key=lambda line: line[1],
                        reverse=True,
                    ),
                ),
                np.array([np.cumsum(sorted(imp, reverse=True))]).transpose(),
            ),
        ),
        columns=[
            'Feature name',
            'Importance',
            'Normalized importance',
            'Cumulated importance',
        ],
    )

    df_importances['Importance'] = pd.to_numeric(df_importances['Importance'])
    df_importances['Normalized importance'] = pd.to_numeric(
        df_importances['Normalized importance'],
    )
    df_importances['Cumulated importance'] = pd.to_numeric(
        df_importances['Cumulated importance'],
    )

    with pd.ExcelWriter(  # pylint: disable=abstract-class-instantiated
        importance_file,
        date_format='YYYY-MM-DD',
        datetime_format='YYYY-MM-DD HH:MM:SS',
        engine='xlsxwriter',
    ) as writer:

        df_importances.to_excel(
            writer,
            sheet_name='Features importance',
            index=False,
        )

        # https://stackoverflow.com/a/40535454
        worksheet = writer.sheets['Features importance']

        for idx, col in enumerate(df_importances):  # loop through all columns
            series = df_importances[col]
            max_len = max(
                (
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name)),  # len of column name/header
                ),
            ) + 2  # adding a little extra space

            worksheet.set_column(idx, idx, max_len)  # set column width


if __name__ == '__main__':
    # Create the script arguments parser
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=True)

    parser.add_argument(
        'input_vectors_info',
        type=str,
        help='one hot vector info path (used to get the prefix)',
    )

    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        help='path to the model',
        default=None,
    )

    parser.add_argument(
        '-fp',
        '--files_prefix',
        type=str,
        help='Prefix of the files names.',
        default=None,
    )

    parser.add_argument(
        '--split_train',
        action='store_true',
        help=(
            'split the dataset in 80/20 partitions for training and validation'
            ' (proportion can be changed using the test_size argument)'
        ),
        default=False,
    )

    parser.add_argument(
        '--test_size',
        type=float,
        help=(
            'proportion of the training set to use for validation'
            ' (by default 0.2)'
        ),
        default=0.2,
    )

    parser.add_argument(
        '-c',
        '--configuration',
        type=str,
        help=(
            'path to a ".json" or ".config" containing the XGBoost '
            'configuration'
        ),
        default=None,
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

    print('The following arguments have been parsed:')
    for k, v in vars(args).items():
        print('{0}: {1}'.format(k, v))

    if not args.no:
        print()
        if not args.yes:
            try:
                while True:
                    conti = input('Do you want to run the program (yes/no)? ')
                    if conti in ['y', 'yes', 'Y', 'YES']:
                        break

                    elif conti in ['n', 'no', 'N', 'NO']:
                        exit()

            except KeyboardInterrupt:
                print(
                    '\nThe user requested the end of the program'
                    ' (KeyboardInterrupt).',
                )

                exit()
    else:
        exit()

    # Get repo information
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
    except Exception:
        sha = 'none'

    # Generate an unique prefix
    date = datetime.datetime.now()
    date_str = date.strftime('%Y-%m-%d_%Hh%Mm')

    prefix = os.path.basename(args.input_vectors_info)
    prefix = prefix[:prefix.rfind('.')]

    # Set model argument
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = MODEL_PATH

    if args.files_prefix is not None:
        files_prefix = args.files_prefix
    else:
        files_prefix = '{0}_({1})'.format(date_str, prefix)

    # Generate filename and path for the info file
    info_filename = os.path.join(
        model_path, '{0}.info'.format(files_prefix),
    )

    path = os.path.dirname(os.path.abspath(args.input_vectors_info))

    print('Create the information file ({0})'.format(info_filename))
    with open(info_filename, 'w+') as info_file:
        info_file.write('[input vectors path]\n')
        info_file.write('{0}\n'.format(path))

        info_file.write('[prefix]\n')
        info_file.write('{0}\n'.format(prefix))

        info_file.write('[sys.argv]\n')
        info_file.write('{0}\n'.format(str(sys.argv)))

        info_file.write('[parsed arguments]\n')
        for key, value in vars(args).items():
            info_file.write('{0}: {1}\n'.format(key, value))

    main(
        args=args,
        path=path,
        prefix=prefix,
        files_prefix=files_prefix,
        model_path=model_path,
        date_str=date_str,
    )
