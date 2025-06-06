import os
from zipfile import ZipFile
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import argparse

# For reproducibility
np.random.seed(88)

def download_url(url, save_path, chunk_size = 128):
    """ Download data util function"""
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size = chunk_size):
            fd.write(chunk)

def data_preparation():
    """Download, unzip and partition ECG5000 dataset"""
    
    # Parsing input arguments
    desc_str = "Data downloading and partitioning"
    parser = argparse.ArgumentParser(description = desc_str)

    # Arguments
    down_str = "Download data 1, otherwise 0"
    tr_str = "Percentage of normal instances to be placed in the training set."
    val_str_n = ("Percentage of normal instances to be placed in the validation set:"
                 "half of these are used to control model training, "
                 "the remaining ones for model selection.")
    val_str_a = ("Percentage of anomalous instances "
                 "w.r.t. normal instances in the training set"
                 "used for model selection"
                 "(e.g, if the training set contains 95 normal instances,"
                 " if you set this parameter equal to 0.05, then,"
                 "5 anomalous instances will be selected)."
                 "The remamining anomalous instances are placed in the test set.")
    parser.add_argument("download", type = int, help = down_str)
    parser.add_argument("perc_tr_n", type = float, help = tr_str)
    parser.add_argument("perc_val_n", type = float, help = val_str_n)
    parser.add_argument("perc_val_an", type = float, help = val_str_a )
    args = parser.parse_args()

    # Creating folder
    if args.download:
        data_path = "data/ECG5000"
        if not os.path.exists(data_path):
            os.mkdir(data_path)
            print('Create ECG5000 folder')

        # Data dowloading
        url = 'http://www.timeseriesclassification.com/Downloads/ECG5000.zip'
        save_path = 'data/ECG5000.zip'
        print('### Starting downloading ECG5000 data ###')
        download_url(url, save_path)
        print('### Download done! ###')

        # Unzipping
        file_name = "data/ECG5000.zip"
        save_path = "data/ECG5000"
        with ZipFile(file_name, 'r') as zip:
            print('Extracting all the files now...')
            zip.extractall(save_path)
            print('Extraction done!')

        # Removing useless files
        os.remove('data/ECG5000.zip')
        os.remove('data/ECG5000/ECG5000_TRAIN.arff')
        os.remove('data/ECG5000/ECG5000_TRAIN.ts')
        os.remove('data/ECG5000/ECG5000_TEST.ts')
        os.remove('data/ECG5000/ECG5000_TEST.arff')
        os.remove('data/ECG5000/ECG5000.txt')

    # Creating folder where to save numpy data
    data_path = "/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy"
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        print('Create ECG5000/numpy folder')

    
    # Loading data
    train = pd.read_table('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/ECG5000_TEST.txt', sep=r'\s{2,}', engine='python', header=None)
    test = pd.read_table('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/ECG5000_TRAIN.txt', sep=r'\s{2,}', engine='python', header=None)

    # Concatenating
    df = pd.concat([train, test])
    new_columns = list(df.columns)
    new_columns[0] = 'Class'
    df.columns = new_columns

    # Dividing in normal and not normal data
    normal = df.loc[df.Class == 1]
    anomaly = df.loc[df.Class != 1]

    # Splitting normal data in training, validation and test set
    X_train_n, X_val_n = train_test_split(normal, random_state = 88, test_size = 1 - args.perc_tr_n)
    X_val_n, X_test_n = train_test_split(X_val_n, random_state = 88, test_size = 1- args.perc_val_n)

    # Splitting validation data into two folds: the former to control model training, the latter
    # for model selection

    X_val_nA, X_val_nB = train_test_split(X_val_n, random_state=88, test_size = 0.5)

    # Splitting anomalous data in validation and test set
    perc_anol_all = args.perc_val_an
    n_anol = len(X_train_n) * perc_anol_all / (1 - perc_anol_all)
    perc_anol_val_a = n_anol / len(anomaly)
    perc_anol_test_a = 1 - perc_anol_val_a
    X_val_a, X_test_a = train_test_split(anomaly,
                                         random_state = 88,
                                         test_size = perc_anol_test_a,
                                         stratify = anomaly.Class)

    # Splitting anomalous validation data into two splitting: the former for model training the latter
    # for model selection
    X_val_aA, X_val_aB = train_test_split(X_val_a, random_state = 88, test_size = 0.5)


    # Training data ONLY NORMAL
    X_train = X_train_n.iloc[:, 1:].values
    y_train = X_train_n.iloc[:, 0].values

    # Training data NORMAL + ANOL
    X_train_p = pd.concat([X_train_n.iloc[:, 1:], X_val_aA.iloc[:, 1:]]).values
    y_train_p = pd.concat([X_train_n.iloc[:, 0], X_val_aA.iloc[:, 0]]).values

    # Validation data only normal to control model training
    X_val = X_val_n.iloc[:, 1:].values
    y_val = X_val_n.iloc[:, 0].values

    # Validation data: both normal and anomalous data for model selection: AUC LOSS
    X_val_p = pd.concat([X_val_nB.iloc[:, 1:], X_val_aB.iloc[:, 1:]]).values
    y_val_p = pd.concat([X_val_nB.iloc[:, 0], X_val_aB.iloc[:, 0]]).values

    # Validation data: both normal and anomalous data for model selection: NO AUC LOSS
    X_val_p_full = pd.concat([X_val_nB.iloc[:, 1:], X_val_a.iloc[:, 1:]]).values
    y_val_p_full = pd.concat([X_val_nB.iloc[:, 0], X_val_a.iloc[:, 0]]).values

    # Test data
    X_test = pd.concat([X_test_n.iloc[:, 1:], X_test_a.iloc[:, 1:]]).values
    y_test = pd.concat([X_test_n.iloc[:, 0], X_test_a.iloc[:, 0]]).values

    # Saving training data (only normal instances)
    np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/X_train.npy', X_train)
    np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/y_train.npy', y_train)

    # Saving training data (normal instances + anomalous)
    np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/X_train_p.npy', X_train_p)
    np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/y_train_p.npy', y_train_p)

    # Saving validation data (only normal instances to control model training)
    np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/X_val.npy', X_val)
    np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/y_val.npy', y_val)
    
    # Saving validation data (normal + anomalous instances to perform model selection yes AUC)
    np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/X_val_p.npy', X_val_p)
    np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/y_val_p.npy', y_val_p)

    # Saving validation data (normal + anomalous instances to perform model selection no AUC)
    np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/X_val_p_full.npy', X_val_p_full)
    np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/y_val_p_full.npy', y_val_p_full)
    
    # Saving test data (normal + anomalous instances)
    np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/X_test.npy', X_test)
    np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/y_test.npy', y_test)

    print('Saved data in numpy')
    
    
    
def convert_ucr_to_numpy(root_dir, val_size=0.1, random_state=88):
        """
        For each UCR dataset folder in root_dir, read
        DATASETNAME/DATASETNAME_TRAIN.tsv and DATASETNAME/DATASETNAME_TEST.tsv,
        split the TRAIN set into train+val, and save:
            X_train.npy, y_train.npy,
            X_val.npy,   y_val.npy,
            X_test.npy,  y_test.npy
        into a `numpy/` subfolder of each dataset.
        """
        for ds in os.listdir(root_dir):
            ds_path = os.path.join(root_dir, ds)
            if not os.path.isdir(ds_path):
                continue

            train_file = os.path.join(ds_path, f"{ds}_TRAIN.tsv")
            test_file  = os.path.join(ds_path, f"{ds}_TEST.tsv")
            if not (os.path.exists(train_file) and os.path.exists(test_file)):
                continue

            # load
            df_train = pd.read_csv(train_file, sep='\t', header=None)
            df_test  = pd.read_csv(test_file,  sep='\t', header=None)

            # split features/labels
            X_full = df_train.iloc[:, 1:].values
            y_full = df_train.iloc[:, 0].values
            X_test = df_test.iloc[:, 1:].values
            y_test = df_test.iloc[:, 0].values

            # # train/val split (stratified)
            # X_train, X_val, y_train, y_val = train_test_split(
            #     X_full, y_full,
            #     test_size=val_size,
            #     random_state=random_state,
            #     stratify=y_full
            # )
            
            # check if stratified split is possible
            # if not, use non-stratified split
            n_samples = len(y_full)
            n_classes = len(np.unique(y_full))
            n_val     = int(np.ceil(val_size * n_samples))

            unique, counts = np.unique(y_full, return_counts=True)

            # only stratify if we have enough samples AND every class ≥2
            if n_val < n_classes:
                stratify_param = None
                print(f"[{ds}] warning: val_size too small for {n_classes} classes ({n_val} samples). skipping stratify.")
            elif np.any(counts < 2):
                stratify_param = None
                print(f"[{ds}] warning: some classes have <2 samples; skipping stratify.")
            else:
                stratify_param = y_full

            # now split
            X_train, X_val, y_train, y_val = train_test_split(
                X_full, y_full,
                test_size=val_size,
                random_state=random_state,
                stratify=stratify_param
            )

            # make output folder
            out_folder = os.path.join(ds_path, 'numpy')
            os.makedirs(out_folder, exist_ok=True)

            # save arrays
            np.save(os.path.join(out_folder, 'X_train.npy'), X_train)
            np.save(os.path.join(out_folder, 'y_train.npy'), y_train)
            np.save(os.path.join(out_folder, 'X_val.npy'),   X_val)
            np.save(os.path.join(out_folder, 'y_val.npy'),   y_val)
            np.save(os.path.join(out_folder, 'X_test.npy'),  X_test)
            np.save(os.path.join(out_folder, 'y_test.npy'),  y_test)

            print(f"[{ds}] saved numpy arrays to {out_folder}")


if __name__ == '__main__':
    print('### Data preparation ###')
    convert_ucr_to_numpy('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data', val_size=0.2, random_state=88)
    #data_preparation()
    print('Data preparation done!')






# # Creating folder where to save numpy data
#     data_path = "/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy"
#     if not os.path.exists(data_path):
#         os.mkdir(data_path)
#         print('Create ECG5000/numpy folder')

    
#     # Loading data
#     train = pd.read_table('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/ECG5000_TEST.txt', sep=r'\s{2,}', engine='python', header=None)
#     test = pd.read_table('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/ECG5000_TRAIN.txt', sep=r'\s{2,}', engine='python', header=None)

#     # Concatenating
#     df = pd.concat([train, test])
#     new_columns = list(df.columns)
#     new_columns[0] = 'Class'
#     df.columns = new_columns

#     # Dividing in normal and not normal data
#     normal = df.loc[df.Class == 1]
#     anomaly = df.loc[df.Class != 1]

#     # Splitting normal data in training, validation and test set
#     X_train_n, X_val_n = train_test_split(normal, random_state = 88, test_size = 1 - args.perc_tr_n)
#     X_val_n, X_test_n = train_test_split(X_val_n, random_state = 88, test_size = 1- args.perc_val_n)

#     # Splitting validation data into two folds: the former to control model training, the latter
#     # for model selection

#     X_val_nA, X_val_nB = train_test_split(X_val_n, random_state=88, test_size = 0.5)

#     # Splitting anomalous data in validation and test set
#     perc_anol_all = args.perc_val_an
#     n_anol = len(X_train_n) * perc_anol_all / (1 - perc_anol_all)
#     perc_anol_val_a = n_anol / len(anomaly)
#     perc_anol_test_a = 1 - perc_anol_val_a
#     X_val_a, X_test_a = train_test_split(anomaly,
#                                          random_state = 88,
#                                          test_size = perc_anol_test_a,
#                                          stratify = anomaly.Class)

#     # Splitting anomalous validation data into two splitting: the former for model training the latter
#     # for model selection
#     X_val_aA, X_val_aB = train_test_split(X_val_a, random_state = 88, test_size = 0.5)


#     # Training data ONLY NORMAL
#     X_train = X_train_n.iloc[:, 1:].values
#     y_train = X_train_n.iloc[:, 0].values

#     # Training data NORMAL + ANOL
#     X_train_p = pd.concat([X_train_n.iloc[:, 1:], X_val_aA.iloc[:, 1:]]).values
#     y_train_p = pd.concat([X_train_n.iloc[:, 0], X_val_aA.iloc[:, 0]]).values

#     # Validation data only normal to control model training
#     X_val = X_val_n.iloc[:, 1:].values
#     y_val = X_val_n.iloc[:, 0].values

#     # Validation data: both normal and anomalous data for model selection: AUC LOSS
#     X_val_p = pd.concat([X_val_nB.iloc[:, 1:], X_val_aB.iloc[:, 1:]]).values
#     y_val_p = pd.concat([X_val_nB.iloc[:, 0], X_val_aB.iloc[:, 0]]).values

#     # Validation data: both normal and anomalous data for model selection: NO AUC LOSS
#     X_val_p_full = pd.concat([X_val_nB.iloc[:, 1:], X_val_a.iloc[:, 1:]]).values
#     y_val_p_full = pd.concat([X_val_nB.iloc[:, 0], X_val_a.iloc[:, 0]]).values

#     # Test data
#     X_test = pd.concat([X_test_n.iloc[:, 1:], X_test_a.iloc[:, 1:]]).values
#     y_test = pd.concat([X_test_n.iloc[:, 0], X_test_a.iloc[:, 0]]).values

#     # Saving training data (only normal instances)
#     np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/X_train.npy', X_train)
#     np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/y_train.npy', y_train)

#     # Saving training data (normal instances + anomalous)
#     np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/X_train_p.npy', X_train_p)
#     np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/y_train_p.npy', y_train_p)

#     # Saving validation data (only normal instances to control model training)
#     np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/X_val.npy', X_val)
#     np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/y_val.npy', y_val)
    
#     # Saving validation data (normal + anomalous instances to perform model selection yes AUC)
#     np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/X_val_p.npy', X_val_p)
#     np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/y_val_p.npy', y_val_p)

#     # Saving validation data (normal + anomalous instances to perform model selection no AUC)
#     np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/X_val_p_full.npy', X_val_p_full)
#     np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/y_val_p_full.npy', y_val_p_full)
    
#     # Saving test data (normal + anomalous instances)
#     np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/X_test.npy', X_test)
#     np.save('/Users/jessekroll/Desktop/Bachelor Thesis 24/Repos for TSCAR models/Recurrent-Autoencoder/data/ECG5000/numpy/y_test.npy', y_test)

#     print('Saved data in numpy')