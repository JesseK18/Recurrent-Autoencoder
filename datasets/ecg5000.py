"""
ECG5000 Dataloader implementation, used in RNN_Autoencoder
"""

import numpy as np
from utils.samplers import StratifiedSampler

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset



class ECG500DataLoader:
    def __init__(self, config):
        self.config = config

        # Loading training data
        if self.config.training_type == 'one_class':
            # If loss without AUC penalty is used
            X_train = np.load(self.config.data_folder + self.config.X_train).astype(np.float32)
            y_train = np.load(self.config.data_folder + self.config.y_train).astype(np.float32)
        else:
            # If loss with AUC penalty is used
            X_train = np.load(self.config.data_folder + self.config.X_train_p).astype(np.float32)
            y_train = np.load(self.config.data_folder + self.config.y_train_p).astype(np.float32)
        
        # Loading validation data to control model training
        X_val = np.load(self.config.data_folder + self.config.X_val).astype(np.float32)
        y_val = np.load(self.config.data_folder + self.config.y_val).astype(np.float32)

        # From numpy to torch
        if X_train.ndim < 3:
            X_train = torch.from_numpy(X_train).unsqueeze(2)
            X_val = torch.from_numpy(X_val).unsqueeze(2)
        else:
            X_train = torch.from_numpy(X_train)
            X_val = torch.from_numpy(X_val)

        y_train = torch.from_numpy(y_train)
        y_val = torch.from_numpy(y_val)

        # Tensordataset
        training = TensorDataset(X_train, y_train)
        validation = TensorDataset(X_val, y_val)

        # Dataloader
        if self.config.training_type == 'one_class':

            self.train_loader = DataLoader(training, batch_size = self.config.batch_size, shuffle = True)
        else:

            sampler = StratifiedSampler(y_train,
                                        batch_size =self.config.batch_size,
                                        random_state =self.config.sampler_random_state)
            self.train_loader = DataLoader(training, batch_sampler = sampler)

        self.valid_loader = DataLoader(validation, batch_size = self.config.batch_size_val, shuffle = False)
        # also test_loader
        X_test = np.load(self.config.data_folder + self.config.X_test).astype(np.float32)
        y_test = np.load(self.config.data_folder + self.config.y_test).astype(np.float32)

        if X_test.ndim < 3:
            X_test = torch.from_numpy(X_test).unsqueeze(2)
        else:
            X_test = torch.from_numpy(X_test)
        y_test = torch.from_numpy(y_test)

        test_dataset = TensorDataset(X_test, y_test)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.config.batch_size,
                                      shuffle=False)
        # --- end new code ---
        # Number of batches
        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)
        self.test_iterations = len(self.test_loader)
    
    def finalize(self):
        pass

    
class UCRDataLoader:
    """
    General UCR dataset loader. 
    Set config.folderdata to point to the dataset folder, which must contain:
    X_train.npy, y_train.npy,
    X_val.npy,   y_val.npy,
    X_test.npy,  y_test.npy
    """
    def __init__(self, config):
        self.config = config
        # placeholder string for the dataset directory
        self.folderdata = config.data_folder

        # load numpy arrays
        X_train = np.load(f"{self.folderdata}/X_train.npy").astype(np.float32)
        y_train = np.load(f"{self.folderdata}/y_train.npy").astype(np.float32)
        X_val   = np.load(f"{self.folderdata}/X_val.npy").astype(np.float32)
        y_val   = np.load(f"{self.folderdata}/y_val.npy").astype(np.float32)
        X_test  = np.load(f"{self.folderdata}/X_test.npy").astype(np.float32)
        y_test  = np.load(f"{self.folderdata}/y_test.npy").astype(np.float32)

        # ensure shape [N, T, 1] for torch RNNs
        def _to_tensor(x):
            t = torch.from_numpy(x)
            return t.unsqueeze(2) if t.ndim < 3 else t

        X_train, X_val, X_test = map(_to_tensor, (X_train, X_val, X_test))
        y_train, y_val, y_test = map(lambda a: torch.from_numpy(a), 
                                    (y_train, y_val, y_test))

        # build TensorDatasets
        train_ds = TensorDataset(X_train, y_train)
        val_ds   = TensorDataset(X_val,   y_val)
        test_ds  = TensorDataset(X_test,  y_test)

        # training loader (shuffled or stratified)
        if getattr(self.config, "training_type", "one_class") == "one_class":
            self.train_loader = DataLoader(
                train_ds,
                batch_size=self.config.batch_size,
                shuffle=True
            )
        else:
            sampler = StratifiedSampler(
                y_train,
                batch_size=self.config.batch_size,
                random_state=self.config.sampler_random_state
            )
            self.train_loader = DataLoader(
                train_ds,
                batch_sampler=sampler
            )

        # validation & test loaders
        self.valid_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size_val,
            shuffle=False
        )
        self.test_loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        # iteration counts
        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)
        self.test_iterations  = len(self.test_loader)

    


