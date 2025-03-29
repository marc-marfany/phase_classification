import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class CSV_load_to_torch_tensor(Dataset):
    """  This class will take the training, validation and test data from the csv files and return it as torch tensors of size [batch_size, 1, 924] for the images and 
    [batch_size, 1] for the labels.
    """

    def __init__(self,csv_file_path, mask_path=None, shuffle=True, train=True):

        data=pd.read_csv(csv_file_path, header=None).values # Read the data from the csv file

        # Assuming the first column is the label and the rest are features.
        self.labels = data[:, 0]
        self.features = data[:, 1:]
        self.num_samples=self.features.shape[0]

        if train:
            if shuffle:
                if mask_path and os.path.exists(mask_path):
                    self.mask=np.load(mask_path)
                
                else:
                    self.mask=np.arange(self.num_samples)
                    np.random.seed(1997)
                    np.random.shuffle(self.mask)

                    # Save the mask if a path is provided
                    if mask_path:
                        np.save(mask_path, self.mask)

            # Apply the shuffled mask
            self.features = self.features[self.mask]
            self.labels = self.labels[self.mask]

    def __len__(self):
        """This function returns the length of the features (total number of images).
            Returns:
                - Total number of images.
        """
        
        return len(self.features)
    
    def __getitem__(self,idx):
        """This funcion return a simple image/feature as a torch tensor with the corresponding format.
            Returns:
                - x (torch.tensor (float32), size=[1, dim_features]): Torch tensor of the feature.
                - y (torch tensor (torch.long, size=1)): Torch tensor of the label
        """

        x=torch.tensor(self.features[idx], dtype=torch.float32).unsqueeze(0) #By doing unsqueeze we add an extra dimension in the begginig, 1, which corresponds to the channel
        y=torch.tensor(self.labels[idx], dtype=torch.long)

        return x,y 

class Dataloaders:
    """This class will create the training, validation and test dataloaders. """

    def __init__(self, train_path, val_path, test_path, batch_size_train=None, batch_size_val=None, batch_size_test=None, shuffle=False):
        """
        Args:
            train_path (string): Path to the training CSV file.
            val_path (string): Path to the validation CSV file.
            test_path (string): Path to the testing CSV file.
            batch_size_train (int): Number of validation samples per batch.
            batch_size_val (int): Number of validation samples per batch.
            batch_size_test (int): Number of test samples per batch.
            shuffle (bool): Whether to shuffle the training data.
        """
        # Store main variables
        self.batch_size_train=batch_size_train
        self.batch_size_val=batch_size_val
        self.batch_size_test=batch_size_test
        self.shuffle=shuffle
    
        # Create Dataset instances for each split
        self.train_dataset = CSV_load_to_torch_tensor(train_path, mask_path=None, shuffle=self.shuffle, train=True)
        self.val_dataset = CSV_load_to_torch_tensor(val_path, mask_path=None, shuffle=True, train=False)
        self.test_dataset = CSV_load_to_torch_tensor(test_path, mask_path=None, shuffle=True, train=False)

        # Create dataloaders
        self.train_dataloader=DataLoader(self.train_dataset, batch_size=batch_size_train, shuffle=self.shuffle)
        self.val_dataloader=DataLoader(self.val_dataset, batch_size=batch_size_val, shuffle=self.shuffle)
        self.test_dataloader=DataLoader(self.test_dataset, batch_size=batch_size_test, shuffle=self.shuffle)

    def get_loaders(self):
        """ This function will return all the dataloaders:
        Returns:
            - train_dataloader (loader): training dataloader.
            - val_dataloader (loader): validation dataloader.
            - test_dataloader (loader): Test dataloader.
        """
        return self.train_dataloader, self.val_dataloader, self.test_dataloader

    def single_test_loader(self):
        """ This function will return a single batch test_dataloader:
            Returns:
                - test_dataloader (loader): Single test dataloader.
        """
        
        return DataLoader(self.test_dataset, batch_size=1, shuffle=self.shuffle)
        

        
       
