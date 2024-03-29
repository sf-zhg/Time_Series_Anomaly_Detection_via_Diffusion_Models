import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MaxAbsScaler

from typing import Tuple
from torch import Tensor


class SineWaveDataset(Dataset):
    def __init__(self, csv_file: str):
        """
        iterate over dataset and gets us x,y tuple
        """
        super(SineWaveDataset, self).__init__()

        self.scaler = MaxAbsScaler()

        self.raw_data = pd.read_csv(csv_file)
        self.data = self.raw_data.iloc[:, 1:]
        self.data_scaled = self.scaler.fit_transform(self.data.iloc[:, 1:].T.values)
        self.data_tensor = torch.tensor(self.data_scaled, dtype=torch.float32)

        if "train" in csv_file:
            self.label_vec = torch.zeros((len(self.data_tensor),), dtype=torch.float32)
        elif "test" in csv_file:
            labels = torch.zeros((len(self.data_tensor),), dtype=torch.float32)
            labels[len(self.data_tensor) // 2 :] = 1
            self.label_vec = labels
        else:
            raise ValueError(f"only train or test")

    def __len__(self):
        """
        get length of dataset
        """
        return len(self.data_tensor)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        get x,y tuple at specific inx
        """
        x = self.data_tensor[idx]
        y = self.label_vec[idx]
        return x, y
