from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd


class PensionData(Dataset):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = pd.read_csv("./raw_data/401k/401k.csv")
        self.data_size = len(data.index)
        self.y = torch.FloatTensor(np.array(data['net_tfa'], dtype=np.float32).reshape(self.data_size, 1)).to(device)
        self.treat = torch.FloatTensor(np.array(data['e401'], dtype=np.float32)).to(device)
        self.x = torch.FloatTensor(np.array(data.loc[:, ~data.columns.isin(['net_tfa', 'e401'])], dtype=np.float32)).to(device)

    def __len__(self):
        return int(self.data_size)

    def __getitem__(self, idx):
        y = self.y[idx]
        x = self.x[idx]
        treat = self.treat[idx]
        return y, treat, x
