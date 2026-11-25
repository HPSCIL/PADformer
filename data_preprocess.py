# PADformer source; data and trained artifacts are not included.

from torch.utils.data import Dataset
import numpy as np
import torch

class MyDataset(Dataset):

    def __init__(self, X, Y, Wind_X, Wind_Y):
        self.X = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
        self.Y = torch.from_numpy(Y) if isinstance(Y, np.ndarray) else Y
        self.Wind_X = torch.from_numpy(Wind_X) if isinstance(Wind_X, np.ndarray) else Wind_X
        self.Wind_Y = torch.from_numpy(Wind_Y) if isinstance(Wind_Y, np.ndarray) else Wind_Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx], self.Wind_X[idx])

def pro_data(data_pollutant, data_wind, args):
    trainX, trainY = ([], [])
    windX, windY = ([], [])
    seq_len = args.lag
    pred_len = args.pre_len
    total_len = data_pollutant.shape[0]
    for i in range(total_len - seq_len - pred_len + 1):
        x_train = data_pollutant[i:i + seq_len, :, :]
        target_train = data_pollutant[i + seq_len:i + seq_len + pred_len, :, :]
        w_train = data_wind[i:i + seq_len, :, :]
        w_target = data_wind[i + seq_len:i + seq_len + pred_len, :, :]
        trainX.append(x_train)
        trainY.append(target_train)
        windX.append(w_train)
        windY.append(w_target)
    trainX = np.array(trainX).transpose(0, 2, 3, 1)
    trainY = np.array(trainY).transpose(0, 2, 3, 1)
    windX = np.array(windX).transpose(0, 2, 3, 1)
    windY = np.array(windY).transpose(0, 2, 3, 1)
    return (trainX, trainY, windX, windY)
