from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

device = "cuda" if torch.cuda.is_available() else "cpu"

# data import: these original training datasets are supplied separately.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "generated_data"
INPUT_FILE = DATA_DIR / "rnn_input_z.csv"
OUTPUT_FILE = DATA_DIR / "rnn_output_z.csv"
for dataset_file in (INPUT_FILE, OUTPUT_FILE):
    if not dataset_file.is_file():
        raise FileNotFoundError(
            f"Missing training dataset: {dataset_file}. Supply the original "
            "rnn_input_z.csv and rnn_output_z.csv in data/generated_data; "
            "outputs must contain interleaved z, vz, az samples."
        )

data_in = np.genfromtxt(INPUT_FILE, delimiter=",", dtype=np.float32)[:, :1000]
data_out = np.genfromtxt(OUTPUT_FILE, delimiter=",", dtype=np.float32)[:, :3000]
nb_data = data_in.shape[0]

# data standardization

def normalize(input_array):
    min_val = np.min(input_array)
    max_val = np.max(input_array)
    
    return (input_array - min_val) / (max_val - min_val)

def standardize(input_array):
    return (input_array - np.mean(input_array)) / np.std(input_array), np.mean(input_array), np.std(input_array)
def mimic_standardize(input_array, mean_mimic, std_mimic):
    return (input_array - mean_mimic) / std_mimic

# dataset module

class DataModule(Dataset):
    def __init__(self, data_u, data_y):
        super().__init__()
        self.N = data_u.shape[0]
        self.data_u = data_u
        self.data_y = data_y
        
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return self.data_u[index], self.data_y[index]

batch_size = 32

az_data = data_out[:, 2::3]

train_size = int(0.9*nb_data)
test_size = nb_data - train_size

input_train, input_mean, input_std = standardize(data_in[:train_size])
input_test = mimic_standardize(data_in[train_size:nb_data], input_mean, input_std)

az_train, az_mean, az_std = standardize(az_data[:train_size])
az_test = mimic_standardize(az_data[train_size:nb_data], az_mean, az_std)

print("VALUES TO KEEP :")
print(f"input_mean = {input_mean}, input_std = {input_std}")
print(f"az_mean = {az_mean}, az_std = {az_std}")

az_train_torch = torch.tensor(az_train, dtype=torch.float32)
az_test_torch = torch.tensor(az_test, dtype=torch.float32)

train_dataset = DataModule(torch.tensor(input_train, dtype=torch.float32), az_train_torch)
test_dataset = DataModule(torch.tensor(input_test, dtype=torch.float32), az_test_torch)

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size)


# model definition

class ModelNN(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        output_size: int = 1,
        rnn_type: str = "RNN",  # "RNN", "LSTM", "GRU"
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Select and initialize the RNN type
        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
        elif rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
        elif rnn_type.upper() == "RNN":
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity='tanh',
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
        
        self.head = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h=None):
        out, h_new = self.rnn(x, h)
        return self.head(out), h_new
    
hidden_layer_size = 32
model = ModelNN(4, 4, "RNN", hidden_layer_size, 3)