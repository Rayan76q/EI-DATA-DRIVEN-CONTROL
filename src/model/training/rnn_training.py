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
        h_new, _ = self.rnn(x, h)
        return self.head(h_new), h_new
    
hidden_layer_size = 32
model = ModelNN(1, 1, "RNN", hidden_layer_size, 1)
model = model.to(device)

epochs = 700

total_steps = epochs * len(train_dataloader)
warmup_steps = int(0.1 * total_steps)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler_warmup = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps)
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])

criterion = nn.MSELoss()

# training loop

for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    for i, (u_batch, y_batch) in enumerate(train_dataloader):
        u_batch = u_batch.unsqueeze(-1).to(device)
        y_batch = y_batch.unsqueeze(-1).to(device)
        
        optimizer.zero_grad()
        outputs = model(u_batch)[0]
        
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        running_loss += loss.item()
        
        optimizer.step()
        scheduler.step()
    
    print(f"Epoch : {epoch+1}, Loss : {(running_loss / len(train_dataloader)):.5f}")
    print(f"Current Learning Rate : {scheduler.get_last_lr()[0]:.6f}")
    
    model.eval()
    running_loss = 0.0
    for u_batch, y_batch in test_dataloader:
        with torch.no_grad():
            u_batch = u_batch.unsqueeze(-1).to(device)
            y_batch = y_batch.unsqueeze(-1).to(device)
            
            outputs = model(u_batch)[0]
                    
            loss = criterion(outputs, y_batch)
            running_loss += loss.item()
    
    print(f"Test Loss : {(running_loss / len(test_dataloader)):.5f}\n --------")

# saving the model

# saving as onnx
model = model.to("cpu")
model.eval()

torch.save(model.state_dict(), PROJECT_ROOT / "src/model/checkpoints/rnn_z.pt")

from torch.export import Dim
import torch

batch_dim = Dim("batch")
time_dim = Dim("time")

example_inputs = (
    torch.randn(1, 1, 1), 
    torch.zeros(1, 1, hidden_layer_size)
)

onnx_program = torch.onnx.export(
    model, 
    example_inputs, 
    dynamo=True,
    dynamic_shapes={
        "x": {0: batch_dim, 1: time_dim}, 
        "h": {1: batch_dim}
    },
)
onnx_program.save(PROJECT_ROOT / "src/model/checkpoints/rnn_z.onnx")