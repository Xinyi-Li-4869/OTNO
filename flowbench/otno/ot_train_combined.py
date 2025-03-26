import numpy as np
import torch
import random
from timeit import default_timer
import os
import sys
sys.path.append('/home/xinyili/neuraloperator')
sys.path.append('/home/xinyili/OTNO')
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from utils import UnitGaussianNormalizer, LpLoss, count_model_params, Normals, DictDataset, DictDatasetWithConstant
from neuralop.models import FNO, SFNO
from neuralop.layers.channel_mlp import ChannelMLP as NeuralopMLP
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.spherical_convolution import SphericalConv
import csv
import time
import argparse

class TransportFNO(FNO):
    def __init__(
            self,
            n_modes,
            hidden_channels,
            in_channels=4,
            out_channels=1,
            lifting_channels=128,
            projection_channels=128,
            n_layers=4,
            positional_embedding=None,
            **kwargs
    ):        
        super().__init__(
            n_modes = n_modes,
            hidden_channels = hidden_channels,
            in_channels = in_channels,
            out_channels = out_channels,
            lifting_channels = lifting_channels,
            projection_channels = projection_channels,
            n_layers = n_layers,
            positional_embedding=positional_embedding,
            **kwargs
        )

        self.projection = NeuralopMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            n_dim=1,
        )

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    # transports: (1, in_channels, n_s_sqrt, n_s_sqrt)
    def forward(self, transports, idx_decoder):
        """TFNO's forward pass"""
        transports = self.lifting(transports)

        if self.domain_padding is not None:
            transports = self.domain_padding.pad(transports)

        for layer_idx in range(self.n_layers):
            transports = self.fno_blocks(transports, layer_idx)

        if self.domain_padding is not None:
            transports = self.domain_padding.unpad(transports) # (1, hidden_channels, n_s_sqrt, n_s_sqrt)

        transports = transports.reshape(self.hidden_channels, -1).permute(1, 0) # (n_s, hidden_channels)

        out = transports[idx_decoder].permute(1,0) # (hidden_channel, n_t)

        out = out.unsqueeze(0)
        out = self.projection(out).squeeze()
        return out.T

def square(n):
    x = torch.linspace(0, 1, n)
    y = torch.linspace(0, 1, n)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    return torch.stack([grid_x, grid_y]).permute(1,2,0)

def ring(n, r_min=0.5, r_max=1.0):
    # Generate theta values evenly spaced around the circle
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Generate radial distances evenly spaced between r_min and r_max
    r = np.linspace(r_min, r_max, n)
    
    # Create meshgrid for the theta and radial distances
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    
    # Convert polar coordinates (R, Theta) to Cartesian coordinates (x, y)
    x = R * np.cos(Theta)
    y = R * np.sin(Theta)
    
    # Stack the x and y coordinates into a single tensor and return
    grid = np.stack((x, y), axis=-1)
    
    return torch.tensor(grid, dtype=torch.float32)

################################################################
# configs
################################################################
n_train = 2400
n_test = 600

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train a model for different resolutions and group names.")

# Add arguments for resolution and group_name
parser.add_argument('--metric', type=str, choices=['boundary', 'fullspace'], required=True, help="Latent shape.")
parser.add_argument('--resolution', type=int, choices=[128, 256, 512], required=True, help="Resolution size.")
parser.add_argument('--expand_factor', type=int, choices=[1,2,3,4], required=True, help="Expand factor.")
parser.add_argument('--latent_shape', type=str, choices=['square', 'ring'], required=True, help="Latent shape.")

# Parse the arguments
args = parser.parse_args()

metric = args.metric
resolution = args.resolution
expand_factor = args.expand_factor
latent_shape = args.latent_shape
reg=1e-06

file_name = f'results/otno_{metric}_3fields_{resolution}_allgroups_{latent_shape}_expand{expand_factor}_bigmodel.csv'
best_model_path = f'saved_models/{metric}/LDC_NS_2D_{resolution}_allgroups_{latent_shape}_expand{expand_factor}_bigmodel.pth'
data_path = f'ot-data/LDC_NS_2D_{metric}_{resolution}_allgroups_{latent_shape}_expand{expand_factor}_reg1e-6_combined.pt'
print(resolution, latent_shape, expand_factor)
device = torch.device('cuda:0')
seed=0
torch.manual_seed(seed)  # Sets the seed for CPU
torch.cuda.manual_seed(seed)  # Sets the seed for CUDA on the current device

################################################################
# load data & normalize
################################################################
data = torch.load(data_path)
train_inputs = data['inputs'][0:n_train]
train_outputs = data['outs'][0:n_train]
train_indices_decoder = data['indices_decoder'][0:n_train]

cat_train_outputs = torch.cat([train_outputs[i] for i in range(n_train)], dim=0)
cat_train_inputs = torch.cat([train_inputs[i].reshape(-1,7) for i in range(n_train)], dim=0)
output_encoder = UnitGaussianNormalizer(cat_train_outputs)
input_encoder = UnitGaussianNormalizer(cat_train_inputs)

train_outputs = [output_encoder.encode(train_outputs[i]) for i in range(n_train)]
train_inputs = [input_encoder.encode(train_inputs[i].reshape(-1,7)).reshape(train_inputs[i].shape[0],train_inputs[i].shape[1],7) for i in range(n_train)]

output_encoder.to(device)
test_inputs = data['inputs'][n_train:]
test_outputs = data['outs'][n_train:]
test_indices_decoder = data['indices_decoder'][n_train:]

test_inputs = [input_encoder.encode(test_inputs[i].reshape(-1,7)).reshape(test_inputs[i].shape[0],test_inputs[i].shape[1],7) for i in range(n_test)]

train_dict = {'inputs': train_inputs, 'outputs': train_outputs, 'indices_decoder': train_indices_decoder}
test_dict = {'inputs': test_inputs, 'outputs': test_outputs, 'indices_decoder': test_indices_decoder}

train_dataset = DictDataset(train_dict)
test_dataset = DictDataset(test_dict)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = TransportFNO(n_modes=(64 ,64), hidden_channels=16*expand_factor, in_channels=6, out_channels=3, lifting_channels=16*expand_factor, projection_channels=16*expand_factor, n_layers=10)
print(count_model_params(model))
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

epochs = 200
#myloss = LpLoss(d=3,size_average=False)
train_losses, test_losses, epoch_times = [], [], []
best_l2 = float('inf')
for ep in range(epochs):
    t1 = default_timer()
    test_l2 = 0.0
    train_l2 = 0.0
    model.train()
    for batch_data in train_loader:
        optimizer.zero_grad()

        input = batch_data['inputs'][0].to(device) # torch.Size([n_s, n_s, 7])
        output = batch_data['outputs'][0].to(device) # torch.Size([n_t, 3])
        indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)
        
        predict = model(input[:,:,:6].permute(2,0,1).unsqueeze(0).to(dtype=torch.float32,device=device), indices_decoder) # torch.Size([n_t, 3])
        node_count = predict.shape[0]
        loss = F.mse_loss(predict, output, reduction='none')
        loss = loss.sum((0,1)) / node_count
        loss.backward()
        optimizer.step()

        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    with torch.no_grad():
        for i,batch_data in enumerate(test_loader):
            input = batch_data['inputs'][0].to(device) # torch.Size([n_s, n_s, 7])
            output = batch_data['outputs'][0].to(device) # torch.Size([n_t, 3])
            indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)
        
            predict = model(input[:,:,:6].permute(2,0,1).unsqueeze(0).to(dtype=torch.float32,device=device), indices_decoder) # torch.Size([n_t, 3])
            predict = output_encoder.decode(predict)
            node_count = predict.shape[0]
            loss = F.mse_loss(predict, output, reduction='none')
            loss = loss.sum((0,1)) / node_count
            test_l2 += loss.item()

    train_l2 /= n_train
    test_l2 /= n_test

    # Calculate time for this epoch
    t2 = default_timer()
    epoch_time = t2 - t1
    epoch_times.append(epoch_time)
    
    # Print the results
    print(f"Epoch {ep+1}/{epochs}, Time: {epoch_time:.4f}s, Train Loss: {train_l2:.6f}, Test Loss: {test_l2:.6f}")
    #sys.exit(0)
    # Save losses and times
    train_losses.append(train_l2)
    test_losses.append(test_l2)
    if test_l2 < best_l2:
        best_l2 = test_l2
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with test error: {best_l2:.8f}")

# Saving the results to a CSV file
with open(file_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Epoch Time', 'Total Time'])

    # Write data for each epoch
    for ep in range(epochs):
        writer.writerow([ep + 1, train_losses[ep], test_losses[ep], epoch_times[ep], sum(epoch_times[:ep+1])])

# Total time for training
total_time = sum(epoch_times)
print(f"Total Training Time: {total_time:.4f}s")