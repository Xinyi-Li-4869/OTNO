import numpy as np
import torch
from timeit import default_timer
import os
import sys
sys.path.append('/home/xinyili/neuraloperator')
sys.path.append('/home/xinyili/OTNO')
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
            use_mlp=False,
            mlp=None,
            non_linearity=torch.nn.functional.gelu,
            norm=None,
            preactivation=False,
            fno_skip="linear",
            mlp_skip="soft-gating",
            separable=False,
            factorization=None,
            rank=1,
            joint_factorization=False,
            fixed_rank_modes=False,
            implementation="factorized",
            decomposition_kwargs=dict(),
            domain_padding=None,
            domain_padding_mode="one-sided",
            fft_norm="forward",
            SpectralConv=SpectralConv,
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
            use_channel_mlp = use_mlp,
            channel_mlp_dropout= mlp['dropout'],
            channel_mlp_expansion= mlp['expansion'],
            non_linearity = non_linearity,
            norm = norm,
            preactivation = preactivation,
            fno_skip = fno_skip,
            mlp_skip = mlp_skip,
            separable = separable,
            factorization = factorization,
            rank = rank,
            joint_factorization = joint_factorization,
            fixed_rank_modes = fixed_rank_modes,
            implementation = implementation,
            decomposition_kwargs = decomposition_kwargs,
            domain_padding = domain_padding,
            domain_padding_mode = domain_padding_mode,
            fft_norm = fft_norm,
            SpectralConv = SpectralConv,
            **kwargs
        )

        self.projection = NeuralopMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            non_linearity=non_linearity,
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
n_train = 800
n_test = 200

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train a model for different resolutions and group names.")

# Add arguments for resolution and group_name
parser.add_argument('--resolution', type=int, choices=[128, 256, 512], required=True, help="Resolution size.")
parser.add_argument('--expand_factor', type=int, choices=[1,2,3,4], required=True, help="Expand factor.")
parser.add_argument('--group_name', type=str, choices=['nurbs', 'harmonics', 'skelneton'], required=True, help="Group name.")
#parser.add_argument('--predict', type=str, choices=['velocity_x', 'velocity_y', 'pressure'], required=True, help="Prediction type.")
parser.add_argument('--latent_shape', type=str, choices=['square', 'ring'], required=True, help="Latent shape.")

# Parse the arguments
args = parser.parse_args()

resolution = args.resolution
group_name = args.group_name
expand_factor = args.expand_factor
latent_res = int(np.sqrt(resolution*resolution*expand_factor))
shape_map = {'square': square, 'ring': ring}
generate_latent_grid = shape_map[args.latent_shape]
#predict_map = {'velocity_x': 0, 'velocity_y': 1, 'pressure': 2}
#predict_idx = predict_map[args.predict]
reg=1e-06

file_name = f'results/otno_boundary_3fields_{resolution}_{group_name}_{args.latent_shape}_expand{expand_factor}.csv'
best_model_path = f'saved_models/boundary/LDC_NS_2D_{resolution}_{group_name}_{args.latent_shape}_expand{expand_factor}.pth'
data_path = f'/data/xinyili/datasets/flowbench/ot-data/LDC_NS_2D_boundary_{resolution}_{group_name}_{args.latent_shape}_expand{expand_factor}_reg1e-6_combined.pt'
data = torch.load(data_path)
print(resolution, group_name, args.latent_shape, expand_factor)
device = torch.device('cuda:0')
seed=0
torch.manual_seed(seed)  # Sets the seed for CPU
torch.cuda.manual_seed(seed)  # Sets the seed for CUDA on the current device

################################################################
# load data & normalize
################################################################
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

model = TransportFNO(n_modes=(32, 32), hidden_channels=64, in_channels=7, out_channels=3, norm='group_norm',
                     use_mlp=True, mlp={'expansion': 1.0, 'dropout': 0}, domain_padding=0.125,
                     factorization='tucker', rank=0.4)
print(count_model_params(model))
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

epochs = 400
myloss = LpLoss(d=3,size_average=False)
train_losses, test_losses, epoch_times = [], [], []
best_l2 = float('inf')
for ep in range(epochs):
    t1 = default_timer()
    test_l2 = 0.0
    train_l2 = 0.0
    model.train()
    for batch_data in train_loader:
        optimizer.zero_grad()

        input = batch_data['inputs'].to(device)
        output = batch_data['outputs'].to(device)
        indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)
        
        predict = model(input.permute(0,3,1,2).to(dtype=torch.float32,device=device), indices_decoder)
        
        loss = myloss(output, predict)
        loss.backward()
        optimizer.step()

        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    with torch.no_grad():
        for i,batch_data in enumerate(test_loader):
            input = batch_data['inputs'].to(device)
            output = batch_data['outputs'].to(device)
            indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)
            
            predict = model(input.permute(0,3,1,2).to(dtype=torch.float32,device=device), indices_decoder)
            predict = output_encoder.decode(predict)
            loss = myloss(output, predict)
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
        print(f"New best model saved with test relative L2: {best_l2:.8f}")

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