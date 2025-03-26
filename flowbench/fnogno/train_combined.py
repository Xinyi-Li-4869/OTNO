import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from torch.utils.data import DataLoader,Dataset
import operator
from functools import reduce
import sys
sys.path.append('/home/xinyili/neuraloperator')
from neuralop.models import FNOGNO
import argparse
import csv
import time
from scipy.interpolate import RegularGridInterpolator

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

class DictDataset(Dataset):
    def __init__(self, data_dict:dict):
        self.data_dict = data_dict
        for k, v in data_dict.items():
            assert len(v) == len(
                data_dict[list(data_dict.keys())[0]]
            ), "All data must have the same length"

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data_dict.items()}

    def __len__(self):
        return len(self.data_dict[list(self.data_dict.keys())[0]])
    
#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

################################################################
# configs
################################################################
Ntotal = 1000
ntrain = 800
ntest = 200

batch_size = 1
learning_rate = 0.001

epochs = 200
step_size = 50
gamma = 0.5

modes = 32
hidden_channel = 64

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train a model for different resolutions and group names.")

# Add arguments for resolution and group_name
parser.add_argument('--resolution', type=int, choices=[128, 256, 512], required=True, help="Resolution size.")
#parser.add_argument('--expand_factor', type=int, choices=[1,2,3,4], required=True, help="Expand factor.")
parser.add_argument('--group_name', type=str, choices=['nurbs', 'harmonics', 'skelneton'], required=True, help="Group name.")
#parser.add_argument('--predict', type=str, choices=['velocity_x', 'velocity_y', 'pressure'], required=True, help="Prediction type.")

# Parse the arguments
args = parser.parse_args()

resolution = args.resolution
group_name = args.group_name
expand_factor = 1
latent_res = 64 #128 for boundary

file_name = f'results/fnogno_fullspace_{resolution}_{group_name}_3fields.csv'

################################################################
# load data and data normalization
################################################################
t_start = default_timer()
x = torch.linspace(0, 2, resolution)
y = torch.linspace(0, 2, resolution)
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
grid = torch.stack([grid_x, grid_y]).to(dtype=torch.float32).reshape(2,-1).T

# Create a finer 256x256 grid
x_latent = torch.linspace(0, 2, latent_res)
y_latent = torch.linspace(0, 2, latent_res)
X_latent, Y_latent = torch.meshgrid(x_latent, y_latent, indexing='ij')
latent_grid = torch.stack([X_latent, Y_latent]).to(dtype=torch.float32).reshape(2,-1).T

inputs = np.load('/data/xinyili/datasets/flowbench/LDC_NS_2D/' + str(resolution) + 'x' + str(resolution) + '/' + group_name + '_lid_driven_cavity_X.npz')
outputs = np.load('/data/xinyili/datasets/flowbench/LDC_NS_2D/' + str(resolution) + 'x' + str(resolution) + '/' + group_name + '_lid_driven_cavity_Y.npz')
input_data = inputs['data']
output_data = outputs['data']
#sdf_data = inputs['data'][:,1,:,:]  # sdf
#output_data = outputs['data'][:,predict_idx,:,:] # velocity_x, velocity_y, press, temperature

input_xy, input_s, input_sdf = [], [], []
for i in range(1000):
    out = output_data[i,0:3,:,:]
    sdf = input_data[i,1,:,:]
    out_flattened = torch.from_numpy(out).to(dtype=torch.float32).reshape(3,-1).T
    sdf_flattened = torch.from_numpy(sdf).to(dtype=torch.float32).flatten()
    indices = torch.nonzero(sdf_flattened >= 0).squeeze() #) & (sdf_flattened < 0.2)
    input_s.append(out_flattened[indices])
    input_xy.append(grid[indices])

    # Interpolate the SDF data to the new grid using 2D interpolation
    #interpolator = RegularGridInterpolator((x, y), sdf, method='linear') 
    #sdf_latent = interpolator(latent_grid).reshape(latent_res, latent_res, 1)
    features = input_data[i,:,:,:]
    downsampled_features = features[:, ::8, ::8] #4 for boundary
    input_sdf.append(torch.from_numpy(downsampled_features).permute(1,2,0).to(dtype=torch.float32))

train_s = input_s[:ntrain]
test_s = input_s[-ntest:]
train_xy = input_xy[:ntrain]
test_xy = input_xy[-ntest:]
train_sdf = input_sdf[:ntrain]
test_sdf = input_sdf[-ntest:]

train_data = {"xy": train_xy, "sdf": train_sdf, "s": train_s}
test_data = {"xy": test_xy, "sdf": test_sdf, "s": test_s}
train_dataset = DictDataset(train_data)
test_dataset = DictDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
print(f"Dataloader took time: {default_timer()-t_start} seconds.")
################################################################
# training and evaluation
################################################################
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

model = FNOGNO(
    in_channels=3,
    out_channels=3,
    projection_channels=256,
    gno_coord_dim=2,
    gno_pos_embed_type='transformer',
    gno_transform_type="linear",
    fno_n_modes=(modes, modes),
    fno_hidden_channels=hidden_channel,
    fno_lifting_channel_ratio=4,
    fno_n_layers=4,
    # Other GNO params
    gno_embed_channels=32,
    gno_embed_max_positions=10000,
    gno_radius=0.07,#0.05 for boundary
    gno_channel_mlp_hidden_layers=[512,256],
    gno_channel_mlp_non_linearity=F.gelu,
    gno_use_open3d=False,
    gno_batched=False,
    # Other FNO params
    fno_resolution_scaling_factor=None,
    fno_incremental_n_modes=None,
    fno_block_precision="full",
    fno_channel_mlp_dropout=0,
    fno_channel_mlp_expansion=0.5,
    fno_rank=0.4,
).cuda()

# model_iphi = FNO2d(modes, modes, width, in_channels=2, out_channels=2, is_skip=True).cuda()
print(count_params(model))

params = list(model.parameters()) 
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
# Store data to save later
train_losses, test_losses, epoch_times = [], [], []
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_reg = 0
    for data in train_loader:
        sigma, mesh, sdf = data["s"][0].cuda(), data["xy"][0].cuda(), data["sdf"][0].cuda()

        optimizer.zero_grad()
        out = model(latent_grid.reshape(latent_res,latent_res,2).cuda(), mesh, sdf)
        loss = myloss(out.reshape(batch_size, -1), sigma.reshape(batch_size, -1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for data in test_loader:
            sigma, mesh, sdf = data["s"][0].cuda(), data["xy"][0].cuda(), data["sdf"][0].cuda()
            out = model(latent_grid.reshape(latent_res,latent_res,2).cuda(), mesh, sdf)
            test_l2 += myloss(out.reshape(batch_size, -1), sigma.reshape(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    # Calculate time for this epoch
    t2 = default_timer()
    epoch_time = t2 - t1
    epoch_times.append(epoch_time)
    
    # Print the results
    print(f"Epoch {ep+1}/{epochs}, Time: {epoch_time:.4f}s, Train Loss: {train_l2:.4f}, Test Loss: {test_l2:.4f}")

    # Save losses and times
    train_losses.append(train_l2)
    test_losses.append(test_l2)

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