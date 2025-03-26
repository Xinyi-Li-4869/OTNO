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
from neuralop.models import FNO
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

epochs = 400
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

file_name = f'results/fno_fullspace_{resolution}_{group_name}_3fields.csv'

################################################################
# load data and data normalization
################################################################
t_start = default_timer()
x = torch.linspace(0, 2, resolution)
y = torch.linspace(0, 2, resolution)
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
grid = torch.stack([grid_x, grid_y]).to(dtype=torch.float32).reshape(2,-1).T

data_X = np.load('/data/xinyili/datasets/flowbench/LDC_NS_2D/' + str(resolution) + 'x' + str(resolution) + '/' + group_name + '_lid_driven_cavity_X.npz')
data_Y = np.load('/data/xinyili/datasets/flowbench/LDC_NS_2D/' + str(resolution) + 'x' + str(resolution) + '/' + group_name + '_lid_driven_cavity_Y.npz')
input_data = data_X['data']
output_data = data_Y['data']
#sdf_data = inputs['data'][:,1,:,:]  # sdf
#output_data = outputs['data'][:,predict_idx,:,:] # velocity_x, velocity_y, press, temperature

inputs, outputs, indices = [], [], []
for i in range(Ntotal):
    out = output_data[i,0:3,:,:]
    inp = input_data[i,:,:,:]
    sdf = input_data[i,1,:,:]
    out_flattened = torch.from_numpy(out).to(dtype=torch.float32).reshape(3,-1).T
    sdf_flattened = torch.from_numpy(sdf).to(dtype=torch.float32).flatten()
    idx = torch.nonzero(sdf_flattened >= 0).squeeze() #& (sdf_flattened < 0.2)
    outputs.append(out_flattened[idx])
    inputs.append(torch.from_numpy(inp).to(dtype=torch.float32))
    indices.append(idx)


train_outputs = outputs[:ntrain]
test_outputs = outputs[-ntest:]
train_inputs = inputs[:ntrain]
test_inputs = inputs[-ntest:]
train_indices = indices[:ntrain]
test_indices = indices[-ntest:]

train_data = {"out": train_outputs, "inp": train_inputs, "idx": train_indices}
test_data = {"out": test_outputs, "inp": test_inputs, "idx": test_indices}
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

model = FNO(
    n_modes=(modes, modes),
    in_channels=3,
    out_channels=3,
    hidden_channels=hidden_channel,
    n_layers=4,
    lifting_channel_ratio=2,
    projection_channel_ratio=2,
    positional_embedding=None,
    rank=0.4
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
        target, inp, idx = data["out"].cuda(), data["inp"].cuda(), data["idx"][0].cuda()

        optimizer.zero_grad()
        out = model(inp)
        out = out.reshape(3,-1).T
        out = out[idx]
        loss = myloss(out.reshape(1,-1), target.reshape(1,-1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for data in test_loader:
            target, inp, idx = data["out"].cuda(), data["inp"].cuda(), data["idx"][0].cuda()
            out = model(inp)
            out = out.reshape(3,-1).T
            out = out[idx]
            loss = myloss(out.reshape(1,-1), target.reshape(1,-1))
            test_l2 += loss.item()

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