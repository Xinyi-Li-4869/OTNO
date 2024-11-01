import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset
from ..utils import UnitGaussianNormalizer, LpLoss, count_model_params
from .TransportFNO import TransportFNO
from timeit import default_timer

class DictDataset(Dataset):
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        for k, v in data_dict.items():
            assert len(v) == len(
                data_dict[list(data_dict.keys())[0]]
            ), "All data must have the same length"

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data_dict.items()}

    def __len__(self):
        return len(self.data_dict[list(self.data_dict.keys())[0]])


class DictDatasetWithConstant(DictDataset):
    def __init__(self, data_dict: dict, constant_dict: dict):
        super().__init__(data_dict)
        self.constant_dict = constant_dict

    def __getitem__(self, index):
        return_dict = {k: v[index] for k, v in self.data_dict.items()}
        return_dict.update(self.constant_dict)
        return return_dict

def compute_torus_normals(n_s_sqrt, R, r):
    theta = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    phi = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')

    # Partial derivatives
    # dP/dtheta
    dx_dtheta = -r * torch.sin(theta) * torch.cos(phi)
    dy_dtheta = -r * torch.sin(theta) * torch.sin(phi)
    dz_dtheta = r * torch.cos(theta)

    # dP/dphi
    dx_dphi = -(R + r * torch.cos(theta)) * torch.sin(phi)
    dy_dphi = (R + r * torch.cos(theta)) * torch.cos(phi)
    dz_dphi = torch.zeros_like(dx_dphi)

    # Cross product to find normal vectors
    nx = dy_dtheta * dz_dphi - dz_dtheta * dy_dphi
    ny = dz_dtheta * dx_dphi - dx_dtheta * dz_dphi
    nz = dx_dtheta * dy_dphi - dy_dtheta * dx_dphi

    # Stack and normalize
    normals = torch.stack((nx, ny, nz), axis=-1)
    norm = torch.linalg.norm(normals, dim=2, keepdim=True)
    normals = normals / norm

    return normals


def create_torus_grid(n_s_sqrt, R, r):
    theta = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    phi = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')

    x = (R + r * torch.cos(theta)) * torch.cos(phi)
    y = (R + r * torch.cos(theta)) * torch.sin(phi)
    z = r * torch.sin(theta)

    return torch.stack((x, y, z), axis=-1)

t1 = default_timer()

"""Config"""
expand_factor = 3.0
n_t = 3586
n_s_sqrt = int(np.sqrt(expand_factor)*np.ceil(np.sqrt(n_t)))
R = 1.5
r = 1

n_train = 500
n_test = 111

epochs = 151

"""Load data"""
data_path = '/central/groups/tensorlab/xinyi/OT/ot-data/torus_OTmean_geomloss_expand'+str(expand_factor)+'.pt'
data = torch.load(data_path)
print(data_path)
device = torch.device('cuda')

train_transports = data['transports'][0:n_train, ...]
train_normals = data['normals'][0:n_train, ...]
train_pressures = data['pressures'][0:n_train, ...]
train_points = data['points'][0:n_train, ...]
train_indices_encoder = data['indices_encoder'][0:n_train, ...]
train_indices_decoder = data['indices_decoder'][0:n_train, ...]

# normalization
pressure_encoder = UnitGaussianNormalizer(train_pressures, reduce_dim=[0,1])
transport_encoder = UnitGaussianNormalizer(train_transports, reduce_dim=[0, 2, 3])

train_pressures = pressure_encoder.encode(train_pressures)
train_transports = transport_encoder.encode(train_transports)

pressure_encoder.to(device)
test_transports = data['transports'][n_train:, ...]
test_normals = data['normals'][n_train:, ...]
test_pressures = data['pressures'][n_train:, ...]
test_points = data['points'][n_train:, ...]
test_indices_encoder = data['indices_encoder'][n_train:, ...]
test_indices_decoder = data['indices_decoder'][n_train:, ...]

test_transports = transport_encoder.encode(test_transports)

train_dict = {'transports': train_transports, 'pressures': train_pressures, 'points': train_points, 'normals':train_normals, 'indices_encoder': train_indices_encoder, 'indices_decoder': train_indices_decoder}
test_dict = {'transports': test_transports, 'pressures': test_pressures, 'points': test_points, 'normals':test_normals, 'indices_encoder': test_indices_encoder, 'indices_decoder': test_indices_decoder}

# combine constant data: torus girds and torus normals
pos_embed = create_torus_grid(n_s_sqrt, R, r)
torus_normals = compute_torus_normals(n_s_sqrt, R, r)

train_dataset = DictDatasetWithConstant(train_dict, {'pos': pos_embed, 'nor':torus_normals})
test_dataset = DictDatasetWithConstant(test_dict, {'pos': pos_embed, 'nor':torus_normals})

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

"""Initialize model"""
model = TransportFNO(n_modes=(32, 32), hidden_channels=120, in_channels=9, norm='group_norm',
                     use_mlp=True, mlp={'expansion': 1.0, 'dropout': 0}, domain_padding=0.125,
                     factorization='tucker', rank=0.4)

print(count_model_params(model))
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
myloss = LpLoss(size_average=False)

for ep in range(epochs):
    t = default_timer()
    test_l2 = 0.0
    train_l2 = 0.0
    model.train()
    for batch_data in train_loader:
        optimizer.zero_grad()

        transports = batch_data['transports'].to(device)
        pressures = batch_data['pressures'].to(device)
        normals = batch_data['normals'][0].to(device)
        indices_encoder = batch_data['indices_encoder'][0].to(dtype=torch.long, device=device)
        indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)

        # compute normal feature
        normals = normals[indices_encoder]
        torus_normals = batch_data['nor'].reshape(-1,3).to(device)
        normal_features = torch.cross(normals, torus_normals, dim=1).reshape(n_s_sqrt,n_s_sqrt,3).permute(2,0,1).unsqueeze(0)
       
        # cat input features
        transports = torch.cat((transports, batch_data['pos'].permute(0,3,1,2).to(device), normal_features), dim=1)

        out = model(transports, indices_decoder)

        loss = myloss(out, pressures)
        loss.backward()

        optimizer.step()

        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    with torch.no_grad():
        for batch_data in test_loader:
            transports = batch_data['transports'].to(device)
            pressures = batch_data['pressures'].to(device)
            normals = batch_data['normals'][0].to(device)
            indices_encoder = batch_data['indices_encoder'][0].to(dtype=torch.long, device=device)
            indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)
            normals = normals[indices_encoder]
            torus_normals = batch_data['nor'].reshape(-1,3).to(device)
            normal_features = torch.cross(normals, torus_normals, dim=1).reshape(n_s_sqrt,n_s_sqrt,3).permute(2,0,1).unsqueeze(0)
        
            transports = torch.cat((transports, batch_data['pos'].permute(0,3,1,2).to(device), normal_features), dim=1)

            out = model(transports, indices_decoder)
            out = pressure_encoder.decode(out)

            test_l2 += myloss(out, pressures).item()

    train_l2 /= n_train
    test_l2 /= n_test

    print(ep, train_l2, test_l2, default_timer() - t)

print(f"Total time: {default_timer()-t1:.2f} seconds.")
