import torch

from torch.utils.data import DataLoader, Dataset
from neuralop.utils import UnitGaussianNormalizer
from src.networks.TransportFNO import TransportFNO
from timeit import default_timer
from src.losses import LpLoss

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


t = torch.linspace(0, 1, 85)[:-1]
pos_embed = torch.stack(torch.meshgrid(t, t, indexing='ij'))

n_train = 800
n_test = 889-n_train
data = torch.load('torus_data.pt')

device = torch.device('cuda')

train_transports = data['transports'][0:n_train,...]
train_couplings = data['couplings'][0:n_train]
train_pressures = data['pressures'][0:n_train,...]

pressure_encoder = UnitGaussianNormalizer(train_pressures, reduce_dim=[0,1])
transport_encoder = UnitGaussianNormalizer(train_transports, reduce_dim=[0,2,3])

train_pressures = pressure_encoder.encode(train_pressures)
train_transports = transport_encoder.encode(train_transports)

pressure_encoder.to(device)
test_transports = data['transports'][n_train:,...]
test_couplings = data['couplings'][n_train:]
test_pressures = data['pressures'][n_train:,...]

test_transports = transport_encoder.encode(test_transports)

train_dict = {'transports': train_transports, 'couplings': train_couplings, 'pressures': train_pressures}
test_dict = {'transports': test_transports, 'couplings': test_couplings, 'pressures': test_pressures}

train_dataset = DictDatasetWithConstant(train_dict, {'pos': pos_embed})
test_dataset = DictDatasetWithConstant(test_dict, {'pos': pos_embed})

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = TransportFNO(n_modes=(24,24), hidden_channels=120, in_channels=5, norm='group_norm',
                    use_mlp=True, mlp={'expansion': 1.0, 'dropout': 0}, domain_padding=0.125,
                    factorization='tucker', rank=0.4)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

epochs = 300
myloss = LpLoss(size_average=False)

for ep in range(epochs):
    t = default_timer()
    test_l2 = 0.0
    train_l2 = 0.0
    model.train()
    for batch_data in train_loader:
        optimizer.zero_grad()

        transports = batch_data['transports'].to(device)
        couplings = batch_data['couplings']
        pressures = batch_data['pressures'].to(device)

        bsize = transports.shape[0]
        transports = torch.cat((transports, batch_data['pos'].to(device)), dim=1)

        out = model(transports, couplings)

        loss = myloss(out, pressures)
        loss.backward()

        optimizer.step()

        train_l2 += loss.item()
    
    scheduler.step()

    model.eval()
    with torch.no_grad():
        for batch_data in test_loader:
            transports = batch_data['transports'].to(device)
            couplings = batch_data['couplings']
            pressures = batch_data['pressures'].to(device)

            bsize = transports.shape[0]
            transports = torch.cat((transports, batch_data['pos'].to(device)), dim=1)

            out = model(transports, couplings)
            out = pressure_encoder.decode(out)

            test_l2 += myloss(out, pressures).item()
    
    train_l2 /= n_train
    test_l2 /= n_test

    print(ep, train_l2, test_l2, default_timer() - t)





