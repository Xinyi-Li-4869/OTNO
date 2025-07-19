import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils import count_model_params
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from drivaernet.otno.TransportFNOCd import TransportFNOCd
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.metrics import r2_score

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


class DictDatasetWithConstant(DictDataset):
    def __init__(self, data_dict: dict, constant_dict: dict):
        super().__init__(data_dict)
        self.constant_dict = constant_dict

    def __getitem__(self, index):
        return_dict = {k: v[index] for k, v in self.data_dict.items()}
        return_dict.update(self.constant_dict)
        return return_dict

def torus_grid(n_s_sqrt):
    theta = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    phi = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    # Create a grid using meshgrid
    X, Y = torch.meshgrid(theta, phi, indexing='ij')
    points = torch.stack((X, Y)).reshape((2, -1)).T

    r = 1.0
    R = 1.5
    x = (R + r * torch.cos(points[:, 0])) * torch.cos(points[:, 1])
    y = (R + r * torch.cos(points[:, 0])) * torch.sin(points[:, 1])
    z = r * torch.sin(points[:, 0])

    return torch.stack((z, x, y), axis=1).reshape(n_s_sqrt,n_s_sqrt,3).permute(2,0,1)

def compute_torus_normals(n_t, r=1.0, R=1.5):
    # Define the angles for the parameterization
    theta = torch.linspace(0, 2*np.pi, n_t + 1)[:-1]
    phi = torch.linspace(0, 2*np.pi, n_t + 1)[0:-1]
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')
    
    # Calculate partial derivatives
    # Partial derivatives with respect to theta
    x_theta = -r * torch.sin(theta) * torch.cos(phi)
    y_theta = -r * torch.sin(theta) * torch.sin(phi)
    z_theta = r * torch.cos(theta)
    
    # Partial derivatives with respect to phi
    x_phi = -(R + r * torch.cos(theta)) * torch.sin(phi)
    y_phi = (R + r * torch.cos(theta)) * torch.cos(phi)
    z_phi = torch.zeros_like(x_phi)
    
    # Compute the cross product to get normals
    normals = torch.cross(torch.stack((z_theta, x_theta, y_theta), dim=-1),
                          torch.stack((z_phi, x_phi, y_phi), dim=-1), dim=2)
    
    # Normalize the normals
    norm = torch.sqrt((normals ** 2).sum(dim=2, keepdim=True))
    normals_normalized = normals / norm

    return normals_normalized.reshape(n_t**2, 3)

def test(test_loader, model, modeltype, device):
    model.eval()
    t_test = default_timer()

    test_mse, test_mae, max_mae = 0.0, 0.0, 0.0
    all_outputs, all_targets = [], []

    # Start evaluation
    with torch.no_grad():
        total_inference_time = 0  # To accumulate the total inference time
        for batch_data in test_loader:
            normals = batch_data['normals'][0].to(device)
            indices_encoder = batch_data['indices_encoder'][0].to(dtype=torch.long, device=device)
            indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)
            weights = batch_data['weights'][0].to(device)
            Cd = batch_data['Cd'][0].to(device)

            n_s_sqrt = int(np.sqrt(len(indices_encoder)))
            n = batch_data['transports'].shape[2]
            transports = batch_data['transports'].reshape(n_s_sqrt,n_s_sqrt,n).permute(2,0,1).unsqueeze(0).to(device)

            # compute features
            normals = normals[indices_encoder]
            torus_normals = batch_data['torus_nor'][0].to(device)
            normal_features = torch.cross(normals, torus_normals, dim=1).reshape(n_s_sqrt,n_s_sqrt,3).permute(2,0,1).unsqueeze(0)
            weights_features = weights[indices_encoder].reshape(n_s_sqrt,n_s_sqrt,1).permute(2,0,1).unsqueeze(0)
        
            # cat input features
            transports = torch.cat((transports, batch_data['pos'].to(device), normal_features), dim=1)

            # Start the timer just before the model inference
            start_time = default_timer()
            out = model(transports, indices_decoder, weights_features)

            # Stop the timer after the inference
            inference_time = default_timer() - start_time
            total_inference_time += inference_time

            # normalizer decode
            out = Cds_encoder.inverse_transform(out)
            
            loss = (out - Cd) ** 2
            abs_error = torch.abs(out - Cd)

            test_mse += loss.item()
            test_mae += abs_error.item()
            max_mae = max(max_mae, abs_error.item())  # Update max MAE
            
            # Store predictions and actual values for R² calculation
            all_outputs.append(out.unsqueeze(0).cpu())
            all_targets.append(Cd.unsqueeze(0).cpu())

    # Compute final MSE and MAE
    test_mse /= n_test
    test_mae /= n_test
    average_inference_time = total_inference_time/n_test

    # Convert lists to tensors for R² computation
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    # Calculate R-squared using sklearn's functionality
    r2 = r2_score(all_targets.numpy(), all_outputs.numpy())

    # Output results
    print('\n')
    print(modeltype)
    print(f"Test MSE: {test_mse:.8f}")
    print(f"Test MAE: {test_mae:.8f}")
    print(f"Max MAE: {max_mae:.8f}")
    print(f"Test R²: {r2:.8f}")
    print(f"Test Time: {default_timer() - t_test:.2f}s")
    print(f"Total Inference Time over 595 test data: {total_inference_time:.6f} seconds")
    print(f"Average Inference Time per Batch: {average_inference_time:.6f} seconds")
    

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(all_targets.numpy(), all_outputs.numpy(), color='blue', label='Predictions')
    plt.plot([all_targets.min(), all_targets.max()], [all_targets.min(), all_targets.max()], '-.', lw=2, label='Ideal Prediction')
    plt.xlabel('Ground Truth Cd')
    plt.ylabel('Predicted Cd')
    plt.title('Prediction vs Ground Truth')
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_save_path+modeltype+'.png', format='png')

def create_model():
    return TransportFNOCd(n_modes=(32, 32), hidden_channels=120, in_channels=9, norm='group_norm',
                     use_mlp=True, mlp={'expansion': 1.0, 'dropout': 0}, domain_padding=0.125,
                     factorization='tucker', rank=0.4)

if __name__ == '__main__':
    t1 = default_timer()

    # Configs
    latent_shape = 'torusXpole'
    voxel_size = 0.1 
    reg = 1e-06
    expand_factor = 3.0
    path = '/path_to_drivaer/STL200k_'+ latent_shape + '_meanOTidx_voxelsize' + str(voxel_size) + '_reg' + str(reg) + '_expand' + str(expand_factor)
    best_model_path = f'drivaernet/otno/saved_models/Cd_200kstl_{latent_shape}_meanOTidx_voxelsize{voxel_size}_expand{expand_factor}.pth'
    figure_save_path = f'drivaernet/otno/visualization/Cd_{latent_shape}_meanOTidx_voxelsize{voxel_size}_expand{expand_factor}_'

    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(figure_save_path), exist_ok=True)

    grid = torus_grid
    train_data = torch.load(path+'/train_design_ids.pt')
    val_data = torch.load(path+'/val_design_ids.pt')
    test_data = torch.load(path+'/test_design_ids.pt')
    device = torch.device('cuda:0')
    print(device)
    print(path)

    n_train = train_data['Cd'].shape[0]
    train_data['pos'] = [grid(train_data['transports'][i].shape[1]) for i in range(n_train)]
    train_data['weights'] = [torch.matmul(train_data['normals'][i],torch.tensor([1, 0, 0], dtype=torch.float32))  for i in range(n_train)]
    train_data['torus_nor'] = [compute_torus_normals(train_data['transports'][i].shape[1]) for i in range(n_train)]

    n_test = test_data['Cd'].shape[0]
    test_data['pos'] = [grid(test_data['transports'][i].shape[1]) for i in range(n_test)]
    test_data['weights'] = [torch.matmul(test_data['normals'][i],torch.tensor([1, 0, 0], dtype=torch.float32))  for i in range(n_test)]
    test_data['torus_nor'] = [compute_torus_normals(test_data['transports'][i].shape[1]) for i in range(n_test)]

    n_val = val_data['Cd'].shape[0]
    val_data['pos'] = [grid(val_data['transports'][i].shape[1]) for i in range(n_val)]
    val_data['weights'] = [torch.matmul(val_data['normals'][i],torch.tensor([1, 0, 0], dtype=torch.float32))  for i in range(n_val)]
    val_data['torus_nor'] = [compute_torus_normals(val_data['transports'][i].shape[1]) for i in range(n_val)]

    n = train_data['transports'][0].shape[0]
    print(n)
    cat_train_transports = torch.cat([train_data['transports'][i].permute(1,2,0).reshape(-1, n) for i in range(n_train)], dim=0)
    transports_encoder = UnitGaussianNormalizer(dim=[0])
    transports_encoder.fit(cat_train_transports)
    Cds_encoder = UnitGaussianNormalizer(dim=[0])
    Cds_encoder.fit(train_data['Cd'])

    train_data['transports'] = [transports_encoder.transform(train_data['transports'][i].permute(1,2,0).reshape(-1,n)) for i in range(n_train)]
    test_data['transports'] = [transports_encoder.transform(test_data['transports'][i].permute(1,2,0).reshape(-1,n)) for i in range(n_test)]
    val_data['transports'] = [transports_encoder.transform(val_data['transports'][i].permute(1,2,0).reshape(-1,n)) for i in range(n_val)]

    train_data['Cd'] = Cds_encoder.transform(train_data['Cd'])
    Cds_encoder.to(device)

    train_dataset = DictDataset(train_data)
    test_dataset = DictDataset(test_data)
    val_dataset = DictDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    print(n_train, n_test, n_val)

    model = create_model()
    print(count_model_params(model))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())#, lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)

    epochs = 51
    best_mse = float('inf')

    for ep in range(epochs):
        t_train = default_timer()
        train_mse = 0.0
        model.train()
        for batch_data in train_loader:
            optimizer.zero_grad()
            
            normals = batch_data['normals'][0].to(device)
            indices_encoder = batch_data['indices_encoder'][0].to(dtype=torch.long, device=device)
            indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)
            weights = batch_data['weights'][0].to(device)
            Cd = batch_data['Cd'][0].to(device)

            n_s_sqrt = int(np.sqrt(len(indices_encoder)))
            transports = batch_data['transports'].reshape(n_s_sqrt,n_s_sqrt,n).permute(2,0,1).unsqueeze(0).to(device)

            normals = normals[indices_encoder]
            torus_normals = batch_data['torus_nor'][0].to(device)
            normal_features = torch.cross(normals, torus_normals, dim=1).reshape(n_s_sqrt,n_s_sqrt,3).permute(2,0,1).unsqueeze(0)
            weights_features = weights[indices_encoder].reshape(n_s_sqrt,n_s_sqrt,1).permute(2,0,1).unsqueeze(0)
        
            transports = torch.cat((transports, batch_data['pos'].to(device), normal_features), dim=1)
            
            out = model(transports, indices_decoder, weights_features)

            loss = (out - Cd)**2
            loss.backward()

            optimizer.step()

            train_mse += loss

        
        train_mse /= n_train
        print(f"Epoch {ep+1} Training loss: {train_mse.item():.8f} Time: {default_timer() - t_train:.2f}s")

        model.eval()
        t_val = default_timer()
        val_l2 = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                normals = batch_data['normals'][0].to(device)
                indices_encoder = batch_data['indices_encoder'][0].to(dtype=torch.long, device=device)
                indices_decoder = batch_data['indices_decoder'][0].to(dtype=torch.long, device=device)
                weights = batch_data['weights'][0].to(device)
                Cd = batch_data['Cd'][0].to(device)

                n_s_sqrt = int(np.sqrt(len(indices_encoder)))
                transports = batch_data['transports'].reshape(n_s_sqrt,n_s_sqrt,n).permute(2,0,1).unsqueeze(0).to(device)

                normals = normals[indices_encoder]
                torus_normals = batch_data['torus_nor'][0].to(device)
                normal_features = torch.cross(normals, torus_normals, dim=1).reshape(n_s_sqrt,n_s_sqrt,3).permute(2,0,1).unsqueeze(0)
                weights_features = weights[indices_encoder].reshape(n_s_sqrt,n_s_sqrt,1).permute(2,0,1).unsqueeze(0)
            
                transports = torch.cat((transports, batch_data['pos'].to(device), normal_features), dim=1)

                out = model(transports, indices_decoder, weights_features)
               
                out = Cds_encoder.inverse_transform(out)
                loss = (out - Cd)**2

                val_l2 += loss

        val_l2 /= n_val
        print(f"Epoch {ep+1} Validation loss: {val_l2.item():.8f} Time: {default_timer() - t_val:.2f}s")
        scheduler.step(val_l2)

        if val_l2 < best_mse:
            best_mse = val_l2
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with MSE: {best_mse.item():.8f}")

        if ep==epochs-1:
            test(test_loader, model, 'final', device)
            
        
    # test
    best_model = create_model()
    best_model.load_state_dict(torch.load(best_model_path))
    best_model = best_model.to(device)

    test(test_loader, best_model, 'best', device)
    print(f"Total time: {default_timer() - t1:.2f} seconds")
