import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from neuralop.utils import UnitGaussianNormalizer
from neuralop.models.fnogno import FNOGNO
from timeit import default_timer
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
import numpy as np
import os
import open3d as o3d
import pyvista as pv
import pandas as pd
from sklearn.metrics import r2_score
from tqdm import tqdm
import wandb
from neuralop.utils import get_wandb_api_key

#torch.manual_seed(1)
config = {
        'cuda': True,
        'exp_name': 'Cd_fnogno',
        'epochs': 100,
        'aero_coeff': '/pscratch/sd/z/zongyili/drivaer/AeroCoefficient_DrivAerNet_Filtered.csv',
        'subset_dir': '/global/homes/z/zongyili/OTFNO/DrivAerNet/train_val_test_splits_part',
        'dataset_path':  '/pscratch/sd/z/zongyili/drivaer/DrivAerNet_STLs_Combined',
        'voxel_size': 0.2,
        'query_res': [64,64,64],
        'gno_radius': 0.03,
        'gno_pos_embed_type': 'transformer',
        'lr': 0.001
    }
device = torch.device("cuda" if torch.cuda.is_available() and config['cuda'] else "cpu")
config['exp_name'] = config['exp_name'] + '_res' + str(config['query_res'][0]) + '_radius' + str(config['gno_radius']) + '_voxelsize' + str(config['voxel_size']) + '_' + config['gno_pos_embed_type'] #
print(config['exp_name'])
wandb.login(key=get_wandb_api_key())
wandb.init(config=config, name=config['exp_name'], project="drivaer-fnogno")

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

def load_and_filter_csv_to_tensor(csv_file_path, design_ids):
    data = pd.read_csv(csv_file_path, index_col='Design')
    value_mapping = data['Average Cd'].to_dict()
    valid_ids = [id for id in design_ids if id in value_mapping]
    values = [value_mapping[id] for id in valid_ids]
    values_tensor = torch.tensor(values, dtype=torch.float32)
    return values_tensor, valid_ids

def range_normalize(data, min_b, max_b, new_min, new_max):
    data = (data - min_b) / (max_b - min_b)
    data = (new_max - new_min) * data + new_min

    return data

def compute_global_bounds(mesh_path):
    # Initialize bounds as None
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])

    # List all STL files in the directory
    for file_name in os.listdir(mesh_path):
        if file_name.endswith('.stl'):
            full_path = os.path.join(mesh_path, file_name)
            try:
                # Read the mesh file
                mesh = pv.read(full_path)
                # Update global bounds
                bounds = mesh.bounds
                current_min = np.array([bounds[0], bounds[2], bounds[4]])
                current_max = np.array([bounds[1], bounds[3], bounds[5]])
                global_min = np.minimum(global_min, current_min)
                global_max = np.maximum(global_max, current_max)
            except Exception as e:
                print(f"Failed to read {full_path}: {str(e)}")

    return global_min, global_max

def compute_distances(mesh, query_points, signed_distance):
    if not isinstance(mesh, o3d.t.geometry.TriangleMesh):
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)

    if signed_distance:
        dist = scene.compute_signed_distance(query_points).numpy()
    else:
        dist = scene.compute_distance(query_points).numpy()

    closest = scene.compute_closest_points(query_points)["points"].numpy()

    return dist, closest

def are_all_meshes_watertight_open3d(mesh_path):
    for file_name in os.listdir(mesh_path):
        if file_name.endswith('.stl'):
            full_path = os.path.join(mesh_path, file_name)
            try:
                mesh = o3d.io.read_triangle_mesh(full_path)
                if not mesh.is_watertight():
                    return False
            except Exception as e:
                print(f"Failed to read {full_path}: {str(e)}")
                return False
    return True

def get_dataloader(config):
    min_b, max_b = np.array([-1.15150928e+00, -1.02194536e+00,  3.87394584e-06]), np.array([4.09356737, 1.02188468, 1.76187801]) #compute_global_bounds(mesh_path)
    are_watertight = False #are_all_meshes_watertight_open3d(mesh_path)
    #[-1.15150928e+00 -1.02194536e+00  3.87394584e-06] [4.09356737 1.02188468 1.76187801]
    #False
    #print(min_b, max_b)
    #print(are_watertight)
    all_dataloader = []

    for subset_name in ['test_design_ids', 'val_design_ids', 'train_design_ids']:
        print("Load:", subset_name)
        file_path = os.path.join(config['subset_dir'], f"{subset_name}.txt")
        with open(file_path, 'r') as file:
            design_ids = [line.strip() for line in file.readlines()]

        all_aero_coeff, valid_ids = load_and_filter_csv_to_tensor(config['aero_coeff'], design_ids)
        if subset_name=='train_design_ids':
            Cd_normalizer = UnitGaussianNormalizer(all_aero_coeff, reduce_dim=[0])
            all_aero_coeff = Cd_normalizer.encode(all_aero_coeff)

        all_vertices = []
        all_queries = []
        all_df = []
        all_vertex_normals = []

        for design_id in tqdm(valid_ids):
            file_name = f"{design_id}.stl"
            full_path = os.path.join(config['dataset_path'], file_name)
            try:
                pv_mesh = pv.read(full_path)
            except FileNotFoundError:
                print(f"File not found: {full_path}")
            vertices = np.asarray(pv_mesh.points)
            faces = np.asarray(pv_mesh.faces).reshape(-1, 4)[:, 1:4]  # PyVista faces include a count at the start

            # Create an Open3D TriangleMesh object
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

            # Compute normals
            #o3d_mesh.compute_vertex_normals()
            #normals = np.asarray(o3d_mesh.vertex_normals)
            # Convert numpy array to Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)

            # Voxel downsampling
            down_pcd = pcd.voxel_down_sample(config['voxel_size'])
            down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=config['voxel_size'] * 2, max_nn=30))
            #poisson_mesh, double_vector = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(down_pcd, depth=9)

            vertices = np.asarray(down_pcd.points)
            normals = np.asarray(down_pcd.normals)

            # create query points
            tx = np.linspace(min_b[0], max_b[0], config['query_res'][0])
            ty = np.linspace(min_b[1], max_b[1], config['query_res'][1])
            tz = np.linspace(min_b[2], max_b[2], config['query_res'][2])
            query_points = np.stack(
                np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            ).astype(np.float32)

            df, closted = compute_distances(o3d_mesh, query_points, are_watertight)

            vertices = range_normalize(vertices, min_b, max_b, 0, 1)
            queries = range_normalize(query_points, min_b, max_b, 0, 1)
            all_vertices.append(torch.from_numpy(vertices).to(torch.float32))
            all_queries.append(torch.from_numpy(queries).to(torch.float32))
            all_vertex_normals.append(torch.from_numpy(normals).to(torch.float32))
            all_df.append(torch.from_numpy(np.expand_dims(df, -1)).to(torch.float32))

        data_dict = {'Cds': all_aero_coeff, 'vertices': all_vertices, 'queries': all_queries, 'df': all_df, 'vertex_normals': all_vertex_normals}
        dataset = DictDataset(data_dict)
        all_dataloader.append(DataLoader(dataset))

    return all_dataloader, Cd_normalizer

def train_and_evaluate(model: torch.nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, Cd_normalizer, config: dict):
    train_losses, val_losses = [], []
    training_start_time = default_timer()  # Start timing for training

    # Initialize optimizer and schedular
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)

    best_mse = float('inf')  # Initialize the best MSE as infinity

    # Training loop over the specified number of epochs
    for epoch in range(config['epochs']):
        epoch_start_time = default_timer()  # Start timing for this epoch
        model.train()  # Set the model to training mode
        total_loss = 0

        # Iterate over the training data
        for data in train_dataloader:

            optimizer.zero_grad()
            targets = data['Cds'][0].to(device)
            outputs = model(data['queries'][0].to(device), data['vertices'][0].to(device), data['df'][0].to(device))

            weights = torch.matmul(data['vertex_normals'][0].to(dtype=torch.float32, device=device), torch.tensor([1, 0, 0], dtype=torch.float32, device=device))
            outputs = torch.mean(outputs.squeeze()*weights)
            
            loss = (targets-outputs)**2
            #print(outputs,loss)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()  # Accumulate the loss

        epoch_duration = default_timer() - epoch_start_time
        # Calculate and print the average training loss for the epoch
        avg_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Training Loss: {avg_loss:.8f} Time: {epoch_duration:.2f}s")

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0
        all_outputs, all_targets = [], []
        inference_times = []

        # No gradient computation needed during validation
        with torch.no_grad():
            # Iterate over the validation data
            for data in val_dataloader:
                inference_start_time = default_timer()
                targets = data['Cds'][0].to(device)
                outputs = model(data['queries'][0].to(device), data['vertices'][0].to(device), data['df'][0].to(device))

                weights = torch.matmul(data['vertex_normals'][0].to(dtype=torch.float32, device=device), torch.tensor([1, 0, 0], dtype=torch.float32, device=device))
                outputs = torch.mean(outputs.squeeze()*weights)
                outputs = Cd_normalizer.decode(outputs)
                loss = (targets-outputs)**2
                val_loss += loss.item()
                inference_duration = default_timer() - inference_start_time
                inference_times.append(inference_duration)

                all_outputs.append(outputs.unsqueeze(0).cpu())
                all_targets.append(targets.unsqueeze(0).cpu())


        # Calculate and print the average validation loss for the epoch
        avg_val_loss = val_loss / len(val_dataloader)

        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        r2 = r2_score(all_targets.numpy(), all_outputs.numpy())

        val_losses.append(avg_val_loss)
        #avg_inference_time = sum(inference_times) / len(inference_times)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.8f}, Validation R²: {r2:.6f}, Inference Time: {sum(inference_times):.4f}s")
        wandb.log({
            "Training Loss": avg_loss,
            "Validation Loss": avg_val_loss,
            "Validation R²": r2,
            "Time": default_timer()-training_start_time
        })
        # Check if this is the best model based on MSE
        if avg_val_loss < best_mse:
            best_mse = avg_val_loss
            best_model_path = os.path.join('models', f'{config["exp_name"]}_best_model.pth')
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with MSE: {best_mse:.8f}")

        # Step the scheduler based on the validation loss
        scheduler.step(avg_val_loss)


    training_duration = default_timer() - training_start_time
    print(f"Total training time: {training_duration:.2f}s")
    with open(os.path.join('results', f'{config["exp_name"]}.txt'), 'a') as file:
        file.write(f"Total training time: {training_duration:.2f}s")
    # Save the final model state to disk
    model_path = os.path.join('models', f'{config["exp_name"]}_final_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    # Save losses for plotting
    np.save(os.path.join('models', f'{config["exp_name"]}_train_losses.npy'), np.array(train_losses))
    np.save(os.path.join('models', f'{config["exp_name"]}_val_losses.npy'), np.array(val_losses))

def test_model(model: torch.nn.Module, test_dataloader: DataLoader, Cd_normalizer, device):
    model.eval()  # Set the model to evaluation mode
    total_mse, total_mae, max_mae = 0, 0, 0
    all_outputs, all_targets = [], []
    total_inference_time = 0  # To track total inference time

    # Disable gradient calculation
    with torch.no_grad():
        for data in test_dataloader:
            start_time = default_timer()  # Start time for inference

            targets = data['Cds'][0].to(device)
            outputs = model(data['queries'][0].to(device), data['vertices'][0].to(device), data['df'][0].to(device))

            weights = torch.matmul(data['vertex_normals'][0].to(dtype=torch.float32, device=device), torch.tensor([1, 0, 0], dtype=torch.float32, device=device))
            outputs = torch.mean(outputs.squeeze()*weights)
            outputs = Cd_normalizer.decode(outputs)

            end_time = default_timer()  # End time for inference
            inference_time = end_time - start_time
            total_inference_time += inference_time  # Accumulate total inference time

            mse = (targets-outputs)**2 #Mean Squared Error (MSE)
            mae = torch.abs(outputs - targets) #Mean Absolute Error (MAE)

            # Accumulate metrics to compute averages later
            total_mse += mse.item()
            total_mae += mae.item()
            max_mae = max(max_mae, mae.item())

            all_outputs.append(outputs.unsqueeze(0).cpu())
            all_targets.append(targets.unsqueeze(0).cpu())

    # Compute average metrics over the entire test set
    avg_mse = total_mse / len(test_dataloader)
    avg_mae = total_mae / len(test_dataloader)

    # Convert lists to tensors for R² computation
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    # Calculate R-squared using sklearn's functionality
    r2 = r2_score(all_targets.numpy(), all_outputs.numpy())

    with open(os.path.join('results', f'{config["exp_name"]}.txt'), 'a') as file:
        # Write the test results
        file.write(f"Test MSE: {avg_mse:.8f}, Test MAE: {avg_mae:.8f}, Max MAE: {max_mae:.8f}, Test R²: {r2:.6f}\n")
        file.write(f"Total inference time: {total_inference_time:.2f}s for {len(test_dataloader)} samples\n")

    # Output test results
    print(f"Test MSE: {avg_mse:.8f}, Test MAE: {avg_mae:.8f}, Max MAE: {max_mae:.8f}, Test R²: {r2:.6f}")
    print(f"Total inference time: {total_inference_time:.2f}s for {len(test_dataloader)} samples")

def load_and_test_model(config, model_path, dataloader, normalizer, device):
    model = create_model(config)
    #model.load_state_dict(torch.load(model_path))
    #model.to(device)
    state_dict = torch.load(model_path)
    new_state_dict = model.state_dict()

    # Filter out unnecessary keys
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in new_state_dict and new_state_dict[k].size() == v.size()}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)
    test_model(model, dataloader, normalizer, device)

def create_model(config):
    return FNOGNO(
        in_channels=1,
        out_channels=1,
        projection_channels=128,
        gno_coord_dim=3,
        gno_coord_embed_dim=32,
        gno_pos_embed_type = config['gno_pos_embed_type'],
        gno_radius=config['gno_radius'],
        gno_channel_mlp_hidden_layers=[256, 128],
        gno_use_open3d=True,
        gno_transform_type='linear',  # Options: linear_kernelonly, linear, nonlinear_kernelonly, nonlinear
        fno_n_modes=[32, 32, 32],
        fno_hidden_channels=32,
        fno_lifting_channels=128,
        fno_use_channel_mlp=True,
        fno_norm='group_norm',
        fno_ada_in_features=32,
        fno_factorization='tucker',
        fno_rank=0.4,
        fno_domain_padding=0.125,
        fno_channel_mlp_expansion=1.0,
        fno_output_scaling_factor=1,
        use_torch_scatter=True,
    )


if __name__ == "__main__":
    all_dataloader, Cd_normalizer = get_dataloader(config)
    test_dataloader = all_dataloader[0]
    train_dataloader = all_dataloader[1]
    val_dataloader = all_dataloader[2]

    # Train and evaluate the model
    model = create_model(config)
    model.to(device)
    train_and_evaluate(model, train_dataloader, val_dataloader, Cd_normalizer, config)

    # Test the final model
    final_model_path = os.path.join('models', f'{config["exp_name"]}_final_model.pth')
    with open(os.path.join('results', f'{config["exp_name"]}.txt'), 'a') as file:
        file.write("Testing the final model:\n")
    print("Testing the final model:")
    load_and_test_model(config, final_model_path, test_dataloader, Cd_normalizer, device)

    # Test the best model
    best_model_path = os.path.join('models', f'{config["exp_name"]}_best_model.pth')
    with open(os.path.join('results', f'{config["exp_name"]}.txt'), 'a') as file:
        file.write("Testing the best model:\n")
    print("Testing the best model:")
    load_and_test_model(config, best_model_path, test_dataloader, Cd_normalizer, device)
