import torch
from timeit import default_timer
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pyvista as pv
import os

def plot(x_in, out, path):
    x = x_in[:, 0]  # X coordinates
    y = x_in[:, 1]  # Y coordinates
    z = x_in[:, 2]  # Z coordinates
    values = out  # Values used for color mapping

    # Normalize the values for consistent scaling across different datasets
    range_values = values.max() - values.min()
    mid_values = (values.max() + values.min()) / 2.0
    values = (values - mid_values) / range_values  # Normalize values

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Sort points by depth to improve the visual effect of transparency
    order = np.argsort(z)
    x = x[order]
    y = y[order]
    z = z[order]
    values = values[order]

    # Create a scatter plot with smaller points and semi-transparency
    scatter = ax.scatter(x, y, z, c=values, cmap='viridis', alpha=0.6, s=15)

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Remove all axis labels and ticks
    #ax.set_axis_off()

    # Remove grid and box (by setting visibility to False)
    #ax.grid(False)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    plt.title('Random sampling')

    # Save the plot to a file without displaying it
    plt.savefig('/central/groups/tensorlab/xinyi/GINO-OT/image/' + path + '.png', format='png')
    plt.close()  # Close the plot window

    # Plot for input data
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, color='blue', alpha=0.2, s=15)  # Plotting as blue points
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    #ax.set_axis_off()
    #ax.grid(False)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    plt.title('Random sampling')
    plt.savefig(f'/central/groups/tensorlab/xinyi/GINO-OT/image/{path}_mesh.png', format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

path = '/central/groups/tensorlab/xinyi/GINO-main/datasets/drivaer-net-dataset'
file_name = f"DrivAer_F_D_WM_WW_{str(2).zfill(4)}.vtk"
full_path = os.path.join(path, file_name)

mesh = pv.read(full_path)
target = np.asarray(mesh.points,dtype=np.float32) #(n_in,3) n_in=413460
pressure = np.asarray(mesh.point_data['p'],dtype=np.float32) #(3586,)
n_t = target.shape[0]
#shuffled_indices = torch.randperm(n_t)
#target = target[shuffled_indices]
#pressure = pressure[shuffled_indices]
selected_indices = []
samples = 100
for batch in range(n_t//samples):
    indices = [batch*samples]
    selected_indices.extend(indices)
x_in = target[selected_indices]
out = pressure[selected_indices]

plot(x_in,out,'drivaer_sample')