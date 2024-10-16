import numpy as np
import torch

def compute_gaussian_curvature_torus(n_s_sqrt, R, r):
    # Create theta and phi angles with torch.linspace
    theta = torch.linspace(0, 2 * torch.pi, n_s_sqrt, dtype=torch.float32)
    phi = torch.linspace(0, 2 * torch.pi, n_s_sqrt, dtype=torch.float32)
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')

    # Compute the Gaussian curvature
    gaussian_curvature = torch.cos(theta) / (R + r * torch.cos(theta))

    return gaussian_curvature

# Example usage
n_s_sqrt = 100  # resolution of the grid, providing a finer resolution for better visualization
R = 1.5  # Major radius
r = 1  # Minor radius
gaussian_curvature = compute_gaussian_curvature_torus(n_s_sqrt, R, r)

# Convert to numpy for visualization or further processing
gaussian_curvature = gaussian_curvature.numpy()

#print(gaussian_curvature)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm

theta, phi = np.meshgrid(np.linspace(0, 2 * np.pi, n_s_sqrt), np.linspace(0, 2 * np.pi, n_s_sqrt), indexing='ij')
x = (R + r * np.cos(theta)) * np.cos(phi)
y = (R + r * np.cos(theta)) * np.sin(phi)
z = r * np.sin(theta)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
norm = Normalize(vmin=gaussian_curvature.min(), vmax=gaussian_curvature.max())  # Adjust the range
mappable = ScalarMappable(norm=norm, cmap=cm.jet)

# Apply the colormap to the Gaussian curvature values for color mapping
colors = mappable.to_rgba(gaussian_curvature)

# Plot the surface with the mapped colors
surf = ax.plot_surface(x, y, z, facecolors=colors, linewidth=0, antialiased=False)

# Add a color bar using the ScalarMappable to ensure correct color mapping
fig.colorbar(mappable, ax=ax, orientation='vertical')

ax.set_title('Gaussian Curvature on a Torus')
plt.savefig('/central/groups/tensorlab/xinyi/OT/test.png')


'''
import pyvista as pv

def is_genus_zero(file_path):
    # Load the mesh from a VTK file
    mesh = pv.read(file_path)
    
    # Ensure the mesh is a single connected component
    connected_components = mesh.connectivity(extraction_mode='largest')
    if mesh.n_points != connected_components.n_points:
        print("Mesh is not a single connected component.")
        return False
    
    # Calculate the Euler characteristic
    V = mesh.n_points
    #E = mesh.n_edges
    F = mesh.n_faces_strict
    
    print(V,F)
    chi = V - F/2
    # Genus calculation for a closed mesh without boundary
    genus = (2 - chi) / 2
    
    return genus == 0

# Example usage
file_path = '/central/groups/tensorlab/xinyi/GINO-main/datasets/drivaer-net-dataset/DrivAer_F_D_WM_WW_0001.vtk'
result = is_genus_zero(file_path)
print("Is the mesh genus-0?", result)

N = 611
path = '/central/groups/tensorlab/xinyi/OT/sphere_data_largeot.pt'
data = torch.load(path)
all_vertices = data["vertices"]
all_car_to_sphere = data["car_to_sphere"]
all_pressure = data["pressure"]
all_pressure_valid = []
all_vertices_valid = []

for index in range(N):
    vertices = torch.tensor(all_vertices[index], dtype=torch.float32)
    all_vertices_valid.append(vertices)
    pressure = all_pressure[index]
    pressure = torch.concatenate((pressure[0:16], pressure[112:]), axis=0)
    print(pressure.shape)
    all_pressure_valid.append(torch.tensor(pressure, dtype=torch.float32))
    print(index)

save_path = '/central/groups/tensorlab/xinyi/OT/sphere_data_largeot_valid.pt'

torch.save({'vertices':all_vertices_valid,'car_to_sphere':all_car_to_sphere,'pressure':all_pressure_valid},save_path)
'''
'''import matplotlib.pyplot as plt
import numpy as np

def plot_2d_grid_with_points():
    path = '/central/groups/tensorlab/xinyi/OT/grid_points.png'
    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(6, 6))

    # Generate grid points
    x = np.arange(0, 21, 1)
    y = np.arange(0, 21, 1)
    X, Y = np.meshgrid(x, y)

    # Plot each grid point
    ax.scatter(X, Y, color='blue', s=10)  # `s` controls the size of the points
    

    # Set the limits of the plot
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-0.5, 20.5)

    # Hide axes and tick labels
    ax.axis('off')
    # Draw grid lines
    ax.grid(True, which='both', color='black', linestyle='-', linewidth=1)
    # Save the plot to a file
    plt.savefig(path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

plot_2d_grid_with_points()'''

'''import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_spherical_grid(n_s_sqrt):
    # Generate grid
    theta = np.arccos(np.linspace(-1, 1, n_s_sqrt, endpoint=False) + 1/n_s_sqrt)  # Polar angle (colatitude)
    print(np.linspace(-1, 1, n_s_sqrt, endpoint=False))
    phi = np.linspace(0, 2 * np.pi, n_s_sqrt, endpoint=False)  # Azimuthal angle

    X, Y = np.meshgrid(theta, phi, indexing='ij')
    points = np.stack((X.flatten(), Y.flatten()), axis=1)
    print(points.shape)

    R = 1  # Radius of the sphere
    x = R * np.sin(points[:, 0]) * np.cos(points[:, 1])  # Convert to Cartesian coordinates
    y = R * np.sin(points[:, 0]) * np.sin(points[:, 1])
    z = R * np.cos(points[:, 0])

    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for spherical points
    ax.scatter(x, y, z, color='b', alpha=0.6, s=10)  # Adjust size and transparency as needed

    # Labeling
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Aspect ratio for sphere
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

    # Set limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_title('Arccos Spherical Grid')

    # Show plot
    plt.savefig('/central/groups/tensorlab/xinyi/OT/arccos_grid.png')
    plt.close

# Call the function with desired grid size
plot_spherical_grid(103)  # For example, a grid size of 30
'''