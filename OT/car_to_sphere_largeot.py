import numpy as np
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from neuralop.utils import UnitGaussianNormalizer
from timeit import default_timer

import sys
sys.path.append('/central/groups/tensorlab/xinyi/OT/Large-Scale-OT')
sys.path.append('/central/groups/tensorlab/xinyi/OT/Large-Scale-OT/StochasticOTClasses')
from StochasticOTClasses.StochasticOTDiscrete import PyTorchStochasticDiscreteOT


def sphere_grid(n_s,R=1.0):
    n_t_sqrt = int(np.ceil(np.sqrt(n_s)))

    theta = np.linspace(0, np.pi, n_t_sqrt)  # Polar angle
    phi = np.linspace(0, 2*np.pi, n_t_sqrt + 1)[:-1]  # Azimuthal angle

    X, Y = np.meshgrid(theta, phi, indexing='ij')
    points = np.stack((X.flatten(), Y.flatten()), axis=1)

    # Convert spherical to Cartesian coordinates
    x = R * np.sin(points[:, 0]) * np.cos(points[:, 1]) 
    y = R * np.sin(points[:, 0]) * np.sin(points[:, 1]) 
    z = R * np.cos(points[:, 0])

    xt = np.stack((x, y, z), axis=1).astype(np.float32)

    return xt

def cubed_sphere_grid(n_s, R=1.0):
    n_t_sqrt = int(np.ceil(np.sqrt(n_s) / np.sqrt(6)))  # Divided by sqrt(6) because there are 6 faces

    # Generate grid points for the range [-1, 1] on each face
    lin = np.linspace(-1, 1, n_t_sqrt)
    x, y = np.meshgrid(lin, lin)
    x = x.ravel()
    y = y.ravel()

    # Six faces of the cube
    faces = [
        (x, y, np.ones_like(x) * R),   # Face 1, z = R
        (x, y, -np.ones_like(x) * R),  # Face 2, z = -R
        (x, np.ones_like(x) * R, y),   # Face 3, y = R
        (x, -np.ones_like(x) * R, y),  # Face 4, y = -R
        (np.ones_like(x) * R, x, y),   # Face 5, x = R
        (-np.ones_like(x) * R, x, y),  # Face 6, x = -R
    ]

    # Project each point from the cube to the sphere
    all_points = []
    for face in faces:
        face_points = np.vstack(face).T
        norms = np.linalg.norm(face_points, axis=1)
        spherical_points = (face_points.T / norms).T  # Normalize points to lie on the sphere
        all_points.append(spherical_points)

    # Concatenate points from all faces
    all_points = np.vstack(all_points)
    return all_points.astype(np.float32)

def sphere_uniform_random(n_s,R):
    n = int(np.ceil(np.sqrt(n_s)))
    # Generate random angles
    theta = np.random.uniform(0, 2 * np.pi, n)  # azimuthal angle
    phi = np.arccos(1 - 2 * np.random.uniform(0, 1, n))  # polar angle, using the inversion method# Convert spherical coordinates to Cartesian coordinates for 3D points
    x = R * np.sin(phi) * np.cos(theta)
    y = R * np.sin(phi) * np.sin(theta)
    z = R * np.cos(phi)

    # Stack them into a (n, 3) array
    points = np.stack((x, y, z), axis=-1)
    return points


t1 = default_timer()
path = '/central/groups/tensorlab/xinyi/GINO-main/datasets/car-pressure-data'
device_index=0

N = 611  #N=306 Total time: 48899.67 seconds. N=305 Total time: 44075.44 seconds.
all_car_to_sphere = []
all_sphere_grid = []
all_pressure = []
all_vertices = []

num_samples = 3600
R_sphere = 1.2
#xt = sphere_grid(num_samples, R_sphere)
xt = sphere_grid(num_samples,R_sphere)
wt = np.full(len(xt), 1 / len(xt))
print(len(xt))
filename="/central/groups/tensorlab/xinyi/GINO-main/datasets/car-pressure-data/watertight_meshes.txt"
with open(filename, "r") as fp:
    mesh_ind = fp.read().split("\n")
    mesh_ind = [int(a) for a in mesh_ind]

for index in mesh_ind:
    tt1 = default_timer()
    mesh_path = path + '/data/mesh_' + str(index).zfill(3) + '.ply'
    press_path = path + '/data/press_' + str(index).zfill(3) + '.npy'
    
    pressure = np.load(press_path).reshape((-1,)).astype(np.float32)
    pressure = np.concatenate((pressure[0:16], pressure[112:]), axis=0)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # xs--source:car
    # xt--target:sub-manifold
    xs = np.asarray(mesh.vertices).squeeze() 
    normalization = UnitGaussianNormalizer(torch.Tensor(xs), eps=1e-6, reduce_dim=[0], verbose=False)
    xs = normalization.encode(torch.Tensor(xs))
    print(xs.shape)
    all_vertices.append(xs)
    xs = xs.detach().numpy() 
    ws = np.full(len(xs), 1 / len(xs))

    discreteOTComputer = PyTorchStochasticDiscreteOT(xs, ws, xt, wt, reg_type='entropy', reg_val=0.5, device_type='gpu', device_index=device_index)
    history = discreteOTComputer.learn_OT_dual_variables(epochs=10, batch_size=200, lr=0.0001)
    bp_history = discreteOTComputer.learn_barycentric_mapping(epochs=3, batch_size=200, lr=0.0001)
    xsf = discreteOTComputer.evaluate_barycentric_mapping(xs)
    
    device = torch.device('cuda:%d' % (device_index,))
    points = torch.tensor(xsf, dtype=torch.float32).to(device)

    # Calculate distances from the origin
    distances = torch.norm(points, dim=1)
    print(torch.min(distances), torch.max(distances))
    scale = 1.0 / distances

    # Apply scaling
    points *= scale.unsqueeze(1)

    all_car_to_sphere.append(points.cpu())
    #all_car_to_sphere.append(torch.tensor(xsf, dtype=torch.float32))
    all_pressure.append(torch.tensor(pressure, dtype=torch.float32))
    #all_sphere_grid.append(torch.tensor(xt, dtype=torch.float32))

    print(index+1, default_timer()-tt1)

save_path = '/central/groups/tensorlab/xinyi/OT/sphere_data_largeot.pt'
torch.save({'vertices':all_vertices, 'car_to_sphere':all_car_to_sphere,'pressure':all_pressure},save_path)   #'sphere_grid':all_sphere_grid, 
print(f"Total time: {default_timer()-t1:.2f} seconds.") #5551.25 seconds.


#test
data =  torch.load(save_path)

#points = data['car_to_sphere'][0]
points = torch.tensor(xsf, dtype=torch.float32)
distances = torch.norm(points, dim=1)
print(torch.min(distances), torch.max(distances))
points = points.numpy()
pred = data['pressure'][0].numpy()

# Ensure tensors are detached and converted to numpy arrays
x = points[:, 0]  # X coordinates
y = points[:, 1]  # Y coordinates
z = points[:, 2]  # Z coordinates
values = pred.squeeze() # Values used for color mapping

# Normalize the values for consistent scaling across different datasets
range_values = values.max() - values.min()
mid_values = (values.max() + values.min()) / 2.0
values = (values - mid_values) / range_values  # Normalize values

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Sort points by depth to improve the visual effect of transparency
order = np.argsort(z)
x = -x[order]
y = y[order]
z = z[order]
values = values[order]

# Create a scatter plot with smaller points and semi-transparency
scatter = ax.scatter(x, y, z, c=values, cmap='viridis', alpha=0.6, s=15)

# Set the same scale for all axes
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Add a color bar which maps values to colors.
#cbar = fig.colorbar(scatter, ax=ax, extend='both')
#cbar.set_label('True Values')

#ax.set_xlabel('X Coordinates')
#ax.set_ylabel('Y Coordinates')
#ax.set_zlabel('Z Coordinates')
ax.set_box_aspect([1,1,1])  # Equal aspect ratio


plt.savefig('/central/groups/tensorlab/xinyi/OT/test.png')

# 关闭图形，防止在服务器上消耗资源
plt.close(fig)