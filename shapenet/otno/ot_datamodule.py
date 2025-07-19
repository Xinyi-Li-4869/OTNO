import numpy as np
import torch
import open3d as o3d
from ot.bregman import empirical_sinkhorn2_geomloss
from timeit import default_timer

def torus_grid(n_s_sqrt):
    theta = torch.linspace(0, 2 * torch.pi, n_s_sqrt + 1)[:-1]
    phi = torch.linspace(0, 2 * torch.pi, n_s_sqrt + 1)[:-1]
    # Create a grid using meshgrid
    X, Y = torch.meshgrid(theta, phi, indexing='ij')
    points = torch.stack((X, Y)).reshape((2, -1)).T

    r = 0.5
    R = 1.0
    x = (R + r * torch.cos(points[:, 0])) * torch.cos(points[:, 1])
    y = (R + r * torch.cos(points[:, 0])) * torch.sin(points[:, 1])
    z = r * torch.sin(points[:, 0])
    grid = torch.stack((x, y, z), axis=1)
    
    return grid


tt1 = default_timer()
path = '/your_path_to_datasets/car-pressure-data/data/'

def read_indices(file_path):
    with open(file_path, 'r') as file:
        indices = [line.strip() for line in file if line.strip().isdigit()]
    return indices


device = torch.device('cuda')
N = 611
expand_factor = 2.0 # expand the pysical size (car, target) to latent size (torus, source)
reg = 1e-06
n_t = 3586 # number of target samples
n_s_sqrt = int(np.sqrt(expand_factor)*np.ceil(np.sqrt(n_t))) # sqrt of the number of source samples
source = torus_grid(n_s_sqrt) # build source gird

# initialize
all_points = torch.zeros((N,3586,3), dtype=torch.float32)
all_normals = torch.zeros((N,3586,3), dtype=torch.float32)
all_transports = torch.zeros((N,3,n_s_sqrt,n_s_sqrt), dtype=torch.float32)
all_presures = torch.zeros((N,3586), dtype=torch.float32)
all_indices_encoder = torch.zeros((N,n_s_sqrt**2), dtype=torch.float32)
all_indices_decoder = torch.zeros((N,3586), dtype=torch.float32)
non_surjective = 0

# Read indices from the mesh.txt file
mesh_indices = read_indices('/your_path_to_datasets/car-pressure-data/watertight_meshes.txt')

for k,index in enumerate(mesh_indices):
    t1 = default_timer()
    # load mesh and pressure data
    mesh = o3d.io.read_triangle_mesh(path + 'mesh_' + index.zfill(3) + '.ply')
    target = torch.from_numpy(np.asarray(mesh.vertices).squeeze().astype(np.float32())).to(device) #(3586,3) car vertices
    mesh.compute_vertex_normals()
    normal = torch.from_numpy(np.asarray(mesh.vertex_normals).astype(np.float32()))
    pressure = np.load(path + 'press_' + index.zfill(3) + '.npy') # (3682,)
    pressure = np.concatenate((pressure[0:16], pressure[112:]), axis=0) #(3586,)
    pressure = torch.from_numpy(pressure.astype(np.float32()))

    # OT
    _, log = empirical_sinkhorn2_geomloss(X_s=source.to(device), X_t=target, reg=reg, log=True) # utilize weighted Sinkhorn a=a.to(device), b=b.to(device),
    gamma = log['lazy_plan'][:].detach() # convert the lazy tensor to torch.tensor (dense)

    # normalize the OT plan matrix by column
    row_norms = torch.norm(gamma, p=1, dim=1, keepdim=True)
    gamma_encoder = gamma / row_norms

    # transport target to source
    transport = torch.mm(gamma_encoder, target)

    # encoder: target -> source
    distances = torch.cdist(transport, target)
    indices_encoder = torch.argmin(distances, dim=1) # find the closest point in "target" (car vertices) to each point in "transport" (latent grids)
    
    # reset the transport as the closest point in target
    transport = target[indices_encoder]
    unique = len(torch.unique(indices_encoder))
    if unique!=3586:
        non_surjective += 1

    # decoder: source -> target
    indices_decoder = torch.argmin(distances, dim=0) # # find the closest point in "transport" (latent grids) to each point in "target" (car vertices)
    transport = transport.T.reshape((3,n_s_sqrt,n_s_sqrt))
    
    # save ot data
    all_points[k,...] = target.cpu()
    all_indices_encoder[k,...] = indices_encoder.cpu()
    all_indices_decoder[k,...] = indices_decoder.cpu()
    all_presures[k,...] = pressure
    all_normals[k,...] = normal
    all_transports[k,...] = transport.cpu()

    print(unique, default_timer()-t1, k+1)

save_path = '/your_path_to_shapenet/ot-data/torus_OTmean_geomloss' + '_expand' + str(expand_factor) + '.pt' 
torch.save({'points': all_points, 'transports': all_transports, 'pressures': all_presures, 'indices_encoder': all_indices_encoder, 'indices_decoder': all_indices_decoder, 'normals': all_normals}, save_path)

print("non_surjective:", non_surjective)
print(f"Total time: {default_timer()-tt1:.2f} seconds.") 
