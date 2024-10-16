'''import debugpy

# Listen for incoming connections from the debugger on a specified port
debugpy.listen(('0.0.0.0', 5678))  # 0.0.0.0 allows connections from any IP
print("Debugger is waiting for connection on port 5678...")
debugpy.wait_for_client()  # Blocks execution until the debugger is attached'''

import ot
import numpy as np
#np.__config__.show()  # Show linked BLAS/LAPACK libraries
#import pyvista as pv
import open3d as o3d
import matplotlib.pyplot as plt
import torch
from timeit import default_timer
tt1 = default_timer()
#path = '/home/nikola/HDD/car-cfd/data/'
path = '/central/groups/tensorlab/xinyi/GINO-main/datasets/car-pressure-data/data/'

def read_indices(file_path):
    with open(file_path, 'r') as file:
        indices = [line.strip() for line in file if line.strip().isdigit()]
    return indices

# Read indices from the mesh.txt file
mesh_indices = read_indices('/central/groups/tensorlab/xinyi/GINO-main/datasets/car-pressure-data/watertight_meshes.txt')

N = 611
all_points = np.zeros((N,3586,3), dtype=np.float32)
all_transports = np.zeros((N,3,103,103), dtype=np.float32)
all_presures = np.zeros((N,3586), dtype=np.float32)
all_couplings = []

for k,index in enumerate(mesh_indices):
    #print(k,index)
    t1 = default_timer()

    #mesh = pv.read(path + 'mesh_' + str(k+1).zfill(3) + '.vtk').extract_surface()
    #target = np.asarray(mesh.points)
    mesh = o3d.io.read_triangle_mesh(path + 'mesh_' + index.zfill(3) + '.ply')
    target = np.asarray(mesh.vertices).squeeze() #(3586,3)
    #pressure = np.asarray(mesh['point_scalars'])
    pressure = np.load(path + 'press_' + index.zfill(3) + '.npy') # (3682,)
    pressure = np.concatenate((pressure[0:16], pressure[112:]), axis=0) #(3586,)
    n_t = target.shape[0]

    expand_factor = np.sqrt(3.0)
    n_s_sqrt = int(expand_factor*np.ceil(np.sqrt(n_t)))
    n_s = n_s_sqrt**2
    theta = np.arccos(np.linspace(-1, 1, n_s_sqrt, endpoint=False) + 1/n_s_sqrt)  # Uniform in cosine of the angle
    #print(theta)
    phi = np.linspace(0, 2*np.pi, n_s_sqrt, endpoint=False)  # Azimuthal angle

    X, Y = np.meshgrid(theta, phi, indexing='ij')
    points = np.stack((X.flatten(), Y.flatten()), axis=1)

    R = 1
    x = R * np.sin(points[:,0]) * np.cos(points[:,1])
    y = R * np.sin(points[:,0]) * np.sin(points[:,1])
    z = R * np.cos(points[:,0])

    source = np.stack((x,y,z), axis=1)

    a, b = np.ones((n_s,)) / n_s, np.ones((n_t,)) / n_t
    
    M = ot.dist(source, target)
    gamma = ot.emd(a, b, M, numItermax=1000000)

    '''
    coupling = np.zeros((n_s,3))
    for j in range(n_s):
        gamma[j,:] /= np.sum(gamma[j,:])
        coupling[j,:] = np.dot(target.T, gamma[j,:])
        
        for j in range(n_s):
        gamma[j,...] /= np.sum(gamma[j,...])
        for j_t in range(n_t):
            if gamma[j,j_t] < 1e-16:
                gamma[j,j_t] = 0
        gamma[j,...] /= np.sum(gamma[j,...])
        ind = gamma[j,:].argmax()
        transport[j,...] = target[ind,...]
    '''

    transport = np.zeros((n_s,3))
    for j in range(n_s):
        ind = gamma[j,:].argmax()
        transport[j,...] = target[ind,...]
        gamma[j,...] /= np.sum(gamma[j,...])
    
    #sparsity = np.count_nonzero(gamma) / gamma.size
    #print("sparsity:",sparsity)
    

    unique = np.unique(transport, axis=0).shape[0]

    transport = transport.T.reshape((3,n_s_sqrt,n_s_sqrt))

    all_points[k,...] = target.astype(np.float32())
    all_couplings.append(torch.from_numpy(gamma.astype(np.float32())).to_sparse())
    all_presures[k,...] = pressure.astype(np.float32())
    all_transports[k,...] = transport.astype(np.float32())

    print(unique, default_timer()-t1, k+1)

save_path = '/central/groups/tensorlab/xinyi/OT/sphere_data_shapenet3_uniform+1.pt' 
torch.save({'points': torch.from_numpy(all_points), 'transports': torch.from_numpy(all_transports), 'pressures': torch.from_numpy(all_presures), 'couplings': all_couplings}, save_path)
tt2 = default_timer()
print(f"Total time: {tt2-tt1:.2f} seconds.")
#shapenet3_uniform 13174.27 seconds
#uniform+1: 13095.77 seconds.
'''
T = 60
movement = np.zeros((T,n_s,3))

for j in range(n_s):
    tx = np.linspace(source[j,0], coupling[j,0], T).reshape((T,1))
    ty = np.linspace(source[j,1], coupling[j,1], T).reshape((T,1))
    tz = np.linspace(source[j,2], coupling[j,2], T).reshape((T,1))
    movement[:,j,:] = np.concatenate((tx,ty,tz), axis=1)


plotter = pv.Plotter()
plotter.open_movie('torus_to_car.mp4', framerate=20)
plotter.add_points(source, render_points_as_spheres=True, point_size=5, name='points')
plotter.camera_position = 'xy'
plotter.camera.azimuth= 135
plotter.camera.elevation= 30
#plotter.camera.zoom(0.9)

plotter.write_frame()
for j in range(1,T):
    plotter.add_points(movement[j,...], render_points_as_spheres=True, point_size=5, name='points')
    plotter.write_frame()

plotter.close()
'''