'''import debugpy

# Listen for incoming connections from the debugger on a specified port
debugpy.listen(('0.0.0.0', 5678))  # 0.0.0.0 allows connections from any IP
print("Debugger is waiting for connection on port 5678...")
debugpy.wait_for_client()  # Blocks execution until the debugger is attached'''

import ot
import os
import numpy as np
#np.__config__.show()  # Show linked BLAS/LAPACK libraries
import pyvista as pv
import open3d as o3d
import matplotlib.pyplot as plt
import torch
from timeit import default_timer

def compute_sinkhorn_ot(source, target, initial_reg=0.1, threshold=1e-16):
    n_s = source.shape[0]
    n_t = target.shape[0]
    a, b = np.ones(n_s) / n_s, np.ones(n_t) / n_t
    
    M = ot.dist(source, target, metric='sqeuclidean')
    M /= M.max()  # Normalize the cost matrix
    M += 1e-9  # Avoid division by zero

    reg = initial_reg
    gamma = None
    try:
        gamma = ot.sinkhorn(a, b, M, reg)
    except RuntimeError as e:
        print("Increasing regularization due to:", e)
        reg *= 10  # Increase regularization
        gamma = ot.sinkhorn(a, b, M, reg)
    # Apply threshold to induce sparsity
    gamma[gamma < threshold] = 0
    return gamma


#path = '/home/nikola/HDD/car-cfd/data/'
path = '/central/groups/tensorlab/xinyi/GINO-main/datasets/drivaer-net-dataset'
device = torch.device('cuda')

N = 61
all_points = []
all_transports = []
all_presures = []
all_couplings = []

k=0
index=1
#1,65,129,195,259,325,389,455,519,585(62)

while k<N:
    t1 = default_timer()
    file_name = f"DrivAer_F_D_WM_WW_{str(index).zfill(4)}.vtk"
    full_path = os.path.join(path, file_name)
            
    if not os.path.exists(full_path): 
        index +=1
        continue 

    mesh = pv.read(full_path)
    target = np.asarray(mesh.points,dtype=np.float32) #(n_in,3) n_in=413460
    pressure = np.asarray(mesh.point_data['p'],dtype=np.float32) #(3586,)
    n_t = target.shape[0]

    #expand_factor = np.sqrt(2.0)
    # = int(expand_factor*np.ceil(np.sqrt(n_t)))
    n_s_sqrt = 80
    theta = np.linspace(0, 2*np.pi, n_s_sqrt + 1)[:-1]
    phi = np.linspace(0, 2*np.pi, n_s_sqrt + 1)[0:-1]
    n_s = n_s_sqrt**2

    X,Y = np.meshgrid(theta, phi, indexing='ij')
    points = np.stack((X,Y)).reshape((2,-1)).T

    r = 1.0
    R = 1.5
    x = (R + r*np.cos(points[:,0]))*np.cos(points[:,1])
    y = (R + r*np.cos(points[:,0]))*np.sin(points[:,1])
    z = r*np.sin(points[:,0])

    source = np.stack((x,y,z), axis=1).astype(np.float32)

    a, b = np.ones((n_s,)) / n_s, np.ones((n_t,)) / n_t
    
    M = ot.dist(source, target)
    M /= M.max()  # Normalize the cost matrix
    M += 1e-16  # Avoid division by zero
    reg=0.01
    gamma = ot.sinkhorn(a, b, M, reg, numItermax=100000) #reg=0.1 sparsity: 0.001001306051371354    4867 724.3637338879053 1    
    #gamma = ot.emd(a, b, M, numItermax=1000000)  #numItermax = 1000000, sparse: 4MB
    #gamma = compute_sinkhorn_ot(source, target)
 
    '''
    coupling = np.zeros((n_s,3))
    for j in range(n_s):
        gamma[j,:] /= np.sum(gamma[j,:])
        coupling[j,:] = np.dot(target.T, gamma[j,:])
    '''
#reg=0.001 sparsity:0.34625268025951084    7327 18907.22374934703 1
    for l in range(n_t):  #Total time: 3494.14 seconds. sparsity: 0.010062358276643991  3160 5.882869478315115 611
        # Calculate number of elements to retain (1%)
        sparse = 0.001
        num_top = int(np.ceil(sparse * n_s))
        top_indices = np.argsort(gamma[:, l])[-num_top:]
        
        # Zero out all but the top 1%
        new_col = np.zeros(n_s)
        new_col[top_indices] = gamma[top_indices, l]
        
        # Normalize the new column to make the sum equal to 1
        if new_col.sum() > 0:  # Only normalize if the sum is not zero to avoid division by zero
            new_col /= new_col.sum()
        
        # Update the matrix
        gamma[:, l] = new_col


    transport = np.zeros((n_s,3))
    for j in range(n_s):
        ind = gamma[j,:].argmax()
        transport[j,...] = target[ind,...]
    
    sparsity = np.count_nonzero(gamma) / gamma.size
    print("sparsity:",sparsity)
    
    unique = np.unique(transport, axis=0).shape[0]

    transport = transport.T.reshape((3,n_s_sqrt,n_s_sqrt))

    all_points.append(torch.from_numpy(target.astype(np.float32())))
    all_couplings.append(torch.from_numpy(gamma.astype(np.float32())).to_sparse())
    all_presures.append(torch.from_numpy(pressure.astype(np.float32())))
    all_transports.append(torch.from_numpy(transport.astype(np.float32())))
    
    print(unique, default_timer()-t1, k+1)
    k+=1
    index+=1
    '''
    # Calculate memory usage
    nnz = torch.from_numpy(gamma.astype(np.float32())).to_sparse()._nnz()
    index_size = 4  # int32 for each index
    num_dims = 2  # since it's a 2D matrix
    value_size = 4  # float32 for each value

    memory_usage = (nnz * index_size * num_dims) + (nnz * value_size)
    print(f"Estimated Memory Usage: {memory_usage / 1024 ** 2:.2f} MB")
    '''
save_path = '/central/groups/tensorlab/xinyi/OT/torus_drivaer_sinkorn_reg' + str(reg) + '_sparse' + str(sparse) + '_1.pt'
torch.save({'points': all_points, 'transports': all_transports, 'pressures': all_presures, 'couplings': all_couplings}, save_path)

#sparsity: 0.00109375  reg=0.01  unique:6007 time:1575.0389659777284 1
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