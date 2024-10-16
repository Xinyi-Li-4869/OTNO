'''import debugpy

# Listen for incoming connections from the debugger on a specified port
debugpy.listen(('0.0.0.0', 5678))  # 0.0.0.0 allows connections from any IP
print("Debugger is waiting for connection on port 5678...")
debugpy.wait_for_client()  # Blocks execution until the debugger is attached'''
import sys
import ot
from ot.bregman import empirical_sinkhorn2_geomloss
import numpy as np
#np.__config__.show()  # Show linked BLAS/LAPACK libraries
#import pyvista as pv
import open3d as o3d
import matplotlib.pyplot as plt
import torch
from timeit import default_timer
from pykeops.torch import LazyTensor

def topk_column_normalized(lazy_tensor, k):
    """
    Returns a tensor with the same shape as the input lazy_tensor where each column contains the
    k largest elements normalized by the sum of these elements, with other elements set to zero.
    
    Parameters:
    - lazy_tensor (LazyTensor): The LazyTensor object.
    - k (int): Number of top elements to keep in each column.

    Returns:
    - torch.Tensor: A PyTorch tensor with the same shape, where non-top-k elements in each column are zero.
    """
    num_rows = lazy_tensor.shape[0]
    num_cols = lazy_tensor.shape[1]
    full_matrix = torch.zeros((num_rows, num_cols), dtype=torch.double, device=device)  # Initialize a tensor with zeros of the same shape

    for j in range(num_cols):
        col_values = lazy_tensor[:, j].clone().detach()   # Evaluate the entire column j and detach from graph
        if col_values.ndim > 1:
            col_values = col_values.squeeze()  # Ensure it's a 1D tensor
        
        # Get the indices of the top k values in the column
        top_k_values, indices = torch.topk(col_values, k)
        
        # Place top k values in their respective positions in the full matrix
        full_matrix[indices, j] = top_k_values

    # Normalize the columns where the top k values are placed
    column_sums = full_matrix.sum(dim=0, keepdim=True)
    full_matrix = (full_matrix / column_sums).clone().detach()  # Normalize to make the sum of each column 1 and detach

    return full_matrix


def topk_row_normalized(lazy_tensor, k):
    """
    Returns a tensor with the same shape as the input lazy_tensor where each row contains the
    two largest elements normalized by the sum of these two elements, with other elements set to zero.
    
    Parameters:
    - lazy_tensor (LazyTensor): The LazyTensor object.

    Returns:
    - torch.Tensor: A PyTorch tensor with the same shape, where non-top-2 elements in each row are zero.
    """
    num_rows = lazy_tensor.shape[0]
    num_cols = lazy_tensor.shape[1]
    full_matrix = torch.zeros((num_rows, num_cols), dtype=torch.double, device=device)  # Initialize a tensor with zeros of the same shape

    for i in range(num_rows):
        row_values = lazy_tensor[i, :].clone().detach()  # Evaluate the entire row i and detach from graph
        if row_values.ndim > 1:
            row_values = row_values.squeeze()  # Ensure it's a 1D tensor
        
        # Get the indices of the top two values in the row
        top_k_values, indices = torch.topk(row_values, k)
        
        # Place top two values in their respective positions in the full matrix
        full_matrix[i, indices] = top_k_values

    # Normalize the rows where the top two values are placed
    row_sums = full_matrix.sum(dim=1, keepdim=True)
    full_matrix = (full_matrix / row_sums).clone().detach()  # Normalize to make the sum of each row 1 and detach

    return full_matrix

def estimate_sparsity_lazy_tensor(lazy_tensor, threshold=1e-15):
    shape = lazy_tensor.shape
    total = shape[0] * shape[1]
    non_zero = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if lazy_tensor[i,j]>threshold:
                non_zero+=1
    sparsity = non_zero/total
    
    return sparsity


def plot_large_matrix(matrix, file_name='geomloss_fullmatrix_expand3', cmap='viridis', title='Large Matrix Visualization'):
    plt.figure(figsize=(8, 16))  # Adjust size as needed
    plt.imshow(matrix, aspect='auto', cmap=cmap, interpolation='none')
    plt.colorbar()  # Add a color bar to interpret values
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.grid(False)  # Turn off the grid for clarity
    plt.savefig('/central/groups/tensorlab/xinyi/OT/'+file_name+'.png')
    plt.close()

def mean_pooling(matrix, block_size=(10, 10)):
    """
    Reduce matrix size by mean pooling.

    Parameters:
        matrix (np.ndarray): Input matrix to pool.
        block_size (tuple): The dimensions (height, width) of blocks over which to compute the mean.

    Returns:
        np.ndarray: Downsampled matrix.
    """
    # Get the shape of the new downsampled matrix
    n_rows, n_cols = matrix.shape
    pooled_rows = n_rows // block_size[0]
    pooled_cols = n_cols // block_size[1]

    # Initialize the downsampled matrix
    pooled_matrix = np.zeros((pooled_rows, pooled_cols))

    for i in range(pooled_rows):
        for j in range(pooled_cols):
            start_row = i * block_size[0]
            end_row = start_row + block_size[0]
            start_col = j * block_size[1]
            end_col = start_col + block_size[1]
            # Compute the mean of each block and assign it to the pooled matrix
            pooled_matrix[i, j] = np.mean(matrix[start_row:end_row, start_col:end_col])

    return pooled_matrix

tt1 = default_timer()
#path = '/home/nikola/HDD/car-cfd/data/'
path = '/central/groups/tensorlab/xinyi/GINO-main/datasets/car-pressure-data/data/'

def read_indices(file_path):
    with open(file_path, 'r') as file:
        indices = [line.strip() for line in file if line.strip().isdigit()]
    return indices

# Read indices from the mesh.txt file
mesh_indices = read_indices('/central/groups/tensorlab/xinyi/GINO-main/datasets/car-pressure-data/watertight_meshes.txt')
device = torch.device('cuda')
N = 611
expand_factor = 3.0
topk = 2
reg = 1e-04
n_t = 3586
n_s_sqrt = int(np.sqrt(expand_factor)*np.ceil(np.sqrt(n_t)))
all_points = np.zeros((N,3586,3), dtype=np.float32)
all_normals = np.zeros((N,3586,3), dtype=np.float32)
all_transports = np.zeros((N,3,n_s_sqrt,n_s_sqrt), dtype=np.float32)
all_presures = np.zeros((N,3586), dtype=np.float32)
all_couplings = []

for k,index in enumerate(mesh_indices):
    #print(k,index)
    t1 = default_timer()

    #mesh = pv.read(path + 'mesh_' + str(k+1).zfill(3) + '.vtk').extract_surface()
    #target = np.asarray(mesh.points)
    mesh = o3d.io.read_triangle_mesh(path + 'mesh_' + index.zfill(3) + '.ply')
    target = np.asarray(mesh.vertices).squeeze() #(3586,3)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    #pressure = np.asarray(mesh['point_scalars'])
    pressure = np.load(path + 'press_' + index.zfill(3) + '.npy') # (3682,)
    pressure = np.concatenate((pressure[0:16], pressure[112:]), axis=0) #(3586,)

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

    value, log = empirical_sinkhorn2_geomloss(torch.tensor(source).to(device), torch.tensor(target).to(device), reg, log=True) #reg=0.1, Total time: 2406.70 seconds.
    gamma = log['lazy_plan']
    #fullmatrix = gamma[:].cpu().detach().numpy()
    #pool_matrix = mean_pooling(fullmatrix, block_size=(100,100))
    #plot_large_matrix(pool_matrix, file_name='geomloss_fullmatrix_expand3_pooling')
    #sparsity = estimate_sparsity_lazy_tensor(gamma)
    #print(sparsity)
    gamma = topk_column_normalized(gamma, topk)
    #pool_topk = mean_pooling(gamma.to_dense().cpu().detach().numpy(), block_size=(100,100))
    #plot_large_matrix(pool_topk, file_name='geomloss_fullmatrix_top2_expand3_pooling')
    #sys.exit(1)
    non_zero_counts = torch.count_nonzero(gamma, dim=1)

    # Find the maximum count among the rows (this tells us the row with the maximum number of non-zero entries)
    max_count = non_zero_counts.max().item()
    print("Maximum non-zero count in any row:", max_count)

    # Calculate the unique counts of non-zero entries
    unique, counts = non_zero_counts.unique(return_counts=True)

    # Calculate how many rows have each possible count of non-zero entries
    print("Non-zero counts per column:")
    unique_counts = {}
    for count, num_rows in zip(unique, counts):
        unique_counts[count.item()] = num_rows.item()
        print(num_rows.item(), count.item())

    # Identify empty rows specifically (rows with zero non-zero entries)
    empty_rows = (non_zero_counts == 0).sum().item()
    print("Empty rows:", empty_rows)
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
        #gamma[j,...] /= np.sum(gamma[j,...])
    
    #sparsity = torch.count_nonzero(gamma) / gamma.size
    #print("sparsity:",sparsity)
    

    unique = np.unique(transport, axis=0).shape[0]

    transport = transport.T.reshape((3,n_s_sqrt,n_s_sqrt))

    all_points[k,...] = target.astype(np.float32())
    all_normals[k,...] = normals.astype(np.float32())
    all_couplings.append(torch.tensor(gamma, dtype=torch.float32).to_sparse())
    all_presures[k,...] = pressure.astype(np.float32())
    all_transports[k,...] = transport.astype(np.float32())

    print(unique, default_timer()-t1, k+1)

save_path = '/central/groups/tensorlab/xinyi/OT/feature_uniformsphere_geomloss' + '_reg' + str(reg) + '_columntopk' + str(topk) + '_expand' + str(expand_factor) + '.pt' 
torch.save({'points': torch.from_numpy(all_points),'normals':torch.from_numpy(all_normals), 'transports': torch.from_numpy(all_transports), 'pressures': torch.from_numpy(all_presures), 'couplings': all_couplings}, save_path)
tt2 = default_timer()
print(f"Total time: {tt2-tt1:.2f} seconds.") 
''' Maximum non-zero count in any row: 2
Non-zero counts per column:
3519 0
7008 1
82 2
Empty rows: 3519
3586 3.18506152741611 611
Total time: 1939.51 seconds.'''


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