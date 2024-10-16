import torch
import numpy as np
import open3d as o3d
from neuralop.utils import UnitGaussianNormalizer
from tqdm import tqdm

def estimate_curvature(mesh):
    # Assuming vertex normals are already computed
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    curvatures = np.zeros(len(vertices))

    # Iterate over all triangles
    for triangle in np.asarray(mesh.triangles):
        # Get vertices of the triangle
        v1, v2, v3 = vertices[triangle]
        n1, n2, n3 = normals[triangle]

        # Calculate the angle between normals (simple curvature estimate)
        curvature12 = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
        curvature23 = np.dot(n2, n3) / (np.linalg.norm(n2) * np.linalg.norm(n3))
        curvature31 = np.dot(n3, n1) / (np.linalg.norm(n3) * np.linalg.norm(n1))

        # Aggregate the curvature for each vertex
        curvatures[triangle[0]] += curvature12 + curvature31
        curvatures[triangle[1]] += curvature12 + curvature23
        curvatures[triangle[2]] += curvature23 + curvature31

    # Normalize the curvature values
    curvatures /= curvatures.max()

    return curvatures

def compute_gaussian_curvature(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    n_vertices = len(vertices)
    gaussian_curvature = np.zeros(n_vertices)

    # Angle deficit
    for triangle in triangles:
        for i in range(3):
            v_i = vertices[triangle[i]]
            v_j = vertices[triangle[(i + 1) % 3]]
            v_k = vertices[triangle[(i + 2) % 3]]
            # Edge vectors
            e_ij = v_j - v_i
            e_ik = v_k - v_i
            e_jk = v_k - v_j
            # Angles
            cosine_value = np.clip(np.dot(e_ij, e_ik) / (np.linalg.norm(e_ij) * np.linalg.norm(e_ik)), -1.0, 1.0)
            angle_i = np.arccos(cosine_value)
            gaussian_curvature[triangle[i]] += angle_i

    # 2π - sum of angles at vertex
    gaussian_curvature = 2 * np.pi - gaussian_curvature
    return gaussian_curvature / np.max(np.abs(gaussian_curvature))  # Normalize for visualization

N = 611
path = '/central/groups/tensorlab/xinyi/GINO-main/datasets/car-pressure-data'
data = torch.load('/central/groups/tensorlab/xinyi/OT/torus_data_geomloss0.0001.pt')
all_normals = []
all_curvatures = []
all_gaussian_curvatures = []
filename="/central/groups/tensorlab/xinyi/GINO-main/datasets/car-pressure-data/watertight_meshes.txt"
with open(filename, "r") as fp:
    mesh_ind = fp.read().split("\n")
    mesh_ind = [int(a) for a in mesh_ind]

for index in tqdm(mesh_ind):
    mesh_path = path + '/data/mesh_' + str(index).zfill(3) + '.ply'

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    #print(normals.shape)
    all_normals.append(normals)
    curvatures = estimate_curvature(mesh)
    #print(curvatures.shape)
    all_curvatures.append(curvatures)
    gaussian_curvatures = compute_gaussian_curvature(mesh)
    #print(gaussian_curvatures.shape)
    all_gaussian_curvatures.append(gaussian_curvatures)
    #print(index)
data['normals'] = all_normals
data['curvatures'] = all_curvatures
data['gaussian_curvature'] = all_gaussian_curvatures
save_path = '/central/groups/tensorlab/xinyi/OT/torus_data_geomloss0.0001_features.pt'
torch.save(data ,save_path)