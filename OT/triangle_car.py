import ot
import numpy as np
import pyvista as pv


def triangle_weights(points, triangles):
    A, B, C = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]

    centroids = (A + B + C) / 3  
    areas = np.sqrt(np.sum(np.cross(B - A, C - A) ** 2, 1)) / 2 

    return centroids, areas/np.sum(areas)

path = '/home/nikola/HDD/car-cfd/data/'

mesh_number = 1
mesh = pv.read(path + 'mesh_' + str(mesh_number).zfill(3) +'.vtk').extract_surface().triangulate()

del_ind = np.array([4*j  for j in range(int((len(mesh.faces) - 4)//4) + 1)])
triangles = np.asarray(mesh.faces)
triangles = np.delete(triangles, del_ind).reshape((-1,3))
points = np.asarray(mesh.points)

centers, weights = triangle_weights(points, triangles)

target = centers
n_t = target.shape[0]

n_s = int(np.ceil(np.cbrt(n_t)))
#n_s = 24
tx = np.linspace(target[:,0].min(), target[:,0].max(), n_s)
ty = np.linspace(target[:,1].min(), target[:,1].max(), n_s)
tz = np.linspace(target[:,2].min(), target[:,2].max(), n_s)

n_s = n_s**3



source = np.stack(np.meshgrid(tx, ty, tz, indexing='ij'))
source = source.reshape(3, n_s).T

sphere = pv.Sphere().triangulate()
source = np.asarray(sphere.points)
n_s = source.shape[0]

a, b = np.ones((n_s,)) / n_s, weights

M = ot.dist(source, target)
gamma = ot.emd(a, b, M, numItermax=1000000)
coupling = np.zeros((n_s,3))

for j in range(n_s):
    coupling[j,...] = target[gamma[j,:].argmax(),...] 

T = 60
movement = np.zeros((T,n_s,3))

for j in range(n_s):
    tx = np.linspace(source[j,0], coupling[j,0], T).reshape((T,1))
    ty = np.linspace(source[j,1], coupling[j,1], T).reshape((T,1))
    tz = np.linspace(source[j,2], coupling[j,2], T).reshape((T,1))
    movement[:,j,:] = np.concatenate((tx,ty,tz), axis=1)

print(np.unique(movement[-1,...], axis=0).shape)


plotter = pv.Plotter()
plotter.open_movie('car4.mp4', framerate=10)
plotter.add_points(movement[0,...], scalars=source, show_scalar_bar=False, render_points_as_spheres=True, point_size=5, name='points')
plotter.camera_position = 'zy'
plotter.camera.azimuth = 235
plotter.camera.elevation = 20
plotter.camera.zoom(0.3)
plotter.write_frame()

for j in range(1,T):
    plotter.add_points(movement[j,...], scalars=source, show_scalar_bar=False, render_points_as_spheres=True, point_size=5, name='points')
    plotter.write_frame()

plotter.close()
