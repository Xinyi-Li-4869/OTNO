import numpy as np
import matplotlib.pyplot as plt
import ot
import matplotlib.animation as animation
import pyvista as pv
pv.start_xvfb()
import sys

np.set_printoptions(threshold=sys.maxsize)

path = '/central/groups/tensorlab/xinyi/GINO-main/datasets/drivaer-net-dataset/'
#path = '/home/nikola/HDD/ShapeNetCore.v2/02958343/1a1de15e572e039df085b75b20c2db33/models/'

mesh_number = 1
mesh = pv.read(path + 'DrivAer_F_D_WM_WW_' + str(mesh_number).zfill(4) +'.vtk').extract_surface() #.triangulate()
#mesh = pv.read(path + 'model_normalized.obj').extract_surface()
#mesh = mesh.connectivity(largest=True)

target = np.asarray(mesh.points)
#n_t = target.shape[0]
n_t = 40000
indices = np.random.choice(len(target), size=n_t, replace=False)
target = target[indices]

n_s = int(np.ceil(np.cbrt(n_t)))
#n_s = 24
tx = np.linspace(target[:,0].min(), target[:,0].max(), n_s)
ty = np.linspace(target[:,1].min(), target[:,1].max(), n_s)
tz = np.linspace(target[:,2].min(), target[:,2].max(), n_s)

n_s = n_s**3
source = np.stack(np.meshgrid(tx, ty, tz, indexing='ij'))
source = source.reshape(3, n_s).T

a, b = np.ones((n_s,)) / n_s, np.ones((n_t,)) / n_t

M = ot.dist(source, target)
gamma = ot.emd(a, b, M, numItermax=1000000)
#gamma = M
coupling = np.zeros((n_s,3))

for j in range(n_s):
    coupling[j,...] = target[gamma[j,:].argmax(),...] 
    #coupling[j,...] = target[gamma[j,:].argmin(),...] 

T = 60
movement = np.zeros((T,n_s,3))

for j in range(n_s):
    tx = np.linspace(source[j,0], coupling[j,0], T).reshape((T,1))
    ty = np.linspace(source[j,1], coupling[j,1], T).reshape((T,1))
    tz = np.linspace(source[j,2], coupling[j,2], T).reshape((T,1))
    movement[:,j,:] = np.concatenate((tx,ty,tz), axis=1)



plotter = pv.Plotter()
plotter.open_movie('car2.mp4', framerate=10)
plotter.add_points(movement[0,...], scalars=source, show_scalar_bar=False, render_points_as_spheres=True, point_size=5, name='points')
plotter.camera_position = 'zy'
plotter.camera.azimuth = 235
plotter.camera.elevation = 20
plotter.camera.zoom(0.7)
plotter.write_frame()

for j in range(1,T):
    plotter.add_points(movement[j,...], scalars=source, show_scalar_bar=False, render_points_as_spheres=True, point_size=5, name='points')
    plotter.write_frame()

plotter.close()