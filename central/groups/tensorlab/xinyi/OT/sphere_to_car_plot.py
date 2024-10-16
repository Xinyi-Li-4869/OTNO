import ot
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

path = '/home/nikola/HDD/car-cfd/data/'

mesh = pv.read(path + 'mesh_001.vtk').extract_surface()
target = np.asarray(mesh.points)
pressure = np.asarray(mesh['point_scalars'])

    
n_t = target.shape[0]


expand_factor = np.sqrt(2)
n_s_sqrt = int(expand_factor*np.ceil(np.sqrt(n_t)))
theta = np.linspace(0, 2*np.pi, n_s_sqrt + 1)[:-1]
phi = np.linspace(0, np.pi, n_s_sqrt + 1)[0:-1]
n_s = n_s_sqrt**2

X,Y = np.meshgrid(theta, phi, indexing='ij')
points = np.stack((X,Y)).reshape((2,-1)).T

r = 1.0
x = r*np.cos(points[:,0])*np.sin(points[:,1])
y = r*np.sin(points[:,0])*np.sin(points[:,1])
z = r*np.cos(points[:,1])

source = np.stack((x,y,z), axis=1)

a, b = np.ones((n_s,)) / n_s, np.ones((n_t,)) / n_t

M = ot.dist(source, target)
gamma = ot.emd(a, b, M, numItermax=1000000)

'''
coupling = np.zeros((n_s,3))
for j in range(n_s):
    gamma[j,:] /= np.sum(gamma[j,:])
    coupling[j,:] = np.dot(target.T, gamma[j,:])

'''

coupling = np.zeros((n_s,3))
pressure_pullback = np.zeros((n_s,))
print(n_s, n_t)
for j in range(n_s):
    ind = gamma[j,:].argmax()
    coupling[j,...] = target[ind,...]
    pressure_pullback[j] = pressure[ind] 

print(np.unique(coupling, axis=0).shape[0])

#pl = pv.Plotter()
#pl.add_points(source, scalars=pressure_pullback)
#pl.show()

cx, cy, cz = np.moveaxis(coupling.T.reshape((3,n_s_sqrt,n_s_sqrt)), 0, 0)

plt.subplot(1,3,1)
plt.imshow(cx)
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(cy)
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(cz)
plt.colorbar()
plt.show()

exit()

print(X.shape, Y.shape)
print(cx.shape, cy.shape, cz.shape)

fig = plt.figure()

ax = fig.add_subplot(projection='mollweide')
im = ax.pcolormesh(cx)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.show()
'''
'''
print(np.unique(coupling, axis=0).shape[0])

T = 60
movement = np.zeros((T,n_s,3))

for j in range(n_s):
    tx = np.linspace(source[j,0], coupling[j,0], T).reshape((T,1))
    ty = np.linspace(source[j,1], coupling[j,1], T).reshape((T,1))
    tz = np.linspace(source[j,2], coupling[j,2], T).reshape((T,1))
    movement[:,j,:] = np.concatenate((tx,ty,tz), axis=1)


plotter = pv.Plotter()
plotter.open_movie('sphere_to_car_barrycenters.mp4', framerate=20)
plotter.add_points(source, render_points_as_spheres=True, point_size=5, name='points')
plotter.camera_position = 'xy'
plotter.camera.azimuth= 135
plotter.camera.elevation= 30
plotter.camera.zoom(0.8)

plotter.write_frame()
for j in range(1,T):
    plotter.add_points(movement[j,...], render_points_as_spheres=True, point_size=5, name='points')
    plotter.write_frame()

plotter.close()
