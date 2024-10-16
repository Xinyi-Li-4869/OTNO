import numpy as np
import matplotlib.pyplot as plt
import ot
import matplotlib.animation as animation

def update_plot(i, data, scat):
    scat.set_offsets(data[i,...])
    return scat,

n = 32**2

t = np.linspace(-1,1,32)

target = np.stack(np.meshgrid(t, t, indexing='ij'))
target = target.reshape(2, 32**2).T

t = np.linspace(0,2*np.pi,32**2 + 1)[:-1]

source = np.concatenate((np.cos(t).reshape(-1,1), np.sin(t).reshape(-1,1)), axis=1)

a, b = np.ones((n,)) / n, np.ones((n,)) / n

M = ot.dist(source, target)
gamma = ot.emd(a, b, M)

coupling = np.zeros((n,2))

for j in range(n):
    coupling[j,...] = target[gamma[j,:].argmax(),...] 


T = 60
movement = np.zeros((T,n,2))

for j in range(n):
    tx = np.linspace(source[j,0], coupling[j,0], T).reshape((T,1))
    ty = np.linspace(source[j,1], coupling[j,1], T).reshape((T,1))
    movement[:,j,:] = np.concatenate((tx,ty), axis=1)


fig = plt.figure()
scat = plt.scatter(source[:,0], source[:,1])
ani = animation.FuncAnimation(fig, update_plot, frames=range(1, T),
                                  fargs=(movement, scat))

ani.save('circle_to_uniform.gif', fps=20)