import classes
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib as mat
import time

counter = time.perf_counter_ns()

filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S')


finish_time = 600

num_asteroids = 2000

asteroids_per_solve = 8

fig, ax1 = plt.subplots()

mass=0.7

saturn_initial_conditions = [60,20,-0.25,0]
output=classes.gal_solver(finish_time, saturn_initial_conditions,[], saturn_mass=mass).y


cmap = mat.cm.get_cmap('jet')


for n in np.linspace(1,6,num=int(num_asteroids/asteroids_per_solve)):
    out = classes.gal_solver(finish_time, saturn_initial_conditions,classes.asteroid_generator([n],n=asteroids_per_solve, CW=False), saturn_mass=mass)
    
    intermediate= np.full((out.y.shape[0]-8, 1000),np.nan)
    intermediate[:,:out.y.shape[1]] = out.y[8:,:]

    output = np.concatenate((output,intermediate), axis=0)
    print('Simulated {:.0f} of {:.0f} objects'.format(output.shape[0]/4, num_asteroids +2))
        
output = output.reshape((int(output.shape[0]/4),4,-1))


#plot asteroid final positions

colors = np.linalg.norm(output[2:,0:2,0], axis=1)

im = ax1.scatter(output[2:,0,output.shape[2]-1], output[2:,1,output.shape[2]-1], linewidth=0.5,c=colors, s=5, cmap=cmap)

#plot galaxy paths
ax1.plot(output[0,0,:],output[0,1,:],'k', label="Galaxy 1", color="black", linewidth="1.5")
ax1.plot(output[1,0,:],output[1,1,:],'k:', label="Galaxy 2", linewidth="1.5")

cbar = fig.colorbar(im, ax=ax1)
cbar.set_label('Initial Orbit Radius')


sun_displacement = np.linalg.norm(output[:,0:2,:] - output[1,0:2,:], axis = 1)
saturn_displacement = np.linalg.norm(output[:,0:2,:] - output[0,0:2,:], axis = 1)


min_displacement = saturn_displacement[1,:].min()

print('Closest approach at {}'.format(min_displacement))

ax1.set_aspect(1)
ax1.set_ylabel("Y")
ax1.set_xlabel("X")
ax1.legend(loc='upper left')


np.savetxt(filename + 'final_positions' + '.csv', output[:,:,output.shape[2]-1], delimiter=',')
np.savetxt(filename + 'initial_positions' + '.csv', output[:,:,0], delimiter=',')

print((time.perf_counter_ns() - counter) * 10e-9)

print('Plotting.....')

plt.tight_layout()
plt.show()
