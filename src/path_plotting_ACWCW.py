import classes
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

plt.rcParams.update({'font.size': 12})

finish_time = 800

num_asteroids = 200

asteroids_per_solve = 8

fig, (ax1, ax2) = plt.subplots(1,2)

saturn_initial_conditions = [60,20,-0.25,0]

output=classes.gal_solver(finish_time, saturn_initial_conditions,[]).y

for CW in (True, False):
    for n in np.linspace(1,6,num=int(num_asteroids/asteroids_per_solve/2)):
        out = classes.gal_solver(finish_time, saturn_initial_conditions,classes.asteroid_generator([n],n=asteroids_per_solve, CW=CW))
        
        intermediate= np.full((out.y.shape[0]-8, 1000),np.nan)
        intermediate[:,:out.y.shape[1]] = out.y[8:,:]

        output = np.concatenate((output,intermediate), axis=0)
        

output = output.reshape((int(output.shape[0]/4),4,-1))


#plot asteroid paths
ax1.plot(output[2:int((output.shape[0]-2)/2),0,:].T, output[2:int((output.shape[0]-2)/2),1,:].T, linewidth=0.5, color='coral')

ax2.plot(output[int((output.shape[0]-2)/2):,0,:].T, output[int((output.shape[0]-2)/2):,1,:].T, linewidth=0.5,color='slateblue')

#plot galaxy paths
ax1.plot(output[0,0,:],output[0,1,:],'k', label="Galaxy 2", color="black", linewidth="1.5")
ax1.plot(output[1,0,:],output[1,1,:],'r', label="Galaxy 1", linewidth="1.5")
ax2.plot(output[0,0,:],output[0,1,:],'k', label="Galaxy 2", color="black", linewidth="1.5")
ax2.plot(output[1,0,:],output[1,1,:],'r', label="Galaxy 1", linewidth="1.5")

sun_displacement = np.linalg.norm(output - output[1,0:2,:], axis = 1)
saturn_displacement = np.linalg.norm(output - output[0,0:2,:], axis = 1)

min_displacement = saturn_displacement[1,:].min()

print('Closest approach at ' + '{}'.format(min_displacement))

ax1.set_aspect(1)
ax1.set_ylabel("Y/arbitraty units")
ax1.set_xlabel("X/arbitraty units")
ax1.set_title('Clockwise')
ax1.legend()
ax1.set_xlim(-50,50)
ax1.set_ylim(-70,30)

ax2.set_aspect(1)
ax2.set_ylabel("Y/arbitraty units")
ax2.set_xlabel("X/arbitraty units")
ax2.set_title('Anticlockwise')
ax2.legend()
ax2.set_xlim(-50,50)
ax2.set_ylim(-70,30)





print('Plotting.....')

plt.tight_layout()
plt.show()
