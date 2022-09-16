import classes
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S')


finish_time = 600

num_asteroids = 4000

asteroids_per_solve = 8

fig, ax1 = plt.subplots()


saturn_initial_conditions = [60,3,-0.25,0]
output=classes.gal_solver(finish_time, saturn_initial_conditions,[]).y


for n in np.linspace(1,6,num=int(num_asteroids/asteroids_per_solve)):
    out = classes.gal_solver(finish_time, saturn_initial_conditions,classes.asteroid_generator([n],n=asteroids_per_solve, CW=False))
    
    intermediate= np.full((out.y.shape[0]-8, 1000),np.nan)
    intermediate[:,:out.y.shape[1]] = out.y[8:,:]

    output = np.concatenate((output,intermediate), axis=0)
    print(output.shape[0])
        
np.savetxt(filename + 'full data' + '.csv', output, delimiter=',')

output = output.reshape((int(output.shape[0]/4),4,-1))


zero_displacement = np.linalg.norm(output[:,0:2,output.shape[2]-1] - output[1,0:2,output.shape[2]-1], axis = 1)


#plot asteroid final displacements

ax1.hist(zero_displacement,bins=np.linspace(0,300,num=400))
ax1.set_xlabel('Distance from galaxy 1')
ax1.set_ylabel('')

np.savetxt(filename + 'final distances' + '.csv', zero_displacement, delimiter=",")



print('Closest approach at ' + 'none')


print('Plotting.....')

plt.tight_layout()
plt.show()
