The base src code is in classes.py; all the other files were produced to either output results (to a file if many or to the command line if not), and to plot figures.

Often they contain lots of leftover code from modification from a different file/plot and from testing. The basis of each is essentially a loop such as this: 


import classes
import numpy as np
from datetime import datetime

filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S')


finish_time = 600

num_asteroids = 2000

asteroids_per_solve = 8


mass=0.7

saturn_initial_conditions = [60,20,-0.25,0]


output=classes.gal_solver(finish_time, saturn_initial_conditions,[], saturn_mass=mass).y


for n in np.linspace(1,6,num=int(num_asteroids/asteroids_per_solve)):
    out = classes.gal_solver(finish_time, saturn_initial_conditions,classes.asteroid_generator([n],n=asteroids_per_solve, CW=False), saturn_mass=mass)
    
    intermediate= np.full((out.y.shape[0]-8, 1000),np.nan)
    intermediate[:,:out.y.shape[1]] = out.y[8:,:]

    output = np.concatenate((output,intermediate), axis=0)
    print('Simulated {:.0f} of {:.0f} objects'.format(output.shape[0]/4, num_asteroids +2))
        
output = output.reshape((int(output.shape[0]/4),4,-1))

np.savetxt(filename + 'data' + '.csv', output[0,:,:], delimiter=',')