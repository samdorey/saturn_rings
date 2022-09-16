import classes
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

for y0 in np.linspace(5,9.5,num=10):

    filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S')


    finish_time = 600

    num_asteroids = 4000

    asteroids_per_solve = 8


    saturn_initial_conditions = [60,y0,-0.25,0]
    output=classes.gal_solver(finish_time, saturn_initial_conditions,[]).y


    for n in np.linspace(1,6,num=int(num_asteroids/asteroids_per_solve)):
        out = classes.gal_solver(finish_time, saturn_initial_conditions,classes.asteroid_generator([n],n=asteroids_per_solve, CW=False))
        
        intermediate= np.full((out.y.shape[0]-8, 1000),np.nan)
        intermediate[:,:out.y.shape[1]] = out.y[8:,:]

        output = np.concatenate((output,intermediate), axis=0)
            
    output = output.reshape((int(output.shape[0]/4),4,-1))


    sun_dist = np.linalg.norm(output[:,0:2,output.shape[2]-1] - output[1,0:2,output.shape[2]-1], axis=1)

    bool_array = sun_dist < 70

    count = np.count_nonzero(bool_array)

    print('{}'.format(y0) + ' ,' + '{}'.format(count))

plt.show()