import classes
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

for mass in np.linspace(0.81,1,num=20):

    filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S')


    finish_time = 600

    num_asteroids = 400

    asteroids_per_solve = 8


    saturn_initial_conditions = [60,20,-0.25,0]
    output=classes.gal_solver(finish_time, saturn_initial_conditions,[], saturn_mass=mass).y


    for n in np.linspace(1,6,num=int(num_asteroids/asteroids_per_solve)):
        out = classes.gal_solver(finish_time, saturn_initial_conditions,classes.asteroid_generator([n],n=asteroids_per_solve, CW=False),saturn_mass=mass)
        
        intermediate= np.full((out.y.shape[0]-8, 1000),np.nan)
        intermediate[:,:out.y.shape[1]] = out.y[8:,:]

        output = np.concatenate((output,intermediate), axis=0)
            
    output = output.reshape((int(output.shape[0]/4),4,-1))


    sun_dist = np.linalg.norm(output[:,0:2,output.shape[2]-1] - output[1,0:2,output.shape[2]-1], axis=1)
    sat_dist = np.linalg.norm(output[:,0:2,output.shape[2]-1] - output[0,0:2,output.shape[2]-1], axis=1)

    sat_sun = np.linalg.norm(output[0,0:2,:] - output[1,0:2,:], axis=0)


    plt.scatter(output[2:,0,output.shape[2]-1], output[2:,1,output.shape[2]-1], linewidth=0.5, s=5)
    plt.plot(output[0,0,:],output[0,1,:],'k', label="Galaxy 1", color="black", linewidth="1.5")
    plt.plot(output[1,0,:],output[1,1,:],'k:', label="Galaxy 2", linewidth="1.5")

    plt.show()
    sun_bool_array = sun_dist < 70
    sat_bool_array = sat_dist < 30
    sun_count = np.count_nonzero(sun_bool_array)
    sat_count = np.count_nonzero(sat_bool_array)


    print('{}, '.format(mass) + '{}, '.format(sun_count) + '{}, '.format(sat_count) + '{}'.format(sat_sun.min()))

plt.show()