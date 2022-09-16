import classes
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S')


finish_time = 800

num_asteroids = 200

asteroids_per_solve = 8

fig, (ax1,ax2) = plt.subplots(1,2)

axes = (ax1, ax2)

ax_num=-1

for mass in [0.1,0.2]:

    ax_num += 1

    axis = axes[ax_num]

    saturn_initial_conditions = [60,20,-0.25,0]
    output=classes.gal_solver(finish_time, saturn_initial_conditions,[], saturn_mass=mass).y




    for n in np.linspace(1,6,num=int(num_asteroids/asteroids_per_solve/2)):
        out = classes.gal_solver(finish_time, saturn_initial_conditions,classes.asteroid_generator([n],n=asteroids_per_solve),saturn_mass=mass)
        
        intermediate= np.full((out.y.shape[0]-8, 1000),np.nan)
        intermediate[:,:out.y.shape[1]] = out.y[8:,:]

        output = np.concatenate((output,intermediate), axis=0)
        
        print("Simulated {ast_per_cycle:.0f} asteroids at radius {asteroid_radius:.2f} with mass = {y0:.2f}. {percent_complete:.1f}% complete".format(
            ast_per_cycle = asteroids_per_solve,
            asteroid_radius = n,
            y0 = mass,
            percent_complete = 100* ((output.shape[0]-8)/4)/num_asteroids)
        )
    output = output.reshape((int(output.shape[0]/4),4,-1))




    #plot asteroid paths
    axis.plot(output[2:,0,:].T, output[2:,1,:].T, linewidth=0.5, color='slateblue')

    #plot galaxy paths
    axis.plot(output[0,0,:],output[0,1,:],'k', label="Galaxy 1", color="black", linewidth="1.5")
    axis.plot(output[1,0,:],output[1,1,:],'r', label="Galaxy 2", linewidth="1.5")


    sun_displacement = np.linalg.norm(output[:,0:2,:] - output[1,0:2,:], axis = 1)
    saturn_displacement = np.linalg.norm(output[:,0:2,:] - output[0,0:2,:], axis = 1)

    min_displacement = saturn_displacement[1,:].min()

    print('Closest approach at ' + '{}'.format(min_displacement))

    axis.set_aspect(1)
    axis.set_ylabel("Y")
    axis.set_xlabel("X")
    axis.set_title("Mass = {:.2f}".format(mass))
    axis.legend()
    axis.set_xlim(-50,50)
    axis.set_ylim(-70,30)

    print('Plotting.....')

plt.tight_layout()
plt.show()
