import classes
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import rcParams, cycler

filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S')


finish_time = 800

num_asteroids = 200

asteroids_per_solve = 8

fig, ((ax1, ax2,), (ax3, ax4)) = plt.subplots(2,2)

axes = (ax1, ax2, ax3, ax4)

ax_num=-1

for y in np.linspace(10,30,num=4):

    ax_num += 1

    axis = axes[ax_num]

    saturn_initial_conditions = [60,y,-0.25,0]
    output=classes.gal_solver(finish_time, saturn_initial_conditions,[]).y




    for CW in (True, False):
        for n in np.linspace(1,6,num=int(num_asteroids/asteroids_per_solve/2)):
            out = classes.gal_solver(finish_time, saturn_initial_conditions,classes.asteroid_generator([n],n=asteroids_per_solve, CW=CW))
            
            intermediate= np.full((out.y.shape[0]-8, 1000),np.nan)
            intermediate[:,:out.y.shape[1]] = out.y[8:,:]

            output = np.concatenate((output,intermediate), axis=0)
            
            print("Simulated {ast_per_cycle:.0f} asteroids at radius {asteroid_radius:.2f} going {direct} with y0 = {y0:.2f}. {percent_complete:.1f}% complete".format(
                ast_per_cycle = asteroids_per_solve,
                asteroid_radius = n,
                direct= 'CW' if CW else 'ACW',
                y0 = y,
                percent_complete = 100* ((output.shape[0]-8)/4)/num_asteroids)
            )


    output = output.reshape((int(output.shape[0]/4),4,-1))


    sun_displacement = np.linalg.norm(output - output[1,0:2,:], axis = 1)
    saturn_displacement = np.linalg.norm(output - output[0,0:2,:], axis = 1)

    min_displacement = saturn_displacement[1,:].min()

    print('Closest approach at ' + '{}'.format(min_displacement))


    axis.plot(out.t, sun_displacement[2:int((sun_displacement.shape[0]-2)/2),:].T, linewidth=0.5, color='coral')
    axis.plot(out.t, sun_displacement[int((sun_displacement.shape[0]-2)/2):,:].T, linewidth=0.5, color='slateblue')
    axis.set_title("Closest approach = {min:.2f}".format(min=min_displacement))
    axis.set_xlabel("Time")
    axis.set_ylabel("Object Displacement")
    axis.set_ylim(top=160, bottom=0)
    axis.set_xlim(left=0,right=out.t.max())
    axis.plot(out.t,sun_displacement[0,:].T,'k--', label="Galaxy 1", color="black", linewidth="1.5" )
    axis.legend()

    

    print('Plotting.....')

plt.tight_layout
plt.show()
