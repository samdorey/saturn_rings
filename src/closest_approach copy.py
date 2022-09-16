import classes
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import rcParams

plt.rcParams.update({'font.size': 12})

filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

finish_time = 600

min_displacement = []

masses = [0.8]

for mass in masses:

    saturn_initial_conditions = [60,20,-0.25,0]
    output= classes.gal_solver(finish_time, saturn_initial_conditions,[], saturn_mass=mass).y.reshape((2,4,-1))

    displacement = np.linalg.norm(output - output[1,:,:], axis = 1)

    min_displacement.append(displacement[0,:].min())
    print(min_displacement)

out = np.stack((masses,min_displacement))

np.savetxt(filename + 'close_approach_data' + '.csv', out.T, delimiter=',')


plt.plot(np.linspace(5,50,num=len(masses)),min_displacement)
plt.xlabel('$y_0$/arbitrary units')
plt.ylabel('Distance of closest approach/arbitrary units')
plt.xlim(10,50)
plt.ylim(0,np.max(min_displacement))
print('Plotting.....')


plt.tight_layout
plt.show()
