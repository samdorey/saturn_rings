import classes
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import rcParams

plt.rcParams.update({'font.size': 12})

filename = datetime.now().strftime('%Y-%m-%d %H-%M-%S')

finish_time = 500

min_displacement = []

y0s = [3]

for y0 in y0s:

    saturn_initial_conditions = [60,y0,-0.25,0]
    output= classes.gal_solver(finish_time, saturn_initial_conditions,[]).y.reshape((2,4,-1))

    displacement = np.linalg.norm(output - output[1,:,:], axis = 1)

    min_displacement.append(displacement[0,:].min())
    print(min_displacement)

out = np.stack((y0s,min_displacement))

np.savetxt(filename + 'close_approach_data' + '.csv', out.T, delimiter=',')


plt.plot(np.linspace(5,50,num=len(y0s)),min_displacement)
plt.xlabel('$y_0$/arbitrary units')
plt.ylabel('Distance of closest approach/arbitrary units')
plt.xlim(10,50)
plt.ylim(0,np.max(min_displacement))
print('Plotting.....')


plt.tight_layout
plt.show()
