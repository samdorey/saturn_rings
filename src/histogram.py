from typing import no_type_check


import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('2021-04-26 15-44-25sundist.csv', delimiter=',')

max_dist=300


plt.scatter()
plt.xlabel('Distance from galaxy 1')
plt.ylabel('Frequency')
plt.xlim(0,max_dist)
plt.show()