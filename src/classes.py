import numpy as np
import scipy.integrate as integrate
import numpy.random as rand
from numba import jit

G = 1

def asteroid_generator(radii,n=10, CW= False):
    """
    Creates initial positions and velocities of asteriods at radii given with 
    n being the number of asteroids at each given radius.
    returns tuple of vectors. CW is boolean that sets the direction of the asteroids
    """
    initial_vectors = []

    rand.seed(seed=1234) #Set seed for consistency

    CW_factor = 1 if CW else -1

    for r in radii:
        for theta in rand.rand(n)*2*np.pi:
            initial_vectors.extend([
                r*np.cos(theta),
                r*np.sin(theta),
                CW_factor*np.sin(theta)*np.sqrt(G/r),
                -CW_factor * np.cos(theta)*np.sqrt(G/r)])
    
    return np.array(initial_vectors)



def saturn_function(t,y, saturn_mass=0.1):
    """
    The function that is passed to the solver to calculate the accelereations of the galaxies and the asteroids.
    galaxies affect all other objects but asteroids are not concidered to affect the acceleration of others.
    y can be vectorised ie. the function can evaluate at multiple points in one function call.
    """

    saturn = y[None,0:4]
    sun = y[None,4:8]
    
    asteroids = y[8:len(y)].reshape((int((len(y)-8)/4), 4, -1)) if len(y) > 8  else None
    
    saturn_sun_displacement = saturn[:,0:2] - sun[:,0:2]
    
    if not(isinstance(asteroids, type(None))): #checks wether to evaluate any asteroids or not
        asteroid_sun_displacement = asteroids[:,0:2] - sun[:,0:2]
        asteroid_saturn_displacement = asteroids[:,0:2] - saturn[:,0:2]
        
        
        asteroids_derivative = np.concatenate((
            asteroids[:,2:4],
            -G * asteroid_sun_displacement / np.expand_dims(np.linalg.norm(asteroid_sun_displacement, axis=1 ), axis=1)**3 + -G * asteroid_saturn_displacement / np.expand_dims(np.linalg.norm(asteroid_saturn_displacement, axis=1 ), axis=1)**3
        ), axis=1)

        output = np.concatenate((
            np.concatenate((saturn[:,2:4],-G * saturn_sun_displacement / np.expand_dims(np.linalg.norm(saturn_sun_displacement, axis=1 ), axis=1)**3), axis=1), 

            np.concatenate((sun[:,2:4], -saturn_mass * -G * saturn_sun_displacement / np.expand_dims(np.linalg.norm(saturn_sun_displacement, axis=1 ),axis=1 )**3), axis=1), 

            asteroids_derivative
            ), axis=0).reshape(y.shape)
    else:
        output = np.concatenate((
            np.concatenate((saturn[:,2:4],-G * saturn_sun_displacement / np.expand_dims(np.linalg.norm(saturn_sun_displacement, axis=1 ), axis=1)**3), axis=1), 

            np.concatenate((sun[:,2:4], -saturn_mass * -G * saturn_sun_displacement / np.expand_dims(np.linalg.norm(saturn_sun_displacement, axis=1 ),axis=1 )**3), axis=1)
            ), axis=0).reshape(y.shape)
    
    return output

# the three following functions are defined to simplify the jacobian, and are compiled using Numba 
# using the @jit decorator to compile them as C code at runtime

@jit
def j1 (a,b,c):
    return G*(-c**2 * G * b + 2*(b - a)**2)/c**4

@jit
def j2 (a,b,c):
    return -2*G*(b[1]-a[1])*(b[0]-a[0])/c**4
@jit
def j3 (a,b,c):
    return -2*G*(b[1]-a[:,1])*(b[0]-a[:,0])/c**4

def jacobian(t,y, saturn_mass=0.1):
    """
    The Jacobian is a function passed to the solver and is used in 'Stiff' solvers 
    returns derivatives of f(y) with respect to each y. 
    """
    sat = y[0:4]
    sun = y[4:8]
    
    ast = y[8:].reshape((-1,4)) if len(y) > 8  else None

    sat_sun_dist = np.linalg.norm(sun[0:2] - sat[0:2])

    if not(isinstance(ast, type(None))):
        ast_sun_dist = np.linalg.norm(ast[:,0:2] - sun[0:2].T, axis = 1)
        ast_sat_dist = np.linalg.norm(ast[:,0:2] - sat[0:2].T, axis = 1)

    saturn_jac = np.zeros((4, y.size))
    
    saturn_jac[0,2] = 1
    saturn_jac[1,3] = 1
    saturn_jac[2,0] = j1(sat[0], sun[0], sat_sun_dist)
    saturn_jac[3,0] = j2(sat, sun, sat_sun_dist)
    saturn_jac[2,1] = j2(sat, sun, sat_sun_dist)
    saturn_jac[3,1] = j1(sat[1], sun[1], sat_sun_dist)

    saturn_jac[2:4,4:6] = -saturn_jac[2:4,0:2]

    sun_jac = np.zeros((4, y.size))
    
    sun_jac[0,2] = 1
    sun_jac[1,3] = 1


    sun_jac[2,0] = saturn_mass * j1(sun[0], sat[0], sat_sun_dist)
    sun_jac[3,0] = saturn_mass * j2(sun, sat, sat_sun_dist)
    sun_jac[2,1] = saturn_mass * j2(sun, sat, sat_sun_dist)
    sun_jac[3,1] = saturn_mass * j1(sun[1], sat[1], sat_sun_dist)

    sun_jac[2:4,4:6] = -sun_jac[2:4,0:2]


    if not(isinstance(ast, type(None))):
        ast_jac = np.zeros((y.size-8, y.size)).reshape((-1,4,y.size)) if not(isinstance(ast, type(None))) else None

        ast_jac[:,0,2] = 1
        ast_jac[:,1,3] = 1

        ast_jac[:,2,0] = j1(ast[:,0], sun[0], ast_sun_dist) + saturn_mass * j1(ast[:,0], sat[0], ast_sat_dist)
        ast_jac[:,3,0] = j3(ast, sun, ast_sun_dist) + saturn_mass * j3(ast, sat, ast_sat_dist)
        ast_jac[:,2,1] = j3(ast, sun, ast_sun_dist) + saturn_mass * j3(ast, sat, ast_sat_dist)
        ast_jac[:,3,1] = j1(ast[:,1], sun[1], ast_sun_dist) + saturn_mass * j1(ast[:,1], sat[1], ast_sat_dist)

        
        ast_jac[:,2:4,4:6] = -ast_jac[:,2:4,0:2]


        ast_jac = ast_jac.reshape(-1,y.size)

    output = np.concatenate((saturn_jac,sun_jac), axis=0) if isinstance(ast, type(None)) else np.concatenate((np.concatenate((saturn_jac,sun_jac), axis=0), ast_jac))

    return output

def gal_solver(t_max,saturn_initial_cond, asteroid_initial_cond= None, num_eval=1000,method='DOP853',saturn_mass=0.1):
    """
    Solves the path of the Sun, Saturn and asteroids. Essentially a wrapper for the np.sovlve_ivp() function that takes arguments that are useful for calculating in a loop via another script.
    Default method is DOP853
    """
    y0 = np.concatenate((saturn_initial_cond, [0,0,-saturn_mass * saturn_initial_cond[2],-saturn_mass * saturn_initial_cond[3]],asteroid_initial_cond)) if not(isinstance(asteroid_initial_cond, type(None))) else np.concatenate((saturn_initial_cond, [0,0,-saturn_mass * saturn_initial_cond[2],-saturn_mass * saturn_initial_cond[3]]))


    return integrate.solve_ivp(
                fun = saturn_function,
                t_span = [0,t_max],
                t_eval=np.linspace(0,t_max,num=num_eval),
                y0 = y0,
                method= method,
                vectorized=True,
                jac=jacobian,
                args=(saturn_mass,)
                )