from math_utils import *
from energy import calculate_total_kinetic_energy


@njit(fastmath=True)
def calculate_temperature(momentums, mass, k):
    N, _ = np.shape(momentums)
    kinetic_energy = calculate_total_kinetic_energy(momentums, mass)
    temperature = kinetic_energy * (2/(3*N*k))

    return temperature
  
