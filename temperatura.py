from math_utils import *
from energy import calculate_total_kinetic_energy


@njit
def calculate_temperature(momentums, mass, k):
    N, _ = np.shape(momentums)
    kinetic_energy = calculate_total_kinetic_energy(momentums, mass)
    temperature = kinetic_energy * (2/(3*N*k))

    return temperature


@njit
def calculate_avg_temperature(S0, Sd, temperatures):
    sum_of_temperatures = 0
    for s in range(S0, S0+Sd, 1):
        sum_of_temperatures += temperatures[s]

    avg_temperature = sum_of_temperatures/Sd
    return avg_temperature
  
