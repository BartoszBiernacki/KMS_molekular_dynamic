from math_utils import *
from potential import calculate_total_potential_energy


@njit
def calculate_total_kinetic_energy(momentums, mass):
    momentums_norms_squared = calculate_norms(momentums) ** 2
    return np.sum(momentums_norms_squared) / (2 * mass)


@njit
def calculate_hamiltonian(momentums, mass, positions, epsilon, R, f, L):
    return calculate_total_kinetic_energy(momentums, mass) + calculate_total_potential_energy(
        positions, epsilon, R, f, L)


@njit
def calculate_all_kinetic_energies(momentums, mass):
    momentums_norms_squared = calculate_norms(momentums) ** 2
    return momentums_norms_squared / (2 * mass)


