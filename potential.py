from math_utils import *


@njit
def calculate_total_sphere_potential(positions, f, L):
    # gets one 2D numpy array of vectors (vector=position) (vectors in rows) as input, returns scalar equals to sum sphere potatnials for all particles
    positions_norms = calculate_norms(positions)
    num_of_positions = get_number_of_vectors_from_2D_nparray(positions)
    potentials = np.zeros(num_of_positions)
    for i in prange(num_of_positions):
        if positions_norms[i] < L:
            potentials[i] = 0
        else:
            potentials[i] = 0.5 * f * (L - positions_norms[i]) ** 2

    sphere_potential = np.sum(potentials)
    return sphere_potential


@njit(fastmath=True)
def calculate_interaction_potential(ri, rj, epsilon, R):
    rij_norm = calculate_norm(ri - rj)
    return epsilon * ((R/rij_norm)**12 - 2*(R/rij_norm)**6)


@njit(fastmath=True, parallel=True)
def calculate_total_particle_interaction_potential(positions, epsilon, R):
    # gets one 2D numpy array of vectors (vector=position) (vectors in rows) as input, returns scalar equals to the sum of particle interaction potentials for all particles
    total_particle_interaction_potential = 0
    num_of_positions = get_number_of_vectors_from_2D_nparray(positions)

    for i in prange(1, num_of_positions):
        for j in range(i):
            total_particle_interaction_potential +=\
                calculate_interaction_potential(ri=positions[i], rj=positions[j], epsilon=epsilon, R=R)

    return total_particle_interaction_potential


@njit
def calculate_total_potential_energy(positions, epsilon, R, f, L):
    # gets one 2D numpy array of vectors (vectors in  rows) as input
    return calculate_total_sphere_potential(positions, f, L) +\
           calculate_total_particle_interaction_potential(positions, epsilon, R)
