from math_utils import *


@njit(fastmath=True, parallel=True)
def calculate_sphere_forces(positions, f, L):
	# gets one 2D numpy array of vectors (vector=position) (vectors in rows) as input, returns 2D array in which i-th row is a sphere interaction force acting on i-th particle
	positions_norms = calculate_norms(positions)   # list of norms of position vectors
	num_of_positions = get_number_of_vectors_from_2D_nparray(positions)
	sphere_forces = np.zeros_like(positions)
	for i in prange(num_of_positions):
		ri = positions[i]
		ri_norm = positions_norms[i]

		if positions_norms[i] < L:
			sphere_forces[i] = np.array([0., 0., 0.])
		else:
			sphere_forces[i] = f * (L - ri_norm) * (ri / ri_norm)

	return sphere_forces


@njit(fastmath=True, parallel=True)
def calculate_interaction_net_forces_acting_on_i_th_particle(i, positions, epsilon, R):
	num_of_positions = get_number_of_vectors_from_2D_nparray(positions)
	net_force = np.zeros(3)

	for j in prange(num_of_positions):
		if i != j:
			rij = positions[i] - positions[j]
			rij_norm = calculate_norm(rij)

			net_force += 12*epsilon*((R/rij_norm)**12 - (R/rij_norm)**6) * (rij/(rij_norm**2))
	return net_force


@njit(fastmath=True)
def calculate_interaction_net_forces_acting_on_all_particles(positions, epsilon, R):
	num_of_positions = get_number_of_vectors_from_2D_nparray(positions)
	net_forces = np.zeros_like(positions)

	for i in prange(num_of_positions):
		net_forces[i] = calculate_interaction_net_forces_acting_on_i_th_particle(i=i, positions=positions, epsilon=epsilon, R=R)

	return net_forces


@njit(fastmath=True)
def calculate_all_forces_acting_on_all_particles(positions, epsilon, R, f, L):
	return calculate_sphere_forces(positions, f, L) + calculate_interaction_net_forces_acting_on_all_particles(positions, epsilon, R)


@njit(fastmath=True)
def calculate_pressure(positions, f, L):
	# gets one 2D numpy array of vectors (vectors in  rows) as input
	sphere_forces = calculate_sphere_forces(positions, f, L)
	sphere_forces_norms = calculate_norms(sphere_forces)
	pressure = np.sum(sphere_forces_norms) / (4 * np.pi * L ** 2)
	return pressure


@njit(fastmath=True)
def calculate_avg_pressure(S0, Sd, pressures):
	sum_of_pressures = 0
	for s in prange(S0, S0 + Sd, 1):
		sum_of_pressures += pressures[s]
	avg_pressure = sum_of_pressures / Sd
	return avg_pressure
