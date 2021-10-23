from math_utils import *
from forces import calculate_all_forces_acting_on_all_particles


@njit(fastmath=True)
def calculate_new_positions_and_momentums(tau, positions, momentums, epsilon, R, f, L, mass, last_known_forces, num_of_step):
    # decide if forces calculated in previous step can be used
    if num_of_step != 0:
        forces = last_known_forces
    else:
        forces = calculate_all_forces_acting_on_all_particles(positions=positions, epsilon=epsilon, R=R, f=f, L=L)

    # half momentums
    half_momentums = momentums + 0.5*forces*tau

    # new positions
    new_positions = positions + (1/mass)*half_momentums*tau

    # new momentums
    new_forces = calculate_all_forces_acting_on_all_particles(positions=new_positions, epsilon=epsilon, R=R, f=f, L=L)
    new_momentums = half_momentums + 0.5*new_forces*tau

    return new_positions, new_momentums, new_forces


