from math_utils import *
from equations_of_motion import calculate_new_positions_and_momentums
from energy import calculate_all_kinetic_energies
from energy import calculate_total_kinetic_energy
from potential import calculate_total_potential_energy
from temperatura import calculate_temperature
from forces import calculate_pressure
import os
from tqdm import trange


def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def prepare_state_parameters_file(filename):
    with open(filename, "a") as myfile:
        myfile.write(f"time,hamiltonian,potential_energy,temperature,pressure\n")


def write_state_parameters_to_file(filename, t, H, V, T, P):
    with open(filename, "a") as myfile:
        myfile.write(f"{t:.3e}, {H:.3e}, {V:.3e}, {T:.3e}, {P:.3e}\n")


def write_positions_and_energys_to_file(filename, positions, kinetic_energies):
    with open(filename, "a") as myfile:
        N, _ = positions.shape
        myfile.write(str(N) + '\n' + '\n')
        for position in positions:
            myfile.write(f"Ar {position[0]:.3e} {position[1]:.3e} {position[2]:.3e}\n")


def run_simulation(num_of_steps, tau, positions, momentums, epsilon, R, f, L, mass, k, S_out, S_xyz):
    simulation_time = 0.
    filename_output_state_parameters = "result_state_parametres.txt"
    filename_output_positions = "result_positions.xyz"
    remove_file(filename_output_state_parameters)
    remove_file(filename_output_positions)
    prepare_state_parameters_file(filename=filename_output_state_parameters)

    V = calculate_total_potential_energy(positions, epsilon, R, f, L)
    H = calculate_total_kinetic_energy(momentums, mass) + V
    write_state_parameters_to_file(filename=filename_output_state_parameters, t=simulation_time,
                                   H=H,
                                   V=V,
                                   T=calculate_temperature(momentums, mass, k),
                                   P=calculate_pressure(positions, f, L))

    write_positions_and_energys_to_file(filename=filename_output_positions, positions=positions,
                                        kinetic_energies=calculate_all_kinetic_energies(momentums,
                                                                                        mass))

    for step in trange(num_of_steps):
        simulation_time += tau
        if step == 0:
            new_positions, new_momentums, forces = \
              calculate_new_positions_and_momentums(tau, positions, momentums, epsilon, R,  f, L, mass, np.zeros_like(positions), step)
        else:
            new_positions, new_momentums, forces =\
              calculate_new_positions_and_momentums(tau, positions, momentums, epsilon, R,  f, L, mass, forces, step)

        positions = new_positions
        momentums = new_momentums
        if step != 0:
            if step % S_out == 0 and step % S_xyz == 0:
                V = calculate_total_potential_energy(positions, epsilon, R, f, L)
                H = calculate_total_kinetic_energy(momentums, mass) + V
                write_state_parameters_to_file(filename=filename_output_state_parameters,
                                               t=simulation_time,
                                               H=H,
                                               V=V,
                                               T=calculate_temperature(momentums, mass, k),
                                               P=calculate_pressure(positions, f, L))

                write_positions_and_energys_to_file(filename=filename_output_positions,
                                                    positions=positions,
                                                    kinetic_energies=calculate_all_kinetic_energies(
                                                        momentums, mass))

            elif step % S_out == 0:
                V = calculate_total_potential_energy(positions, epsilon, R, f, L)
                H = calculate_total_kinetic_energy(momentums, mass) + V
                write_state_parameters_to_file(filename=filename_output_state_parameters,
                                               t=simulation_time,
                                               H=H,
                                               V=V,
                                               T=calculate_temperature(momentums, mass, k),
                                               P=calculate_pressure(positions, f, L))
            elif step % S_xyz == 0:
                write_positions_and_energys_to_file(filename=filename_output_positions,
                                                    positions=positions,
                                                    kinetic_energies=calculate_all_kinetic_energies(
                                                        momentums, mass))

