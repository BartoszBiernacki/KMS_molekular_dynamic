from math_utils import *
from equations_of_motion import calculate_new_positions_and_momentums
from energy import calculate_total_kinetic_energy
from potential import calculate_total_particle_interaction_potential
from potential import calculate_total_sphere_potential

from temperatura import calculate_temperature
from forces import calculate_pressure
import os
from tqdm import trange
import matplotlib.pyplot as plt
import pandas as pd


def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def prepare_state_parameters_file(out_dir, n, T0):
    filename = "RESULTS_PARAMETERS_n=" + str(n) + "_T0=" + str(T0) + ".csv"
    filepath = out_dir + filename
    remove_file(filepath)
    with open(filepath, "w") as myfile:
        myfile.write(f"time,hamiltonian,interaction_potential,potential_energy,temperature,pressure\n")
    return filepath


def prepare_positions_file(out_dir, n, T0):
    filename = "RESULTS_POSITIONS_n=" + str(n) + "_T0=" + str(T0) + ".xyz"
    filepath = out_dir + filename
    remove_file(filepath)
    return filepath


def write_state_parameters_to_file(filename, t, H, V_interaction, V, T, P):
    with open(filename, "a") as myfile:
        myfile.write(f"{t:.15f}, {H:.15f}, {V_interaction:.15f},{V:.15f}, {T:.15f}, {P:.15f}\n")


def write_positions_to_file(filename, positions):
    with open(filename, "a") as myfile:
        N, _ = positions.shape
        myfile.write(str(N) + '\n' + '\n')
        for position in positions:
            myfile.write(f"Ar {position[0]:.15f} {position[1]:.15f} {position[2]:.15f}\n")


def get_n_from_result_filename(csv_filepath):
    n_start = csv_filepath.find('_n=') + 3
    n_end = csv_filepath.find('_T0=')
    n = int(csv_filepath[n_start: n_end])

    return n


def save_plot_of_given_data(data_filename, out_dir, T0, variable_name, variable_symbol,
                            unit_symbol_in_Tex_style, unit_system_multiplier):
    # ####################     DATA TO PLOT #####################################
    df = pd.read_csv(data_filename)
    x = df["time"]
    y = df[variable_name] * unit_system_multiplier

    tail_starting_index = int(len(y) / 2)
    data_tail = np.array(y[tail_starting_index:])
    tail_mean = np.mean(data_tail)
    # ################# PLOTTING ##############################
    fig = plt.figure(figsize=(25, 10))
    ax1 = fig.add_subplot(111)

    x_true = x
    y_true = y
    y_avg_const = tail_mean * np.ones_like(y_true)

    ax1.plot(x_true, y_true, 'o', color='blue', markersize=10, label="Raw simulation data", zorder=0)
    y_axis_min, y_axis_max = ax1.get_ylim()
    x_axis_min, x_axis_max = ax1.get_xlim()
    ax1.plot(x_true, y_avg_const, '-', color='black', linewidth=4,
             label=r"${}_{{avg}}={:.2f}$ based on last 50% points".format(variable_symbol, tail_mean), zorder=1)

    ax1.set_xticks(np.arange(round(min(x)), round(max(x) + 1), 1.0))
    ax1.set_xlim([x_axis_min, x_axis_max])
    ax1.set_ylim([y_axis_min, y_axis_max])
    ax1.legend(loc="best", scatterpoints=30, prop={'size': 25})

    ############################################################################
    # ############## APPLY FANCY TITLES AND STYLE TO PLOT ####################
    n = get_n_from_result_filename(csv_filepath=data_filename)

    max_delta_value = abs(y_true.max() - y_true.min())
    main_title = str(
        r"${}\left(t\right), \  \max\left(\Delta {}\right) \approx {:.2f};\ \ \left(n={}, T_0\approx{}\right)$".format(
            variable_symbol, variable_symbol, max_delta_value, n, T0))
    y_label = str(r"$\mathrm{{{}\  {}\ }} \left({}\right)$".format(variable_name.capitalize(), variable_symbol,
                                                                   unit_symbol_in_Tex_style))
    x_label = str(r"$\mathrm{time \ t \ (ps)}$")

    ax1.set_title(main_title, fontsize=50)
    ax1.set_xlabel(x_label, fontsize=40)
    ax1.set_ylabel(y_label, fontsize=40)

    plt.rcParams["font.weight"] = "bold"
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(4)
    ax1.tick_params(axis="both", width=4, length=10)

    ax1.ticklabel_format(useOffset=False)  # without it sometimes y axis scale is in bad-looking scientific notation
    plt.tight_layout()
    filename = "{}(t)_n=".format(variable_symbol) + str(n) + "_T0=" + str(T0) + ".pdf"
    filepath = out_dir + filename
    plt.savefig(filepath)
    plt.close()


def run_simulation(num_of_steps, tau, positions, momentums, epsilon, R, f, L, mass, k, S_out, S_xyz, T0,
                   out_dynamic_dir, out_images_dir):
    simulation_time = 0.
    state_filename = prepare_state_parameters_file(out_dir=out_dynamic_dir,
                                                   n=round(get_number_of_vectors_from_2D_nparray(momentums) ** (1 / 3)),
                                                   T0=T0)
    positions_filename = prepare_positions_file(out_dir=out_dynamic_dir,
                                                n=round(get_number_of_vectors_from_2D_nparray(momentums) ** (1 / 3)),
                                                T0=T0)

    V_interaction = calculate_total_particle_interaction_potential(positions, epsilon, R)
    V = V_interaction + calculate_total_sphere_potential(positions, f, L)
    H = calculate_total_kinetic_energy(momentums, mass) + V
    write_state_parameters_to_file(filename=state_filename, t=simulation_time,
                                   H=H,
                                   V_interaction=V_interaction,
                                   V=V,
                                   T=calculate_temperature(momentums, mass, k),
                                   P=calculate_pressure(positions, f, L))

    write_positions_to_file(filename=positions_filename, positions=positions)

    forces = np.empty_like(positions)
    for step in trange(num_of_steps):
        simulation_time += tau
        positions, momentums, forces = calculate_new_positions_and_momentums(tau=tau, positions=positions,
                                                                             momentums=momentums, epsilon=epsilon, R=R,
                                                                             f=f, L=L, mass=mass,
                                                                             last_known_forces=forces, num_of_step=step)

        if step != 0:
            if step % S_out == 0 and step % S_xyz == 0:
                V_interaction = calculate_total_particle_interaction_potential(positions, epsilon, R)
                V = V_interaction + calculate_total_sphere_potential(positions, f, L)
                H = V + calculate_total_kinetic_energy(momentums, mass)
                write_state_parameters_to_file(filename=state_filename,
                                               t=simulation_time,
                                               H=H,
                                               V_interaction=V_interaction,
                                               V=V,
                                               T=calculate_temperature(momentums, mass, k),
                                               P=calculate_pressure(positions, f, L))

                write_positions_to_file(filename=positions_filename, positions=positions)

            elif step % S_out == 0:
                V_interaction = calculate_total_particle_interaction_potential(positions, epsilon, R)
                V = V_interaction + calculate_total_sphere_potential(positions, f, L)
                H = V + calculate_total_kinetic_energy(momentums, mass)
                write_state_parameters_to_file(filename=state_filename,
                                               t=simulation_time,
                                               H=H,
                                               V=V,
                                               V_interaction=V_interaction,
                                               T=calculate_temperature(momentums, mass, k),
                                               P=calculate_pressure(positions, f, L))
            elif step % S_xyz == 0:
                write_positions_to_file(filename=positions_filename, positions=positions)

    save_plot_of_given_data(data_filename=state_filename, out_dir=out_images_dir, T0=T0,
                            variable_name='temperature', variable_symbol='T', unit_symbol_in_Tex_style=r'\mathrm{K}',
                            unit_system_multiplier=1)

    save_plot_of_given_data(data_filename=state_filename, out_dir=out_images_dir, T0=T0,
                            variable_name='hamiltonian', variable_symbol='H',
                            unit_symbol_in_Tex_style=r'\frac{\mathrm{kJ}}{\mathrm{mol}}', unit_system_multiplier=1)

    save_plot_of_given_data(data_filename=state_filename, out_dir=out_images_dir, T0=T0,
                            variable_name='pressure', variable_symbol='P', unit_symbol_in_Tex_style=r'\mathrm{atm}',
                            unit_system_multiplier=16.6)

    return state_filename
