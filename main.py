from momentums import generate_n3_random_3D_momentums
from simulation import run_simulation
from initial_position import create_initial_pos_file_in_xyz_format
from pathlib import Path


if __name__ == '__main__':
    out_initialization_dir = "results/initialization/"
    out_dynamic_dir = "results/dynamic/"
    out_images_dir = "results/images/"

    Path(out_initialization_dir).mkdir(parents=True, exist_ok=True)  # create directory if not exist
    Path(out_dynamic_dir).mkdir(parents=True, exist_ok=True)   # create directory if not exist
    Path(out_images_dir).mkdir(parents=True, exist_ok=True)   # create directory if not exist
    # DEFAULT PARAMETERS *********************************************************************************************
    n = 5  # OK
    k = 8.31e-3  # OK
    T0 = 0  # OK
    mass = 40  # OK
    a = 0.38  # ?
    epsilon = 1  # OK
    R = 0.38  # OK
    f = 10e4  # OK
    L = 2.3 * a * (n - 1)
    tau = 1e-3  # ?

    num_of_steps = 1000
    S_out = 10
    S_xyz = 100
    # *****************************************************************************************************************

    # variable parameters *********************************************************************************************
    n_values = [6]
    T0_values = [80, 100, 120]
    # *****************************************************************************************************************

    for n in n_values:
        for T0 in T0_values:
            momentums = generate_n3_random_3D_momentums(n, k, T0, mass)
            positions = create_initial_pos_file_in_xyz_format(n, a, out_initialization_dir)

            print(f"n={n}, T0={T0}, steps={num_of_steps}")
            run_simulation(num_of_steps, tau, positions, momentums, epsilon, R, f, L, mass, k, S_out, S_xyz, T0,
                           out_dynamic_dir=out_dynamic_dir, out_images_dir=out_images_dir)
