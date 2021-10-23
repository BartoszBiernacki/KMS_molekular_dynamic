import cProfile
import pstats

from initial_position import create_initial_pos_file_in_xyz_format
from momentums import generate_n3_random_3D_momentums
from simulation import run_simulation


# 1000 sim step in 184s (5 steps per second) on ryzen 5 3600
# -----------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("NEW MAIN RUN")
    n = 10  # OK
    k = 8.31e-3  # OK
    T0 = 1000  # OK
    mass = 40  # OK
    a = 0.38  # ?
    epsilon = 1  # OK
    R = 0.38  # OK
    f = 10e4  # OK
    L = 2.8 * a * (n - 1)  # Ok
    tau = 1e-3  # ?

    num_of_steps = 1000
    S_out = 1
    S_xyz = 10

    momentums = generate_n3_random_3D_momentums(n, k, T0, mass, force_new=True)
    positions = create_initial_pos_file_in_xyz_format(n=n, a=a, create_new_initial_file=True)

    # run new simulation with one step only just to compile numba functions
    run_simulation(1, tau, positions, momentums, epsilon, R, f, L, mass, k, S_out, S_xyz)

    # run real simulation with many time steps and see how fast it goes
    with cProfile.Profile() as pr:
        run_simulation(num_of_steps, tau, positions, momentums, epsilon, R, f, L, mass, k, S_out, S_xyz)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()

    print("\nDone")
