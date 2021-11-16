from math_utils import *


@njit
def initialize_position(n, a):
    b0 = np.array([a, 0., 0.])
    b1 = np.array([a / 2, (a * np.sqrt(3)) / 2., 0.])
    b2 = np.array([a / 2, (a * np.sqrt(3)) / 6., a * np.sqrt(2/3)])

    r = np.zeros((n**3, 3))

    for i_0 in range(n):
            for i_1 in range(n):
                    for i_2 in range(n):
                            i = i_0 + i_1*n + i_2*n*n
                            r[i] = (i_0 - (n - 1) / 2) * b0 + (i_1 - (n - 1) / 2) * b1 + (i_2 - (n - 1) / 2) * b2
    return r
    


def create_initial_pos_file_in_xyz_format(n, a, out_dir):
    N = round(n**3)
    filename = out_dir + "initial_file_n={}.xyz".format(n)
    positions = initialize_position(n, a)

    with open(filename, "w") as outfile:
        outfile.write(str(N) + '\n' + '\n')
        for vector in positions:
            outfile.write('Ar {vector[0]} {vector[1]} {vector[2]} \n')

    return positions
