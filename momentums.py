from math_utils import *
import matplotlib.pyplot as plt


@njit
def generate_energy(k, T0):
    Ex = -0.5 * k * T0 * np.log(np.random.uniform(0, 1))
    Ey = -0.5 * k * T0 * np.log(np.random.uniform(0, 1))
    Ez = -0.5 * k * T0 * np.log(np.random.uniform(0, 1))

    return np.array([Ex, Ey, Ez])


@njit
def generate_momentum(k, T0, m):
    E = generate_energy(k, T0)
    possible_signs = np.array([-1, 1])

    sign = np.random.choice(possible_signs)
    px = sign * np.sqrt(m * E[0])

    sign = np.random.choice(possible_signs)
    py = sign * np.sqrt(m * E[1])

    sign = np.random.choice(possible_signs)
    pz = sign * np.sqrt(m * E[2])

    return np.array([px, py, pz])


@njit
def generate_momentum_for_all_particles(N, k, T0, m):
    momentums = np.zeros((N, 3))
    for i in prange(N):
        momentums[i] = generate_momentum(k=k, T0=T0, m=m)

    return momentums


@njit
def normalize_momentum_for_all_particles(momentums):
    P_sum = np.zeros(3)
    N, _ = momentums.shape
    # getting sum of momentum
    for i in prange(N):
        P_sum += momentums[i]
    # normalization
    for i in prange(N):
        normalized_vector = momentums[i] - (1 / N) * P_sum
        momentums[i] = normalized_vector
    return momentums


def save_normalized_momentums_to_text_file(momentums):
    np.savetxt("momentums.txt", X=momentums, delimiter=' ', fmt='%.15f')
    print("Normalised momentums saved to ,,momentums.txt''")


def generate_n3_random_3D_momentums(n, k, T0, m, force_new=False):
    line_count = 0
    filename = "momentums.txt"
    try:
        with open(filename) as file:
            for line in file:
              if line != "\n":
                line_count += 1
    except Exception:
        print(f"Problem with {filename} in generate_n3_random_3D_momentums function")

    if line_count != n**3 or force_new:
        momentums = generate_momentum_for_all_particles(N=n ** 3, k=k, T0=T0, m=m)
        momentums = normalize_momentum_for_all_particles(momentums=momentums)
        save_normalized_momentums_to_text_file(momentums=momentums)
        return np.array(momentums)
    else:
        print(f"I am going to use already exist {filename} file to get momentums")
        with open(filename) as file:
            momentums = [[float(digit) for digit in line.split()] for line in file]
    return np.array(momentums)


def show_px():
    with open('momentums.txt') as file:
        momentums = [[float(digit) for digit in line.split()] for line in file]

        px = []
        for momentum in momentums:
            px.append(momentum[0])

    fig, ax = plt.subplots()
    num_bins = 100
    # the histogram of the data
    ax.hist(px, num_bins, density=True)
    plt.savefig('px.png')


def show_py():
    with open('momentums.txt') as file:
        momentums = [[float(digit) for digit in line.split()] for line in file]

        px = []
        for momentum in momentums:
            px.append(momentum[1])

    fig, ax = plt.subplots()
    num_bins = 100
    # the histogram of the data
    ax.hist(px, num_bins, density=True)
    plt.savefig('py.png')


def show_pz():
    with open('momentums.txt') as file:
        momentums = [[float(digit) for digit in line.split()] for line in file]

        px = []
        for momentum in momentums:
            px.append(momentum[2])

    fig, ax = plt.subplots()
    num_bins = 100
    # the histogram of the data
    ax.hist(px, num_bins, density=True)
    plt.savefig('pz.png')






