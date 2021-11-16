from math_utils import *


@njit(fastmath=True)
def generate_energy(k, T0):
    Ex = -0.5 * k * T0 * np.log(np.random.uniform(0, 1))
    Ey = -0.5 * k * T0 * np.log(np.random.uniform(0, 1))
    Ez = -0.5 * k * T0 * np.log(np.random.uniform(0, 1))

    return np.array([Ex, Ey, Ez])


@njit(fastmath=True)
def generate_momentum(k, T0, m):
    E = generate_energy(k, T0)
    possible_signs = np.array([-1, 1])

    sign = np.random.choice(possible_signs)
    px = sign * np.sqrt(2*m * E[0])

    sign = np.random.choice(possible_signs)
    py = sign * np.sqrt(2*m * E[1])

    sign = np.random.choice(possible_signs)
    pz = sign * np.sqrt(2*m * E[2])

    return np.array([px, py, pz])


@njit(fastmath=True)
def generate_momentum_for_all_particles(N, k, T0, m):
    momentums = np.zeros((N, 3))
    for i in prange(N):
        momentums[i] = generate_momentum(k=k, T0=T0, m=m)
    return momentums


@njit(fastmath=True)
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


@njit(fastmath=True)
def generate_n3_random_3D_momentums(n, k, T0, m):
    momentums = generate_momentum_for_all_particles(N=n**3, k=k, T0=T0, m=m)
    normalized_momentums = normalize_momentum_for_all_particles(momentums=momentums)
    return normalized_momentums
























