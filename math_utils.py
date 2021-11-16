from numba import njit, jit, prange, vectorize
import numpy as np


@njit(fastmath=True)
def get_number_of_vectors_from_2D_nparray(array):
    rows, cols = array.shape
    return rows


@njit(fastmath=True)
def calculate_norm(vector):
    return np.sqrt(np.sum(vector * vector))


@njit(fastmath=True)
def calculate_norms(vectors):
    return np.sqrt(np.sum(vectors*vectors, axis=1))

