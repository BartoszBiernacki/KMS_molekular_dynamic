from numba import njit, jit, prange, vectorize
import numpy as np


@njit
def get_number_of_vectors_from_2D_nparray(array):
    rows, cols = array.shape
    return rows


@njit
def calculate_norm(vector):
    return np.sqrt(np.sum(vector * vector))


@njit()
def calculate_norms(vectors):
    return np.sqrt(np.sum(vectors*vectors, axis=1))

