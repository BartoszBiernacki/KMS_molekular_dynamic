from math_utils import *
import time
import cProfile
import pstats


@njit
def my_func(n):
	pass





inner_iterations = int(1e1)
outer_iterations = int(1e7)

my_func(inner_iterations)


with cProfile.Profile() as pr:
	for k in range(outer_iterations):
		my_func(inner_iterations)
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()

with cProfile.Profile() as pr:
	for k in np.arange(outer_iterations):
		my_func(inner_iterations)
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()
