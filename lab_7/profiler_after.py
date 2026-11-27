import cProfile
import pstats
from polybius_optimized import main

if __name__ == "__main__":
    cProfile.run("main()", "profile_after.prof")
    p = pstats.Stats("profile_after.prof")
    p.sort_stats("cumulative").print_stats(15)
