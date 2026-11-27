import cProfile
import pstats
from polybius_slow import main

if __name__ == "__main__":
    cProfile.run("main()", "profile_before.prof")
    p = pstats.Stats("profile_before.prof")
    p.sort_stats("cumulative").print_stats(15)
