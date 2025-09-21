import sys

from functions import run_experiment

if __name__ == "__main__":
    path = sys.argv[1]
    num_procs = int(sys.argv[2])
    run_experiment(path, num_procs)
