import multiprocessing
from simulate_d3_benchmark import run_d3_benchmark

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_d3_benchmark(num_games=10)
