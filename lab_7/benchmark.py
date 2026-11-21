import cProfile
import io
import pstats
import sys
import time
import tracemalloc
import main


def run_benchmark(input_file: str = "sample_profiles.txt") -> None:
    sys.argv = ["main.py", input_file]

    # Замер CPU через cProfile
    profiler = cProfile.Profile()
    t_start = time.perf_counter()
    profiler.enable()
    main.main()
    profiler.disable()
    elapsed = time.perf_counter() - t_start

    profiler.dump_stats("profile_result.prof")

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("tottime")
    stats.print_stats(10)

    print(f"\n=== Время выполнения: {elapsed:.4f} с ===")
    print(stream.getvalue())

    # Замер пиковой RAM через tracemalloc
    tracemalloc.start()
    main.main()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"=== Пиковое потребление RAM: {peak / (1024 * 1024):.2f} MiB ===")


if __name__ == "__main__":
    run_benchmark()