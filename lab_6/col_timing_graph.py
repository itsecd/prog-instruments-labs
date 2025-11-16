import time

import matplotlib.pyplot as plt

from card_finder import *


def find_card_number_time(process_count: int) -> float:
    start = time.time()

    for bin_code in BINS:
        card_nums = gen_card_nums(bin_code)

        with multiprocessing.Pool(process_count) as pool:
            results = pool.imap_unordered(check_hash, card_nums, chunksize=500)

            for result in results:
                if result is not None:
                    pool.close()
                    pool.join()
                    end = time.time()
                    return end - start

            pool.close()
            pool.join()

    end = time.time()
    return end - start


def collect_times() -> tuple[list[int], list[float]]:
    max_cpu = int(get_cpu_count() * 1.5)
    process_counts = list(range(1, max_cpu + 1))
    timings = []

    for count in process_counts:
        duration = find_card_number_time(count)
        timings.append(duration)

    return process_counts, timings


def get_timing_graph(proc_list: list[int], time_list: list[float]) -> None:
    best_time = min(time_list)
    optimal_idx = time_list.index(best_time)
    optimal_proc = proc_list[optimal_idx]

    plt.figure(figsize=(12, 6))
    plt.plot(proc_list, time_list, marker='o', linestyle='-', color="dodgerblue", label="Время подбора (сек)")
    plt.scatter(optimal_proc, best_time, color="crimson", label=f"Оптимум: {optimal_proc} поток(ов)")

    plt.title("Эффективность подбора номера карты по количеству процессов", fontsize=14, weight='bold')
    plt.xlabel("Количество процессов", fontsize=12)
    plt.ylabel("Время выполнения (сек)", fontsize=12)

    plt.xticks(proc_list)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()