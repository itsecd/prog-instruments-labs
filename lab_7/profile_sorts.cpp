#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

#include "sorts.h"

using Clock = std::chrono::steady_clock;
constexpr std::size_t input_size = 50'000;
constexpr int benchmark_repetitions = 5;
constexpr int profile_before_repetitions = 5;
constexpr int profile_after_repetitions = 5'000;

enum class Mode { compare, before, after };

std::vector<int> make_input() {
    std::mt19937 engine(42);
    std::uniform_int_distribution<int> distribution(-100'000, 100'000);
    std::vector<int> values(input_size);
    for (int& value : values) {
        value = distribution(engine);
    }
    return values;
}

double measure_seconds(std::vector<int>& values, bool optimized) {
    const auto started = Clock::now();
    if (optimized) {
        optimized_sort(values);
    } else {
        bubble_sort(values);
    }
    return std::chrono::duration<double>(Clock::now() - started).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

Mode parse_mode(int argc, char* argv[]) {
    if (argc == 1) return Mode::compare;
    if (argc == 2 && std::string_view(argv[1]) == "--before") return Mode::before;
    if (argc == 2 && std::string_view(argv[1]) == "--after") return Mode::after;
    std::cerr << "Usage: profile_sorts [--before | --after]\n";
    std::exit(2);
}

int run_profile(Mode mode, const std::vector<int>& input) {
    const bool optimized = mode == Mode::after;
    const int repetitions = optimized ? profile_after_repetitions : profile_before_repetitions;
    std::vector<int> expected;
    std::size_t comparisons = 0;

    for (int run = 0; run < repetitions; ++run) {
        auto values = input;
        if (optimized) {
            optimized_sort(values);
        } else {
            comparisons = bubble_sort(values).comparison_count;
        }

        if (!std::is_sorted(values.begin(), values.end())) {
            std::cerr << "Output is not sorted.\n";
            return 1;
        }
        if (run == 0) {
            expected = values;
        } else if (values != expected) {
            std::cerr << "Sorting result changed between runs.\n";
            return 1;
        }
    }

    std::cout << (optimized ? "After" : "Before") << " profile: "
              << repetitions << " sorts of " << input_size << " elements\n"
              << "Comparisons: " << comparisons << '\n'
              << "Output is sorted and deterministic: yes\n";
    return 0;
}

int run_comparison(const std::vector<int>& input) {
    std::vector<double> bubble_times;
    std::vector<double> optimized_times;
    std::size_t bubble_comparisons = 0;
    std::vector<int> expected;

    for (int run = 0; run < benchmark_repetitions; ++run) {
        auto bubble_values = input;
        auto optimized_values = input;

        const auto bubble_started = Clock::now();
        const auto bubble_stats = bubble_sort(bubble_values);
        const auto bubble_seconds = std::chrono::duration<double>(
            Clock::now() - bubble_started
        ).count();
        const auto optimized_seconds = measure_seconds(optimized_values, true);

        if (bubble_values != optimized_values) {
            std::cerr << "Sorting results differ.\n";
            return 1;
        }
        if (run == 0) {
            expected = bubble_values;
        } else if (bubble_values != expected) {
            std::cerr << "Sorting result changed between runs.\n";
            return 1;
        }

        bubble_comparisons = bubble_stats.comparison_count;
        bubble_times.push_back(bubble_seconds);
        optimized_times.push_back(optimized_seconds);
    }

    const double bubble_seconds = median(bubble_times);
    const double optimized_seconds = median(optimized_times);
    std::cout << "Input size: " << input_size << '\n'
              << "Runs: " << benchmark_repetitions << " (median)\n"
              << "Bubble Sort: " << bubble_seconds << " s; comparisons="
              << bubble_comparisons << '\n'
              << "std::sort:   " << optimized_seconds << " s\n"
              << "Speedup:     " << bubble_seconds / optimized_seconds << "x\n"
              << "Same sorted output: yes\n";
    return 0;
}

int main(int argc, char* argv[]) {
    const Mode mode = parse_mode(argc, argv);
    const auto input = make_input();
    if (mode == Mode::compare) return run_comparison(input);
    return run_profile(mode, input);
}
