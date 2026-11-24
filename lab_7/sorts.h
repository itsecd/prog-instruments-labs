#ifndef SORTS_H
#define SORTS_H

#include <iostream>
#include <vector>
#include "LinkedList.h"

// Структура для сбора статистики о работе алгоритмов сортировки
struct Stats
{
    size_t comparison_count = 0;
    size_t copy_count = 0;

    // Перегрузка оператора += для суммирования статистик
    Stats& operator+=(const Stats& rhs)
    {
        comparison_count += rhs.comparison_count;
        copy_count += rhs.copy_count;
        return *this;
    }
    
    // Перегрузка оператора /= для усреднения статистик
    Stats& operator/=(int a)
    {
        if (a != 0) {
            comparison_count /= a;
            copy_count /= a;
        }
        return *this;
    }
};

// Перегрузка оператора вывода для структуры Stats
std::ostream& operator<<(std::ostream& os, Stats s)
{
    return os << s.comparison_count << " " << s.copy_count << "\n";
    //return os;
}

// Сортировка пузырьком для вектора
Stats bubble_sort(std::vector<int>& arr) {
    Stats s; 
    int n = static_cast<int>(arr.size());
    bool swapped; 

    for (int i = 0; i < n - 1; i++) {
        swapped = false;

        for (int j = 0; j < n - i - 1; j++) {
            s.comparison_count++; 
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                s.copy_count += 3;
                swapped = true;
            }
        }
        if (!swapped) break;
    }
    return s;
}

// Сортировка расческой (Comb sort)
Stats comb_sort(std::vector<int>& arr) { // std::span | * + size |* + *
    Stats s;
    int n = static_cast<int>(arr.size());
    int gap = n; 
    bool swapped = true;
    const double shrink = 1.3;

    while (gap > 1 || swapped) {
        gap = (gap > 1) ? static_cast<int>(static_cast<double>(gap) / shrink) : 1;
        swapped = false;

        for (int i = 0; i < n - gap; i++) {
            s.comparison_count++;
            if (arr[i] > arr[i + gap]) {
                int temp = arr[i];
                arr[i] = arr[i + gap];
                arr[i + gap] = temp;
                s.copy_count += 3;
                swapped = true;
            }
        }
    }

    return s;
}

// Вспомогательная функция для пирамидальной сортировки (Heap sort)
// Преобразует поддерево с корнем i в двоичную кучу (max-heap)
void heapify(std::vector<int>& arr, int n, int i, Stats& s) {
    int largest = i;   
    int left = 2 * i + 1;   
    int right = 2 * i + 2; 
    
    // Сравниваем корень с левым потомком
    ++s.comparison_count;
    if (left < n && arr[left] > arr[largest]) {
        largest = left;
        //++s.copy_count; 
    }
    
    // Сравниваем корень с правым потомком
    ++s.comparison_count;
    if (right < n && arr[right] > arr[largest]) {
        largest = right;
        //++s.copy_count; 
    }

    // Если наибольший элемент не в корне меняем корень с наибольшим потомком
    if (largest != i) {
        int temp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = temp;
        s.copy_count += 3;
        
        heapify(arr, n, largest, s);
    }
}

// Пирамидальная сортировка (Heap sort)
// Алгоритм строит max-heap из массива, затем извлекает элементы
Stats heap_sort(std::vector<int>& arr) {
    Stats s;
    int n = static_cast<int>(arr.size());

    // Построение max-heap
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i, s);

    // Извлечение элементов из кучи
    for (int i = n - 1; i > 0; i--) {
        // Перемещаем текущий корень в конец
        int temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;
        s.copy_count += 3;
        
        // Вызываем heapify для уменьшенной кучи
        heapify(arr, i, 0, s);
    }
    return s;
}

// Перегрузка оператора вывода для вектора любого типа
template <class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& arr) {
    for (const T& num : arr) {
        os << num << " ";
    }
    os << "\n";
    return os;
}

// Перегрузка оператора вывода для вектора статистик
std::ostream& operator<<(std::ostream& os, const std::vector<Stats>& arr) {
    for (const Stats& num : arr) {
        os << num;
    }
    return os;
}

#endif