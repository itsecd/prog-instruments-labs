#include <iostream>
#include <random>
#include <fstream>
#include "sorts.h"

// Функция для получения статистики работы алгоритмов на случайных данных
void get_stats(size_t n, int seed, Stats& s1, Stats& s2, Stats& s3)
{
    std::vector<int> vec1(n), vec2(n), vec3(n);  

    std::default_random_engine eng(seed);
    std::uniform_int_distribution<int> dist(-100000, 100000);  

    size_t count = 5;

    // Запускаем сортировки count раз и усредняем результаты
    for (size_t i = 0; i < count; i++)
    {
        // Заполняем массивы одинаковыми случайными числами
        for (size_t j = 0; j < n; j++)
        {
            int val = dist(eng);
            vec1[j] = vec2[j] = vec3[j] = val;
        }

        // Запускаем все три алгоритма сортировки
        s1 += bubble_sort(vec1);
        s2 += comb_sort(vec2);
        s3 += heap_sort(vec3);
    }

    // Усредняем статистики
    s1 /= static_cast<int>(count);
    s2 /= static_cast<int>(count);
    s3 /= static_cast<int>(count);
}

int main() 
{
    // Векторы для хранения статистик по трем типам тестов
    std::vector<Stats> s1; 
    std::vector<Stats> s2;  
    std::vector<Stats> s3;  

    // Подготовка тестовых данных
    const size_t max_size = 50000;
    std::vector<int> sorted_array(max_size);          
    std::vector<int> reverse_sorted_array(max_size);  

    // Заполняем тестовые массивы
    for (size_t i = 0; i < max_size; i++)
    {
        sorted_array[i] = static_cast<int>(i);               
        reverse_sorted_array[i] = static_cast<int>(max_size - i - 1); 
    }

    // ТЕСТ 1: СРЕДНИЙ СЛУЧАЙ (случайные массивы)
    std::cout << "Calculating average case 1-10k...\n";
    for (size_t i = 1; i <= 10; i++)
    {
        Stats tmp1, tmp2, tmp3;
        get_stats(i * 1000, static_cast<int>(i), tmp1, tmp2, tmp3);
        s1.push_back(tmp1);
        s2.push_back(tmp2);
        s3.push_back(tmp3);
    }
    
    // Тестируем большие массивы: 25000 и 50000
    std::cout << "Calculating average case 25k, 50k...\n";
    for (size_t i = 1; i <= 2; i++)
    {
        Stats tmp1, tmp2, tmp3;
        get_stats(25 * i * 1000, static_cast<int>(i), tmp1, tmp2, tmp3);
        s1.push_back(tmp1);
        s2.push_back(tmp2);
        s3.push_back(tmp3);
    }

    // ТЕСТ 2: УЖЕ ОТСОРТИРОВАННЫЙ МАССИВ 
    std::cout << "Calculating sorted case 1-10k...\n";
    for (size_t i = 1; i <= 10; i++)
    {
        std::vector<int> vec1(i * 1000), vec2(i * 1000), vec3(i * 1000);

        // Заполняем массивы отсортированными значениями
        for (size_t j = 0; j < i * 1000; j++) {
            vec1[j] = vec2[j] = vec3[j] = sorted_array[j];
        }

        // Запускаем сортировки и сохраняем статистику
        s1.push_back(bubble_sort(vec1));
        s2.push_back(comb_sort(vec2));
        s3.push_back(heap_sort(vec3));
    }
    
    // Тестируем большие отсортированные массивы
    std::cout << "Calculating sorted case 25k, 50k...\n";
    for (size_t i = 1; i <= 2; i++)
    {
        size_t current_size = 25 * i * 1000;
        std::vector<int> vec1(current_size), vec2(current_size), vec3(current_size);

        for (size_t j = 0; j < current_size; j++) {
            vec1[j] = vec2[j] = vec3[j] = sorted_array[j];
        }

        s1.push_back(bubble_sort(vec1));
        s2.push_back(comb_sort(vec2));
        s3.push_back(heap_sort(vec3));
    }

    // ТЕСТ 3: ОБРАТНО ОТСОРТИРОВАННЫЙ МАССИВ
    std::cout << "Calculating reverse sorted case 1-10k...\n";
    for (size_t i = 1; i <= 10; i++)
    {
        std::vector<int> vec1(i * 1000), vec2(i * 1000), vec3(i * 1000);

        for (size_t j = 0; j < i * 1000; j++) {
            vec1[j] = vec2[j] = vec3[j] = reverse_sorted_array[j];
        }

        s1.push_back(bubble_sort(vec1));
        s2.push_back(comb_sort(vec2));
        s3.push_back(heap_sort(vec3));
    }
    
    // Тестируем большие обратно отсортированные массивы
    std::cout << "Calculating reverse sorted case 25k, 50k...\n";
    for (size_t i = 1; i <= 2; i++)
    {
        size_t current_size = 25 * i * 1000;
        std::vector<int> vec1(current_size), vec2(current_size), vec3(current_size);

        for (size_t j = 0; j < current_size; j++) {
            vec1[j] = vec2[j] = vec3[j] = reverse_sorted_array[j];
        }

        s1.push_back(bubble_sort(vec1));
        s2.push_back(comb_sort(vec2));
        s3.push_back(heap_sort(vec3));
    }

    // СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В ФАЙЛЫ 
    std::ofstream f;
    f.open("res1.txt");
    f << s1;  
    f.close();
    
    f.open("res2.txt");
    f << s2;  
    f.close();
    
    f.open("res3.txt");
    f << s3;  
    f.close();

    // ДЕМОНСТРАЦИЯ РАБОТЫ С LinkedList
    std::cout << "\nTesting LinkedList bubble sort:\n";
    LinkedList<int> list;
    
    for (size_t i = 0; i < 10; i++)
    {
        list.push_tail(static_cast<int>(10 - i));
    }
    
    std::cout << "Before sorting: " << list;
    bubble_sort(list); 
    std::cout << "After sorting: " << list;

    return 0;
}