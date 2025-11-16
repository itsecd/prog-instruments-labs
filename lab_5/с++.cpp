#include <iostream>
#include <fstream>
#include <random>
#include <bitset>
#include <ctime>

using namespace std;

int main()
{
    // Получаем начальное значение (seed) от текущего времени
    unsigned seed = static_cast<unsigned>(time(nullptr));
    mt19937 generator(seed);              // Генератор псевдослучайных чисел (MT19937)
    uniform_int_distribution<int> dist(0, 1);  // Равномерное распределение между 0 и 1

    // Создаем контейнер для хранения 128 бит
    bitset<128> bits;

    // Генерируем 128 случайных битов
    for (int i = 0; i < 128; ++i)
    {
        bits.set(i, dist(generator));      // Устанавливаем каждый бит случайным значением
    }

    // Выводим результат в консоль
    cout << "C++ GPCSH (128 bit):\n" << bits << endl;

    // Открываем файл для сохранения последовательности
    ofstream output_file("bin_sequence_cpp.txt");
    if (output_file.is_open())
    {
        output_file << bits;               // Записываем данные в файл
        output_file.close();                // Закрываем файл
        cout << "Sequence saved to bin_sequence_cpp.txt." << endl;
    }
    else
    {
        cerr << "Error: Could not open the file for writing!" << endl;
        return 1;                          // Завершаем программу с ошибкой
    }

    return 0;
}