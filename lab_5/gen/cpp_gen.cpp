#include <iostream>
#include <random>
#include <limits>

using namespace std;

void generate() {

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> distribution(numeric_limits<int>::min(), numeric_limits<int>::max());
    for (int i = 0; i < 128; i++) {
        cout << abs(distribution(gen)) % 2;
    }

}

int main() {

    generate();
    return 0;

}

// 00100101001001100110101111101011001000100100110101011110000101010111111010001111110111111111101010000100100111011001001110001001