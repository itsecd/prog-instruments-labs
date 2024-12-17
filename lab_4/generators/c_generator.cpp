#include <iostream>
#include <random>
#include <bitset>
#include <time.h>

using namespace std;

bitset<128> generateRandomSequence() {
    srand(time(0));
    bitset<128> random_sequence;
    for (int i = 0; i < 128; ++i) {
        random_sequence[i] = rand() % 2; 
    }
    return random_sequence;
}

int main() {
    std::bitset<128> random_sequence = generateRandomSequence();
    std::cout << "Random sequence: " << random_sequence << std::endl;
    return 0;
}