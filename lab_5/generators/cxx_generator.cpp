#include <iostream>
#include <vector>
#include <random>
#include <fstream>

using namespace std;

vector<int> generate(size_t size) {
    vector<int> sequence;
    for (int i = 0; i < size; i++) {
        sequence.push_back(rand() % 2);
    }
    return sequence;
}

int main() {
    vector<int> result = generate(128);
    for (int i = 0; i < result.size(); i++) {
        cout << result[i];
    }
    return 0;
}

//11001000001111111010100100100110101011101101101110100111111001000000000101000110110000001001011000111110001010110001111000101110