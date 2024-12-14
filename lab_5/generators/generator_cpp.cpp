#include <iostream>
#include <random>
#include <sstream>

/**
 * \brief Generates a random binary sequence of the specified size.
 * \param size The size of the sequence to generate.
 * \return A string representing the random binary sequence.
 */
std::string generateRandomBinarySequence(int size) {
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<> dis(0, 1);

    std::stringstream ss;

    // Generate random binary numbers and append them to the string stream.
    for (int i = 0; i < size; ++i) {
        ss << dis(gen);
    }

    return ss.str(); 
}

/**
 * \brief Main function of the program.
 * \return Return code.
 */
int main() {
    int size;

    // Request the size of the sequence from the user.
    std::cout << "Enter the size of the sequence: ";
    std::cin >> size;

    // Generate a random binary sequence.
    std::string randomBinarySequence = generateRandomBinarySequence(size);

    // Output the generated sequence to the screen.
    std::cout << "Random binary sequence: " << randomBinarySequence << std::endl;

    return 0;
}