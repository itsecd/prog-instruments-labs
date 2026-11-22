#include "CppGenerate.h"

#include <random>
#include <cstdlib>
#include <ctime> 
#include <fstream>
#include <string>
using namespace std;


int read_constant(const char* NAME, const char* filepath) {
	ifstream file(filepath);
	if (!file.is_open()) {
		throw std::invalid_argument("Uncorrect file or path!");
	}

	string word;
	while (file >> word) {
		if (word == NAME) {
			file >> word;
			if (word != "=") {
				file.close();
				throw std::invalid_argument("Uncorrect constants file format!");
			}
			file >> word;
			file.close();
			return stoi(word);
		}
	}
	file.close();
}


// ����������� ������� ���������������� ���� ��� ��� ������ ������ �������
static std::random_device rd;                    // �������� ��������
static std::mt19937 gen(rd());                 // ��������� Mersenne Twister
static std::uniform_int_distribution<int> dis(0, 1); // �������������: 0 ��� 1

int generateDRV() {
	return dis(gen); // ���������� 0 ��� 1 � ������ ������������ (50 %)
}



vector<int> generateDRVec() {
	const int SEQUENCE_LENGTH = read_constant("SEQUENCE_LENGTH", "D:\\UniversityLabs\\inform-security-base\\lab_2\\generators\\constants.txt");
	vector<int> result(SEQUENCE_LENGTH);
	for (int i = 0; i < SEQUENCE_LENGTH; i++) {
		result[i] = generateDRV();
	}
	return result;
}


void write_file(const char* path,int n) {
	ofstream outputFile(path);

	if (!outputFile.is_open()) {
		cerr << "������ �������� ����� ��� ������!";
		throw std::invalid_argument("Uncorrect file path!");
	}

	for (size_t i = 0; i < n; ++i) {
		vector<int> DRVec = generateDRVec();
		for (auto u : DRVec) {
			outputFile << u;
			cout << u;
		}
		outputFile << "\n";
	}
	outputFile.close();
}