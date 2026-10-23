#ifndef CPP_GENERATE_H
#define CPP_GENERATE_H
#include <iostream>
#include <vector>

int read_constant(const char* NAME, const char* filepath);

int generateDRV();
std::vector<int> generateDRVec();

void write_file(const char* path,int n);
#endif