#ifndef UTILS_HPP
#define UTILS_HPP

#include "Types.hpp"

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <zlib.h>

inline std::string strategyToString(Strategy strategy) {
    switch (strategy) {
        case Random:
            return "Random";
        case Payoff:
            return "Payoff";
        case Proximal:
            return "Proximal";
        case Prestige:
            return "Prestige";
        case Conformity:
            return "Conformity";
        default:
            throw std::invalid_argument("Unknown strategy");
    }
}

inline std::vector<std::vector<size_t>> binaryStringToAdjacencyMatrix(const std::string& str) {
    std::string binaryStr = str;

    // Calculate the dimension of the matrix
    int n = static_cast<int>(std::sqrt(binaryStr.size()));

    std::vector<std::vector<size_t>> matrix(n, std::vector<size_t>(n));
    
    for (int row = 0; row < n; ++row) {
        for (int column = 0; column < n; ++column) {
            // Convert the character '0' or '1' to integer 0 or 1
            matrix[row][column] = binaryStr[row * n + column] - '0';
        }
    }

    return matrix;
}

inline std::vector<std::vector<std::vector<size_t>>> readAdjacencyMatrices(int num_nodes) {
    std::string filePath = "../data/adj_mat_" + std::to_string(num_nodes) + ".csv";
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filePath);
    }

    std::vector<std::vector<std::vector<size_t>>> matrices;
    std::string line;
    
    while (std::getline(file, line)) {
        matrices.push_back(binaryStringToAdjacencyMatrix(line));
    }
    
    std::cout << "Loaded " << matrices.size() << " adjacency matrices." << '\n';

    return matrices;
}

inline void writeAndCompressCSV(const std::string& outputDir, int n, const std::vector<std::string>& csvData) {
    // Construct the output CSV file path
    std::string outputCsvPath = outputDir + "/expected_steps_" + std::to_string(n) + ".csv";

    // Write results to CSV
    std::ofstream csvFile(outputCsvPath);
    if (!csvFile.is_open()) {
        std::cerr << "Failed to open file for writing: " << outputCsvPath << '\n';
        return;
    }
    for (const auto& line : csvData) {
        csvFile << line << "\n";
    }
    csvFile.close();

    // Compress the CSV file using gzip
    std::string compressedFilePath = outputCsvPath + ".gz";
    FILE* source = fopen(outputCsvPath.c_str(), "rb");
    gzFile dest = gzopen(compressedFilePath.c_str(), "wb");
    if ((source == nullptr) || (dest == nullptr)) {
        std::cerr << "Failed to open files for compression\n";
        if (source != nullptr) fclose(source);
        if (dest != nullptr) gzclose(dest);
        return;
    }

    char buffer[8192];
    int bytesRead = 0;
    while ((bytesRead = fread(buffer, 1, sizeof(buffer), source)) > 0) {
        gzwrite(dest, buffer, bytesRead);
    }

    fclose(source);
    gzclose(dest);

    // Remove the original uncompressed file
    if (std::remove(outputCsvPath.c_str()) != 0) {
        std::cerr << "Failed to remove original file: " << outputCsvPath << '\n';
    }
}

#endif // UTILS_HPP