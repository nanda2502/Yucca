#ifndef UTILS_HPP
#define UTILS_HPP

#include "Types.hpp"

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <zlib.h>
#include <sstream>

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

inline std::vector<std::vector<double>> binaryStringToWeightedMatrix(const std::string& str) {
    // For backward compatibility, convert binary string to weighted matrix
    // where 1s become 1.0 and 0s become 0.0
    
    int n = static_cast<int>(std::sqrt(str.size()));
    std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
    
    for (int row = 0; row < n; ++row) {
        for (int column = 0; column < n; ++column) {
            // Convert the character '0' or '1' to double 0.0 or 1.0
            matrix[row][column] = (str[row * n + column] == '1') ? 1.0 : 0.0;
        }
    }
    
    return matrix;
}

inline std::vector<std::vector<double>> parseMatrixString(const std::string& str) {
    // Check if the string contains any commas (weighted comma format)
    bool isWeightedCommaFormat = str.find(',') != std::string::npos;
    
    if (isWeightedCommaFormat) {
        // Parse as weighted comma-separated format
        std::vector<double> flatValues;
        std::stringstream ss(str);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            // Handle empty cell (if there are consecutive commas)
            if(cell.empty()) {
                flatValues.push_back(0.0);
            } else {
                try {
                    flatValues.push_back(std::stod(cell));
                } catch (const std::exception& e) {
                    // If conversion fails, default to 0.0
                    flatValues.push_back(0.0);
                }
            }
        }
        
        // Calculate dimension (assuming square matrix)
        int n = static_cast<int>(std::sqrt(flatValues.size()));
        
        // Reshape into n×n matrix
        std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                size_t index = i * n + j;
                if (index < flatValues.size()) {
                    matrix[i][j] = flatValues[index];
                } else {
                    // Handle case where there are not enough values
                    matrix[i][j] = 0.0;
                }
            }
        }
        
        return matrix;
    } 
    else if (str.find_first_not_of("01") == std::string::npos) {
        // Parse as binary format (only contains 0s and 1s)
        int n = static_cast<int>(std::sqrt(str.size()));
        std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // Convert '0'/'1' character to 0.0/1.0 double
                matrix[i][j] = (str[i * n + j] == '1') ? 1.0 : 0.0;
            }
        }
        
        return matrix;
    }
    else {
        // Parse as compact weighted format (single digits representing tenths)
        int n = static_cast<int>(std::sqrt(str.size()));
        std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // Convert single digit character to double value with one decimal
                char digit = str[i * n + j];
                int value = digit - '0';  // Convert char to int
                matrix[i][j] = value / 10.0;  // Convert to double with one decimal
            }
        }
        
        return matrix;
    }
}

inline std::vector<std::vector<std::vector<double>>> readWeightedAdjacencyMatrices(int num_nodes) {
    std::string filePath = "../data/adj_mat_" + std::to_string(num_nodes) + ".csv";
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filePath);
    }

    std::vector<std::vector<std::vector<double>>> matrices;
    std::string line;
    
    while (std::getline(file, line)) {
        matrices.push_back(parseMatrixString(line));
    }
    
    std::cout << "Loaded " << matrices.size() << " weighted adjacency matrices." << '\n';

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