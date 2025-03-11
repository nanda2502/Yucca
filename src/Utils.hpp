#ifndef UTILS_HPP
#define UTILS_HPP

#include "Types.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <unordered_set>
#include <numeric>
#include <random>
#include <zlib.h>

std::string strategyToString(Strategy strategy) {
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

bool isUnconstrained(const std::vector<std::vector<size_t>>& adjMatrix) {
    // Skip the first row
    for (size_t row = 1; row < adjMatrix.size(); ++row) {
        // Check each element in this row
        for (size_t col : adjMatrix[row]) {
            // If any element is true (nonzero), return false
            if (col != 0) {
                return false;
            }
        }
    }
    // If we've checked all rows and found no nonzero values, return true
    return true;
}

std::string adjMatrixToBinaryString(const std::vector<std::vector<size_t>>& adjMatrix) {
    std::string binaryString;
    binaryString.reserve(adjMatrix.size() * adjMatrix[0].size());

    for (const auto& row : adjMatrix) {
        for (size_t entry : row) {
            binaryString += entry == 1 ? '1' : '0';
        }
    }
    return binaryString;
}

std::vector<std::vector<size_t>> binaryStringToAdjacencyMatrix(const std::string& str) {
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

std::vector<std::vector<std::vector<size_t>>> readAdjacencyMatrices() {
    std::string filePath = "../adj_mat.csv";
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

size_t factorial(size_t num) {
    size_t result = 1;
    for (size_t i = 2; i <= num; ++i) {
        result *= i;
    }
    return result;
}

std::vector<std::vector<size_t>> makeShuffles(int n) {
    std::vector<std::vector<size_t>> shuffleSequences;
    
    // For small n (10 or less), generate all permutations
    if (n <= 10) {
        std::vector<size_t> perm(n - 1);
        std::iota(perm.begin(), perm.end(), 0);
        size_t sequenceCount = factorial(n - 1);
        shuffleSequences.reserve(sequenceCount);
        
        do {
            shuffleSequences.push_back(perm);
        } while (std::ranges::next_permutation(perm).found);
    }
    // For larger n, generate a limited number of random permutations
    else {
        // Generate random unique permutations of trait indices (excluding root)
        size_t maxPermutations = 10000;
        
        // Create a set to store unique permutations (as strings for easy comparison)
        std::unordered_set<std::string> uniquePermsSet;
        
        // Create a random number generator
        std::random_device rd;
        std::mt19937 g(rd());
        
        // Generate the base permutation
        std::vector<size_t> basePerm(n - 1);
        std::iota(basePerm.begin(), basePerm.end(), 0);
        
        // Try to generate maxPermutations unique permutations
        while (uniquePermsSet.size() < maxPermutations) {
            // Create a new permutation by shuffling the base
            std::vector<size_t> perm = basePerm;
            std::shuffle(perm.begin(), perm.end(), g);
            
            // Convert to string for uniqueness check
            std::string permStr;
            for (size_t val : perm) {
                permStr += std::to_string(val) + ",";
            }
            
            // Add to set and vector if unique
            if (uniquePermsSet.insert(permStr).second) {
                shuffleSequences.push_back(perm);
                
                // Progress output
                if (shuffleSequences.size() % 500 == 0) {
                    std::cout << "Generated " << shuffleSequences.size() << " unique permutations..." << '\n';
                }
            }
            
            // Safety check to avoid infinite loop
            if (uniquePermsSet.size() == factorial(n - 1)) {
                std::cout << "Generated all possible permutations (" << uniquePermsSet.size() << ")" << '\n';
                break;
            }
        }
    }
    
    return shuffleSequences;
}

std::vector<double> returnSlopeVector(Strategy strategy) {
    switch (strategy) {
        case Random:
            return {0.0};
        default:
            return {0.0, 1.0, 1.25, 2.0, 2.5, 5.0};	

    }
}

std::vector<ParamCombination> makeCombinations(
    const std::vector<std::vector<std::vector<size_t>>>& adjacencyMatrices, 
    int replications
) {
    std::vector<ParamCombination> combinations;
        
    std::vector<Strategy> strategies = {
        Strategy::Random,
        Strategy::Payoff,
        Strategy::Proximal,
        Strategy::Prestige,
        Strategy::Conformity
    };

    for (const auto& adjMatrix : adjacencyMatrices) {
        std::string adjMatrixBinary = adjMatrixToBinaryString(adjMatrix);
        size_t n = adjMatrix.size();
        auto shuffleSequences = makeShuffles(n);
        // Determine which shuffle sequences to use
        std::vector<std::vector<size_t>> usedShuffleSequences;
        if (isUnconstrained(adjMatrix)) {
            // For unconstrained adjacency matrix, use only the first shuffle sequence
            usedShuffleSequences = {shuffleSequences[0]};
        } else {
            // For constrained adjacency matrix, use all shuffle sequences
            usedShuffleSequences = shuffleSequences;
        }
        
        // Create combinations with default parameters
        for (const auto& strategy : strategies) {
                auto slopes = returnSlopeVector(strategy);
                
                // Base cases: default values for all parameters, but vary the slopes
                for (const auto& slope : slopes) {
                    for (int repl = 0; repl < replications; ++repl) {
                        combinations.push_back({
                            adjMatrix, 
                            adjMatrixBinary, 
                            strategy, 
                            slope, 
                            usedShuffleSequences
                        });
                    }
                }
                
        
            } 
        }
    
    return combinations;
}

void writeAndCompressCSV(const std::string& outputDir, int n, const std::vector<std::string>& csvData) {
    // Construct the output CSV file path
    std::string outputCsvPath = outputDir + "../output/expected_steps_" + std::to_string(n) + ".csv";

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