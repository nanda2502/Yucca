#ifndef PARAMS_HPP
#define PARAMS_HPP

#include "Types.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <random>

inline std::string adjMatrixToBinaryString(const std::vector<std::vector<size_t>>& adjMatrix) {
    std::string binaryString;
    binaryString.reserve(adjMatrix.size() * adjMatrix[0].size());

    for (const auto& row : adjMatrix) {
        for (size_t entry : row) {
            binaryString += entry == 1 ? '1' : '0';
        }
    }
    return binaryString;
}

inline bool isUnconstrained(const std::vector<std::vector<size_t>>& adjMatrix) {
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

inline double factorial(size_t n) {
    double result = 1.0;
    for (size_t i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

inline std::string getNodeFingerprint(const std::vector<std::vector<size_t>>& adjMatrix, size_t nodeIdx) {
    std::string fingerprint = "U:";
    
    // Add upstream nodes
    for (size_t i = 0; i < adjMatrix.size(); ++i) {
        if (adjMatrix[i][nodeIdx] == 1) {
            fingerprint += std::to_string(i) + ",";
        }
    }
    
    fingerprint += "D:";
    
    // Add downstream nodes
    for (size_t i = 0; i < adjMatrix[nodeIdx].size(); ++i) {
        if (adjMatrix[nodeIdx][i] == 1) {
            fingerprint += std::to_string(i) + ",";
        }
    }
    
    return fingerprint;
}

struct ShuffleResult {
    std::vector<std::vector<size_t>> shuffles;
    std::vector<double> weights;
};

inline size_t estimateRequiredPermutations(int n, 
    const std::unordered_map<std::string, std::vector<size_t>>& equivalenceClasses) {
        if (n == 121) return 1;
    
    // Calculate the number of unique equivalence classes
    size_t numClasses = equivalenceClasses.size();
    
    // If there are very few classes, we might need very few permutations
    if (numClasses <= 4) {
        // Calculate exact number of representative permutations
        double representativePerms = factorial(numClasses);
        
        // For tiny graphs, we can just use the exact number
        if (representativePerms < 500) {
            return static_cast<size_t>(representativePerms);
        }
        // For slightly larger ones, add a buffer
        return std::min(static_cast<size_t>(representativePerms * 2.0), static_cast<size_t>(15000));
    }
    
    // For larger numbers of classes, we need to consider the multinomial coefficient
    
    // Collect the sizes of all equivalence classes
    std::vector<size_t> classSizes;
    for (const auto& [fingerprint, nodes] : equivalenceClasses) {
        classSizes.push_back(nodes.size());
    }
    
    // Sort class sizes in descending order
    std::sort(classSizes.begin(), classSizes.end(), std::greater<size_t>());
        
    // Calculate base on the first few largest classes (to avoid numerical issues)
    size_t consideredClasses = std::min(numClasses, static_cast<size_t>(6));
    
    // Simplified approach: ratio of max weight to min weight is approximately
    // equal to factorial of the minimum of (largest class size, numClasses)
    size_t maxFromOneClass = std::min(classSizes[0], consideredClasses);
    double weightRatio = factorial(maxFromOneClass);
    
    // Estimate samples needed based on weight ratio
    // If the weight ratio is N, we need approximately N samples
    // to expect to see the lowest-weight permutation once
    size_t baseEstimate = static_cast<size_t>(weightRatio * 2.0); // 2x for safety buffer
    
    // Consider class count as another factor
    if (numClasses > 10) {
        // For many small classes, we need more samples to cover the space
        baseEstimate = std::max(baseEstimate, numClasses * 100);
    }
    
    // Cap at 15,000
    return std::min(baseEstimate, static_cast<size_t>(5000));
}

inline ShuffleResult makeShuffles(const std::vector<std::vector<size_t>>& adjMatrix, int n) {
    ShuffleResult result;
    std::vector<std::vector<size_t>>& shuffleSequences = result.shuffles;
    std::vector<double>& shuffleWeights = result.weights;
    
    // Group nodes by their connectivity patterns (excluding node 0 which is assumed to be the root)
    std::unordered_map<std::string, std::vector<size_t>> equivalenceClasses;
    
    for (size_t i = 1; i < static_cast<size_t>(n); ++i) {  // Skip node 0 (root)
        std::string fingerprint = getNodeFingerprint(adjMatrix, i);
        equivalenceClasses[fingerprint].push_back(i - 1);  // Adjust index by -1 for perm vector
    }
    
    // Construct a representative ordering for each equivalence class
    std::vector<size_t> uniquePositions;
    
    // Store the mapping of fingerprints for each representative node
    std::unordered_map<size_t, std::string> representativeFingerprints;
    
    for (const auto& [fingerprint, nodes] : equivalenceClasses) {
        size_t rep = nodes[0];
        uniquePositions.push_back(rep);                  // Representative node for this class
        representativeFingerprints[rep] = fingerprint;   // Store the fingerprint for lookup
    }
    
    // Calculate permutation weights based on the multinomial coefficient formula
    auto calculateWeight = [](const std::vector<size_t>& perm, 
                                               const std::vector<std::string>& permFingerprints) -> double {
        // Count occurrences of each fingerprint
        std::unordered_map<std::string, size_t> fingerprintCounts;
        for (const auto& fp : permFingerprints) {
            fingerprintCounts[fp]++;
        }
        
        // Calculate numerator: product of factorial of counts
        double numerator = 1.0;
        for (const auto& [fingerprint, count] : fingerprintCounts) {
            numerator *= factorial(count);
        }
        
        // Calculate denominator: factorial of total positions
        double denominator = factorial(perm.size());
        
        return numerator / denominator;
    };
    
    // For small numbers of unique positions, generate all permutations
    size_t uniqueCount = uniquePositions.size();
    
    if (uniqueCount <= 8) {
        std::vector<size_t> perm = uniquePositions;
        size_t sequenceCount = factorial(uniqueCount);
        shuffleSequences.reserve(sequenceCount);
        shuffleWeights.reserve(sequenceCount);
        
        double totalWeight = 0.0;
        
        do {
            // For each permutation, collect the fingerprints of each position
            std::vector<std::string> permFingerprints;
            for (size_t pos : perm) {
                permFingerprints.push_back(representativeFingerprints[pos]);
            }
            
            // Expand the permutation to include all equivalent nodes
            std::vector<size_t> expandedPerm;
            for (size_t pos : perm) {
                const std::string& fingerprint = representativeFingerprints[pos];
                for (size_t node : equivalenceClasses[fingerprint]) {
                    expandedPerm.push_back(node);
                }
            }
            shuffleSequences.push_back(expandedPerm);
            
            // Calculate the weight for this specific permutation
            double weight = calculateWeight(perm, permFingerprints);
            shuffleWeights.push_back(weight);
            totalWeight += weight;
            
        } while (std::next_permutation(perm.begin(), perm.end()));
        
        // Normalize weights to sum to 1
        for (double& weight : shuffleWeights) {
            weight /= totalWeight;
        }
    } 
    // For larger counts, generate a limited number of random permutations
    else {
        // Generate random unique permutations of representative indices
        size_t maxPermutations = estimateRequiredPermutations(n, equivalenceClasses);
        
        // Create a set to store unique permutations
        std::unordered_set<std::string> uniquePermsSet;
        
        // Create a random number generator
        std::random_device rd;
        std::mt19937 g(rd());
        
        // Generate the base permutation of representative nodes
        std::vector<size_t> basePerm = uniquePositions;
        
        double totalWeight = 0.0;
        
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
                // For each permutation, collect the fingerprints of each position
                std::vector<std::string> permFingerprints;
                for (size_t pos : perm) {
                    permFingerprints.push_back(representativeFingerprints[pos]);
                }
                
                // Expand the permutation to include all equivalent nodes
                std::vector<size_t> expandedPerm;
                for (size_t pos : perm) {
                    const std::string& fingerprint = representativeFingerprints[pos];
                    for (size_t node : equivalenceClasses[fingerprint]) {
                        expandedPerm.push_back(node);
                    }
                }
                shuffleSequences.push_back(expandedPerm);
                
                // Calculate the weight for this specific permutation using multinomial coefficient
                double weight = calculateWeight(perm, permFingerprints);
                shuffleWeights.push_back(weight);
                totalWeight += weight;
            }
            
            // Safety check to avoid infinite loop
            if (uniquePermsSet.size() == factorial(uniqueCount)) {
                break;
            }
        }
        
        // Normalize weights to sum to 1
        for (double& weight : shuffleWeights) {
            weight /= totalWeight;
        }
    }
    
    return result;
}

inline std::vector<double> returnSlopeVector(Strategy strategy) {
    switch (strategy) {
        case Random:
            return {0.0};
        default:
            return {2.0};	
    }
}

inline std::vector<ParamCombination> makeCombinations(
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
        auto shuffleResult = makeShuffles(adjMatrix, n);
        // Determine which shuffle sequences to use

        auto usedShuffleSequences = shuffleResult.shuffles;

        // Fixed weights for the graph from Hosseinioun et al. (2025)
        if (n == 121) {
            usedShuffleSequences = {shuffleResult.shuffles[0]};
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
                        shuffleResult.shuffles,
                        shuffleResult.weights

                    });
                }
            }
                
        
            } 
        }
    
    return combinations;
}

#endif // PARAMS_HPP