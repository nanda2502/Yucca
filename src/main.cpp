#include "Population.hpp"
#include "Params.hpp"
#include "Learners.hpp"
#include "Types.hpp"
#include "Utils.hpp"
#include <string>
#include <iostream>
#include <vector>
#include <cmath>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <adjacency_matrix_index> <num_nodes>" << std::endl;
        return 1;
    }
    int adj_int = std::stoi(argv[1]);
    int num_nodes = std::stoi(argv[2]);

    auto allMatrices = readWeightedAdjacencyMatrices(num_nodes);
    std::vector<std::vector<std::vector<double>>> matrices(1, allMatrices[adj_int]);

    // Simulate background population
    Population population(matrices[0]);
    population.learn();

    std::vector<ParamCombination> combinations = makeCombinations(matrices, 1); 

    std::cout << "Number of combinations: " << combinations.size() << '\n';
    std::cout << "Number of shuffles: " << combinations[0].shuffleSequences.size() << '\n';
    std::vector<AccumulatedResult> accumulatedResults(combinations.size());

    std::vector<size_t> indices(combinations.size());
    std::iota(indices.begin(), indices.end(), 0);

    for (size_t idx : indices) {
        const ParamCombination& comb = combinations[idx];

        size_t maxSteps = comb.adjMatrix.size() * 2;
        std::vector<double> payoffs(maxSteps, 0.0);
        std::vector<double> successRates(maxSteps, 0.0);
        double completionTime = 0.0;

        for (size_t i = 0; i < comb.shuffleSequences.size(); ++i) {
            const auto& shuffleSequence = comb.shuffleSequences[i];
            Learners learners = Learners(
                comb.adjMatrix,
                population.repertoires,
                comb.slope,
                comb.strategy,
                shuffleSequence
            );
            learners.learn();

            // Average outcome measures over shuffles
            for (size_t j = 0; j < payoffs.size(); ++j) {
                payoffs[j] += learners.meanPayoff[j] * comb.shuffleWeights[i];
                successRates[j] += learners.meanSuccessRate[j] * comb.shuffleWeights[i];
            }
            completionTime += learners.meanCompletionTime * comb.shuffleWeights[i];
        }
        AccumulatedResult result;

        result.absorbing = completionTime;
        result.payoffs = payoffs;
        result.successRates = successRates;

        accumulatedResults[idx] = result;
        std::cout << "Completed combination " << idx + 1 << " of " << combinations.size() 
        << " (processed " << comb.shuffleSequences.size() << " shuffles)" << '\n';
    }

    std::string csvHeader = "num_nodes,adj_mat,strategy,steps,step_payoff,step_transitions,slope,absorbing";
    std::vector<std::string> csvData;
    csvData.push_back(csvHeader);

    for (size_t i = 0; i < accumulatedResults.size(); ++i) {
        const AccumulatedResult& accumResult = accumulatedResults[i];
        const ParamCombination& comb = combinations[i];
        for (size_t step = 0; step < comb.adjMatrix.size(); step++) {
            std::string csvLine =
                std::to_string(comb.adjMatrix.size()) + "," +
                comb.adjMatrixBinary + "," +
                strategyToString(comb.strategy) + "," +
                std::to_string(step) + "," + // add 1 since index is 0-based
                std::to_string(accumResult.payoffs[step]) + "," +
                std::to_string(accumResult.successRates[step]) + "," +
                std::to_string(comb.slope) + "," +
                std::to_string(accumResult.absorbing);
            csvData.push_back(csvLine);	
        }
    }
    std::string outputDir = "../output";
    writeAndCompressCSV(outputDir, adj_int, csvData);
}