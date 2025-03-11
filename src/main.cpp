#include "Population.hpp"
#include "Learners.hpp"
#include "Types.hpp"
#include "Utils.hpp"
#include <string>
#include <iostream>
#include <vector>
#include <cmath>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <adjacency_matrix_index>" << std::endl;
        return 1;
    }
    int adj_int = std::stoi(argv[1]);

    
    auto allMatrices = readAdjacencyMatrices();
    std::vector<std::vector<std::vector<size_t>>> matrices(1, allMatrices[adj_int]);

    // Simulate background population
    Population population(matrices[adj_int]);
    population.learn();



    std::vector<ParamCombination> combinations = makeCombinations(matrices, 1); 

    std::cout << "Number of combinations: " << combinations.size() << '\n';
    std::vector<AccumulatedResult> accumulatedResults(combinations.size());

    std::vector<size_t> indices(combinations.size());
    std::iota(indices.begin(), indices.end(), 0);

    #pragma omp parallel for
    for (size_t idx : indices) {
        const ParamCombination& comb = combinations[idx];

        std::vector<double> payoffs(comb.adjMatrix.size() * 2, 0.0);
        std::vector<double> successRates(comb.adjMatrix.size() * 2, 0.0);
        double completionTime = 0.0;

        for (const auto& shuffleSequence : comb.shuffleSequences) {
            Learners learners = Learners(
                comb.adjMatrix,
                population.repertoires,
                comb.slope,
                comb.strategy,
                shuffleSequence
            );
            learners.learn();

            // Average outcome measures over shuffles
            for (size_t i = 0; i < payoffs.size(); ++i) {
                payoffs[i] += learners.meanPayoff[i] / comb.shuffleSequences.size();
                successRates[i] += learners.meanSuccessRate[i] / comb.shuffleSequences.size();
            }
            completionTime += learners.meanCompletionTime / comb.shuffleSequences.size();
        }
        accumulatedResults[idx] = {completionTime, payoffs, successRates};
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
                std::to_string(step + 1) + "," + // add 1 since index is 0-based
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