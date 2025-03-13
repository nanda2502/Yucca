#ifndef TYPES_HPP
#define TYPES_HPP

#include <string>
#include <vector>

enum Strategy { Random, Payoff, Proximal, Prestige, Conformity };

struct ParamCombination {
    std::vector<std::vector<size_t>> adjMatrix;
    std::string adjMatrixBinary;
    Strategy strategy;
    double slope;
    std::vector<std::vector<size_t>> shuffleSequences;
    std::vector<double> shuffleWeights;
};

struct AccumulatedResult {
    double absorbing = 0.0;
    std::vector<double> payoffs;
    std::vector<double> successRates;
};

#endif // TYPES_HPP