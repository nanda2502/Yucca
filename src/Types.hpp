#ifndef TYPES_HPP
#define TYPES_HPP

#include <string>
#include <vector>

enum Strategy { Random, Payoff, Proximal, Prestige, Conformity };
enum Dependency { Fixed, Variable }; // Dependency structure is either fixed per agent or variable per learning attempt

struct ParamCombination {
    std::vector<std::vector<double>> adjMatrix;
    std::string adjMatrixBinary;
    Strategy strategy;
    double slope;
    std::vector<std::vector<size_t>> shuffleSequences;
    std::vector<double> shuffleWeights;
    Dependency dependency = Dependency::Variable;
};

struct AccumulatedResult {
    double absorbing = 0.0;
    std::vector<double> payoffs;
    std::vector<double> successRates;
};

#endif // TYPES_HPP