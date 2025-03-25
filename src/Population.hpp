#ifndef LEARN_HPP
#define LEARN_HPP

#include <random>
#include <vector>
#include "Types.hpp"

// Population of random learners. Used to generate the background population for the Learners class. 
class Population {
public:
Population(const std::vector<std::vector<double>>& adjMatrix);
void learn();
std::vector<std::vector<size_t>> repertoires; // repertoires[agent][trait] = 1 if agent has trait, 0 otherwise

private: 
std::vector<std::vector<double>> adjMatrix;
std::vector<std::vector<std::vector<double>>> adjMatrices; // adjMatrices[agent][parent][trait]. Individual matrix for each agent.
Dependency dependency; 

double resetProb; // Probability of resetting the repertoire each learning attempt. This is tuned to achieve the target omniscience level. 
double targetOmniscience = 0.05;
bool isLearnable(size_t trait, size_t agentIndex, std::mt19937& gen);
};

#endif // LEARN_HPP