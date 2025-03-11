#ifndef LEARN_HPP
#define LEARN_HPP

#include <vector>


// Population of random learners. Used to generate demoRepertoires 
class Population {
public:
Population(const std::vector<std::vector<size_t>>& adjMatrix);
void learn();
std::vector<std::vector<size_t>> repertoires; // repertoires[agent][trait] = 1 if agent has trait, 0 otherwise

private: 
std::vector<std::vector<size_t>> adjMatrix;
double resetProb;
double targetOmniscience = 0.05;
bool isLearnable(size_t trait, size_t agentIndex);

};


#endif // LEARN_HPP