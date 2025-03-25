#include "Population.hpp"

#include <random>

Population::Population(const std::vector<std::vector<double>>& adjMatrix) 
    : adjMatrix(adjMatrix), dependency(Dependency::Variable), resetProb(0.2) // Start with a higher reset probability
{
    // Initialize repertoires for 10000 agents
    const size_t numAgents = 10000;
    const size_t numTraits = adjMatrix.size();
    
    // Create repertoires with all traits set to 0, except the first trait (index 0) set to 1
    repertoires.resize(numAgents, std::vector<size_t>(numTraits, 0));
    for (auto& repertoire : repertoires) {
        repertoire[0] = 1; // Initialize the first trait to 1 for all agents
    }

    // Initialize adjMatrices based on dependency type
    adjMatrices.resize(numAgents);
    if (dependency == Dependency::Fixed) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(0.0, 1.0);
        
        for (size_t agent = 0; agent < numAgents; ++agent) {
            adjMatrices[agent].resize(numTraits, std::vector<double>(numTraits, 0.0));
            for (size_t parent = 0; parent < numTraits; ++parent) {
                for (size_t trait = 0; trait < numTraits; ++trait) {
                    if (adjMatrix[parent][trait] > 0.0) {
                        adjMatrices[agent][parent][trait] = (dist(gen) < adjMatrix[parent][trait]) ? 1.0 : 0.0;
                    }
                }
            }
        }
    } else {
        // For variable dependency, just store references to the original matrix
        for (size_t agent = 0; agent < numAgents; ++agent) {
            adjMatrices[agent] = adjMatrix;
        }
    }
}

// Learn function implementation
void Population::learn() {
    const size_t numAgents = repertoires.size();
    const size_t numTraits = adjMatrix.size();
    
    auto countOmniscientAgents = [this, numTraits]() {
        size_t count = 0;
        for (const auto& repertoire : repertoires) {
            size_t traitCount = 0;
            for (size_t trait : repertoire) {
                traitCount += trait;
            }
            if (traitCount == numTraits) {
                count++;
            }
        }
        return count;
    };
    
    auto adjustResetProbability = [this, &countOmniscientAgents, numAgents]() {
        size_t currentOmniscient = countOmniscientAgents();
        double currentPercentage = static_cast<double>(currentOmniscient) / numAgents;
        
        if (currentPercentage < targetOmniscience * 0.9) {
            resetProb = std::max(0.001, resetProb * 0.9);
        } else if (currentPercentage < targetOmniscience) {
            resetProb = std::max(0.001, resetProb * 0.95);
        } else if (currentPercentage > targetOmniscience * 1.1) {
            resetProb = std::min(0.9, resetProb * 1.2);
        } else if (currentPercentage > targetOmniscience) {
            resetProb = std::min(0.9, resetProb * 1.1);
        }
        
        return std::abs(currentPercentage - targetOmniscience) < 0.002;
    };
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> resetDist(0.0, 1.0);
    std::uniform_int_distribution<size_t> agentDist(0, numAgents - 1);
    std::uniform_int_distribution<size_t> traitDist(1, numTraits - 1);
    
    size_t iterationCount = 0;
    const size_t checkFrequency = 100;
    const size_t maxIterations = 100000;
    bool targetReached = false;
    
    resetProb = 0.2;
    
    while (!targetReached && iterationCount < maxIterations) {
        #pragma omp parallel
        {
            std::random_device thread_rd;
            std::mt19937 thread_gen(thread_rd());
            std::uniform_real_distribution<> thread_resetDist(0.0, 1.0);
            std::uniform_int_distribution<size_t> thread_agentDist(0, numAgents - 1);
            std::uniform_int_distribution<size_t> thread_traitDist(1, numTraits - 1);
            
            #pragma omp for
            for (size_t i = 0; i < numAgents; ++i) {
                size_t agentIndex = thread_agentDist(thread_gen);
                
                if (thread_resetDist(thread_gen) < resetProb) {
                    std::fill(repertoires[agentIndex].begin(), repertoires[agentIndex].end(), 0);
                    repertoires[agentIndex][0] = 1;
                    continue;
                }
                
                size_t traitToLearn = thread_traitDist(thread_gen);
                
                if (repertoires[agentIndex][traitToLearn] == 0) {
                    if (isLearnable(traitToLearn, agentIndex, thread_gen)) {
                        repertoires[agentIndex][traitToLearn] = 1;
                    }
                }
            }
        }
        
        iterationCount++;
        
        if (iterationCount % checkFrequency == 0) {
            targetReached = adjustResetProbability();
            
            if (iterationCount > 5000 && std::abs(countOmniscientAgents() / static_cast<double>(numAgents) - targetOmniscience) < 0.01) {
                for (size_t i = 0; i < numAgents / 10; ++i) {
                    size_t agentIndex = agentDist(gen);
                    
                    if (resetDist(gen) < resetProb) {
                        std::fill(repertoires[agentIndex].begin(), repertoires[agentIndex].end(), 0);
                        repertoires[agentIndex][0] = 1;
                    } else {
                        size_t traitToLearn = traitDist(gen);
                        if (repertoires[agentIndex][traitToLearn] == 0 && isLearnable(traitToLearn, agentIndex, gen)) {
                            repertoires[agentIndex][traitToLearn] = 1;
                        }
                    }
                    
                    if (i % 100 == 0) {
                        double currentPercentage = countOmniscientAgents() / static_cast<double>(numAgents);
                        if (std::abs(currentPercentage - targetOmniscience) < 0.001) {
                            targetReached = true;
                            break;
                        }
                    }
                }
            }
        }
    }
}

bool Population::isLearnable(size_t trait, size_t agentIndex, std::mt19937& gen) {
    const std::vector<size_t>& focalTraits = repertoires[agentIndex];
  
    if (dependency == Dependency::Fixed) {
        // For fixed dependency, use the pre-generated binary matrix
        for (size_t parent = 0; parent < adjMatrices[agentIndex].size(); ++parent) {
            if (adjMatrices[agentIndex][parent][trait] > 0.0 && focalTraits[parent] == 0) {
                return false; // Required prerequisite is missing
            }
        }
    } else {
        // For variable dependency, use probabilistic check with original weights
        std::uniform_real_distribution<> dist(0.0, 1.0);
        for (size_t parent = 0; parent < adjMatrix.size(); ++parent) {
            double weight = adjMatrix[parent][trait];
            if (weight > 0.0) {  // If there is any dependency
                if (focalTraits[parent] == 0) {  // Agent doesn't have the prerequisite
                    // Probabilistic check based on edge weight
                    if (dist(gen) < weight) {
                        return false; // Required prerequisite is missing
                    }
                }
            }
        }
    }
    return true;
}