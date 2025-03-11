#include "Population.hpp"

#include <random>

Population::Population(const std::vector<std::vector<size_t>>& adjMatrix) 
    : adjMatrix(adjMatrix), resetProb(0.2) // Start with a higher reset probability
{
    // Initialize repertoires for 10000 agents
    const size_t numAgents = 10000;
    const size_t numTraits = adjMatrix.size();
    
    // Create repertoires with all traits set to 0, except the first trait (index 0) set to 1
    repertoires.resize(numAgents, std::vector<size_t>(numTraits, 0));
    for (auto& repertoire : repertoires) {
        repertoire[0] = 1; // Initialize the first trait to 1 for all agents
    }
}

// Learn function implementation
void Population::learn() {
    const size_t numAgents = repertoires.size();
    const size_t numTraits = adjMatrix.size();
    
    // Keep track of agents who have all traits
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
    
    // Dynamic adjustment for reset probability
    auto adjustResetProbability = [this, &countOmniscientAgents, numAgents]() {
        size_t currentOmniscient = countOmniscientAgents();
        double currentPercentage = static_cast<double>(currentOmniscient) / numAgents;
        
        // More aggressive adjustment for reset probability based on difference from target
        if (currentPercentage < targetOmniscience * 0.9) {
            resetProb = std::max(0.001, resetProb * 0.9); // Decrease reset probability more aggressively
        } else if (currentPercentage < targetOmniscience) {
            resetProb = std::max(0.001, resetProb * 0.95); // Decrease reset probability slightly
        } else if (currentPercentage > targetOmniscience * 1.1) {
            resetProb = std::min(0.9, resetProb * 1.2);   // Increase reset probability more aggressively
        } else if (currentPercentage > targetOmniscience) {
            resetProb = std::min(0.9, resetProb * 1.1);   // Increase reset probability slightly
        }
        
        // Only return true when we're very close to the target
        return std::abs(currentPercentage - targetOmniscience) < 0.002; // Within 0.2% of target
    };
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> resetDist(0.0, 1.0);
    std::uniform_int_distribution<size_t> agentDist(0, numAgents - 1);
    std::uniform_int_distribution<size_t> traitDist(1, numTraits - 1); // Skip trait 0 as all agents already have it
    
    size_t iterationCount = 0;
    const size_t checkFrequency = 100; // Check progress more frequently
    const size_t maxIterations = 100000; // Safety cap on iterations
    bool targetReached = false;
    
    // Start with a much higher reset probability to prevent excessive omniscience
    resetProb = 0.2;
    
    while (!targetReached && iterationCount < maxIterations) {
        // Parallel execution could be done using OpenMP or similar
        #pragma omp parallel for
        for (size_t i = 0; i < numAgents; ++i) {
            // Each learning attempt applies to a randomly selected agent
            size_t agentIndex = agentDist(gen);
            
            // Reset check - this is crucial for controlling omniscience rate
            if (resetDist(gen) < resetProb) {
                // Reset the agent's repertoire, keeping only trait 0
                std::fill(repertoires[agentIndex].begin(), repertoires[agentIndex].end(), 0);
                repertoires[agentIndex][0] = 1;
                continue;
            }
            
            // Randomly select a trait to learn (excluding trait 0 which all agents already have)
            size_t traitToLearn = traitDist(gen);
            
            // Check if the agent already has this trait
            if (repertoires[agentIndex][traitToLearn] == 0) {
                // Check if the trait is learnable for this agent
                if (isLearnable(traitToLearn, agentIndex)) {
                    // Agent learns the trait
                    repertoires[agentIndex][traitToLearn] = 1;
                }
            }
        }
        
        iterationCount++;
        
        // Check progress more frequently
        if (iterationCount % checkFrequency == 0) {
            targetReached = adjustResetProbability();
            
            // If we're oscillating around the target, slow down the adjustment
            if (iterationCount > 5000 && std::abs(countOmniscientAgents() / static_cast<double>(numAgents) - targetOmniscience) < 0.01) {
                // Fine-tune with smaller learning batches
                for (size_t i = 0; i < numAgents / 10; ++i) {
                    size_t agentIndex = agentDist(gen);
                    
                    if (resetDist(gen) < resetProb) {
                        std::fill(repertoires[agentIndex].begin(), repertoires[agentIndex].end(), 0);
                        repertoires[agentIndex][0] = 1;
                    } else {
                        size_t traitToLearn = traitDist(gen);
                        if (repertoires[agentIndex][traitToLearn] == 0 && isLearnable(traitToLearn, agentIndex)) {
                            repertoires[agentIndex][traitToLearn] = 1;
                        }
                    }
                    
                    // Check if we've reached the target after each mini-batch
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

bool Population::isLearnable(size_t trait, size_t agentIndex) {
    const std::vector<size_t>& focalTraits = repertoires[agentIndex];

    for (size_t parent = 0; parent < adjMatrix[trait].size(); ++parent) {
        if (adjMatrix[parent][trait] == 1) {  
            if (focalTraits[parent] == 0) {  
                return false;
            }
        }
    }
    return true;
}; 