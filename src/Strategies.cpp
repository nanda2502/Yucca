#include "Learners.hpp"
#include <algorithm>
#include <cmath>



int computeDelta(const std::vector<size_t>& agentRepertoire, const std::vector<size_t>& demonstratorState) {
    int delta = 0;
    for (size_t i = 0; i < agentRepertoire.size(); ++i) {
        if (agentRepertoire[i] == 0 && demonstratorState[i] == 1) {
            delta++;
        }
    }
    return delta;
}

std::vector<double> Learners::proximalWeights(
    int agentIndex
) {
    // Get the repertoire for the specified agent
    const auto& repertoire = repertoires[agentIndex];
    
    // Check if we have cached results for this repertoire
    auto it = proximalWeightsCache.find(repertoire);
    if (it != proximalWeightsCache.end()) {
        return it->second;
    }
    
    std::vector<double> w_star(repertoire.size(), 0.0);
    
    // Loop over each trait
    for (size_t trait = 0; trait < repertoire.size(); ++trait) {
        if (repertoire[trait] == 0) { // Agent doesn't have this trait
            // Loop over all potential demonstrator states in the stateFrequencies map
            for (const auto& [state, frequency] : stateFrequencies) {
                if (state[trait] == 1) { // Demonstrator has this trait
                    auto delta = computeDelta(repertoire, state);
                    if (delta > 0) {
                        // Add to the weight using inverse of delta
                        w_star[trait] += frequency * std::pow(delta, -slope);
                    }
                }
            }
        }
    }
    
    // Cache the results before returning
    proximalWeightsCache[repertoire] = w_star;
    return w_star;
}

std::vector<double> Learners::prestigeWeights(
    int agentIndex
) {
    const auto& repertoire = repertoires[agentIndex];
    
    // Check if we have cached results for this repertoire
    auto it = prestigeWeightsCache.find(repertoire);
    if (it != prestigeWeightsCache.end()) {
        return it->second;
    }

    std::vector<double> w_star(repertoire.size(), 0.0);
    
    // First loop over all potential demonstrator states
    for (const auto& state_pair : stateFrequencies) {
        const auto& state = state_pair.first;
        const double frequency = state_pair.second;
        
        auto delta = computeDelta(repertoire, state);
        if (delta > 0) {  // State has at least one trait not in repertoire
            // Count total traits in the demonstrator state
            int totalTraits = 0;
            for (size_t val : state) {
                if (val == 1) {
                    totalTraits++;
                }
            }
            
            // Weight using total traits
            double stateWeight = frequency * std::pow(totalTraits, slope);
            
            // Then loop over traits to assign weights
            for (size_t trait = 0; trait < repertoire.size(); ++trait) {
                if (repertoire[trait] == 0 && state[trait] == 1) {  // Trait is unlearned by agent but present in demonstrator
                    w_star[trait] += stateWeight;
                }
            }
        }
    }

    // Cache the results before returning
    prestigeWeightsCache[repertoire] = w_star;
    return w_star;
}

std::vector<double> Learners::conformityWeights() {
    std::vector<double> w_star(traitFrequencies.size());
    
    std::ranges::transform(traitFrequencies, w_star.begin(), [slope = this->slope](double f) {return std::pow(f, slope);});

    return w_star;
}

std::vector<double> Learners::payoffWeights() {
    std::vector<double> w_star(traitPayoffs.size());
    std::ranges::transform(traitPayoffs, traitFrequencies, w_star.begin(),
        [slope = this->slope](double payoff, double traitFrequency) {
            return traitFrequency * std::pow(payoff, slope);
        });
    return w_star;
}