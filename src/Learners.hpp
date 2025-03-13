#ifndef LEARNERS_HPP
#define LEARNERS_HPP

#include <random>
#include <unordered_map>
#include <vector>
#include "Types.hpp"


struct RepertoireHash {
  size_t operator()(const std::vector<size_t> &repertoire) const {
    size_t hash = 0;
    for (size_t i = 0; i < repertoire.size(); ++i) {
      hash ^= std::hash<size_t>{}(repertoire[i]) + 0x9e3779b9 + (hash << 6) +
              (hash >> 2);
    }
    return hash;
  }
};

std::vector<double> generatePayoffs(const std::vector<std::vector<size_t>> &adjMatrix, const std::vector<size_t> &shuffleSequence);

std::unordered_map<std::vector<size_t>, double, RepertoireHash> calculateStateFrequencies(const std::vector<std::vector<size_t>> &demoRepertoires);

std::vector<double> calculateTraitFrequencies(const std::vector<std::vector<size_t>> &demoRepertoires); 

// Model learners that learn from background population, track the earned
// payoffs, success rate and completion times of the skill trees
class Learners {
private:
  std::vector<double> traitPayoffs;
  std::vector<std::vector<size_t>> adjMatrix;
  std::vector<std::vector<size_t>> repertoires;
  std::vector<double> traitFrequencies;
  std::unordered_map<std::vector<size_t>, double, RepertoireHash>
      stateFrequencies;
  double slope;
  Strategy strategy;

  // these weights do not change during learning
  std::vector<double> conformityBaseWeights;
  std::vector<double> payoffBaseWeights;
  std::vector<double> randomBaseWeights;

public:
  Learners(const std::vector<std::vector<size_t>> &adjMatrix,
           const std::vector<std::vector<size_t>> &demoRepertoires,
           double slope, Strategy strategy,
           const std::vector<size_t> &shuffleSequence);
  // learn function
  void learn();
  // outcome measures indexed by age of the agent
  std::vector<double> meanPayoff;
  std::vector<double> meanSuccessRate;
  double meanCompletionTime;

private:
  // learning strategies
  std::vector<double> prestigeWeights(int agentIndex);
  std::vector<double> proximalWeights(int agentIndex);
  std::vector<double> conformityWeights();
  std::vector<double> payoffWeights();

  // helper functions
  bool isLearnable(size_t trait, size_t agentIndex);
  void updateRepertoire(std::vector<double> weights, size_t agentIndex,
                        std::mt19937 &gen, bool &success);
  double calculateAgentPayoff(size_t agentIndex);
};

#endif // LEARNERS_HPP