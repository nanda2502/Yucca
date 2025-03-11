#include "Learners.hpp"
#include <algorithm>
#include <numeric>
#include <random>

std::vector<double>
generatePayoffs(const std::vector<std::vector<size_t>> &adjMatrix,
                const std::vector<size_t> &shuffleSequence) {
  size_t n = adjMatrix.size();
  size_t non_root_count = n - 1;
  std::vector<double> payoffs(n, 0.0);

  // Root trait always has zero payoff
  payoffs[0] = 0.0;

  std::vector<double> non_root_payoffs(non_root_count);

  // equal spacing approach
  double spacing = 2.0 / (non_root_count + 1);
  for (size_t i = 0; i < non_root_count; ++i) {
    non_root_payoffs[i] = spacing * (i + 1);
  }

  // Apply shuffle sequence
  std::vector<double> shuffled_non_root_payoffs(non_root_count);
  for (size_t i = 0; i < non_root_count; ++i) {
    shuffled_non_root_payoffs[i] = non_root_payoffs[shuffleSequence[i]];
  }

  // Assign to payoffs vector
  size_t idx = 0;
  for (size_t trait = 1; trait < n; ++trait) {
    payoffs[trait] = shuffled_non_root_payoffs[idx++];
  }

  // Adjust non-root payoffs to maintain a mean of 1
  double sum = 0.0;
  size_t count = 0;
  for (size_t trait = 1; trait < n; ++trait) {
    sum += payoffs[trait];
    count++;
  }

  double mean = (count > 0) ? sum / count : 0.0;

  if (mean > 0.0) {
    // Scale all non-root payoffs to get a mean of 1.0
    double scaleFactor = 1.0 / mean;
    for (size_t trait = 1; trait < n; ++trait) {
      payoffs[trait] *= scaleFactor;
    }
  }

  return payoffs;
}

std::vector<double> calculateTraitFrequencies(
    const std::vector<std::vector<size_t>> &demoRepertoires) {
  if (demoRepertoires.empty()) {
    return {};
  }

  const size_t numTraits = demoRepertoires[0].size();
  const size_t numDemos = demoRepertoires.size();

  // Initialize the frequency vector
  std::vector<double> frequencies(numTraits, 0.0);

  // Count how many demonstrators have each trait
  for (size_t trait = 0; trait < numTraits; ++trait) {
    size_t count = 0;
    for (const auto &repertoire : demoRepertoires) {
      if (repertoire[trait] == 1) {
        count++;
      }
    }
    // Calculate frequency as proportion
    frequencies[trait] = static_cast<double>(count) / numDemos;
  }

  return frequencies;
}

std::unordered_map<std::vector<size_t>, double, RepertoireHash>
calculateStateFrequencies(
    const std::vector<std::vector<size_t>> &demoRepertoires) {

  const size_t numDemos = demoRepertoires.size();

  // Map to count occurrences of each unique state
  std::unordered_map<std::vector<size_t>, size_t, RepertoireHash> stateCounts;

  // Count occurrences of each unique repertoire
  for (const auto &repertoire : demoRepertoires) {
    stateCounts[repertoire]++;
  }

  // Convert counts to frequencies
  std::unordered_map<std::vector<size_t>, double, RepertoireHash> frequencies;
  for (const auto &[state, count] : stateCounts) {
    frequencies[state] = static_cast<double>(count) / numDemos;
  }

  return frequencies;
}

Learners::Learners(const std::vector<std::vector<size_t>> &adjMatrix,
                   const std::vector<std::vector<size_t>> &demoRepertoires,
                   double slope, Strategy strategy,
                   const std::vector<size_t> &shuffleSequence) {
  int numAgents = 1000;

  this->adjMatrix = adjMatrix;
  this->traitPayoffs = generatePayoffs(adjMatrix, shuffleSequence);
  this->slope = slope;
  this->strategy = strategy;
  this->repertoires = std::vector<std::vector<size_t>>(
      numAgents, std::vector<size_t>(adjMatrix.size(), 0));
  // set the first trait to 1 for all agents
  for (auto &repertoire : repertoires) {
    repertoire[0] = 1;
  }
  traitFrequencies = calculateTraitFrequencies(demoRepertoires);
  stateFrequencies = calculateStateFrequencies(demoRepertoires);
  conformityBaseWeights = conformityWeights();
  payoffBaseWeights = payoffWeights();
  RandomBaseWeights = std::vector<double>(adjMatrix.size(), 1.0);
};

bool Learners::isLearnable(size_t trait, size_t agentIndex) {
  const std::vector<size_t> &agentRepertoire = repertoires[agentIndex];

  // Check if the agent has all prerequisite traits
  for (size_t parent = 0; parent < adjMatrix.size(); ++parent) {
    if (adjMatrix[parent][trait] == 1) {  // parent is a prerequisite for trait
      if (agentRepertoire[parent] == 0) { // agent doesn't have the prerequisite
        return false;
      }
    }
  }
  return true;
}

double Learners::calculateAgentPayoff(size_t agentIndex) {
  const std::vector<size_t> &agentRepertoire = repertoires[agentIndex];
  double totalPayoff = 0.0;

  for (size_t trait = 0; trait < agentRepertoire.size(); ++trait) {
    if (agentRepertoire[trait] == 1) {
      totalPayoff += traitPayoffs[trait];
    }
  }

  return totalPayoff;
}

void Learners::updateRepertoire(std::vector<double> weights, size_t agentIndex,
                                std::mt19937 &gen, bool &success) {

  std::discrete_distribution<size_t> traitDist(weights.begin(), weights.end());
  size_t traitToLearn = traitDist(gen);

  success = false;
  if (repertoires[agentIndex][traitToLearn] == 0 &&
      isLearnable(traitToLearn, agentIndex)) {
    repertoires[agentIndex][traitToLearn] = 1;
    success = true;
  }
}

void Learners::learn() {
  size_t maxAge = adjMatrix.size() * 2;
  size_t numAgents = repertoires.size();

  meanPayoff.resize(maxAge, 0.0);
  meanSuccessRate.resize(maxAge, 0.0);

  // Vectors to track total payoffs and success counts for each age
  std::vector<double> totalPayoffs(maxAge, 0.0);
  std::vector<size_t> successCounts(maxAge, 0);
  std::vector<size_t> attemptCounts(maxAge, 0);
  std::vector<size_t> completionTimes;
  completionTimes.reserve(numAgents);

  for (size_t agentIndex = 0; agentIndex < numAgents; agentIndex++) {
    std::random_device rd;
    std::mt19937 gen(rd());

    size_t age = 0;
    bool isOmniscient = false;

    while (!isOmniscient) {
      // Calculate and store current payoff for this age (only if age < maxAge)
      if (age < maxAge) {
        double currentPayoff = calculateAgentPayoff(agentIndex);

        totalPayoffs[age] += currentPayoff;
        attemptCounts[age]++;
      }

      std::vector<double> weights;
      switch (strategy) {
      case Strategy::Prestige:
        weights = prestigeWeights(agentIndex);
        break;
      case Strategy::Proximal:
        weights = proximalWeights(agentIndex);
        break;
      case Strategy::Conformity:
        weights = conformityBaseWeights;
        break;
      case Strategy::Payoff:
        weights = payoffBaseWeights;
        break;
      case Strategy::Random:
        weights = RandomBaseWeights;
      }

      // Mask out already learned traits
      for (size_t i = 0; i < repertoires[agentIndex].size(); i++) {
        if (repertoires[agentIndex][i] == 1) {
          weights[i] = 0.0;
        }
      }

      // Check if all traits are learned (omniscient)
      if (std::all_of(repertoires[agentIndex].begin(),
                      repertoires[agentIndex].end(),
                      [](size_t trait) { return trait == 1; })) {
        isOmniscient = true;
        completionTimes.push_back(age);
        break;
      }

      // Normalize weights
      double total = std::accumulate(weights.begin(), weights.end(), 0.0);
      if (total > 0.0) {
        std::ranges::transform(weights, weights.begin(),
                               [total](double w) { return w / total; });

        // Try to learn a new trait
        bool success = false;
        updateRepertoire(weights, agentIndex, gen, success);

        // Track success rate (only if age < maxAge)
        if (age < maxAge) {

          if (success) {
            successCounts[age]++;
          }
        }
      } else {
        // No valid traits to learn, but not omniscient yet
        // This might happen if prerequisites make some traits unreachable
        isOmniscient = true; // Exit the loop

        completionTimes.push_back(age);
        break;
      }

      age++;
    }
  }

  // Compute mean metrics
  for (size_t age = 0; age < maxAge; age++) {
    if (attemptCounts[age] > 0) {
      meanPayoff[age] = totalPayoffs[age] / attemptCounts[age];
      meanSuccessRate[age] =
          attemptCounts[age] > 0
              ? static_cast<double>(successCounts[age]) / attemptCounts[age]
              : 0.0;
    }
  }

  // Calculate mean completion time
  if (!completionTimes.empty()) {
    meanCompletionTime =
        std::accumulate(completionTimes.begin(), completionTimes.end(), 0.0) /
        completionTimes.size();
  }
}