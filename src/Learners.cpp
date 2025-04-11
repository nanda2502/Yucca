#include "Learners.hpp"
#include "Utils.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <omp.h>
#include <random>
#include <iostream>

std::vector<double>
generatePayoffs(const std::vector<std::vector<double>> &adjMatrix,
                const std::vector<size_t> &shuffleSequence) {
  size_t n = adjMatrix.size();
  size_t non_root_count = n - 1;
  std::vector<double> payoffs(n, 0.0);

  if (n == 121) {
    std::string filePath = "../data/payoffs_121.csv";
    std::ifstream file(filePath);

    if (!file.is_open()) {
      throw std::runtime_error("Could not open payoffs file: " + filePath);
    }

    std::string line;
    size_t index = 0;

    while (std::getline(file, line) && index < n) {
      try {
        payoffs[index++] = std::stod(line);
      } catch (const std::exception &e) {
        throw std::runtime_error("Error parsing payoff value at line " +
                                 std::to_string(index) + ": " + e.what());
      }
    }

    if (index < n) {
      throw std::runtime_error("Not enough payoff values in file: expected " +
                               std::to_string(n) + ", got " +
                               std::to_string(index));
    }

    return payoffs;
  }

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

Learners::Learners(const std::vector<std::vector<double>> &adjMatrix,
                   const std::vector<std::vector<size_t>> &demoRepertoires,
                   double slope, Strategy strategy,
                   const std::vector<size_t> &shuffleSequence) {
  size_t numAgents = 1000;

  this->adjMatrix = adjMatrix;
  this->traitPayoffs = generatePayoffs(adjMatrix, shuffleSequence);
  this->slope = slope;
  this->strategy = strategy;
  this->dependency = Dependency::Variable; // Default to variable dependency
  this->repertoires = std::vector<std::vector<size_t>>(
      numAgents, std::vector<size_t>(adjMatrix.size(), 0));
  // set the first trait to 1 for all agents
  for (auto &repertoire : repertoires) {
    repertoire[0] = 1;
  }

  // Initialize adjMatrices based on dependency type
  adjMatrices.resize(numAgents);
  if (dependency == Dependency::Fixed) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);

    for (size_t agent = 0; agent < numAgents; ++agent) {
      adjMatrices[agent].resize(adjMatrix.size(),
                                std::vector<double>(adjMatrix.size(), 0.0));
      for (size_t parent = 0; parent < adjMatrix.size(); ++parent) {
        for (size_t trait = 0; trait < adjMatrix.size(); ++trait) {
          if (adjMatrix[parent][trait] > 0.0) {
            adjMatrices[agent][parent][trait] =
                (dist(gen) < adjMatrix[parent][trait]) ? 1.0 : 0.0;
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

  traitFrequencies = calculateTraitFrequencies(demoRepertoires);
  stateFrequencies = calculateStateFrequencies(demoRepertoires);
  conformityBaseWeights = conformityWeights();
  payoffBaseWeights = payoffWeights();
  randomBaseWeights = traitFrequencies;
}

bool Learners::isLearnable(size_t trait, size_t agentIndex, std::mt19937 &gen) {
  const std::vector<size_t> &agentRepertoire = repertoires[agentIndex];

  if (dependency == Dependency::Fixed) {
    // For fixed dependency, use the pre-generated binary matrix
    for (size_t parent = 0; parent < adjMatrices[agentIndex].size(); ++parent) {
      if (adjMatrices[agentIndex][parent][trait] > 0.0 &&
          agentRepertoire[parent] == 0) {
        return false; // Required prerequisite is missing
      }
    }
  } else {
    // For variable dependency, use probabilistic check with original weights
    std::uniform_real_distribution<> dist(0.0, 1.0);
    for (size_t parent = 0; parent < adjMatrix.size(); ++parent) {
      double edgeWeight = adjMatrix[parent][trait];

      if (edgeWeight > 0.0) { // If there is any dependency relationship
        if (agentRepertoire[parent] ==
            0) { // Agent doesn't have the prerequisite
          // Determine probabilistically if this prerequisite is required
          // Higher edge weight means higher probability of it being required
          if (dist(gen) < edgeWeight) {
            return false; // Prerequisite is required but agent doesn't have it
          }
        }
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
      isLearnable(traitToLearn, agentIndex, gen)) {
    repertoires[agentIndex][traitToLearn] = 1;
    success = true;
  }
}

void Learners::learn() {
  size_t maxAge = adjMatrix.size() * 2;
  size_t numAgents = repertoires.size();
  size_t maxAttempts = maxAge; // Maximum learning attempts per agent

  meanPayoff.resize(maxAge, 0.0);
  meanSuccessRate.resize(maxAge, 0.0);

  // Vectors to track total payoffs and success counts for each age
  std::vector<double> totalPayoffs(maxAge, 0.0);
  std::vector<size_t> successCounts(maxAge, 0);
  std::vector<size_t> attemptCounts(maxAge, 0);

  // Using a vector of vectors to avoid race conditions during parallel
  // execution
  std::vector<std::vector<size_t>> threadCompletionTimes(omp_get_max_threads());

#pragma omp parallel
  {
    // Each thread will have its local accumulation variables
    std::vector<double> localTotalPayoffs(maxAge, 0.0);
    std::vector<size_t> localSuccessCounts(maxAge, 0);
    std::vector<size_t> localAttemptCounts(maxAge, 0);
    int threadId = omp_get_thread_num();

// Parallelize the loop over agents
#pragma omp for
    for (size_t agentIndex = 0; agentIndex < numAgents; agentIndex++) {
      // Create a thread-local random generator with a unique seed
      std::random_device rd;
      std::mt19937 gen(rd() +
                       agentIndex); // Add agentIndex to make seeds different

      size_t age = 0;
      bool isOmniscient = false;

      while (!isOmniscient && age < maxAttempts) {
        // Calculate and store current payoff for this age (only if age <
        // maxAge)
        if (age < maxAge) {
          double currentPayoff = calculateAgentPayoff(agentIndex);

          localTotalPayoffs[age] += currentPayoff;
          localAttemptCounts[age]++;
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
          weights = randomBaseWeights;
          break;
        default:
            throw std::invalid_argument("Invalid strategy");
        }

        if (weights.empty() || weights.size() != repertoires[agentIndex].size()) {
          std::cout << "Debug - Strategy: " << strategyToString(strategy) 
          << ", weights.size(): " << weights.size() 
          << ", repertoires[agentIndex].size(): " << repertoires[agentIndex].size() 
          << ", adjMatrix.size(): " << adjMatrix.size() << std::endl;
            throw std::runtime_error("Weights vector size mismatch");
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
          threadCompletionTimes[threadId].push_back(age);
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
              localSuccessCounts[age]++;
            }
          }
        } else {
          // No valid traits to learn, but not omniscient yet
          isOmniscient = true;
          threadCompletionTimes[threadId].push_back(age);
          break;
        }

        age++;
      }
    }

// Merge thread-local results into shared results
#pragma omp critical
    {
      for (size_t age = 0; age < maxAge; age++) {
        totalPayoffs[age] += localTotalPayoffs[age];
        successCounts[age] += localSuccessCounts[age];
        attemptCounts[age] += localAttemptCounts[age];
      }
    }
  }

  // Merge completion times from all threads
  std::vector<size_t> completionTimes;
  for (const auto &threadTimes : threadCompletionTimes) {
    completionTimes.insert(completionTimes.end(), threadTimes.begin(),
                           threadTimes.end());
  }

  // Compute mean metrics
  for (size_t age = 0; age < maxAge; age++) {
    if (attemptCounts[age] > 0) {
      meanPayoff[age] = totalPayoffs[age] / attemptCounts[age];
      meanSuccessRate[age] =
          static_cast<double>(successCounts[age]) / attemptCounts[age];
    }
  }

  // Calculate mean completion time only for agents who achieved omniscience
  if (!completionTimes.empty()) {
    meanCompletionTime =
        std::accumulate(completionTimes.begin(), completionTimes.end(), 0.0) /
        completionTimes.size();
  }
}