"""
Evolutionary optimization algorithms for CTRNN parameter evolution.

This module implements evolutionary algorithms commonly used in embodied cognition
and evolutionary robotics research:

1. Microbial Genetic Algorithm (Harvey 2009):
   - Extremely simple evolutionary model
   - Asexual reproduction with point mutations
   - Single "tournament" selection mechanism
   - Used in Beer's (2003) categorical perception and many robotic evolution studies
   
2. CMA-ES (Covariance Matrix Adaptation Evolution Strategy):
   - More sophisticated black-box optimizer
   - Good for continuous parameter optimization
   - Adaptation of step-size and covariance structure

References:
    Harvey, I. (2009). The microbial genetic algorithm. In Advances in Artificial Life
        (pp. 126-133). Springer, Berlin, Heidelberg.
    Beyer, H. G., & Schwefel, H. P. (2002). Evolution strategies–a comprehensive
        introduction. Natural Computing, 1(1), 3-52.
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class EvolutionStats:
    """Statistics collected during evolution."""
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    median_fitness: float
    worst_fitness: float


class MicrobialGA:
    """
    Microbial Genetic Algorithm for evolving CTRNN parameters.
    
    The algorithm maintains a population of candidate solutions. Each generation:
    1. Evaluate fitness of all individuals
    2. Select best individual (or tournament winner)
    3. Create mutation of best individual
    4. Replace worst individual with the mutant if mutant is better
    
    This simple algorithm has surprising properties:
    - Can escape local optima despite no sexual recombination
    - Performs well on multimodal fitness landscapes
    - Computationally efficient (only two evaluations per generation)
    - Biologically plausible (asexual reproduction)
    """
    
    def __init__(
        self,
        genotype_size: int,
        fitness_function: Callable[[np.ndarray], float],
        population_size: int = 100,
        mutation_std: float = 0.1,
        param_min: float = -5.0,
        param_max: float = 5.0,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize Microbial GA.
        
        Args:
            genotype_size: Dimensionality of parameter space.
            fitness_function: Function mapping genotype -> fitness (higher is better).
            population_size: Number of individuals in population.
            mutation_std: Standard deviation of Gaussian mutation.
            param_min: Minimum parameter value (for clipping).
            param_max: Maximum parameter value (for clipping).
            seed: Random seed for reproducibility.
        """
        self.genotype_size = genotype_size
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.mutation_std = mutation_std
        self.param_min = param_min
        self.param_max = param_max
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize population with random genotypes
        self.population = np.random.uniform(
            param_min, param_max, (population_size, genotype_size)
        )
        self.fitness = np.zeros(population_size)
        self.generation = 0
        self.history = []  # Track best fitness over time
        
        # Evaluate initial population
        self._evaluate_population()
    
    def _evaluate_population(self) -> None:
        """Evaluate fitness of all individuals in population."""
        for i in range(self.population_size):
            self.fitness[i] = self.fitness_function(self.population[i])
    
    def step(self) -> Tuple[np.ndarray, float]:
        """
        Execute one generation of microbial GA.
        
        Returns:
            best_genotype: Best individual in current population.
            best_fitness: Fitness of best individual.
        """
        # Find best and worst individuals
        best_idx = np.argmax(self.fitness)
        worst_idx = np.argmin(self.fitness)
        best_individual = self.population[best_idx].copy()
        best_fitness = self.fitness[best_idx]
        
        # Create mutation of best individual
        mutant = best_individual + self.mutation_std * np.random.randn(self.genotype_size)
        mutant = np.clip(mutant, self.param_min, self.param_max)
        
        # Evaluate mutant
        mutant_fitness = self.fitness_function(mutant)
        
        # Replace worst if mutant is better
        if mutant_fitness > self.fitness[worst_idx]:
            self.population[worst_idx] = mutant
            self.fitness[worst_idx] = mutant_fitness
            replaced = True
        else:
            replaced = False
        
        # Record statistics
        self.generation += 1
        stats = EvolutionStats(
            generation=self.generation,
            best_fitness=float(np.max(self.fitness)),
            mean_fitness=float(np.mean(self.fitness)),
            std_fitness=float(np.std(self.fitness)),
            median_fitness=float(np.median(self.fitness)),
            worst_fitness=float(np.min(self.fitness))
        )
        self.history.append(stats)
        
        return self.population[np.argmax(self.fitness)].copy(), stats.best_fitness
    
    def run(self, generations: int) -> List[EvolutionStats]:
        """
        Run evolutionary search for specified number of generations.
        
        Args:
            generations: Number of generations to run.
        
        Returns:
            List of EvolutionStats for each generation.
        """
        for _ in range(generations):
            self.step()
        return self.history
    
    def get_best_individual(self) -> np.ndarray:
        """Get best individual found so far."""
        return self.population[np.argmax(self.fitness)].copy()
    
    def get_best_fitness(self) -> float:
        """Get best fitness found so far."""
        return float(np.max(self.fitness))
    
    def get_population_snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current population and fitness values."""
        return self.population.copy(), self.fitness.copy()


class CMAES:
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    
    A more sophisticated evolutionary algorithm that adapts both the mean and
    covariance of the search distribution. Better for high-dimensional optimization
    and complex fitness landscapes.
    
    Note: This is a lightweight wrapper. For production use, consider
    pycma or cma packages.
    """
    
    def __init__(
        self,
        genotype_size: int,
        fitness_function: Callable[[np.ndarray], float],
        population_size: Optional[int] = None,
        initial_sigma: float = 1.0,
        param_min: float = -5.0,
        param_max: float = 5.0,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize CMA-ES optimizer.
        
        Args:
            genotype_size: Dimensionality of parameter space.
            fitness_function: Function mapping genotype -> fitness.
            population_size: Population size (defaults to 4 + floor(3*ln(n))).
            initial_sigma: Initial step-size.
            param_min: Minimum parameter value.
            param_max: Maximum parameter value.
            seed: Random seed.
        """
        self.genotype_size = genotype_size
        self.fitness_function = fitness_function
        self.param_min = param_min
        self.param_max = param_max
        
        if seed is not None:
            np.random.seed(seed)
        
        # Set default population size
        if population_size is None:
            self.population_size = max(4, int(4 + 3 * np.log(genotype_size)))
        else:
            self.population_size = population_size
        
        # CMA-ES parameters
        self.mean = np.zeros(genotype_size)
        self.sigma = initial_sigma
        self.cov = np.eye(genotype_size)
        self.lambda_param = self.population_size
        
        # Learning rates
        self.cc = (4 + self.lambda_param / genotype_size) / (
            4 + genotype_size + 2 * self.lambda_param / genotype_size
        )
        self.cs = (self.lambda_param + 2) / (genotype_size + self.lambda_param + 5)
        self.c1 = 2 / ((genotype_size + 1.3) ** 2 + self.lambda_param)
        self.cmu = min(self.lambda_param, int(self.lambda_param / 2)) / genotype_size
        
        # Evolution paths
        self.pc = np.zeros(genotype_size)
        self.ps = np.zeros(genotype_size)
        
        self.generation = 0
        self.history = []
    
    def step(self) -> Tuple[np.ndarray, float]:
        """
        Execute one generation of CMA-ES.
        
        Returns:
            best_genotype: Best individual in current population.
            best_fitness: Fitness of best individual.
        """
        # Sample population from N(mean, sigma^2 * cov)
        L = np.linalg.cholesky(self.cov + 1e-8 * np.eye(self.genotype_size))
        population = []
        fitness_values = []
        
        for _ in range(self.lambda_param):
            z = np.random.randn(self.genotype_size)
            x = self.mean + self.sigma * np.dot(L, z)
            x = np.clip(x, self.param_min, self.param_max)
            f = self.fitness_function(x)
            population.append(x)
            fitness_values.append(f)
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_values)[::-1]
        best_fitness = fitness_values[sorted_indices[0]]
        best_individual = population[sorted_indices[0]]
        
        # Select top mu individuals
        mu = int(self.lambda_param / 2)
        selected_pop = [population[i] for i in sorted_indices[:mu]]
        
        # Update mean
        old_mean = self.mean.copy()
        self.mean = np.mean(selected_pop, axis=0)
        
        # Update step-size and covariance (simplified CMA-ES update)
        self.ps = (1 - self.cs) * self.ps + \
                  np.sqrt(self.cs * (2 - self.cs)) * (self.mean - old_mean) / self.sigma
        
        self.pc = (1 - self.cc) * self.pc + \
                  np.sqrt(self.cc * (2 - self.cc)) * (self.mean - old_mean) / self.sigma
        
        # Adapt covariance matrix
        self.cov = (1 - self.c1 - self.cmu) * self.cov + \
                   self.c1 * (np.outer(self.pc, self.pc)) + \
                   self.cmu * np.sum([np.outer(selected_pop[i] - old_mean, selected_pop[i] - old_mean) 
                                      for i in range(mu)], axis=0) / (mu * self.sigma ** 2)
        
        # Adapt step-size (simplified)
        self.sigma *= np.exp((self.cs / 2) * (np.linalg.norm(self.ps) / np.sqrt(self.genotype_size) - 1))
        self.sigma = np.clip(self.sigma, 1e-6, 10.0)
        
        self.generation += 1
        stats = EvolutionStats(
            generation=self.generation,
            best_fitness=best_fitness,
            mean_fitness=float(np.mean(fitness_values)),
            std_fitness=float(np.std(fitness_values)),
            median_fitness=float(np.median(fitness_values)),
            worst_fitness=float(np.min(fitness_values))
        )
        self.history.append(stats)
        
        return np.array(best_individual), best_fitness
    
    def run(self, generations: int) -> List[EvolutionStats]:
        """Run for specified number of generations."""
        for _ in range(generations):
            self.step()
        return self.history


class GenotypeDecoder:
    """
    Convert genotype (flat parameter vector) to CTRNN parameters.
    
    Handles encoding/decoding of:
    - Neural time constants (tau)
    - Synaptic weights (w_ij)
    - Neural biases (theta)
    - Gains (optional)
    """
    
    def __init__(
        self,
        num_neurons: int,
        include_gains: bool = True,
        tau_range: Tuple[float, float] = (0.1, 10.0),
        weight_range: Tuple[float, float] = (-16.0, 16.0),
        bias_range: Tuple[float, float] = (-16.0, 16.0),
        gain_range: Tuple[float, float] = (0.1, 10.0)
    ) -> None:
        """
        Initialize genotype decoder.
        
        Args:
            num_neurons: Number of neurons in network.
            include_gains: Whether to include gain parameters.
            tau_range: Range for time constants (will be log-transformed).
            weight_range: Range for synaptic weights.
            bias_range: Range for neural biases.
            gain_range: Range for gains (will be log-transformed).
        """
        self.num_neurons = num_neurons
        self.include_gains = include_gains
        
        self.tau_range = tau_range
        self.weight_range = weight_range
        self.bias_range = bias_range
        self.gain_range = gain_range
        
        # Calculate genotype size
        self.num_tau = num_neurons
        self.num_weights = num_neurons * num_neurons
        self.num_biases = num_neurons
        self.num_gains = num_neurons if include_gains else 0
        
        self.genotype_size = (self.num_tau + self.num_weights + 
                             self.num_biases + self.num_gains)
        
        # Offset indices in genotype vector
        self.tau_start = 0
        self.tau_end = self.tau_start + self.num_tau
        self.weight_start = self.tau_end
        self.weight_end = self.weight_start + self.num_weights
        self.bias_start = self.weight_end
        self.bias_end = self.bias_start + self.num_biases
        self.gain_start = self.bias_end
        self.gain_end = self.gain_start + self.num_gains
    
    def decode(self, genotype: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert genotype vector to CTRNN parameter dictionary.
        
        Args:
            genotype: Flat parameter vector (shape: [genotype_size]).
        
        Returns:
            Dictionary with keys: 'tau', 'weights', 'biases', 'gains'.
        """
        assert genotype.shape == (self.genotype_size,)
        
        # Extract and rescale parameters
        # Time constants: use log-scale for better coverage
        tau_raw = genotype[self.tau_start:self.tau_end]
        tau_min, tau_max = self.tau_range
        tau = tau_min * np.exp(tau_raw * np.log(tau_max / tau_min))
        tau = np.clip(tau, tau_min, tau_max)  # Ensure tau stays in specified range
        
        # Weights: linear scale
        weights_raw = genotype[self.weight_start:self.weight_end]
        w_min, w_max = self.weight_range
        weights = w_min + (w_max - w_min) * (weights_raw + 1.0) / 2.0
        weights = weights.reshape((self.num_neurons, self.num_neurons))
        
        # Biases: linear scale
        biases_raw = genotype[self.bias_start:self.bias_end]
        b_min, b_max = self.bias_range
        biases = b_min + (b_max - b_min) * (biases_raw + 1.0) / 2.0
        
        # Gains: log scale
        result = {
            'tau': tau,
            'weights': weights,
            'biases': biases
        }
        
        if self.include_gains:
            gains_raw = genotype[self.gain_start:self.gain_end]
            g_min, g_max = self.gain_range
            gains = g_min * np.exp(gains_raw * np.log(g_max / g_min))
            result['gains'] = gains
        
        return result
    
    def encode(
        self,
        tau: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        biases: Optional[np.ndarray] = None,
        gains: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert CTRNN parameters to genotype vector.
        
        Args:
            tau, weights, biases, gains: CTRNN parameter arrays.
        
        Returns:
            Genotype vector.
        """
        genotype = np.zeros(self.genotype_size)
        
        if tau is not None:
            tau_min, tau_max = self.tau_range
            tau_raw = np.log(tau / tau_min) / np.log(tau_max / tau_min)
            genotype[self.tau_start:self.tau_end] = tau_raw
        
        if weights is not None:
            w_min, w_max = self.weight_range
            weights_raw = (weights - w_min) / (w_max - w_min) * 2.0 - 1.0
            genotype[self.weight_start:self.weight_end] = weights_raw.flatten()
        
        if biases is not None:
            b_min, b_max = self.bias_range
            biases_raw = (biases - b_min) / (b_max - b_min) * 2.0 - 1.0
            genotype[self.bias_start:self.bias_end] = biases_raw
        
        if gains is not None and self.include_gains:
            g_min, g_max = self.gain_range
            gains_raw = np.log(gains / g_min) / np.log(g_max / g_min)
            genotype[self.gain_start:self.gain_end] = gains_raw
        
        return genotype


class NoveltySearch:
    """
    Novelty Search evolutionary algorithm for open-ended exploration.
    
    Unlike traditional fitness-based evolution which optimizes for a specific objective,
    novelty search maintains an archive of interesting behavioral characterizations
    and drives evolution to discover new behaviors in the behavior space.
    
    Key concepts:
    1. Behavioral characterization: A low-dimensional descriptor of behavior
       (e.g., final position, trajectory statistics, sensor activation patterns)
    2. Archive: Collection of past behavioral characterizations to measure novelty against
    3. Novelty metric: Average distance to k-nearest neighbors in archive space
    4. Evolution pressure: High novelty (far from archive) gets high fitness
    
    Wimsattian rationale for robustness:
    - Novelty search avoids fitness deceptiveness and local optima
    - Open-ended exploration discovers robust solutions (they survive without selection pressure)
    - Behavior space is more interpretable than fitness landscape (more modular, compositional)
    - Archive creates redundancy: multiple behaviors achieve similar functions
    - Robustness emerges from searching in intrinsically meaningful space
    
    References:
        Lehman, J., & Stanley, K. O. (2011). Abandoning objectives: Evolution through
            the search for novelty alone. Evolutionary computation, 19(2), 189-223.
        Pugh, J. K., Soros, L. B., & Stanley, K. O. (2016). Quality diversity: A new
            frontier for evolutionary computation. Frontiers in Robotics and AI, 3, 40.
    """
    
    def __init__(
        self,
        genotype_size: int,
        behavior_characterization_fn: Callable[[np.ndarray], np.ndarray],
        fitness_function: Optional[Callable[[np.ndarray], float]] = None,
        population_size: int = 100,
        archive_addition_threshold: float = 0.5,
        k_nearest_neighbors: int = 15,
        mutation_std: float = 0.1,
        param_min: float = -5.0,
        param_max: float = 5.0,
        fitness_weight: float = 0.0,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize Novelty Search.
        
        Args:
            genotype_size: Dimensionality of parameter space.
            behavior_characterization_fn: Function mapping genotype -> behavior_vector.
                                         Behavior vector should be low-dimensional (e.g., 2-20D).
                                         Example: lambda g: np.array([final_x, final_y, max_speed])
            fitness_function: Optional fitness function for hybrid novelty+fitness search.
                             If None, pure novelty search (fitness_weight is ignored).
            population_size: Number of individuals in population.
            archive_addition_threshold: Minimum novelty required to add to archive.
            k_nearest_neighbors: Number of nearest neighbors for novelty computation.
            mutation_std: Standard deviation of Gaussian mutation.
            param_min: Minimum parameter value (for clipping).
            param_max: Maximum parameter value (for clipping).
            fitness_weight: Weight for combining novelty and fitness (0=pure novelty, 1=pure fitness).
                           If > 0, uses weighted sum: (1-w)*novelty + w*fitness
            seed: Random seed for reproducibility.
        """
        self.genotype_size = genotype_size
        self.behavior_characterization_fn = behavior_characterization_fn
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.archive_addition_threshold = archive_addition_threshold
        self.k_nearest_neighbors = min(k_nearest_neighbors, population_size)
        self.mutation_std = mutation_std
        self.param_min = param_min
        self.param_max = param_max
        self.fitness_weight = fitness_weight
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize population with random genotypes
        self.population = np.random.uniform(
            param_min, param_max, (population_size, genotype_size)
        )
        
        # Compute initial behaviors and novelty
        self.behaviors = np.zeros((population_size, 1))  # Will be resized on first evaluation
        self.novelty = np.zeros(population_size)
        self.fitness = np.zeros(population_size) if fitness_function is not None else None
        
        # Archive of interesting behaviors
        self.archive = []  # List of behavior vectors
        self.archive_genotypes = []  # Corresponding genotypes (optional, for reconstruction)
        
        self.generation = 0
        self.history = []
        self._evaluate_population()
    
    def _evaluate_population(self) -> None:
        """Evaluate behaviors and novelty of all individuals in population."""
        behaviors_list = []
        
        for i in range(self.population_size):
            behavior = self.behavior_characterization_fn(self.population[i])
            if np.isscalar(behavior):
                behavior = np.array([behavior])
            behaviors_list.append(behavior)
        
        # Stack behaviors (handle variable-length behaviors by padding)
        max_behavior_len = max(len(b) for b in behaviors_list)
        padded_behaviors = []
        for b in behaviors_list:
            if len(b) < max_behavior_len:
                b = np.concatenate([b, np.zeros(max_behavior_len - len(b))])
            padded_behaviors.append(b)
        
        self.behaviors = np.array(padded_behaviors)
        
        # Compute novelty
        for i in range(self.population_size):
            self.novelty[i] = self._compute_novelty(self.behaviors[i])
        
        # Optionally compute fitness
        if self.fitness_function is not None:
            for i in range(self.population_size):
                self.fitness[i] = self.fitness_function(self.population[i])
    
    def _compute_novelty(self, behavior: np.ndarray) -> float:
        """
        Compute novelty as average distance to k-nearest neighbors in archive.
        
        If archive is small, compute against population as fallback.
        
        Args:
            behavior: Behavior vector to compute novelty for.
        
        Returns:
            Novelty score (higher = more novel).
        """
        if len(self.archive) > 0:
            # Compute distances to archive
            archive_array = np.array(self.archive)
            distances = np.linalg.norm(archive_array - behavior, axis=1)
            
            # Average distance to k-nearest neighbors
            k = min(self.k_nearest_neighbors, len(self.archive))
            novelty = float(np.mean(np.partition(distances, min(k-1, len(distances)-1))[:k]))
        else:
            # Fallback: compute against population
            distances = np.linalg.norm(self.behaviors - behavior, axis=1)
            k = min(self.k_nearest_neighbors, len(self.behaviors))
            # Remove self (distance 0)
            distances = distances[distances > 1e-6]
            if len(distances) > 0:
                novelty = float(np.mean(np.partition(distances, min(k-1, len(distances)-1))[:k]))
            else:
                novelty = 0.0
        
        return novelty
    
    def step(self) -> Tuple[np.ndarray, float]:
        """
        Execute one generation of novelty search.
        
        Returns:
            best_genotype: Best (most novel or fittest) individual.
            best_score: Novelty or fitness score of best individual.
        """
        # Determine selection criteria
        if self.fitness_weight > 0.0 and self.fitness is not None:
            # Hybrid: combine novelty and fitness
            # Normalize both to [0, 1] before combining
            novelty_norm = (self.novelty - np.min(self.novelty)) / (np.max(self.novelty) - np.min(self.novelty) + 1e-6)
            fitness_norm = (self.fitness - np.min(self.fitness)) / (np.max(self.fitness) - np.min(self.fitness) + 1e-6)
            scores = (1.0 - self.fitness_weight) * novelty_norm + self.fitness_weight * fitness_norm
        else:
            # Pure novelty search
            scores = self.novelty
        
        # Find best individual
        best_idx = np.argmax(scores)
        best_genotype = self.population[best_idx].copy()
        best_score = scores[best_idx]
        best_novelty = self.novelty[best_idx]
        
        # Update archive: add best individual if sufficiently novel
        if best_novelty > self.archive_addition_threshold:
            self.archive.append(self.behaviors[best_idx].copy())
            self.archive_genotypes.append(best_genotype.copy())
        
        # Create mutant from best individual
        mutant = best_genotype + self.mutation_std * np.random.randn(self.genotype_size)
        mutant = np.clip(mutant, self.param_min, self.param_max)
        
        # Evaluate mutant
        mutant_behavior = self.behavior_characterization_fn(mutant)
        if np.isscalar(mutant_behavior):
            mutant_behavior = np.array([mutant_behavior])
        
        # Pad mutant behavior to match current behavior dimensionality
        if len(mutant_behavior) < len(self.behaviors[0]):
            mutant_behavior = np.concatenate([
                mutant_behavior,
                np.zeros(len(self.behaviors[0]) - len(mutant_behavior))
            ])
        
        mutant_novelty = self._compute_novelty(mutant_behavior)
        
        # Replace worst individual if mutant is more novel
        worst_idx = np.argmin(self.novelty)
        if mutant_novelty > self.novelty[worst_idx]:
            self.population[worst_idx] = mutant
            self.behaviors[worst_idx] = mutant_behavior
            self.novelty[worst_idx] = mutant_novelty
            
            if self.fitness_function is not None:
                self.fitness[worst_idx] = self.fitness_function(mutant)
        
        # Record statistics
        self.generation += 1
        stats = EvolutionStats(
            generation=self.generation,
            best_fitness=float(best_score),
            mean_fitness=float(np.mean(scores)),
            std_fitness=float(np.std(scores)),
            median_fitness=float(np.median(scores)),
            worst_fitness=float(np.min(scores))
        )
        self.history.append(stats)
        
        return best_genotype, best_score
    
    def run(self, generations: int) -> List[EvolutionStats]:
        """
        Run novelty search for specified number of generations.
        
        Args:
            generations: Number of generations to run.
        
        Returns:
            List of EvolutionStats for each generation.
        """
        for _ in range(generations):
            self.step()
        return self.history
    
    def get_best_individual(self) -> np.ndarray:
        """
        Get best individual found so far (by novelty or fitness).
        
        Returns:
            Best genotype.
        """
        if self.fitness_weight > 0.0 and self.fitness is not None:
            novelty_norm = (self.novelty - np.min(self.novelty)) / (np.max(self.novelty) - np.min(self.novelty) + 1e-6)
            fitness_norm = (self.fitness - np.min(self.fitness)) / (np.max(self.fitness) - np.min(self.fitness) + 1e-6)
            scores = (1.0 - self.fitness_weight) * novelty_norm + self.fitness_weight * fitness_norm
            best_idx = np.argmax(scores)
        else:
            best_idx = np.argmax(self.novelty)
        
        return self.population[best_idx].copy()
    
    def get_best_fitness(self) -> float:
        """
        Get best score (novelty or fitness-weighted score).
        
        Returns:
            Best score value.
        """
        if self.fitness_weight > 0.0 and self.fitness is not None:
            novelty_norm = (self.novelty - np.min(self.novelty)) / (np.max(self.novelty) - np.min(self.novelty) + 1e-6)
            fitness_norm = (self.fitness - np.min(self.fitness)) / (np.max(self.fitness) - np.min(self.fitness) + 1e-6)
            scores = (1.0 - self.fitness_weight) * novelty_norm + self.fitness_weight * fitness_norm
            return float(np.max(scores))
        else:
            return float(np.max(self.novelty))
    
    def get_archive(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get the archive of novel behaviors and corresponding genotypes.
        
        Returns:
            (archive_behaviors, archive_genotypes) tuple.
        """
        return self.archive, self.archive_genotypes
    
    def get_population_snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current population and novelty values."""
        return self.population.copy(), self.novelty.copy()
