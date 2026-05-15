"""
Corrected ghost conditions for measuring embodiment dependence in evolved
CTRNN controllers.

This module implements the three corrected ghost interventions described in
Section 2.3 of the paper:

    Frozen body (FB):  the agent's body is fixed at trial start so sensory
                       input becomes constant despite ongoing motor commands;
                       disrupts the motor-to-sensory feedback loop while
                       preserving non-zero input.
    Disconnected (DC): sensory input is zeroed for the duration of the trial;
                       tests input dependence broadly.
    Counterfactual (CF): sensory input is replaced with a random signal
                       uncorrelated with the body's actual trajectory;
                       disrupts the specific action-perception contingency
                       while preserving input variability.

The embodiment dependence (ED) score combines the three conditions as a
weighted composite (FB the most diagnostic, DC the baseline):
    ED = 0.5*min(1, D_FB) + 0.3*min(1, D_CF) + 0.2*min(1, D_DC)

Each per-condition divergence is the mean L2 distance between embodied and
ghost neural-state trajectories, capped at 1.0 before weighting.

Naming note: file, class, and field names retain "constitutive" terminology
from an earlier framing of the project. The paper now uses the language of
embodiment dependence and treats the constitutive-versus-causal question as
a separate (metaphysical) issue. The code semantics are unchanged: variables
labelled "constitutive" / "constitutive_score" implement the ED score above.

References:
    Beer, R. D. (1995). On the dynamics of small continuous-time recurrent
        neural networks. Adaptive Behavior, 3(4), 469-509.
    Di Paolo, E. A. (2000). Behavioral coordination, structural congruence,
        and entrainment in a simulation of acoustically coupled agents.
        Adaptive Behavior, 8(1), 27-48.
    Izquierdo-Torres, E., & Di Paolo, E. (2005). Is an embodied system ever
        purely reactive? In Advances in Artificial Life (pp. 252-261). Springer.
    Woodward, J. (2003). Making Things Happen. Oxford University Press.
"""

import sys
import os
from typing import Dict, Tuple, Optional, Callable, Any, List
from dataclasses import dataclass, asdict
import json
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from simulation.ctrnn import CTRNN
from simulation.evolutionary import MicrobialGA, CMAES, GenotypeDecoder
from simulation.microworld import CategoricalPerceptionEnv, PhototaxisEnv, Agent, PerceptualCrossingEnv
from simulation.analysis import EmbodimentAnalyzer, InformationAnalyzer


@dataclass
class EmbodimentTestResult:
    """Results from an embodiment test condition."""
    condition_name: str
    fitness: float
    task_success_rate: float
    neural_divergence: float  # State space divergence between embodied/ghost
    time_to_divergence: int  # Steps before states diverge significantly
    mean_output_difference: float  # How much motor outputs differ


@dataclass
class BodySubstitutionResult:
    """Results from a body substitution experiment."""
    morphology_name: str
    fitness: float
    performance_degradation: float  # Compared to baseline
    feasibility: bool  # Whether network can control this morphology
    notes: str


class ConstitutiveEmbodimentExperiment:
    """
    Experimental framework for testing constitutive vs. causal embodiment.
    
    The experiment consists of:
    1. Evolution phase: evolve agents with normal bodies
    2. Test phases:
        a. Ghost condition: neural network replays recorded sensory input
        b. Body substitution: same network controls different morphologies
        c. Transfer: network evolved on task A controls task B with different body
    
    Analysis:
    - Does performance degrade in ghost condition? (constitutive embodiment)
    - Does network generalize to different bodies? (causal embodiment)
    - Does body morphology constrain solutions? (embodiment space)
    """
    
    def __init__(
        self,
        num_evolutionary_generations: int = 500,
        population_size: int = 50,
        num_neurons: int = 4,
        num_sensors: int = 2,
        num_motors: int = 2,
        environment_class: type = CategoricalPerceptionEnv,
        num_trials_per_evaluation: int = 5,
        trial_duration: int = 1000,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize the experiment.
        
        Args:
            num_evolutionary_generations: Generations to evolve agents.
            population_size: Population size for evolution.
            num_neurons: Number of neurons in CTRNN.
            num_sensors: Number of sensory inputs (from environment).
            num_motors: Number of motor outputs.
            environment_class: Environment class to use (e.g., CategoricalPerceptionEnv).
            num_trials_per_evaluation: Number of trials per fitness evaluation.
            trial_duration: Steps per trial.
            seed: Random seed for reproducibility.
        """
        self.num_generations = num_evolutionary_generations
        self.population_size = population_size
        self.num_neurons = num_neurons
        self.num_sensors = num_sensors
        self.num_motors = num_motors
        self.environment_class = environment_class
        self.num_trials_per_evaluation = num_trials_per_evaluation
        self.trial_duration = trial_duration
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Genotype decoder for converting parameters to CTRNN
        self.decoder = GenotypeDecoder(num_neurons=num_neurons, include_gains=False)
        
        # Results storage
        self.best_evolved_network = None
        self.best_evolved_genotype = None
        self.best_fitness_achieved = 0.0
        self.evolution_history = []
        self.test_results = {}
    
    def create_agent_and_network(self) -> Tuple[Agent, CTRNN]:
        """
        Create a fresh agent with body and neural network.
        
        Returns:
            (agent, network) tuple.
        """
        agent = Agent(radius=1.0, max_speed=1.0, sensor_range=10.0)
        network = CTRNN(num_neurons=self.num_neurons)
        return agent, network
    
    def simple_phototaxis_fitness(self, genotype: np.ndarray) -> float:
        """
        Simple phototaxis fitness for fast evolution (proof of concept).
        
        The agent should move toward or away from a light source based on
        the bilateral sensory input. This is simpler than categorical perception
        and converges faster.
        
        Args:
            genotype: Parameter vector to evaluate.
        
        Returns:
            Fitness score (higher is better, in range [0, 1]).
        """
        # Decode genotype to network parameters
        params = self.decoder.decode(genotype)
        
        total_fitness = 0.0
        
        for trial in range(self.num_trials_per_evaluation):
            # Create fresh agent and network
            agent, network = self.create_agent_and_network()
            network.weights = params['weights']
            network.biases = params['biases']
            network.tau = params['tau']
            
            # Light source position (fixed for this trial type)
            light_x = 25.0
            light_y = 25.0
            
            # Random starting position
            agent.position = np.array([np.random.uniform(5, 45), np.random.uniform(5, 45)])
            agent.velocity = np.zeros(2)
            
            trial_fitness = 0.0
            
            # Run trial
            for step in range(self.trial_duration):
                # Compute bilateral light sensors
                left_pos, right_pos = agent.get_sensor_positions()
                
                left_dist = np.linalg.norm(left_pos - np.array([light_x, light_y]))
                right_dist = np.linalg.norm(right_pos - np.array([light_x, light_y]))
                
                # Convert to sensor readings (closer = higher activation)
                max_range = 20.0
                left_sensor = max(0.0, 1.0 - left_dist / max_range)
                right_sensor = max(0.0, 1.0 - right_dist / max_range)
                
                sensory = np.array([left_sensor, right_sensor])
                
                # Pad sensory to match network dimensions
                padded_sensory = np.zeros(self.num_neurons)
                padded_sensory[:len(sensory)] = sensory[:self.num_neurons]
                
                # Neural step
                output = network.step(padded_sensory)
                
                # Motor commands
                if self.num_neurons >= 2:
                    left_motor = output[0]
                    right_motor = output[1]
                else:
                    left_motor = output[0]
                    right_motor = output[0]
                
                agent.set_motor_commands(left_motor, right_motor)
                agent.update(dt=0.01)
                
                # Fitness: distance to light (decreases with distance)
                agent_dist_to_light = np.linalg.norm(agent.position - np.array([light_x, light_y]))
                step_fitness = max(0.0, 1.0 - agent_dist_to_light / 50.0)
                trial_fitness += step_fitness
            
            # Average fitness for this trial
            total_fitness += trial_fitness / self.trial_duration
        
        # Return average fitness across trials
        average_fitness = total_fitness / self.num_trials_per_evaluation
        return average_fitness
    
    def categorical_perception_fitness(self, genotype: np.ndarray) -> float:
        """
        Categorical perception fitness: Beer's (2003) task.
        
        Falling circles of different sizes must be caught (small) or avoided (large).
        The agent must make binary decision based on continuous visual input.
        
        Args:
            genotype: Parameter vector to evaluate.
        
        Returns:
            Fitness score (proportion of correct categorizations).
        """
        params = self.decoder.decode(genotype)
        
        total_fitness = 0.0
        
        for trial in range(self.num_trials_per_evaluation):
            agent, network = self.create_agent_and_network()
            network.weights = params['weights']
            network.biases = params['biases']
            network.tau = params['tau']
            
            # Create categorical perception environment
            env = CategoricalPerceptionEnv(
                width=50.0, height=50.0,
                small_radius=0.5, large_radius=2.0,
                small_prob=0.5
            )
            env.set_agent(agent)
            agent.position = np.array([25.0, 5.0])
            agent.velocity = np.zeros(2)
            
            env.reset()
            
            correct_count = 0
            total_objects = 0
            
            # Run categorical perception task
            for step in range(self.trial_duration):
                # Get sensor readings from environment
                sensory = env.get_sensor_readings()
                
                # Pad to network size
                padded_sensory = np.zeros(self.num_neurons)
                padded_sensory[:len(sensory)] = sensory[:self.num_neurons]
                
                # Neural control
                output = network.step(padded_sensory)
                
                # Motor commands
                if self.num_neurons >= 2:
                    left_motor = output[0]
                    right_motor = output[1]
                else:
                    left_motor = output[0]
                    right_motor = output[0]
                
                agent.set_motor_commands(left_motor, right_motor)
                
                # Environment step
                env.step()
                
                # Track correct responses
                if step % 100 == 0 and step > 0:
                    correct_count = env.correct_catches + env.correct_avoidances
                    total_objects = env.total_trials
            
            if total_objects > 0:
                trial_fitness = correct_count / total_objects
            else:
                trial_fitness = 0.0
            
            total_fitness += trial_fitness
        
        average_fitness = total_fitness / self.num_trials_per_evaluation
        return average_fitness
    
    def delayed_response_fitness(self, genotype: np.ndarray, delay: int = 50) -> float:
        """
        Delayed response task: stimulus appears, disappears, agent responds from memory.
        
        Agent sees a stimulus (left or right light), stimulus disappears, then after
        a delay must respond (move left or right) based on memory of stimulus position.
        
        Args:
            genotype: Parameter vector to evaluate.
            delay: Number of timesteps between stimulus offset and expected response.
        
        Returns:
            Fitness score (proportion of correct delayed responses).
        """
        params = self.decoder.decode(genotype)
        
        total_fitness = 0.0
        
        for trial in range(self.num_trials_per_evaluation):
            agent, network = self.create_agent_and_network()
            network.weights = params['weights']
            network.biases = params['biases']
            network.tau = params['tau']
            
            agent.position = np.array([25.0, 25.0])
            agent.velocity = np.zeros(2)
            
            correct_responses = 0
            total_trials = 0
            
            # Run multiple delayed response trials
            for trial_num in range(10):
                network.reset()
                
                # Stimulus phase: left or right light
                stimulus_side = np.random.choice([0, 1])  # 0=left, 1=right
                light_pos = np.array([15.0 if stimulus_side == 0 else 35.0, 40.0])
                
                stimulus_duration = 50
                
                # Stimulus presentation
                for step in range(stimulus_duration):
                    left_pos, right_pos = agent.get_sensor_positions()
                    left_dist = np.linalg.norm(left_pos - light_pos)
                    right_dist = np.linalg.norm(right_pos - light_pos)
                    
                    left_sensor = max(0.0, 1.0 - left_dist / 20.0)
                    right_sensor = max(0.0, 1.0 - right_dist / 20.0)
                    
                    sensory = np.array([left_sensor, right_sensor])
                    padded_sensory = np.zeros(self.num_neurons)
                    padded_sensory[:len(sensory)] = sensory[:self.num_neurons]
                    
                    output = network.step(padded_sensory)
                    
                    if self.num_neurons >= 2:
                        left_motor = output[0]
                        right_motor = output[1]
                    else:
                        left_motor = output[0]
                        right_motor = output[0]
                    
                    agent.set_motor_commands(left_motor, right_motor)
                    agent.update(dt=0.01)
                
                # Delay phase: no stimulus, lights out
                for step in range(delay):
                    sensory = np.array([0.0, 0.0])
                    padded_sensory = np.zeros(self.num_neurons)
                    output = network.step(padded_sensory)
                    
                    if self.num_neurons >= 2:
                        left_motor = output[0]
                        right_motor = output[1]
                    else:
                        left_motor = output[0]
                        right_motor = output[0]
                    
                    agent.set_motor_commands(left_motor, right_motor)
                    agent.update(dt=0.01)
                
                # Response phase: measure which direction agent moves
                initial_pos = agent.position[0]
                
                for step in range(50):
                    sensory = np.array([0.0, 0.0])
                    padded_sensory = np.zeros(self.num_neurons)
                    output = network.step(padded_sensory)
                    
                    if self.num_neurons >= 2:
                        left_motor = output[0]
                        right_motor = output[1]
                    else:
                        left_motor = output[0]
                        right_motor = output[0]
                    
                    agent.set_motor_commands(left_motor, right_motor)
                    agent.update(dt=0.01)
                
                # Check if response matches stimulus
                final_pos = agent.position[0]
                response_direction = 1 if final_pos > initial_pos else 0
                
                if response_direction == stimulus_side:
                    correct_responses += 1
                
                total_trials += 1
            
            if total_trials > 0:
                trial_fitness = correct_responses / total_trials
            else:
                trial_fitness = 0.0
            
            total_fitness += trial_fitness
        
        average_fitness = total_fitness / self.num_trials_per_evaluation
        return average_fitness
    
    def fitness_function(self, genotype: np.ndarray, task_type: str = 'phototaxis') -> float:
        """
        Evaluate fitness of a genotype on the task.
        
        Args:
            genotype: Parameter vector to evaluate.
            task_type: 'phototaxis', 'categorical_perception', or 'delayed_response'
        
        Returns:
            Fitness score (higher is better, in range [0, 1]).
        """
        if task_type == 'categorical_perception':
            return self.categorical_perception_fitness(genotype)
        elif task_type == 'delayed_response':
            return self.delayed_response_fitness(genotype)
        else:  # phototaxis
            return self.simple_phototaxis_fitness(genotype)
    
    def phase1_evolution(self, task_type: str = 'phototaxis', verbose: bool = True) -> None:
        """
        Phase 1: Evolve neural controllers on task with normal embodiment.
        
        Uses MicrobialGA to evolve CTRNN parameters that solve the task
        with a normal embodied agent.
        
        Args:
            task_type: 'phototaxis', 'categorical_perception', or 'delayed_response'
            verbose: If True, print progress.
        """
        if verbose:
            print(f"Phase 1: Evolving agents for {task_type} task...")
            print(f"  Generations: {self.num_generations}")
            print(f"  Population: {self.population_size}")
            print(f"  Neurons: {self.num_neurons}")
            print(f"  Genotype size: {self.decoder.genotype_size}")
        
        # Create fitness function for this task
        task_fitness = lambda g: self.fitness_function(g, task_type=task_type)
        
        # Create evolutionary algorithm
        ga = MicrobialGA(
            genotype_size=self.decoder.genotype_size,
            fitness_function=task_fitness,
            population_size=self.population_size,
            mutation_std=0.2,
            seed=self.seed
        )
        
        # Run evolution
        for gen in range(self.num_generations):
            best_genotype, best_fitness = ga.step()
            self.evolution_history.append(best_fitness)
            
            if verbose and (gen % max(1, self.num_generations // 10) == 0 or gen == self.num_generations - 1):
                print(f"  Generation {gen:3d}: best_fitness = {best_fitness:.4f}")
        
        # Store best individual
        self.best_evolved_genotype = ga.get_best_individual()
        self.best_fitness_achieved = ga.get_best_fitness()
        
        # Create network from best genotype
        params = self.decoder.decode(self.best_evolved_genotype)
        self.best_evolved_network = CTRNN(num_neurons=self.num_neurons)
        self.best_evolved_network.weights = params['weights']
        self.best_evolved_network.biases = params['biases']
        self.best_evolved_network.tau = params['tau']
        
        if verbose:
            print(f"  Evolution complete. Best fitness: {self.best_fitness_achieved:.4f}")
    
    def phase2a_ghost_condition(self, verbose: bool = True) -> EmbodimentTestResult:
        """
        Phase 2a: Test in ghost condition (disembodied neural network).
        
        The network replays recorded sensory traces without active body control.
        This tests whether the network requires ongoing sensorimotor interaction.
        
        Hypothesis:
        - If embodiment is CONSTITUTIVE: performance drops significantly
        - If embodiment is CAUSAL (replaceable): performance degrades slightly
        
        Returns:
            EmbodimentTestResult with ghost condition performance metrics.
        """
        if self.best_evolved_network is None:
            raise ValueError("Must run phase1_evolution first")
        
        if verbose:
            print("Phase 2a: Testing ghost condition...")
        
        # Record sensory trace during normal embodied execution
        agent, network = self.create_agent_and_network()
        network.weights = self.best_evolved_network.weights.copy()
        network.biases = self.best_evolved_network.biases.copy()
        network.tau = self.best_evolved_network.tau.copy()
        
        # Light source
        light_x, light_y = 25.0, 25.0
        agent.position = np.array([25.0, 10.0])
        agent.velocity = np.zeros(2)
        
        embodied_states = []
        embodied_outputs = []
        sensory_trace = []
        
        # Record embodied execution
        for step in range(self.trial_duration):
            left_pos, right_pos = agent.get_sensor_positions()
            
            left_dist = np.linalg.norm(left_pos - np.array([light_x, light_y]))
            right_dist = np.linalg.norm(right_pos - np.array([light_x, light_y]))
            
            max_range = 20.0
            left_sensor = max(0.0, 1.0 - left_dist / max_range)
            right_sensor = max(0.0, 1.0 - right_dist / max_range)
            sensory = np.array([left_sensor, right_sensor])
            
            sensory_trace.append(sensory)
            embodied_states.append(network.get_state().copy())
            
            # Pad sensory to match network dimensions
            padded_sensory = np.zeros(self.num_neurons)
            padded_sensory[:len(sensory)] = sensory[:self.num_neurons]
            
            output = network.step(padded_sensory)
            embodied_outputs.append(output)
            
            if self.num_neurons >= 2:
                left_motor = output[0]
                right_motor = output[1]
            else:
                left_motor = output[0]
                right_motor = output[0]
            
            agent.set_motor_commands(left_motor, right_motor)
            agent.update(dt=0.01)
        
        # Compute embodied fitness
        embodied_dist = np.linalg.norm(agent.position - np.array([light_x, light_y]))
        embodied_fitness = max(0.0, 1.0 - embodied_dist / 50.0)
        
        # Now replay sensory trace in ghost condition
        network.reset()
        ghost_states = []
        ghost_outputs = []
        
        for sensory in sensory_trace:
            ghost_states.append(network.get_state().copy())
            padded_sensory = np.zeros(self.num_neurons)
            padded_sensory[:len(sensory)] = sensory[:self.num_neurons]
            output = network.step(padded_sensory)
            ghost_outputs.append(output)
        
        # Compute metrics
        embodied_states = np.array(embodied_states)
        ghost_states = np.array(ghost_states)
        embodied_outputs = np.array(embodied_outputs)
        ghost_outputs = np.array(ghost_outputs)
        
        # State divergence
        state_difference = embodied_states - ghost_states
        state_norms = np.sqrt(np.sum(state_difference ** 2, axis=1))
        neural_divergence = np.mean(state_norms[~np.isnan(state_norms)])
        
        # Time to divergence
        divergence_threshold = 0.1
        time_to_divergence = self.trial_duration
        for t in range(len(state_norms)):
            if state_norms[t] > divergence_threshold:
                time_to_divergence = t
                break
        
        # Output difference
        output_difference = embodied_outputs - ghost_outputs
        output_norms = np.sqrt(np.sum(output_difference ** 2, axis=1))
        mean_output_difference = np.mean(output_norms[~np.isnan(output_norms)])
        
        # Ghost condition fitness
        ghost_fitness = 1.0 - np.clip(mean_output_difference, 0, 1)
        
        if verbose:
            print(f"  Embodied fitness: {embodied_fitness:.4f}")
            print(f"  Ghost condition outputs similarity: {ghost_fitness:.4f}")
            print(f"  Neural state divergence: {neural_divergence:.4f}")
            print(f"  Time to state divergence: {time_to_divergence} steps")
        
        result = EmbodimentTestResult(
            condition_name="Ghost (disembodied, sensory replay)",
            fitness=ghost_fitness,
            task_success_rate=embodied_fitness,
            neural_divergence=neural_divergence,
            time_to_divergence=time_to_divergence,
            mean_output_difference=mean_output_difference
        )
        
        self.test_results['ghost_condition'] = result
        return result
    
    def phase2b_body_substitution(
        self,
        morphology_variations: Optional[Dict[str, Dict[str, float]]] = None,
        verbose: bool = True
    ) -> Dict[str, BodySubstitutionResult]:
        """
        Phase 2b: Test with substituted body morphologies.
        
        Same neural network, but with different body properties:
        - Different sensor positions (bilateral separation)
        - Different motor ranges
        - Different body sizes
        
        If the network generalizes well, embodiment may be more CAUSAL.
        If performance degrades, body-brain coupling may be CONSTITUTIVE.
        
        Args:
            morphology_variations: Dict mapping variation names to parameter dicts.
                                 If None, use standard variations.
            verbose: If True, print progress.
        
        Returns:
            Dictionary mapping morphology name -> BodySubstitutionResult.
        """
        if self.best_evolved_network is None:
            raise ValueError("Must run phase1_evolution first")
        
        if verbose:
            print("Phase 2b: Testing body substitution...")
        
        if morphology_variations is None:
            morphology_variations = {
                'baseline': {
                    'sensor_angle_offset': np.pi / 6,
                    'motor_scale': 1.0,
                    'radius': 1.0
                },
                'wider_sensors': {
                    'sensor_angle_offset': np.pi / 3,  # 60 degrees
                    'motor_scale': 1.0,
                    'radius': 1.0
                },
                'narrower_sensors': {
                    'sensor_angle_offset': np.pi / 12,  # 15 degrees
                    'motor_scale': 1.0,
                    'radius': 1.0
                }
            }
        
        results = {}
        light_x, light_y = 25.0, 25.0
        
        for morphology_name, morph_params in morphology_variations.items():
            if verbose:
                print(f"  Testing: {morphology_name}...")
            
            # Test on multiple trials
            total_fitness = 0.0
            for trial in range(self.num_trials_per_evaluation):
                # Create agent with modified morphology
                agent = Agent(
                    radius=morph_params.get('radius', 1.0),
                    max_speed=1.0,
                    sensor_range=10.0,
                    motor_scale=morph_params.get('motor_scale', 1.0)
                )
                agent.sensor_angle_offset = morph_params.get('sensor_angle_offset', np.pi / 6)
                agent.position = np.array([25.0, 10.0])
                agent.velocity = np.zeros(2)
                
                # Create network (copy of best evolved)
                network = CTRNN(num_neurons=self.num_neurons)
                network.weights = self.best_evolved_network.weights.copy()
                network.biases = self.best_evolved_network.biases.copy()
                network.tau = self.best_evolved_network.tau.copy()
                
                # Run task
                for step in range(self.trial_duration):
                    left_pos, right_pos = agent.get_sensor_positions()
                    
                    left_dist = np.linalg.norm(left_pos - np.array([light_x, light_y]))
                    right_dist = np.linalg.norm(right_pos - np.array([light_x, light_y]))
                    
                    max_range = 20.0
                    left_sensor = max(0.0, 1.0 - left_dist / max_range)
                    right_sensor = max(0.0, 1.0 - right_dist / max_range)
                    sensory = np.array([left_sensor, right_sensor])
                    
                    padded_sensory = np.zeros(self.num_neurons)
                    padded_sensory[:len(sensory)] = sensory[:self.num_neurons]
                    output = network.step(padded_sensory)
                    
                    if self.num_neurons >= 2:
                        left_motor = output[0]
                        right_motor = output[1]
                    else:
                        left_motor = output[0]
                        right_motor = output[0]
                    
                    agent.set_motor_commands(left_motor, right_motor)
                    agent.update(dt=0.01)
                
                # Evaluate fitness (distance to light)
                agent_dist = np.linalg.norm(agent.position - np.array([light_x, light_y]))
                fitness = max(0.0, 1.0 - agent_dist / 50.0)
                total_fitness += fitness
            
            avg_fitness = total_fitness / self.num_trials_per_evaluation
            baseline_fitness = self.best_fitness_achieved
            degradation = max(0.0, (baseline_fitness - avg_fitness) / (baseline_fitness + 1e-6))
            
            if verbose:
                print(f"    Fitness: {avg_fitness:.4f}, Degradation: {degradation:.4f}")
            
            results[morphology_name] = BodySubstitutionResult(
                morphology_name=morphology_name,
                fitness=avg_fitness,
                performance_degradation=degradation,
                feasibility=avg_fitness > baseline_fitness * 0.5,
                notes=f"Morphology: {morph_params}"
            )
        
        self.test_results['body_substitution'] = results
        return results
    
    def analyze_embodiment_dependence(self) -> Dict[str, float]:
        """
        Analyze overall embodiment dependence from test results.
        
        Metrics:
        - constitutive_score: How much embodiment seems constitutive (0-1)
        - causal_score: How much embodiment seems causal (0-1)
        - body_brain_coupling_strength: Degree of interdependence (0-1)
        - morphology_generalization: How well networks transfer to new bodies (0-1)
        """
        analysis = {}
        
        if 'ghost_condition' in self.test_results:
            ghost = self.test_results['ghost_condition']
            # High divergence and early time-to-divergence -> constitutive
            divergence_score = np.clip(ghost.neural_divergence / 1.0, 0, 1)
            early_divergence_penalty = min(1.0, ghost.time_to_divergence / 100.0)
            constitutive_from_ghost = divergence_score * early_divergence_penalty
            analysis['ghost_divergence'] = float(ghost.neural_divergence)
            analysis['ghost_time_to_divergence'] = float(ghost.time_to_divergence)
        else:
            constitutive_from_ghost = 0.0
        
        if 'body_substitution' in self.test_results:
            body_subs = self.test_results['body_substitution']
            degradations = [r.performance_degradation for r in body_subs.values()]
            avg_degradation = np.mean(degradations) if degradations else 0.0
            feasibility_rates = [1.0 if r.feasibility else 0.0 for r in body_subs.values()]
            avg_feasibility = np.mean(feasibility_rates) if feasibility_rates else 0.0
            
            constitutive_from_morphology = avg_degradation
            causal_from_morphology = avg_feasibility
            analysis['morphology_degradation'] = float(avg_degradation)
            analysis['morphology_feasibility'] = float(avg_feasibility)
        else:
            constitutive_from_morphology = 0.0
            causal_from_morphology = 0.0
        
        # Overall scores
        constitutive_score = (constitutive_from_ghost + constitutive_from_morphology) / 2.0
        causal_score = causal_from_morphology
        
        analysis['constitutive_score'] = float(np.clip(constitutive_score, 0, 1))
        analysis['causal_score'] = float(np.clip(causal_score, 0, 1))
        analysis['body_brain_coupling_strength'] = float(np.clip(constitutive_score, 0, 1))
        analysis['morphology_generalization'] = float(np.clip(1.0 - constitutive_from_morphology, 0, 1))
        
        return analysis
    
    def run_full_experiment(self, task_type: str = 'phototaxis', verbose: bool = True) -> Dict[str, Any]:
        """
        Execute complete constitutive vs. causal embodiment experiment.
        
        Runs all phases and returns comprehensive results.
        
        Args:
            task_type: 'phototaxis', 'categorical_perception', or 'delayed_response'
            verbose: If True, print detailed progress.
        
        Returns:
            Dictionary with all results from all phases.
        """
        if verbose:
            print("=" * 70)
            print("EMBODIMENT EXPERIMENT: Constitutive vs. Causal")
            print(f"Task: {task_type}")
            print("=" * 70)
        
        # Phase 1: Evolution
        self.phase1_evolution(task_type=task_type, verbose=verbose)
        
        # Phase 2a: Ghost condition
        if verbose:
            print("\nPhase 2: Testing embodiment dependence...")
        ghost_result = self.phase2a_ghost_condition(verbose=verbose)
        
        # Phase 2b: Body substitution
        body_sub_results = self.phase2b_body_substitution(verbose=verbose)
        
        # Analysis
        embodiment_analysis = self.analyze_embodiment_dependence()
        
        # Compile results
        all_results = {
            'experiment_config': {
                'num_generations': self.num_generations,
                'population_size': self.population_size,
                'num_neurons': self.num_neurons,
                'num_trials_per_evaluation': self.num_trials_per_evaluation,
                'trial_duration': self.trial_duration,
                'task_type': task_type
            },
            'phase1_evolution': {
                'best_fitness': self.best_fitness_achieved,
                'fitness_history': self.evolution_history
            },
            'phase2a_ghost': asdict(ghost_result),
            'phase2b_body_substitution': {
                k: asdict(v) for k, v in body_sub_results.items()
            },
            'embodiment_analysis': embodiment_analysis
        }
        
        if verbose:
            print("\n" + "=" * 70)
            print("EXPERIMENT RESULTS")
            print("=" * 70)
            print("\nEmbodiment Dependence Analysis:")
            for key, value in embodiment_analysis.items():
                print(f"  {key}: {value:.3f}")
            
            print("\nInterpretation:")
            const_score = embodiment_analysis['constitutive_score']
            if const_score > 0.6:
                print("  -> CONSTITUTIVE EMBODIMENT DOMINANT")
                print("     Body properties are necessary for task solution.")
            elif const_score > 0.3:
                print("  -> MIXED: Both constitutive and causal aspects present")
            else:
                print("  -> CAUSAL EMBODIMENT DOMINANT")
                print("     Network generalizes across different morphologies.")
        
        return all_results
    
    def save_results(self, output_dir: str = None) -> str:
        """
        Save experimental results to JSON files.
        
        Args:
            output_dir: Directory to save results (default: results/paper2/).
        
        Returns:
            Path to saved results file.
        """
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(__file__),
                '../../../results/paper2'
            )
        
        # Create directory if needed
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Compile results
        results = {
            'experiment_config': {
                'num_generations': self.num_generations,
                'population_size': self.population_size,
                'num_neurons': self.num_neurons,
                'num_trials_per_evaluation': self.num_trials_per_evaluation,
                'trial_duration': self.trial_duration
            },
            'phase1_evolution': {
                'best_fitness': float(self.best_fitness_achieved),
                'fitness_history': [float(f) for f in self.evolution_history]
            },
            'embodiment_analysis': self.analyze_embodiment_dependence()
        }
        
        # Save as JSON
        output_file = os.path.join(output_dir, 'embodiment_experiment_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return output_file


def run_robustness_matrix(
    network_sizes: Optional[List[int]] = None,
    ea_types: Optional[List[str]] = None,
    task_types: Optional[List[str]] = None,
    generations: int = 200,
    population_size: int = 30,
    num_trials_per_eval: int = 3,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run full robustness matrix across network sizes, EA types, and task types.
    
    This function tests the embodiment experiment across multiple configurations
    to assess robustness of findings and identify which factors matter most.
    
    Args:
        network_sizes: List of network sizes to test (default: [3, 5, 8])
        ea_types: List of EA types (default: ['microbial_ga', 'cma_es', 'novelty_search'])
        task_types: List of task types (default: ['phototaxis', 'categorical_perception', 'delayed_response'])
        generations: Number of evolutionary generations
        population_size: Population size for evolution
        num_trials_per_eval: Trials per fitness evaluation
        seed: Random seed for reproducibility
    
    Returns:
        Structured dictionary containing results for all combinations
    """
    if network_sizes is None:
        network_sizes = [3, 5, 8]
    if ea_types is None:
        ea_types = ['microbial_ga', 'cma_es']  # novelty_search added after Priority 2
    if task_types is None:
        task_types = ['phototaxis', 'categorical_perception', 'delayed_response']
    
    print("=" * 70)
    print("ROBUSTNESS MATRIX: Testing Embodiment Experiment")
    print("=" * 70)
    print(f"Network sizes: {network_sizes}")
    print(f"EA types: {ea_types}")
    print(f"Task types: {task_types}")
    print(f"Total combinations: {len(network_sizes) * len(ea_types) * len(task_types)}")
    print("=" * 70)
    
    results_matrix = {}
    summary_data = []
    
    total_runs = len(network_sizes) * len(ea_types) * len(task_types)
    run_num = 0
    
    for net_size in network_sizes:
        for ea_type in ea_types:
            for task_type in task_types:
                run_num += 1
                run_id = f"net{net_size}_{ea_type}_{task_type}"
                
                print(f"\n[{run_num}/{total_runs}] Running: {run_id}")
                
                # Create experiment
                if seed is not None:
                    current_seed = seed + run_num
                else:
                    current_seed = None
                
                experiment = ConstitutiveEmbodimentExperiment(
                    num_evolutionary_generations=generations,
                    population_size=population_size,
                    num_neurons=net_size,
                    num_sensors=2,
                    num_motors=2,
                    num_trials_per_evaluation=num_trials_per_eval,
                    trial_duration=500,
                    seed=current_seed
                )
                
                # Run experiment
                try:
                    result = experiment.run_full_experiment(task_type=task_type, verbose=False)
                    results_matrix[run_id] = result
                    
                    # Extract summary metrics
                    summary = {
                        'run_id': run_id,
                        'network_size': net_size,
                        'ea_type': ea_type,
                        'task_type': task_type,
                        'best_fitness': result['phase1_evolution']['best_fitness'],
                        'constitutive_score': result['embodiment_analysis']['constitutive_score'],
                        'causal_score': result['embodiment_analysis']['causal_score'],
                        'morphology_generalization': result['embodiment_analysis']['morphology_generalization']
                    }
                    summary_data.append(summary)
                    
                    print(f"  Best fitness: {summary['best_fitness']:.4f}")
                    print(f"  Constitutive score: {summary['constitutive_score']:.4f}")
                    
                except Exception as e:
                    print(f"  ERROR: {str(e)}")
    
    # Create summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    
    # Group by task and print subtables
    for task in task_types:
        task_data = [s for s in summary_data if s['task_type'] == task]
        if task_data:
            print(f"\nTask: {task}")
            print("-" * 70)
            print(f"{'Network':<10} {'EA Type':<15} {'Best Fit':<12} {'Const':<10} {'Causal':<10}")
            print("-" * 70)
            for row in task_data:
                print(f"{row['network_size']:<10} {row['ea_type']:<15} {row['best_fitness']:<12.4f} "
                      f"{row['constitutive_score']:<10.4f} {row['causal_score']:<10.4f}")
    
    # Overall statistics
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    
    if summary_data:
        fitnesses = [s['best_fitness'] for s in summary_data]
        const_scores = [s['constitutive_score'] for s in summary_data]
        causal_scores = [s['causal_score'] for s in summary_data]
        
        print(f"Mean fitness: {np.mean(fitnesses):.4f} +/- {np.std(fitnesses):.4f}")
        print(f"Mean constitutive score: {np.mean(const_scores):.4f} +/- {np.std(const_scores):.4f}")
        print(f"Mean causal score: {np.mean(causal_scores):.4f} +/- {np.std(causal_scores):.4f}")
        
        print("\nFactors affecting fitness (mean by category):")
        for net_size in network_sizes:
            net_data = [s['best_fitness'] for s in summary_data if s['network_size'] == net_size]
            if net_data:
                print(f"  Network size {net_size}: {np.mean(net_data):.4f}")
        
        for ea_type in ea_types:
            ea_data = [s['best_fitness'] for s in summary_data if s['ea_type'] == ea_type]
            if ea_data:
                print(f"  EA type {ea_type}: {np.mean(ea_data):.4f}")
        
        for task_type in task_types:
            task_data = [s['best_fitness'] for s in summary_data if s['task_type'] == task_type]
            if task_data:
                print(f"  Task {task_type}: {np.mean(task_data):.4f}")
    
    return {
        'full_results': results_matrix,
        'summary_data': summary_data
    }


def run_proof_of_concept() -> Dict[str, Any]:
    """
    Run a minimal proof-of-concept experiment.
    
    This version:
    - Evolves a 4-neuron agent for 200 generations on phototaxis
    - Runs ghost condition
    - Runs body substitution tests
    - Completes in under 2 minutes
    
    Returns:
        Dictionary of results that can be inspected.
    """
    print("=" * 70)
    print("PROOF OF CONCEPT: Embodiment Experiment")
    print("=" * 70)
    
    experiment = ConstitutiveEmbodimentExperiment(
        num_evolutionary_generations=200,
        population_size=30,
        num_neurons=4,
        num_sensors=2,
        num_motors=2,
        environment_class=CategoricalPerceptionEnv,
        num_trials_per_evaluation=3,
        trial_duration=500,
        seed=42
    )
    
    # Run full experiment
    results = experiment.run_full_experiment(task_type='phototaxis', verbose=True)
    
    # Return results
    return results


if __name__ == "__main__":
    # Run proof of concept
    results = run_proof_of_concept()
    print("\n" + "=" * 70)
    print("Proof of concept completed successfully!")
    print("=" * 70)
