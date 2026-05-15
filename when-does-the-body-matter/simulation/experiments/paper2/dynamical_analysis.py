"""
Dynamical Analysis of Evolved Networks for Paper 2

Addresses reviewer concern: "The capacity explanation is asserted rather than
demonstrated. Compute Lyapunov exponents, bifurcation structure, or attractor
dimensionality to show that larger networks actually maintain more complex
attractor landscapes."

This script:
1. Re-evolves networks at 3, 5, 8 neurons (quick, 2000 gen)
2. Analyzes the evolved dynamics:
   - Lyapunov exponents (positive = chaotic, near-zero = limit cycle, negative = fixed point)
   - Attractor dimensionality (estimated via correlation dimension of neural trajectories)
   - Number of distinct behavioral modes (clustering of phase-space trajectories)
   - Sensitivity to initial conditions (perturbation analysis)
3. Compares across network sizes to test the capacity hypothesis
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from simulation.ctrnn import CTRNN
from simulation.evolutionary import MicrobialGA, GenotypeDecoder
from simulation.microworld import Agent

# Import the fitness function from the corrected experiment
from simulation.experiments.paper2.phase_a_expanded import (
    phototaxis_fitness, run_embodied_trial, AGENT_DT, AGENT_MAX_SPEED,
    SENSOR_RANGE, ARENA_SIZE, LIGHT_POSITIONS_EVOLUTION
)


def evolve_network(num_neurons, generations=2000, population_size=50, seed=42):
    """Evolve a phototaxis controller and return the best parameters."""
    np.random.seed(seed)
    decoder = GenotypeDecoder(
        num_neurons=num_neurons,
        include_gains=False,
        tau_range=(0.5, 5.0),
        weight_range=(-10.0, 10.0),
        bias_range=(-10.0, 10.0),
    )
    fitness_fn = lambda g: phototaxis_fitness(g, decoder, num_neurons, num_trials=4)

    ea = MicrobialGA(
        genotype_size=decoder.genotype_size,
        fitness_function=fitness_fn,
        population_size=population_size,
        mutation_std=0.2,
        seed=seed
    )

    best_ever_fitness = -np.inf
    best_ever_genotype = None
    for gen in range(generations):
        best_geno, best_fit = ea.step()
        if best_fit > best_ever_fitness:
            best_ever_fitness = best_fit
            best_ever_genotype = best_geno.copy()

    return decoder.decode(best_ever_genotype), best_ever_fitness


def compute_lyapunov_spectrum(params, num_neurons, trial_duration=2000, num_light_positions=4):
    """
    Estimate Lyapunov exponents of the embodied agent-environment system.

    Uses the standard approach: track how small perturbations grow over time
    in the coupled neural-body-environment system.
    """
    lyapunov_estimates = []

    for lp_idx in range(num_light_positions):
        light_pos = np.array(LIGHT_POSITIONS_EVOLUTION[lp_idx])

        # Run baseline trajectory
        agent = Agent(radius=1.0, max_speed=AGENT_MAX_SPEED, sensor_range=SENSOR_RANGE)
        network = CTRNN(num_neurons=num_neurons)
        network.weights = params['weights'].copy()
        network.biases = params['biases'].copy()
        network.tau = params['tau'].copy()
        agent.position = np.array([25.0, 25.0])
        agent.velocity = np.zeros(2)

        # Warm up
        for _ in range(100):
            left_pos, right_pos = agent.get_sensor_positions()
            left_dist = np.linalg.norm(left_pos - light_pos)
            right_dist = np.linalg.norm(right_pos - light_pos)
            padded = np.zeros(num_neurons)
            padded[:2] = [max(0.0, 1.0 - left_dist / SENSOR_RANGE),
                          max(0.0, 1.0 - right_dist / SENSOR_RANGE)]
            output = network.step(padded)
            agent.set_motor_commands(output[0], output[1] if num_neurons >= 2 else output[0])
            agent.update(dt=AGENT_DT)

        # Save state after warmup
        base_state = network.get_state().copy()
        base_pos = agent.position.copy()
        base_angle = agent.angle

        # Perturbation analysis: apply small perturbation to neural state
        epsilon = 1e-6
        lyap_sum = 0.0
        n_steps = 0

        for dim in range(num_neurons):
            # Reset to base state
            network_base = CTRNN(num_neurons=num_neurons)
            network_base.weights = params['weights'].copy()
            network_base.biases = params['biases'].copy()
            network_base.tau = params['tau'].copy()
            network_base.set_state(base_state.copy())

            network_pert = CTRNN(num_neurons=num_neurons)
            network_pert.weights = params['weights'].copy()
            network_pert.biases = params['biases'].copy()
            network_pert.tau = params['tau'].copy()
            perturbed_state = base_state.copy()
            perturbed_state[dim] += epsilon
            network_pert.set_state(perturbed_state)

            agent_base = Agent(radius=1.0, max_speed=AGENT_MAX_SPEED, sensor_range=SENSOR_RANGE)
            agent_base.position = base_pos.copy()
            agent_base.angle = base_angle
            agent_base.velocity = np.zeros(2)

            agent_pert = Agent(radius=1.0, max_speed=AGENT_MAX_SPEED, sensor_range=SENSOR_RANGE)
            agent_pert.position = base_pos.copy()
            agent_pert.angle = base_angle
            agent_pert.velocity = np.zeros(2)

            # Run both for some steps and measure divergence
            for step in range(trial_duration):
                # Base agent
                lp_b, rp_b = agent_base.get_sensor_positions()
                ld_b = np.linalg.norm(lp_b - light_pos)
                rd_b = np.linalg.norm(rp_b - light_pos)
                pad_b = np.zeros(num_neurons)
                pad_b[:2] = [max(0.0, 1.0 - ld_b / SENSOR_RANGE),
                             max(0.0, 1.0 - rd_b / SENSOR_RANGE)]
                out_b = network_base.step(pad_b)
                agent_base.set_motor_commands(out_b[0], out_b[1] if num_neurons >= 2 else out_b[0])
                agent_base.update(dt=AGENT_DT)

                # Perturbed agent
                lp_p, rp_p = agent_pert.get_sensor_positions()
                ld_p = np.linalg.norm(lp_p - light_pos)
                rd_p = np.linalg.norm(rp_p - light_pos)
                pad_p = np.zeros(num_neurons)
                pad_p[:2] = [max(0.0, 1.0 - ld_p / SENSOR_RANGE),
                             max(0.0, 1.0 - rd_p / SENSOR_RANGE)]
                out_p = network_pert.step(pad_p)
                agent_pert.set_motor_commands(out_p[0], out_p[1] if num_neurons >= 2 else out_p[0])
                agent_pert.update(dt=AGENT_DT)

            # Measure final divergence
            state_diff = np.linalg.norm(network_base.get_state() - network_pert.get_state())
            if state_diff > 0 and epsilon > 0:
                lyap = np.log(state_diff / epsilon) / (trial_duration * AGENT_DT)
                lyap_sum += lyap
                n_steps += 1

        if n_steps > 0:
            lyapunov_estimates.append(lyap_sum / n_steps)

    return {
        'max_lyapunov': float(np.max(lyapunov_estimates)) if lyapunov_estimates else 0.0,
        'mean_lyapunov': float(np.mean(lyapunov_estimates)) if lyapunov_estimates else 0.0,
        'std_lyapunov': float(np.std(lyapunov_estimates)) if lyapunov_estimates else 0.0,
        'per_condition': [float(x) for x in lyapunov_estimates],
    }


def compute_trajectory_complexity(params, num_neurons, trial_duration=1000, num_positions=4):
    """
    Measure trajectory complexity in neural state space.

    Computes:
    1. Trajectory entropy (how spread out the trajectory is in state space)
    2. Effective dimensionality (PCA of neural state trajectories)
    3. Autocorrelation timescale (how quickly the system decorrelates)
    """
    all_states = []

    for lp_idx in range(num_positions):
        light_pos = np.array(LIGHT_POSITIONS_EVOLUTION[lp_idx])
        s_trace, n_states, n_outputs, a_pos, fitness = run_embodied_trial(
            params, num_neurons, trial_duration, light_pos
        )
        all_states.append(n_states)

    # Stack all trajectories
    combined_states = np.vstack(all_states)  # [total_steps, num_neurons]

    # 1. Effective dimensionality via PCA
    centered = combined_states - np.mean(combined_states, axis=0)
    cov_matrix = np.cov(centered.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov_matrix))[::-1]
    eigenvalues = eigenvalues[eigenvalues > 0]

    # Participation ratio: effective number of dimensions
    if np.sum(eigenvalues) > 0:
        participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    else:
        participation_ratio = 0.0

    # Fraction of variance explained by each component
    total_var = np.sum(eigenvalues)
    variance_explained = eigenvalues / total_var if total_var > 0 else eigenvalues

    # 2. Trajectory entropy (histogram-based)
    # Bin each neuron's state into 20 bins
    n_bins = 20
    state_entropy = 0.0
    for dim in range(num_neurons):
        hist, _ = np.histogram(combined_states[:, dim], bins=n_bins, density=True)
        hist = hist[hist > 0]
        bin_width = (np.max(combined_states[:, dim]) - np.min(combined_states[:, dim])) / n_bins
        if bin_width > 0:
            state_entropy += -np.sum(hist * np.log(hist + 1e-12) * bin_width)
    state_entropy /= num_neurons  # Average per dimension

    # 3. Autocorrelation timescale
    autocorr_times = []
    for dim in range(num_neurons):
        trace = combined_states[:, dim]
        trace_centered = trace - np.mean(trace)
        var = np.var(trace)
        if var > 0:
            max_lag = min(200, len(trace) // 2)
            for lag in range(1, max_lag):
                corr = np.mean(trace_centered[:-lag] * trace_centered[lag:]) / var
                if corr < 1.0 / np.e:
                    autocorr_times.append(lag)
                    break
            else:
                autocorr_times.append(max_lag)

    mean_autocorr = float(np.mean(autocorr_times)) if autocorr_times else 0.0

    # 4. Inter-trial variability: how different are neural trajectories across light positions?
    inter_trial_distances = []
    for i in range(len(all_states)):
        for j in range(i + 1, len(all_states)):
            min_len = min(len(all_states[i]), len(all_states[j]))
            dist = np.mean(np.sqrt(np.sum(
                (all_states[i][:min_len] - all_states[j][:min_len]) ** 2, axis=1
            )))
            inter_trial_distances.append(dist)

    return {
        'participation_ratio': float(participation_ratio),
        'variance_explained': [float(v) for v in variance_explained[:min(5, len(variance_explained))]],
        'total_variance': float(total_var),
        'state_entropy': float(state_entropy),
        'autocorrelation_timescale': mean_autocorr,
        'inter_trial_distance_mean': float(np.mean(inter_trial_distances)) if inter_trial_distances else 0.0,
        'inter_trial_distance_std': float(np.std(inter_trial_distances)) if inter_trial_distances else 0.0,
    }


def compute_perturbation_sensitivity(params, num_neurons, num_perturbations=20, trial_duration=500):
    """
    Measure sensitivity to perturbations at different points along the trajectory.

    This tests whether the system amplifies or damps perturbations, indicating
    whether it operates in a sensitive or robust dynamical regime.
    """
    light_pos = np.array(LIGHT_POSITIONS_EVOLUTION[0])

    # Run baseline trajectory
    s_trace, n_states, n_outputs, a_pos, fitness = run_embodied_trial(
        params, num_neurons, trial_duration, light_pos
    )

    # At evenly spaced points, apply perturbation and measure growth
    perturbation_points = np.linspace(50, trial_duration - 100, num_perturbations, dtype=int)
    epsilon = 0.01
    growth_rates = []

    for t0 in perturbation_points:
        # Create network at state from time t0
        network_base = CTRNN(num_neurons=num_neurons)
        network_base.weights = params['weights'].copy()
        network_base.biases = params['biases'].copy()
        network_base.tau = params['tau'].copy()
        network_base.set_state(n_states[t0].copy())

        # Create perturbed network
        network_pert = CTRNN(num_neurons=num_neurons)
        network_pert.weights = params['weights'].copy()
        network_pert.biases = params['biases'].copy()
        network_pert.tau = params['tau'].copy()
        pert_state = n_states[t0].copy()
        pert_state += epsilon * np.random.randn(num_neurons)
        network_pert.set_state(pert_state)

        # Both receive the same sensory input from recorded trajectory
        # (This isolates neural sensitivity from sensorimotor effects)
        divergence_trace = []
        for dt_step in range(min(100, trial_duration - t0)):
            t = t0 + dt_step
            if t < len(s_trace):
                padded = np.zeros(num_neurons)
                padded[:2] = s_trace[t, :min(2, num_neurons)]
            else:
                padded = np.zeros(num_neurons)

            network_base.step(padded)
            network_pert.step(padded)

            state_diff = np.linalg.norm(network_base.get_state() - network_pert.get_state())
            divergence_trace.append(state_diff)

        if len(divergence_trace) > 10:
            # Fit exponential growth rate
            initial_div = divergence_trace[0] if divergence_trace[0] > 0 else epsilon
            final_div = divergence_trace[-1] if divergence_trace[-1] > 0 else epsilon
            growth = np.log(final_div / initial_div) / (len(divergence_trace) * 0.01)
            growth_rates.append(growth)

    return {
        'mean_growth_rate': float(np.mean(growth_rates)) if growth_rates else 0.0,
        'std_growth_rate': float(np.std(growth_rates)) if growth_rates else 0.0,
        'max_growth_rate': float(np.max(growth_rates)) if growth_rates else 0.0,
        'min_growth_rate': float(np.min(growth_rates)) if growth_rates else 0.0,
        'fraction_amplifying': float(np.mean([g > 0 for g in growth_rates])) if growth_rates else 0.0,
    }


def run_dynamical_analysis(network_sizes=(3, 5, 8), seed=42, generations=2000):
    """Run complete dynamical analysis across network sizes."""
    print("=" * 70)
    print("DYNAMICAL ANALYSIS OF EVOLVED NETWORKS")
    print("=" * 70)

    results = {}
    for num_neurons in network_sizes:
        print(f"\n{'='*50}")
        print(f"NETWORK SIZE: {num_neurons} neurons")
        print(f"{'='*50}")

        # Evolve
        print("  Evolving...")
        t0 = time.time()
        params, fitness = evolve_network(num_neurons, generations=generations, seed=seed)
        t_evolve = time.time() - t0
        print(f"  Evolved in {t_evolve:.1f}s, fitness={fitness:.4f}")

        # Lyapunov exponents
        print("  Computing Lyapunov exponents...")
        t0 = time.time()
        lyap = compute_lyapunov_spectrum(params, num_neurons, trial_duration=500)
        t_lyap = time.time() - t0
        print(f"  Lyapunov: max={lyap['max_lyapunov']:.4f}, mean={lyap['mean_lyapunov']:.4f} ({t_lyap:.1f}s)")

        # Trajectory complexity
        print("  Computing trajectory complexity...")
        t0 = time.time()
        complexity = compute_trajectory_complexity(params, num_neurons)
        t_complex = time.time() - t0
        print(f"  Participation ratio: {complexity['participation_ratio']:.3f}")
        print(f"  Entropy: {complexity['state_entropy']:.3f}")
        print(f"  Autocorrelation: {complexity['autocorrelation_timescale']:.1f} steps")
        print(f"  Inter-trial distance: {complexity['inter_trial_distance_mean']:.4f} ± {complexity['inter_trial_distance_std']:.4f}")
        print(f"  ({t_complex:.1f}s)")

        # Perturbation sensitivity
        print("  Computing perturbation sensitivity...")
        t0 = time.time()
        perturbation = compute_perturbation_sensitivity(params, num_neurons)
        t_pert = time.time() - t0
        print(f"  Growth rate: {perturbation['mean_growth_rate']:.4f} ± {perturbation['std_growth_rate']:.4f}")
        print(f"  Fraction amplifying: {perturbation['fraction_amplifying']:.2f}")
        print(f"  ({t_pert:.1f}s)")

        results[f'net{num_neurons}'] = {
            'num_neurons': num_neurons,
            'fitness': float(fitness),
            'lyapunov': lyap,
            'trajectory_complexity': complexity,
            'perturbation_sensitivity': perturbation,
            'timing': {
                'evolution_s': t_evolve,
                'lyapunov_s': t_lyap,
                'complexity_s': t_complex,
                'perturbation_s': t_pert,
            }
        }

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: DYNAMICAL COMPLEXITY vs. NETWORK SIZE")
    print("=" * 70)
    print(f"{'Size':>5} {'Fitness':>8} {'MaxLyap':>9} {'PartRatio':>10} {'Entropy':>8} "
          f"{'AutoCorr':>9} {'InterTrial':>11} {'GrowthRate':>11} {'FracAmpl':>9}")
    print("-" * 95)
    for n in network_sizes:
        r = results[f'net{n}']
        print(f"{n:>5} {r['fitness']:>8.4f} "
              f"{r['lyapunov']['max_lyapunov']:>9.4f} "
              f"{r['trajectory_complexity']['participation_ratio']:>10.3f} "
              f"{r['trajectory_complexity']['state_entropy']:>8.3f} "
              f"{r['trajectory_complexity']['autocorrelation_timescale']:>9.1f} "
              f"{r['trajectory_complexity']['inter_trial_distance_mean']:>11.4f} "
              f"{r['perturbation_sensitivity']['mean_growth_rate']:>11.4f} "
              f"{r['perturbation_sensitivity']['fraction_amplifying']:>9.2f}")

    return results


if __name__ == "__main__":
    results = run_dynamical_analysis(
        network_sizes=(3, 5, 8),
        seed=42,
        generations=2000,
    )

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '../../../results/paper2')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'dynamical_analysis_{timestamp}.json')

    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(i) for i in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to: {output_file}")
