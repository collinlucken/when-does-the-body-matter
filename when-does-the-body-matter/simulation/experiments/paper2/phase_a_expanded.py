"""
Paper 2 Phase A Expanded: Robust Embodiment Experiments

Addresses critical review issues from the v0.1 self-review:
1. Only 3 network sizes tested → expanded to 6 (2, 3, 4, 5, 6, 8 neurons)
2. Only 2000 generations → expanded to 10,000 generations
3. No statistical replication → 5 independent seeds per condition
4. No gradient analysis → continuous constitutivity-vs-capacity curve

This produces the core dataset for Paper 2 v0.2.

Design: 6 network sizes × 5 seeds × MicrobialGA = 30 conditions
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from simulation.ctrnn import CTRNN
from simulation.evolutionary import MicrobialGA, GenotypeDecoder
from simulation.microworld import Agent

# ============================================================
# Constants
# ============================================================
AGENT_DT = 0.1
AGENT_MAX_SPEED = 3.0
SENSOR_RANGE = 40.0
ARENA_SIZE = 50.0
TRIAL_STEPS_EVOLUTION = 500
TRIAL_STEPS_TEST = 1000
NUM_EVOLUTION_TRIALS = 4  # Use 4 of 8 positions during evolution (faster, still forces sensor use)
NUM_TEST_TRIALS = 5
DIVERGENCE_THRESHOLD = 0.1

LIGHT_POSITIONS_EVOLUTION = [
    (10.0, 10.0), (40.0, 40.0), (10.0, 40.0), (40.0, 10.0),
    (25.0, 40.0), (25.0, 10.0), (10.0, 25.0), (40.0, 25.0),
]

LIGHT_POSITIONS_TEST = [
    (10.0, 10.0), (40.0, 40.0), (10.0, 40.0), (40.0, 10.0),
    (25.0, 40.0),
]

# Morphology configurations for substitution tests
MORPHOLOGIES = {
    'baseline':        {'sensor_angle_offset': np.pi / 6,  'motor_scale': 1.0, 'radius': 1.0},
    'wider_sensors':   {'sensor_angle_offset': np.pi / 3,  'motor_scale': 1.0, 'radius': 1.0},
    'narrower_sensors':{'sensor_angle_offset': np.pi / 12, 'motor_scale': 1.0, 'radius': 1.0},
    'larger_body':     {'sensor_angle_offset': np.pi / 6,  'motor_scale': 1.0, 'radius': 2.0},
    'faster_motors':   {'sensor_angle_offset': np.pi / 6,  'motor_scale': 2.0, 'radius': 1.0},
    'slower_motors':   {'sensor_angle_offset': np.pi / 6,  'motor_scale': 0.5, 'radius': 1.0},
}


# ============================================================
# Fitness Function
# ============================================================
def phototaxis_fitness(
    genotype: np.ndarray,
    decoder: GenotypeDecoder,
    num_neurons: int,
    num_trials: int = NUM_EVOLUTION_TRIALS,
    trial_duration: int = TRIAL_STEPS_EVOLUTION,
) -> float:
    """Evaluate phototaxis fitness with variable light positions."""
    params = decoder.decode(genotype)
    total_fitness = 0.0

    for trial in range(min(num_trials, len(LIGHT_POSITIONS_EVOLUTION))):
        agent = Agent(radius=1.0, max_speed=AGENT_MAX_SPEED, sensor_range=SENSOR_RANGE)
        network = CTRNN(num_neurons=num_neurons)
        network.weights = params['weights'].copy()
        network.biases = params['biases'].copy()
        network.tau = params['tau'].copy()

        light_x, light_y = LIGHT_POSITIONS_EVOLUTION[trial]
        agent.position = np.array([25.0, 25.0])
        agent.velocity = np.zeros(2)
        initial_dist = np.linalg.norm(agent.position - np.array([light_x, light_y]))

        cumulative_fitness = 0.0
        for step in range(trial_duration):
            left_pos, right_pos = agent.get_sensor_positions()
            left_dist = np.linalg.norm(left_pos - np.array([light_x, light_y]))
            right_dist = np.linalg.norm(right_pos - np.array([light_x, light_y]))

            left_sensor = max(0.0, 1.0 - left_dist / SENSOR_RANGE)
            right_sensor = max(0.0, 1.0 - right_dist / SENSOR_RANGE)

            padded_sensory = np.zeros(num_neurons)
            padded_sensory[:2] = np.array([left_sensor, right_sensor])[:min(2, num_neurons)]

            output = network.step(padded_sensory)

            left_motor = output[0]
            right_motor = output[1] if num_neurons >= 2 else output[0]

            agent.set_motor_commands(left_motor, right_motor)
            agent.update(dt=AGENT_DT)

            agent_dist = np.linalg.norm(agent.position - np.array([light_x, light_y]))
            cumulative_fitness += max(0.0, 1.0 - agent_dist / ARENA_SIZE)

        final_dist = np.linalg.norm(agent.position - np.array([light_x, light_y]))
        approach_score = max(0.0, (initial_dist - final_dist) / initial_dist)
        time_avg = cumulative_fitness / trial_duration
        trial_fitness = 0.5 * time_avg + 0.5 * approach_score

        total_fitness += trial_fitness

    return total_fitness / min(num_trials, len(LIGHT_POSITIONS_EVOLUTION))


# ============================================================
# Embodied and Ghost Trial Functions
# ============================================================
def run_embodied_trial(
    network_params: Dict[str, np.ndarray],
    num_neurons: int,
    trial_duration: int = TRIAL_STEPS_TEST,
    light_pos: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Run embodied trial. Returns (sensory, states, outputs, positions, fitness)."""
    agent = Agent(radius=1.0, max_speed=AGENT_MAX_SPEED, sensor_range=SENSOR_RANGE)
    network = CTRNN(num_neurons=num_neurons)
    network.weights = network_params['weights'].copy()
    network.biases = network_params['biases'].copy()
    network.tau = network_params['tau'].copy()

    light_x, light_y = (light_pos[0], light_pos[1]) if light_pos is not None else (10.0, 40.0)
    agent.position = np.array([25.0, 25.0])
    agent.velocity = np.zeros(2)

    sensory_trace = np.zeros((trial_duration, 2))
    neural_states = np.zeros((trial_duration, num_neurons))
    neural_outputs = np.zeros((trial_duration, num_neurons))
    agent_positions = np.zeros((trial_duration, 2))
    total_fitness = 0.0

    for step in range(trial_duration):
        agent_positions[step] = agent.position.copy()
        left_pos, right_pos = agent.get_sensor_positions()
        left_dist = np.linalg.norm(left_pos - np.array([light_x, light_y]))
        right_dist = np.linalg.norm(right_pos - np.array([light_x, light_y]))
        left_sensor = max(0.0, 1.0 - left_dist / SENSOR_RANGE)
        right_sensor = max(0.0, 1.0 - right_dist / SENSOR_RANGE)
        sensory = np.array([left_sensor, right_sensor])
        sensory_trace[step] = sensory

        neural_states[step] = network.get_state().copy()
        padded = np.zeros(num_neurons)
        padded[:2] = sensory[:min(2, num_neurons)]
        output = network.step(padded)
        neural_outputs[step] = output.copy()

        agent.set_motor_commands(output[0], output[1] if num_neurons >= 2 else output[0])
        agent.update(dt=AGENT_DT)

        agent_dist = np.linalg.norm(agent.position - np.array([light_x, light_y]))
        total_fitness += max(0.0, 1.0 - agent_dist / ARENA_SIZE)

    return sensory_trace, neural_states, neural_outputs, agent_positions, total_fitness / trial_duration


def run_ghost_frozen_body(params, num_neurons, trial_duration, light_pos):
    """Frozen body ghost: constant sensory input from fixed position."""
    network = CTRNN(num_neurons=num_neurons)
    network.weights = params['weights'].copy()
    network.biases = params['biases'].copy()
    network.tau = params['tau'].copy()

    agent = Agent(radius=1.0, max_speed=AGENT_MAX_SPEED, sensor_range=SENSOR_RANGE)
    agent.position = np.array([25.0, 25.0])
    light_x, light_y = light_pos[0], light_pos[1]

    # Compute constant sensory from frozen position
    left_pos, right_pos = agent.get_sensor_positions()
    left_dist = np.linalg.norm(left_pos - np.array([light_x, light_y]))
    right_dist = np.linalg.norm(right_pos - np.array([light_x, light_y]))
    frozen_sensory = np.array([
        max(0.0, 1.0 - left_dist / SENSOR_RANGE),
        max(0.0, 1.0 - right_dist / SENSOR_RANGE)
    ])

    neural_states = np.zeros((trial_duration, num_neurons))
    neural_outputs = np.zeros((trial_duration, num_neurons))

    for step in range(trial_duration):
        neural_states[step] = network.get_state().copy()
        padded = np.zeros(num_neurons)
        padded[:2] = frozen_sensory[:min(2, num_neurons)]
        output = network.step(padded)
        neural_outputs[step] = output.copy()

    return neural_states, neural_outputs


def run_ghost_disconnected(params, num_neurons, trial_duration):
    """Disconnected ghost: zero sensory input."""
    network = CTRNN(num_neurons=num_neurons)
    network.weights = params['weights'].copy()
    network.biases = params['biases'].copy()
    network.tau = params['tau'].copy()

    neural_states = np.zeros((trial_duration, num_neurons))
    neural_outputs = np.zeros((trial_duration, num_neurons))

    for step in range(trial_duration):
        neural_states[step] = network.get_state().copy()
        padded = np.zeros(num_neurons)
        output = network.step(padded)
        neural_outputs[step] = output.copy()

    return neural_states, neural_outputs


def run_ghost_counterfactual(params, num_neurons, trial_duration, rng):
    """Counterfactual ghost: random sensory input."""
    network = CTRNN(num_neurons=num_neurons)
    network.weights = params['weights'].copy()
    network.biases = params['biases'].copy()
    network.tau = params['tau'].copy()

    random_sensory = rng.uniform(0, 0.5, (trial_duration, 2))
    neural_states = np.zeros((trial_duration, num_neurons))
    neural_outputs = np.zeros((trial_duration, num_neurons))

    for step in range(trial_duration):
        neural_states[step] = network.get_state().copy()
        padded = np.zeros(num_neurons)
        padded[:2] = random_sensory[step, :min(2, num_neurons)]
        output = network.step(padded)
        neural_outputs[step] = output.copy()

    return neural_states, neural_outputs


# ============================================================
# Divergence Metrics
# ============================================================
def compute_divergence(emb_states, ghost_states, emb_outputs, ghost_outputs, threshold=DIVERGENCE_THRESHOLD):
    """Compute divergence metrics between embodied and ghost trajectories."""
    state_diff = np.sqrt(np.sum((emb_states - ghost_states) ** 2, axis=1))
    output_diff = np.sqrt(np.sum((emb_outputs - ghost_outputs) ** 2, axis=1))

    time_to_div = len(state_diff)
    for t in range(len(state_diff)):
        if state_diff[t] > threshold:
            time_to_div = t
            break

    return {
        'neural_divergence': float(np.mean(state_diff)),
        'output_divergence': float(np.mean(output_diff)),
        'time_to_divergence': int(time_to_div),
        'max_divergence': float(np.max(state_diff)),
        'divergence_at_end': float(state_diff[-1]) if len(state_diff) > 0 else 0.0,
        'neural_divergence_std': float(np.std(state_diff)),
    }


# ============================================================
# Morphology Substitution
# ============================================================
def run_morphology_test(params, num_neurons, morph_config, light_positions, trial_duration=TRIAL_STEPS_TEST):
    """Run morphology substitution test and return average fitness."""
    fitnesses = []
    for lp in light_positions:
        agent = Agent(
            radius=morph_config['radius'],
            max_speed=AGENT_MAX_SPEED,
            sensor_range=SENSOR_RANGE,
            motor_scale=morph_config['motor_scale']
        )
        agent.sensor_angle_offset = morph_config['sensor_angle_offset']
        agent.position = np.array([25.0, 25.0])
        agent.velocity = np.zeros(2)

        network = CTRNN(num_neurons=num_neurons)
        network.weights = params['weights'].copy()
        network.biases = params['biases'].copy()
        network.tau = params['tau'].copy()

        total_fit = 0.0
        light_x, light_y = lp[0], lp[1]
        for step in range(trial_duration):
            left_pos, right_pos = agent.get_sensor_positions()
            left_dist = np.linalg.norm(left_pos - np.array([light_x, light_y]))
            right_dist = np.linalg.norm(right_pos - np.array([light_x, light_y]))
            left_s = max(0.0, 1.0 - left_dist / SENSOR_RANGE)
            right_s = max(0.0, 1.0 - right_dist / SENSOR_RANGE)
            padded = np.zeros(num_neurons)
            padded[:2] = np.array([left_s, right_s])[:min(2, num_neurons)]
            output = network.step(padded)
            agent.set_motor_commands(output[0], output[1] if num_neurons >= 2 else output[0])
            agent.update(dt=AGENT_DT)
            ad = np.linalg.norm(agent.position - np.array([light_x, light_y]))
            total_fit += max(0.0, 1.0 - ad / ARENA_SIZE)

        fitnesses.append(total_fit / trial_duration)
    return float(np.mean(fitnesses))


# ============================================================
# Single Condition Runner
# ============================================================
def run_condition(num_neurons, generations, population_size, seed, verbose=True):
    """Run a complete condition: evolve → test → ghost → morphology."""
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    decoder = GenotypeDecoder(
        num_neurons=num_neurons,
        include_gains=False,
        tau_range=(0.5, 5.0),
        weight_range=(-10.0, 10.0),
        bias_range=(-10.0, 10.0),
    )

    fitness_fn = lambda g: phototaxis_fitness(g, decoder, num_neurons)

    # ---- EVOLVE ----
    ea = MicrobialGA(
        genotype_size=decoder.genotype_size,
        fitness_function=fitness_fn,
        population_size=population_size,
        mutation_std=0.2,
        seed=seed
    )

    fitness_history = []
    best_ever_fitness = -np.inf
    best_ever_genotype = None

    for gen in range(generations):
        best_geno, best_fit = ea.step()
        if gen % max(1, generations // 20) == 0:
            fitness_history.append({'gen': gen, 'fitness': float(best_fit)})
        if best_fit > best_ever_fitness:
            best_ever_fitness = best_fit
            best_ever_genotype = best_geno.copy()

    # Record final fitness
    fitness_history.append({'gen': generations, 'fitness': float(best_ever_fitness)})
    best_params = decoder.decode(best_ever_genotype)
    best_genotype = best_ever_genotype.copy()  # Save for dynamical analysis

    if verbose:
        print(f"    Evolved: fitness={best_ever_fitness:.4f}")

    # ---- EMBODIED TEST ----
    test_light_positions = [np.array(lp) for lp in LIGHT_POSITIONS_TEST]
    embodied_results = []
    for lp in test_light_positions:
        s_trace, n_states, n_outputs, a_pos, fit = run_embodied_trial(
            best_params, num_neurons, TRIAL_STEPS_TEST, lp
        )
        embodied_results.append((s_trace, n_states, n_outputs, a_pos, fit))

    avg_embodied_fitness = float(np.mean([r[4] for r in embodied_results]))
    emb_states_ref = embodied_results[0][1]  # First trial as reference
    emb_outputs_ref = embodied_results[0][2]
    ref_light = test_light_positions[0]

    # ---- GHOST CONDITIONS ----
    # Average ghost metrics over multiple test light positions for robustness
    ghost_frozen_metrics_all = []
    ghost_disconn_metrics_all = []
    ghost_cf_metrics_all = []

    for i, lp in enumerate(test_light_positions):
        emb_s, emb_st, emb_out, emb_pos, emb_fit = embodied_results[i]

        # Frozen body
        fb_states, fb_outputs = run_ghost_frozen_body(best_params, num_neurons, TRIAL_STEPS_TEST, lp)
        fb_m = compute_divergence(emb_st, fb_states, emb_out, fb_outputs)
        ghost_frozen_metrics_all.append(fb_m)

        # Disconnected
        dc_states, dc_outputs = run_ghost_disconnected(best_params, num_neurons, TRIAL_STEPS_TEST)
        dc_m = compute_divergence(emb_st, dc_states, emb_out, dc_outputs)
        ghost_disconn_metrics_all.append(dc_m)

        # Counterfactual
        cf_rng = np.random.RandomState(seed + 5000 + i)
        cf_states, cf_outputs = run_ghost_counterfactual(best_params, num_neurons, TRIAL_STEPS_TEST, cf_rng)
        cf_m = compute_divergence(emb_st, cf_states, emb_out, cf_outputs)
        ghost_cf_metrics_all.append(cf_m)

    # Average metrics across test positions
    def avg_metrics(metrics_list):
        keys = metrics_list[0].keys()
        return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}

    ghost_frozen_avg = avg_metrics(ghost_frozen_metrics_all)
    ghost_disconn_avg = avg_metrics(ghost_disconn_metrics_all)
    ghost_cf_avg = avg_metrics(ghost_cf_metrics_all)

    # Also compute per-trial std for error bars
    def std_metrics(metrics_list):
        keys = metrics_list[0].keys()
        return {k + '_std': float(np.std([m[k] for m in metrics_list])) for k in keys}

    ghost_frozen_std = std_metrics(ghost_frozen_metrics_all)
    ghost_disconn_std = std_metrics(ghost_disconn_metrics_all)
    ghost_cf_std = std_metrics(ghost_cf_metrics_all)

    # ---- MORPHOLOGY SUBSTITUTIONS ----
    morph_results = {}
    for morph_name, morph_config in MORPHOLOGIES.items():
        morph_fit = run_morphology_test(
            best_params, num_neurons, morph_config,
            test_light_positions, TRIAL_STEPS_TEST
        )
        degradation = max(0.0, (avg_embodied_fitness - morph_fit) / (avg_embodied_fitness + 1e-6))
        morph_results[morph_name] = {
            'fitness': morph_fit,
            'degradation': float(degradation),
        }

    # ---- COMPUTE SCORES ----
    frozen_score = min(1.0, ghost_frozen_avg['neural_divergence'])
    disconn_score = min(1.0, ghost_disconn_avg['neural_divergence'])
    cf_score = min(1.0, ghost_cf_avg['neural_divergence'])

    constitutive_score = float(0.5 * frozen_score + 0.3 * cf_score + 0.2 * disconn_score)

    non_baseline_morphs = {k: v for k, v in morph_results.items() if k != 'baseline'}
    avg_degradation = float(np.mean([v['degradation'] for v in non_baseline_morphs.values()]))
    causal_score = float(1.0 - avg_degradation)

    # Classification
    if constitutive_score > 0.6:
        classification = "CONSTITUTIVE_DOMINANT"
    elif constitutive_score > 0.3:
        classification = "MIXED"
    elif constitutive_score > 0.1:
        classification = "WEAK_CONSTITUTIVE"
    else:
        classification = "CAUSAL_DOMINANT"

    return {
        'config': {
            'num_neurons': num_neurons,
            'generations': generations,
            'population_size': population_size,
            'seed': seed,
        },
        'evolution': {
            'best_fitness': float(best_ever_fitness),
            'fitness_history': fitness_history,
            'best_genotype': best_genotype.tolist(),
        },
        'embodied': {
            'avg_fitness': avg_embodied_fitness,
            'per_trial_fitness': [float(r[4]) for r in embodied_results],
        },
        'ghost_frozen_body': {**ghost_frozen_avg, **ghost_frozen_std},
        'ghost_disconnected': {**ghost_disconn_avg, **ghost_disconn_std},
        'ghost_counterfactual': {**ghost_cf_avg, **ghost_cf_std},
        'morphology': morph_results,
        'scores': {
            'constitutive': constitutive_score,
            'causal': causal_score,
            'classification': classification,
        },
    }


# ============================================================
# Main Experiment
# ============================================================
def run_expanded_phase_a(
    network_sizes=(2, 3, 4, 5, 6, 8),
    seeds=(42, 137, 256, 512, 1024),
    generations=10000,
    population_size=50,
    verbose=True,
):
    """Run expanded Phase A with multiple network sizes and seeds."""
    total_conditions = len(network_sizes) * len(seeds)
    print("=" * 70)
    print("EXPANDED PHASE A: ROBUST EMBODIMENT EXPERIMENTS")
    print("=" * 70)
    print(f"Network sizes: {network_sizes}")
    print(f"Seeds: {seeds}")
    print(f"Generations: {generations}, Population: {population_size}")
    print(f"Total conditions: {total_conditions}")
    print("=" * 70)

    all_results = {}
    run_num = 0
    start_total = time.time()

    for net_size in network_sizes:
        for seed in seeds:
            run_num += 1
            run_id = f"net{net_size}_seed{seed}"
            print(f"\n[{run_num}/{total_conditions}] {run_id}")
            start_time = time.time()

            try:
                result = run_condition(
                    num_neurons=net_size,
                    generations=generations,
                    population_size=population_size,
                    seed=seed,
                    verbose=verbose,
                )
                elapsed = time.time() - start_time
                result['timing'] = {'elapsed_seconds': elapsed}
                all_results[run_id] = result

                if verbose:
                    print(f"    Score: const={result['scores']['constitutive']:.4f}, "
                          f"class={result['scores']['classification']}, "
                          f"time={elapsed:.1f}s")

            except Exception as e:
                elapsed = time.time() - start_time
                print(f"    FAILED after {elapsed:.1f}s: {e}")
                all_results[run_id] = {
                    'config': {'num_neurons': net_size, 'seed': seed},
                    'error': str(e),
                    'timing': {'elapsed_seconds': elapsed},
                }

    total_elapsed = time.time() - start_total

    # ---- AGGREGATE STATISTICS ----
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS BY NETWORK SIZE")
    print("=" * 70)

    aggregate = {}
    for net_size in network_sizes:
        conditions = [v for k, v in all_results.items()
                      if k.startswith(f"net{net_size}_") and 'error' not in v]
        if not conditions:
            continue

        fitnesses = [c['evolution']['best_fitness'] for c in conditions]
        const_scores = [c['scores']['constitutive'] for c in conditions]
        frozen_divs = [c['ghost_frozen_body']['neural_divergence'] for c in conditions]
        disconn_divs = [c['ghost_disconnected']['neural_divergence'] for c in conditions]
        cf_divs = [c['ghost_counterfactual']['neural_divergence'] for c in conditions]
        ttds_frozen = [c['ghost_frozen_body']['time_to_divergence'] for c in conditions]

        agg = {
            'num_conditions': len(conditions),
            'fitness_mean': float(np.mean(fitnesses)),
            'fitness_std': float(np.std(fitnesses)),
            'constitutive_mean': float(np.mean(const_scores)),
            'constitutive_std': float(np.std(const_scores)),
            'frozen_div_mean': float(np.mean(frozen_divs)),
            'frozen_div_std': float(np.std(frozen_divs)),
            'disconn_div_mean': float(np.mean(disconn_divs)),
            'disconn_div_std': float(np.std(disconn_divs)),
            'cf_div_mean': float(np.mean(cf_divs)),
            'cf_div_std': float(np.std(cf_divs)),
            'ttd_frozen_mean': float(np.mean(ttds_frozen)),
            'ttd_frozen_std': float(np.std(ttds_frozen)),
        }
        aggregate[f'net{net_size}'] = agg

        print(f"\n  {net_size} neurons (n={len(conditions)}):")
        print(f"    Fitness:       {agg['fitness_mean']:.4f} ± {agg['fitness_std']:.4f}")
        print(f"    Constitutive:  {agg['constitutive_mean']:.4f} ± {agg['constitutive_std']:.4f}")
        print(f"    Frozen div:    {agg['frozen_div_mean']:.4f} ± {agg['frozen_div_std']:.4f}")
        print(f"    Disconn div:   {agg['disconn_div_mean']:.4f} ± {agg['disconn_div_std']:.4f}")
        print(f"    TTD frozen:    {agg['ttd_frozen_mean']:.1f} ± {agg['ttd_frozen_std']:.1f}")

    # ---- STATISTICAL TESTS ----
    # Spearman correlation: network size vs constitutive score
    all_sizes = []
    all_const = []
    for k, v in all_results.items():
        if 'error' not in v:
            all_sizes.append(v['config']['num_neurons'])
            all_const.append(v['scores']['constitutive'])

    if len(all_sizes) > 2:
        from scipy.stats import spearmanr, pearsonr
        rho_spearman, p_spearman = spearmanr(all_sizes, all_const)
        rho_pearson, p_pearson = pearsonr(all_sizes, all_const)
        print(f"\n  Correlation (size vs constitutive):")
        print(f"    Spearman: rho={rho_spearman:.4f}, p={p_spearman:.6f}")
        print(f"    Pearson:  r={rho_pearson:.4f}, p={p_pearson:.6f}")
        stats_results = {
            'spearman_rho': float(rho_spearman),
            'spearman_p': float(p_spearman),
            'pearson_r': float(rho_pearson),
            'pearson_p': float(p_pearson),
            'n': len(all_sizes),
        }
    else:
        stats_results = {'error': 'insufficient data for correlation'}

    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    return {
        'conditions': all_results,
        'aggregate': aggregate,
        'statistics': stats_results,
        'meta': {
            'network_sizes': list(network_sizes),
            'seeds': list(seeds),
            'generations': generations,
            'population_size': population_size,
            'total_conditions': total_conditions,
            'total_elapsed_seconds': total_elapsed,
        }
    }


def convert_for_json(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    return obj


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Expanded Phase A embodiment experiments")
    parser.add_argument('--generations', type=int, default=5000)
    parser.add_argument('--population-size', type=int, default=50)
    parser.add_argument('--num-seeds', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--quick', action='store_true', help='Quick test (1000 gen, 2 seeds)')
    args = parser.parse_args()

    seeds = [42, 137, 256, 512, 1024][:args.num_seeds]

    if args.quick:
        args.generations = 1000
        seeds = seeds[:2]

    results = run_expanded_phase_a(
        network_sizes=(2, 3, 4, 5, 6, 8),
        seeds=tuple(seeds),
        generations=args.generations,
        population_size=args.population_size,
    )

    # Save results
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), '../../../results/paper2'
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'phase_a_expanded_{timestamp}.json')

    results_json = convert_for_json(results)
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
