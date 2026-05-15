"""
Paper 2 Phase A: 10-Seed Expansion with Multiprocessing

Runs 7 new seeds (512, 1024, 2048, 3141, 4096, 5555, 7777) across all 6 network sizes,
then merges with existing 3-seed results to produce a 60-condition dataset.

Uses multiprocessing with 4 workers for ~4x speedup.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from simulation.experiments.paper2.phase_a_expanded import (
    run_condition, convert_for_json
)

# ============================================================
# Configuration
# ============================================================
NETWORK_SIZES = (2, 3, 4, 5, 6, 8)
ALL_SEEDS = (42, 137, 256, 512, 1024, 2048, 3141, 4096, 5555, 7777)
NEW_SEEDS = (512, 1024, 2048, 3141, 4096, 5555, 7777)
GENERATIONS = 5000
POPULATION_SIZE = 50
NUM_WORKERS = min(4, cpu_count())

# Previous results file (3-seed run from Session 4)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../../results/paper2')


def run_single_condition(args):
    """Worker function for multiprocessing."""
    net_size, seed, generations, population_size, condition_id, total = args
    run_id = f"net{net_size}_seed{seed}"
    print(f"  [{condition_id}/{total}] Starting {run_id}...", flush=True)

    start_time = time.time()
    try:
        result = run_condition(
            num_neurons=net_size,
            generations=generations,
            population_size=population_size,
            seed=seed,
            verbose=False,
        )
        elapsed = time.time() - start_time
        result['timing'] = {'elapsed_seconds': elapsed}
        print(f"  [{condition_id}/{total}] {run_id}: const={result['scores']['constitutive']:.4f}, "
              f"class={result['scores']['classification']}, time={elapsed:.1f}s", flush=True)
        return run_id, result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  [{condition_id}/{total}] {run_id}: FAILED after {elapsed:.1f}s: {e}", flush=True)
        return run_id, {
            'config': {'num_neurons': net_size, 'seed': seed},
            'error': str(e),
            'timing': {'elapsed_seconds': elapsed},
        }


def load_existing_results():
    """Load existing 3-seed results from the most recent JSON file."""
    results_path = Path(RESULTS_DIR)
    if not results_path.exists():
        return {}

    # Find the most recent expanded results file
    json_files = sorted(results_path.glob('phase_a_expanded_*.json'), reverse=True)
    if not json_files:
        return {}

    latest = json_files[0]
    print(f"Loading existing results from: {latest}")
    with open(latest, 'r') as f:
        data = json.load(f)

    return data.get('conditions', {})


def compute_aggregate_statistics(all_results, network_sizes, all_seeds):
    """Compute aggregate statistics from the full dataset."""
    from scipy.stats import spearmanr, pearsonr

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
            'constitutive_median': float(np.median(const_scores)),
            'constitutive_min': float(np.min(const_scores)),
            'constitutive_max': float(np.max(const_scores)),
            'constitutive_iqr': float(np.percentile(const_scores, 75) - np.percentile(const_scores, 25)),
            'frozen_div_mean': float(np.mean(frozen_divs)),
            'frozen_div_std': float(np.std(frozen_divs)),
            'disconn_div_mean': float(np.mean(disconn_divs)),
            'disconn_div_std': float(np.std(disconn_divs)),
            'cf_div_mean': float(np.mean(cf_divs)),
            'cf_div_std': float(np.std(cf_divs)),
            'ttd_frozen_mean': float(np.mean(ttds_frozen)),
            'ttd_frozen_std': float(np.std(ttds_frozen)),
            'per_seed_constitutive': {
                f"seed{c['config']['seed']}": c['scores']['constitutive']
                for c in conditions
            },
        }
        aggregate[f'net{net_size}'] = agg

    # Statistical tests
    all_sizes = []
    all_const = []
    for k, v in all_results.items():
        if 'error' not in v:
            all_sizes.append(v['config']['num_neurons'])
            all_const.append(v['scores']['constitutive'])

    stats = {}
    if len(all_sizes) > 2:
        rho_spearman, p_spearman = spearmanr(all_sizes, all_const)
        rho_pearson, p_pearson = pearsonr(all_sizes, all_const)
        stats = {
            'spearman_rho': float(rho_spearman),
            'spearman_p': float(p_spearman),
            'pearson_r': float(rho_pearson),
            'pearson_p': float(p_pearson),
            'n': len(all_sizes),
        }

        # Also compute per-size group means for Jonckheere-Terpstra trend test (manual)
        # And bootstrap CIs
        from scipy.stats import mannwhitneyu, kruskal

        # Kruskal-Wallis test across all groups
        groups = []
        for ns in network_sizes:
            group_scores = [v['scores']['constitutive'] for k, v in all_results.items()
                           if 'error' not in v and v['config']['num_neurons'] == ns]
            if group_scores:
                groups.append(group_scores)

        if len(groups) >= 2:
            try:
                h_stat, kw_p = kruskal(*groups)
                stats['kruskal_h'] = float(h_stat)
                stats['kruskal_p'] = float(kw_p)
            except Exception:
                pass

        # Pairwise comparisons: small (2-3) vs medium (4-5) vs large (6-8)
        small = [v['scores']['constitutive'] for k, v in all_results.items()
                 if 'error' not in v and v['config']['num_neurons'] in (2, 3)]
        medium = [v['scores']['constitutive'] for k, v in all_results.items()
                  if 'error' not in v and v['config']['num_neurons'] in (4, 5)]
        large = [v['scores']['constitutive'] for k, v in all_results.items()
                 if 'error' not in v and v['config']['num_neurons'] in (6, 8)]

        if small and medium:
            try:
                u_sm, p_sm = mannwhitneyu(small, medium, alternative='less')
                stats['mannwhitney_small_vs_medium_U'] = float(u_sm)
                stats['mannwhitney_small_vs_medium_p'] = float(p_sm)
            except Exception:
                pass
        if medium and large:
            try:
                u_ml, p_ml = mannwhitneyu(medium, large, alternative='less')
                stats['mannwhitney_medium_vs_large_U'] = float(u_ml)
                stats['mannwhitney_medium_vs_large_p'] = float(p_ml)
            except Exception:
                pass
        if small and large:
            try:
                u_sl, p_sl = mannwhitneyu(small, large, alternative='less')
                stats['mannwhitney_small_vs_large_U'] = float(u_sl)
                stats['mannwhitney_small_vs_large_p'] = float(p_sl)
            except Exception:
                pass

        # Bootstrap 95% CI for Spearman rho
        np.random.seed(42)
        n_bootstrap = 10000
        rhos = []
        indices = np.arange(len(all_sizes))
        for _ in range(n_bootstrap):
            boot_idx = np.random.choice(indices, size=len(indices), replace=True)
            boot_sizes = [all_sizes[i] for i in boot_idx]
            boot_const = [all_const[i] for i in boot_idx]
            try:
                r, _ = spearmanr(boot_sizes, boot_const)
                rhos.append(r)
            except Exception:
                pass
        if rhos:
            stats['spearman_bootstrap_ci_lower'] = float(np.percentile(rhos, 2.5))
            stats['spearman_bootstrap_ci_upper'] = float(np.percentile(rhos, 97.5))

    return aggregate, stats


def main():
    print("=" * 70)
    print("EXPANDED PHASE A: 10-SEED REPLICATION")
    print("=" * 70)
    print(f"Network sizes: {NETWORK_SIZES}")
    print(f"All seeds: {ALL_SEEDS}")
    print(f"New seeds to run: {NEW_SEEDS}")
    print(f"Generations: {GENERATIONS}, Population: {POPULATION_SIZE}")
    print(f"Workers: {NUM_WORKERS}")
    print("=" * 70)

    # Load existing results
    existing = load_existing_results()
    print(f"Loaded {len(existing)} existing conditions")

    # Determine which conditions still need to be run
    tasks = []
    condition_id = 0
    for net_size in NETWORK_SIZES:
        for seed in ALL_SEEDS:
            run_id = f"net{net_size}_seed{seed}"
            if run_id in existing and 'error' not in existing[run_id]:
                print(f"  Skipping {run_id} (already completed)")
                continue
            condition_id += 1
            tasks.append((net_size, seed, GENERATIONS, POPULATION_SIZE, condition_id, 0))

    # Update total count
    total = len(tasks)
    tasks = [(t[0], t[1], t[2], t[3], t[4], total) for t in tasks]

    print(f"\nNeed to run {total} new conditions")
    print(f"Estimated time: {total * 190 / NUM_WORKERS / 60:.0f} min with {NUM_WORKERS} workers")
    print("=" * 70)

    if total == 0:
        print("All conditions already completed!")
        new_results = {}
    else:
        # Run with multiprocessing
        start_time = time.time()

        new_results = {}
        with Pool(processes=NUM_WORKERS) as pool:
            results = pool.map(run_single_condition, tasks)

        for run_id, result in results:
            new_results[run_id] = result

        elapsed = time.time() - start_time
        print(f"\nNew conditions completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Merge results
    all_results = {**existing, **new_results}
    print(f"\nTotal conditions: {len(all_results)}")

    # Compute aggregate statistics
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (10 SEEDS PER SIZE)")
    print("=" * 70)

    aggregate, stats = compute_aggregate_statistics(all_results, NETWORK_SIZES, ALL_SEEDS)

    for net_size in NETWORK_SIZES:
        key = f'net{net_size}'
        if key in aggregate:
            agg = aggregate[key]
            print(f"\n  {net_size} neurons (n={agg['num_conditions']}):")
            print(f"    Fitness:       {agg['fitness_mean']:.4f} ± {agg['fitness_std']:.4f}")
            print(f"    Constitutive:  {agg['constitutive_mean']:.4f} ± {agg['constitutive_std']:.4f}")
            print(f"    Median:        {agg['constitutive_median']:.4f}")
            print(f"    Range:         [{agg['constitutive_min']:.4f}, {agg['constitutive_max']:.4f}]")
            print(f"    IQR:           {agg['constitutive_iqr']:.4f}")
            print(f"    Frozen div:    {agg['frozen_div_mean']:.4f} ± {agg['frozen_div_std']:.4f}")
            print(f"    Disconn div:   {agg['disconn_div_mean']:.4f} ± {agg['disconn_div_std']:.4f}")

    if stats:
        print(f"\n  Correlation (size vs constitutive, n={stats.get('n', '?')}):")
        print(f"    Spearman: rho={stats.get('spearman_rho', 0):.4f}, p={stats.get('spearman_p', 1):.6f}")
        if 'spearman_bootstrap_ci_lower' in stats:
            print(f"    95% Bootstrap CI: [{stats['spearman_bootstrap_ci_lower']:.4f}, {stats['spearman_bootstrap_ci_upper']:.4f}]")
        print(f"    Pearson:  r={stats.get('pearson_r', 0):.4f}, p={stats.get('pearson_p', 1):.6f}")
        if 'kruskal_h' in stats:
            print(f"    Kruskal-Wallis: H={stats['kruskal_h']:.4f}, p={stats['kruskal_p']:.6f}")
        if 'mannwhitney_small_vs_large_p' in stats:
            print(f"    Mann-Whitney small vs large: p={stats['mannwhitney_small_vs_large_p']:.6f}")

    # Save comprehensive results
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(RESULTS_DIR, f'phase_a_10seeds_{timestamp}.json')

    final_results = {
        'conditions': convert_for_json(all_results),
        'aggregate': convert_for_json(aggregate),
        'statistics': convert_for_json(stats),
        'meta': {
            'network_sizes': list(NETWORK_SIZES),
            'all_seeds': list(ALL_SEEDS),
            'new_seeds': list(NEW_SEEDS),
            'generations': GENERATIONS,
            'population_size': POPULATION_SIZE,
            'total_conditions': len(all_results),
            'new_conditions_run': len(new_results),
            'existing_conditions_loaded': len(existing),
            'timestamp': timestamp,
        }
    }

    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return final_results


if __name__ == "__main__":
    main()
