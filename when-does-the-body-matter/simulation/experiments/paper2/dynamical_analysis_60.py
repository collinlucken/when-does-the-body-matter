"""
Dynamical Analysis for ALL 60 conditions from the 10-seed Phase A experiment.

Uses saved genotypes where available (42 conditions), re-evolves for the
original 18 conditions that lack saved genotypes.

Computes participation ratio, Lyapunov spectrum, perturbation sensitivity
for every condition, then correlates with constitutive scores.
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

from simulation.experiments.paper2.dynamical_analysis import (
    compute_lyapunov_spectrum,
    compute_trajectory_complexity,
    compute_perturbation_sensitivity,
    evolve_network,
)
from simulation.evolutionary import GenotypeDecoder

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../../results/paper2')
NETWORK_SIZES = (2, 3, 4, 5, 6, 8)
ALL_SEEDS = (42, 137, 256, 512, 1024, 2048, 3141, 4096, 5555, 7777)
NUM_WORKERS = min(4, cpu_count())


def load_all_data():
    """Load the 10-seed results file."""
    results_path = Path(RESULTS_DIR)
    json_files = sorted(results_path.glob('phase_a_10seeds_*.json'), reverse=True)
    if not json_files:
        raise FileNotFoundError("No phase_a_10seeds results found")
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    return data


def analyze_single(args):
    """Worker: run dynamical analysis for one condition."""
    net_size, seed, genotype_list, condition_id, total = args
    run_id = f"net{net_size}_seed{seed}"
    start_time = time.time()

    decoder = GenotypeDecoder(
        num_neurons=net_size,
        include_gains=False,
        tau_range=(0.5, 5.0),
        weight_range=(-10.0, 10.0),
        bias_range=(-10.0, 10.0),
    )

    if genotype_list is not None:
        genotype = np.array(genotype_list)
        params = decoder.decode(genotype)
        source = 'loaded'
    else:
        print(f"  [{condition_id}/{total}] {run_id}: re-evolving (2000 gen)...", flush=True)
        params, _ = evolve_network(net_size, generations=2000, seed=seed)
        source = 'evolved'

    # Compute all measures
    try:
        lyap = compute_lyapunov_spectrum(params, net_size, trial_duration=500)
    except Exception as e:
        lyap = {'max_lyapunov': 0.0, 'mean_lyapunov': 0.0, 'error': str(e)}

    try:
        complexity = compute_trajectory_complexity(params, net_size)
    except Exception as e:
        complexity = {'participation_ratio': 0.0, 'error': str(e)}

    try:
        perturbation = compute_perturbation_sensitivity(params, net_size)
    except Exception as e:
        perturbation = {'mean_growth_rate': 0.0, 'error': str(e)}

    elapsed = time.time() - start_time
    pr = complexity.get('participation_ratio', 0)
    gr = perturbation.get('mean_growth_rate', 0)
    print(f"  [{condition_id}/{total}] {run_id} ({source}): PR={pr:.3f}, growth={gr:.3f}, "
          f"time={elapsed:.1f}s", flush=True)

    return run_id, {
        'num_neurons': net_size,
        'seed': seed,
        'genotype_source': source,
        'lyapunov': lyap,
        'trajectory_complexity': complexity,
        'perturbation_sensitivity': perturbation,
        'timing': {'elapsed_seconds': elapsed},
    }


def main():
    print("=" * 70)
    print("DYNAMICAL ANALYSIS: ALL 60 CONDITIONS")
    print("=" * 70)

    data = load_all_data()
    conditions = data['conditions']

    # Build task list
    tasks = []
    cid = 0
    for ns in NETWORK_SIZES:
        for s in ALL_SEEDS:
            cid += 1
            run_id = f"net{ns}_seed{s}"
            geno = conditions.get(run_id, {}).get('evolution', {}).get('best_genotype', None)
            tasks.append((ns, s, geno, cid, 60))

    print(f"Total conditions: {len(tasks)}")
    with_geno = sum(1 for t in tasks if t[2] is not None)
    print(f"  With saved genotype: {with_geno}")
    print(f"  Need re-evolution: {len(tasks) - with_geno}")
    print(f"Workers: {NUM_WORKERS}")
    print("=" * 70)

    start = time.time()
    results = {}
    with Pool(processes=NUM_WORKERS) as pool:
        outputs = pool.map(analyze_single, tasks)
    for run_id, result in outputs:
        results[run_id] = result

    elapsed = time.time() - start
    print(f"\nAll done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Load constitutive scores for correlation
    const_scores = {}
    for k, v in conditions.items():
        if 'error' not in v and 'scores' in v:
            const_scores[k] = v['scores']['constitutive']

    # Compute correlations
    from scipy.stats import spearmanr, pearsonr

    pr_vals, cs_vals, sizes_vals = [], [], []
    gr_vals, fa_vals, ly_vals = [], [], []
    for run_id, r in results.items():
        if run_id in const_scores and 'error' not in r.get('trajectory_complexity', {}):
            pr_vals.append(r['trajectory_complexity'].get('participation_ratio', 0))
            gr_vals.append(r['perturbation_sensitivity'].get('mean_growth_rate', 0))
            fa_vals.append(r['perturbation_sensitivity'].get('fraction_amplifying', 0))
            ly_vals.append(r['lyapunov'].get('max_lyapunov', 0))
            cs_vals.append(const_scores[run_id])
            sizes_vals.append(r['num_neurons'])

    n = len(pr_vals)
    print(f"\n{'='*70}")
    print(f"CORRELATIONS WITH CONSTITUTIVE SCORE (n={n})")
    print(f"{'='*70}")

    for name, vals in [('Participation Ratio', pr_vals), ('Growth Rate', gr_vals),
                       ('Frac Amplifying', fa_vals), ('Max Lyapunov', ly_vals)]:
        rho, p = spearmanr(vals, cs_vals)
        r_p, p_p = pearsonr(vals, cs_vals)
        print(f"  {name}:")
        print(f"    Spearman: rho={rho:.4f}, p={p:.6f}")
        print(f"    Pearson:  r={r_p:.4f}, p={p_p:.6f}")

    # Also: PR vs network size
    rho_sz, p_sz = spearmanr(sizes_vals, pr_vals)
    print(f"\n  PR vs Network Size: rho={rho_sz:.4f}, p={p_sz:.6f}")

    # Aggregate by size
    print(f"\n{'='*70}")
    print("AGGREGATE BY NETWORK SIZE")
    print(f"{'='*70}")
    for ns in NETWORK_SIZES:
        prs = [results[f"net{ns}_seed{s}"]['trajectory_complexity'].get('participation_ratio', 0)
               for s in ALL_SEEDS if f"net{ns}_seed{s}" in results
               and 'error' not in results[f"net{ns}_seed{s}"].get('trajectory_complexity', {})]
        grs = [results[f"net{ns}_seed{s}"]['perturbation_sensitivity'].get('mean_growth_rate', 0)
               for s in ALL_SEEDS if f"net{ns}_seed{s}" in results]
        fas = [results[f"net{ns}_seed{s}"]['perturbation_sensitivity'].get('fraction_amplifying', 0)
               for s in ALL_SEEDS if f"net{ns}_seed{s}" in results]
        lyaps = [results[f"net{ns}_seed{s}"]['lyapunov'].get('max_lyapunov', 0)
                 for s in ALL_SEEDS if f"net{ns}_seed{s}" in results]

        if prs:
            print(f"\n  {ns} neurons (n={len(prs)}):")
            print(f"    PR:          {np.mean(prs):.3f} ± {np.std(prs):.3f}")
            print(f"    Growth rate: {np.mean(grs):.4f} ± {np.std(grs):.4f}")
            print(f"    Frac ampl:   {np.mean(fas):.2f} ± {np.std(fas):.2f}")
            print(f"    Max Lyap:    {np.mean(lyaps):.4f} ± {np.std(lyaps):.4f}")

    # Save
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = os.path.join(RESULTS_DIR, f'dynamical_analysis_60_{timestamp}.json')

    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(i) for i in obj]
        return obj

    save = {
        'conditions': convert(results),
        'correlations': {
            'pr_vs_constitutive': {'rho': float(spearmanr(pr_vals, cs_vals)[0]),
                                    'p': float(spearmanr(pr_vals, cs_vals)[1])},
            'gr_vs_constitutive': {'rho': float(spearmanr(gr_vals, cs_vals)[0]),
                                    'p': float(spearmanr(gr_vals, cs_vals)[1])},
            'fa_vs_constitutive': {'rho': float(spearmanr(fa_vals, cs_vals)[0]),
                                    'p': float(spearmanr(fa_vals, cs_vals)[1])},
            'pr_vs_size': {'rho': float(rho_sz), 'p': float(p_sz)},
            'n': n,
        },
        'meta': {'timestamp': timestamp, 'total_conditions': len(results)},
    }
    with open(outfile, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"\nSaved to: {outfile}")


if __name__ == "__main__":
    main()
