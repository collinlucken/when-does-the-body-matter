"""
Mechanistic Analysis: Weight Configuration Comparison for High vs Low Embodiment Solutions.

Addresses v0.5 Review Issue #1 (Option A): Why does evolution discover positive
growth rate solutions in larger networks? What weight configuration properties
distinguish high-embodiment from low-embodiment solutions?

Analyzes all 60 conditions from Phase A, comparing:
- Input gain (weights from sensory neurons 0,1 to rest of network)
- Spectral radius of recurrent weight matrix
- Self-connection strengths
- Time constant distributions
- Bias distributions
- Total weight magnitude and connectivity
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu, pearsonr
from scipy.linalg import eigvals

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from simulation.evolutionary import GenotypeDecoder

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../../results/paper2')
NETWORK_SIZES = (2, 3, 4, 5, 6, 8)
ALL_SEEDS = (42, 137, 256, 512, 1024, 2048, 3141, 4096, 5555, 7777)

# Thresholds for classification
HIGH_THRESHOLD = 0.70  # constitutive-dominant
LOW_THRESHOLD = 0.30   # causal-dominant


def load_data():
    """Load phase A results and dynamical analysis."""
    results_path = Path(RESULTS_DIR)

    # Load phase A (constitutive scores + genotypes)
    phase_files = sorted(results_path.glob('phase_a_10seeds_*.json'), reverse=True)
    if not phase_files:
        raise FileNotFoundError("No phase_a_10seeds results found")
    with open(phase_files[0], 'r') as f:
        phase_data = json.load(f)

    # Load dynamical analysis (growth rates etc)
    dyn_files = sorted(results_path.glob('dynamical_analysis_60_*.json'), reverse=True)
    if not dyn_files:
        raise FileNotFoundError("No dynamical_analysis_60 results found")
    with open(dyn_files[0], 'r') as f:
        dyn_data = json.load(f)

    return phase_data, dyn_data


def decode_genotype(genotype_list, num_neurons):
    """Decode genotype into CTRNN parameters."""
    decoder = GenotypeDecoder(
        num_neurons=num_neurons,
        include_gains=False,
        tau_range=(0.5, 5.0),
        weight_range=(-10.0, 10.0),
        bias_range=(-10.0, 10.0),
    )
    genotype = np.array(genotype_list)
    params = decoder.decode(genotype)
    return params


def analyze_network(params, num_neurons):
    """Extract mechanistic properties from decoded CTRNN parameters."""
    weights = params['weights']  # (n, n) matrix
    tau = params['tau']          # (n,) vector
    biases = params['biases']    # (n,) vector

    # --- Input gain analysis ---
    # Sensory input goes to neurons 0 and 1 as external input (Ii)
    # The influence of sensory neurons on the network is through:
    # (a) Their self-connections and connections to other neurons
    # (b) The direct external input weighting

    # Input propagation: weights FROM sensory neurons (0,1) TO interneurons
    if num_neurons > 2:
        # Weights from sensor neurons to non-sensor neurons
        input_weights = weights[2:, :2]  # shape (n-2, 2)
        input_gain_mean = np.mean(np.abs(input_weights))
        input_gain_max = np.max(np.abs(input_weights))
        input_gain_total = np.sum(np.abs(input_weights))
    else:
        # Only 2 neurons: both are sensor+motor
        input_weights = weights[:, :2]
        input_gain_mean = np.mean(np.abs(input_weights))
        input_gain_max = np.max(np.abs(input_weights))
        input_gain_total = np.sum(np.abs(input_weights))

    # --- Spectral radius ---
    eigenvalues = eigvals(weights)
    spectral_radius = np.max(np.abs(eigenvalues))
    max_real_eigenvalue = np.max(np.real(eigenvalues))

    # --- Self-connections ---
    self_connections = np.diag(weights)
    mean_self_connection = np.mean(self_connections)
    max_self_connection = np.max(self_connections)
    min_self_connection = np.min(self_connections)
    positive_self_frac = np.mean(self_connections > 0)

    # --- Recurrent weights (excluding diagonal) ---
    mask = ~np.eye(num_neurons, dtype=bool)
    recurrent_weights = weights[mask]
    recurrent_mean_abs = np.mean(np.abs(recurrent_weights))
    recurrent_max_abs = np.max(np.abs(recurrent_weights))
    recurrent_total = np.sum(np.abs(recurrent_weights))

    # --- Connection density (fraction of weights > 1.0 in absolute value) ---
    strong_connections = np.sum(np.abs(weights[mask]) > 1.0) / len(recurrent_weights)

    # --- Tau distribution ---
    tau_mean = np.mean(tau)
    tau_std = np.std(tau)
    tau_ratio = np.max(tau) / (np.min(tau) + 1e-8)  # dynamic range
    tau_min = np.min(tau)
    tau_max = np.max(tau)

    # Sensor neuron taus
    sensor_tau_mean = np.mean(tau[:2])
    if num_neurons > 2:
        interneuron_tau_mean = np.mean(tau[2:])
    else:
        interneuron_tau_mean = sensor_tau_mean

    # --- Bias distribution ---
    bias_mean = np.mean(biases)
    bias_std = np.std(biases)
    bias_abs_mean = np.mean(np.abs(biases))

    # --- Total weight magnitude ---
    total_weight_magnitude = np.sum(np.abs(weights))
    mean_weight_magnitude = np.mean(np.abs(weights))

    # --- Effective gain (spectral radius / mean tau) ---
    # Higher values mean signals amplify faster relative to decay
    effective_gain = spectral_radius / tau_mean

    return {
        # Input gain
        'input_gain_mean': input_gain_mean,
        'input_gain_max': input_gain_max,
        'input_gain_total': input_gain_total,
        # Spectral properties
        'spectral_radius': spectral_radius,
        'max_real_eigenvalue': max_real_eigenvalue,
        'effective_gain': effective_gain,
        # Self-connections
        'mean_self_connection': mean_self_connection,
        'max_self_connection': max_self_connection,
        'min_self_connection': min_self_connection,
        'positive_self_frac': positive_self_frac,
        # Recurrent weights
        'recurrent_mean_abs': recurrent_mean_abs,
        'recurrent_max_abs': recurrent_max_abs,
        'recurrent_total': recurrent_total,
        'strong_connection_frac': strong_connections,
        # Tau
        'tau_mean': tau_mean,
        'tau_std': tau_std,
        'tau_ratio': tau_ratio,
        'tau_min': tau_min,
        'tau_max': tau_max,
        'sensor_tau_mean': sensor_tau_mean,
        'interneuron_tau_mean': interneuron_tau_mean,
        # Bias
        'bias_mean': bias_mean,
        'bias_std': bias_std,
        'bias_abs_mean': bias_abs_mean,
        # Total
        'total_weight_magnitude': total_weight_magnitude,
        'mean_weight_magnitude': mean_weight_magnitude,
    }


def main():
    print("=" * 70)
    print("MECHANISTIC ANALYSIS: HIGH vs LOW EMBODIMENT SOLUTIONS")
    print("=" * 70)

    phase_data, dyn_data = load_data()
    conditions = phase_data['conditions']
    dyn_conditions = dyn_data['conditions']

    # Collect all data
    all_results = []
    skipped = 0

    for ns in NETWORK_SIZES:
        for s in ALL_SEEDS:
            run_id = f"net{ns}_seed{s}"
            cond = conditions.get(run_id, {})

            # Get constitutive score
            if 'error' in cond or 'scores' not in cond:
                skipped += 1
                continue
            const_score = cond['scores']['constitutive']

            # Get genotype
            genotype = cond.get('evolution', {}).get('best_genotype', None)
            if genotype is None:
                skipped += 1
                continue

            # Get dynamical data
            dyn = dyn_conditions.get(run_id, {})
            growth_rate = dyn.get('perturbation_sensitivity', {}).get('mean_growth_rate', None)
            pr = dyn.get('trajectory_complexity', {}).get('participation_ratio', None)

            # Decode and analyze
            try:
                params = decode_genotype(genotype, ns)
                props = analyze_network(params, ns)
            except Exception as e:
                print(f"  ERROR decoding {run_id}: {e}")
                skipped += 1
                continue

            props['run_id'] = run_id
            props['num_neurons'] = ns
            props['seed'] = s
            props['constitutive_score'] = const_score
            props['growth_rate'] = growth_rate
            props['participation_ratio'] = pr

            all_results.append(props)

    print(f"\nAnalyzed: {len(all_results)} conditions (skipped: {skipped})")

    # Classify
    high = [r for r in all_results if r['constitutive_score'] >= HIGH_THRESHOLD]
    low = [r for r in all_results if r['constitutive_score'] < LOW_THRESHOLD]
    mid = [r for r in all_results if LOW_THRESHOLD <= r['constitutive_score'] < HIGH_THRESHOLD]

    print(f"HIGH embodiment (score >= {HIGH_THRESHOLD}): {len(high)}")
    print(f"LOW embodiment (score < {LOW_THRESHOLD}): {len(low)}")
    print(f"MIXED: {len(mid)}")

    # --- Property comparison ---
    properties_to_compare = [
        'spectral_radius', 'max_real_eigenvalue', 'effective_gain',
        'input_gain_mean', 'input_gain_max', 'input_gain_total',
        'mean_self_connection', 'max_self_connection', 'positive_self_frac',
        'recurrent_mean_abs', 'recurrent_total', 'strong_connection_frac',
        'tau_mean', 'tau_std', 'tau_ratio', 'sensor_tau_mean', 'interneuron_tau_mean',
        'bias_abs_mean', 'bias_std',
        'total_weight_magnitude', 'mean_weight_magnitude',
    ]

    print(f"\n{'='*70}")
    print("PROPERTY COMPARISON: HIGH vs LOW EMBODIMENT")
    print(f"{'='*70}")
    print(f"{'Property':<28} {'High (n={})'.format(len(high)):<18} {'Low (n={})'.format(len(low)):<18} {'Mann-Wh p':<12} {'Effect'}")
    print("-" * 88)

    comparison_results = {}
    for prop in properties_to_compare:
        h_vals = [r[prop] for r in high if r[prop] is not None]
        l_vals = [r[prop] for r in low if r[prop] is not None]

        if len(h_vals) < 3 or len(l_vals) < 3:
            continue

        h_mean = np.mean(h_vals)
        l_mean = np.mean(l_vals)
        h_std = np.std(h_vals)
        l_std = np.std(l_vals)

        # Mann-Whitney U test
        try:
            stat, p_val = mannwhitneyu(h_vals, l_vals, alternative='two-sided')
        except:
            p_val = 1.0

        # Cohen's d
        pooled_std = np.sqrt((h_std**2 + l_std**2) / 2) if (h_std + l_std) > 0 else 1.0
        cohens_d = (h_mean - l_mean) / pooled_std if pooled_std > 0 else 0.0

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        print(f"  {prop:<26} {h_mean:>7.3f}±{h_std:<6.3f} {l_mean:>7.3f}±{l_std:<6.3f} {p_val:<12.4f} d={cohens_d:+.2f} {sig}")

        comparison_results[prop] = {
            'high_mean': h_mean, 'high_std': h_std,
            'low_mean': l_mean, 'low_std': l_std,
            'mann_whitney_p': p_val, 'cohens_d': cohens_d,
        }

    # --- Correlations with constitutive score ---
    print(f"\n{'='*70}")
    print("CORRELATIONS WITH CONSTITUTIVE SCORE (all n={})".format(len(all_results)))
    print(f"{'='*70}")

    cs_vals = [r['constitutive_score'] for r in all_results]
    correlation_results = {}

    for prop in properties_to_compare:
        vals = [r[prop] for r in all_results]
        if any(v is None for v in vals):
            continue
        rho, p = spearmanr(vals, cs_vals)
        r_p, p_p = pearsonr(vals, cs_vals)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {prop:<28} Spearman rho={rho:+.3f} (p={p:.4f}) {sig}  Pearson r={r_p:+.3f}")
        correlation_results[prop] = {'spearman_rho': rho, 'spearman_p': p, 'pearson_r': r_p}

    # --- Within-size analysis (controlling for network size) ---
    print(f"\n{'='*70}")
    print("WITHIN-SIZE CORRELATIONS (controlling for architecture)")
    print(f"{'='*70}")

    key_props = ['spectral_radius', 'effective_gain', 'input_gain_mean',
                 'mean_self_connection', 'tau_ratio', 'recurrent_mean_abs']

    for ns in NETWORK_SIZES:
        subset = [r for r in all_results if r['num_neurons'] == ns]
        if len(subset) < 5:
            continue
        cs_sub = [r['constitutive_score'] for r in subset]
        print(f"\n  Size {ns} (n={len(subset)}):")
        for prop in key_props:
            vals = [r[prop] for r in subset]
            if any(v is None for v in vals):
                continue
            rho, p = spearmanr(vals, cs_sub)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {prop:<24} rho={rho:+.3f} (p={p:.3f}) {sig}")

    # --- Multivariate: what predicts constitutive score? ---
    print(f"\n{'='*70}")
    print("STEPWISE VARIANCE EXPLANATION")
    print(f"{'='*70}")

    # Compute R² for individual predictors and combinations
    from numpy.linalg import lstsq

    cs = np.array(cs_vals)
    n = len(cs)
    ss_total = np.sum((cs - np.mean(cs))**2)

    print("\n  Single predictors (R²):")
    r2_results = {}
    for prop in ['spectral_radius', 'effective_gain', 'input_gain_mean',
                 'mean_self_connection', 'tau_ratio', 'recurrent_mean_abs',
                 'tau_mean', 'bias_abs_mean', 'total_weight_magnitude',
                 'strong_connection_frac', 'positive_self_frac']:
        vals = np.array([r[prop] for r in all_results])
        X = np.column_stack([vals, np.ones(n)])
        beta, _, _, _ = lstsq(X, cs, rcond=None)
        predicted = X @ beta
        ss_res = np.sum((cs - predicted)**2)
        r2 = 1 - ss_res / ss_total
        r2_results[prop] = r2
        print(f"    {prop:<28} R² = {r2:.4f}")

    # Combinations
    print("\n  Two-predictor models (R²):")
    size_vals = np.array([r['num_neurons'] for r in all_results])
    gr_vals = np.array([r['growth_rate'] if r['growth_rate'] is not None else 0 for r in all_results])

    for prop in ['spectral_radius', 'effective_gain', 'input_gain_mean',
                 'mean_self_connection', 'tau_ratio']:
        vals = np.array([r[prop] for r in all_results])
        # Size + property
        X = np.column_stack([size_vals, vals, np.ones(n)])
        beta, _, _, _ = lstsq(X, cs, rcond=None)
        predicted = X @ beta
        r2 = 1 - np.sum((cs - predicted)**2) / ss_total
        print(f"    size + {prop:<22} R² = {r2:.4f}")

    # Growth rate + property
    print("\n  Growth rate + property (R²):")
    for prop in ['spectral_radius', 'effective_gain', 'input_gain_mean',
                 'mean_self_connection', 'tau_ratio']:
        vals = np.array([r[prop] for r in all_results])
        X = np.column_stack([gr_vals, vals, np.ones(n)])
        beta, _, _, _ = lstsq(X, cs, rcond=None)
        predicted = X @ beta
        r2 = 1 - np.sum((cs - predicted)**2) / ss_total
        print(f"    growth_rate + {prop:<18} R² = {r2:.4f}")

    # Full model: size + growth rate + best weight properties
    print("\n  Multi-predictor models:")
    for combo_name, combo_props in [
        ("size + growth_rate", [size_vals, gr_vals]),
        ("size + growth_rate + spectral_radius",
         [size_vals, gr_vals, np.array([r['spectral_radius'] for r in all_results])]),
        ("size + growth_rate + effective_gain",
         [size_vals, gr_vals, np.array([r['effective_gain'] for r in all_results])]),
        ("size + growth_rate + effective_gain + tau_ratio",
         [size_vals, gr_vals,
          np.array([r['effective_gain'] for r in all_results]),
          np.array([r['tau_ratio'] for r in all_results])]),
        ("size + growth_rate + effective_gain + input_gain_mean",
         [size_vals, gr_vals,
          np.array([r['effective_gain'] for r in all_results]),
          np.array([r['input_gain_mean'] for r in all_results])]),
    ]:
        X = np.column_stack(combo_props + [np.ones(n)])
        beta, _, _, _ = lstsq(X, cs, rcond=None)
        predicted = X @ beta
        r2 = 1 - np.sum((cs - predicted)**2) / ss_total
        print(f"    {combo_name:<48} R² = {r2:.4f}")

    # --- Summary by size ---
    print(f"\n{'='*70}")
    print("KEY PROPERTIES BY NETWORK SIZE")
    print(f"{'='*70}")

    for ns in NETWORK_SIZES:
        subset = [r for r in all_results if r['num_neurons'] == ns]
        if not subset:
            continue
        print(f"\n  Size {ns} (n={len(subset)}):")
        print(f"    Constitutive score:  {np.mean([r['constitutive_score'] for r in subset]):.3f} ± {np.std([r['constitutive_score'] for r in subset]):.3f}")
        print(f"    Spectral radius:    {np.mean([r['spectral_radius'] for r in subset]):.3f} ± {np.std([r['spectral_radius'] for r in subset]):.3f}")
        print(f"    Effective gain:     {np.mean([r['effective_gain'] for r in subset]):.3f} ± {np.std([r['effective_gain'] for r in subset]):.3f}")
        print(f"    Input gain (mean):  {np.mean([r['input_gain_mean'] for r in subset]):.3f} ± {np.std([r['input_gain_mean'] for r in subset]):.3f}")
        print(f"    Self-connection:    {np.mean([r['mean_self_connection'] for r in subset]):.3f} ± {np.std([r['mean_self_connection'] for r in subset]):.3f}")
        print(f"    Tau ratio:          {np.mean([r['tau_ratio'] for r in subset]):.3f} ± {np.std([r['tau_ratio'] for r in subset]):.3f}")
        print(f"    Tau mean:           {np.mean([r['tau_mean'] for r in subset]):.3f} ± {np.std([r['tau_mean'] for r in subset]):.3f}")
        if ns > 2:
            print(f"    Sensor tau:         {np.mean([r['sensor_tau_mean'] for r in subset]):.3f} ± {np.std([r['sensor_tau_mean'] for r in subset]):.3f}")
            print(f"    Interneuron tau:    {np.mean([r['interneuron_tau_mean'] for r in subset]):.3f} ± {np.std([r['interneuron_tau_mean'] for r in subset]):.3f}")

    # --- Save results ---
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(i) for i in obj]
        return obj

    save_data = {
        'meta': {'timestamp': timestamp, 'n_conditions': len(all_results),
                 'n_high': len(high), 'n_low': len(low), 'n_mid': len(mid)},
        'comparison': convert(comparison_results),
        'correlations': convert(correlation_results),
        'r2_single': convert(r2_results),
        'conditions': convert(all_results),
    }

    outfile = os.path.join(RESULTS_DIR, f'mechanistic_analysis_{timestamp}.json')
    with open(outfile, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to: {outfile}")


if __name__ == "__main__":
    main()
