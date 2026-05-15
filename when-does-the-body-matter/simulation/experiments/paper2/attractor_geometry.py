"""
Attractor Geometry Analysis for Paper 2 v0.7.

Addresses review improvement #1: Complete mechanistic chain with attractor characterization.

For each decoded genotype (42 conditions with saved genotypes), this script:
1. Reconstructs the CTRNN from decoded parameters
2. Finds fixed points under multiple input conditions (zero, low, medium, high sensory)
3. Computes Jacobian eigenvalues at each fixed point (stability)
4. Classifies attractor type: stable FP, unstable FP, limit cycle, chaotic
5. Measures bifurcation proximity: how close is the system to qualitative change?
6. Validates causal chain: self-connection -> eigenvalue regime -> attractor type -> ED

Key hypotheses being tested:
- H1: High-ED solutions sit near bifurcation boundaries
- H2: High-ED solutions have limit cycle / chaotic attractors; low-ED have stable FPs
- H3: Positive self-connections produce eigenvalues with positive real parts (instability)
- H4: Number of qualitatively distinct attractor regimes correlates with ED
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
from scipy.optimize import fsolve
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from simulation.ctrnn import CTRNN
from simulation.evolutionary import GenotypeDecoder

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../../results/paper2')
NETWORK_SIZES = (2, 3, 4, 5, 6, 8)
ALL_SEEDS = (42, 137, 256, 512, 1024, 2048, 3141, 4096, 5555, 7777)

# Ghost condition thresholds (same as mechanistic_analysis.py)
HIGH_THRESHOLD = 0.70
LOW_THRESHOLD = 0.30

# Input conditions to test: represent different sensory regimes
# In phototaxis, sensory input typically ranges from 0 to ~1.0
INPUT_AMPLITUDES = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]


def load_data():
    """Load phase A results and dynamical analysis."""
    results_path = Path(RESULTS_DIR)
    phase_files = sorted(results_path.glob('phase_a_10seeds_*.json'), reverse=True)
    if not phase_files:
        raise FileNotFoundError("No phase_a_10seeds results found")
    with open(phase_files[0], 'r') as f:
        phase_data = json.load(f)

    dyn_files = sorted(results_path.glob('dynamical_analysis_60_*.json'), reverse=True)
    dyn_data = None
    if dyn_files:
        with open(dyn_files[0], 'r') as f:
            dyn_data = json.load(f)

    mech_files = sorted(results_path.glob('mechanistic_analysis_*.json'), reverse=True)
    mech_data = None
    if mech_files:
        with open(mech_files[0], 'r') as f:
            mech_data = json.load(f)

    return phase_data, dyn_data, mech_data


def build_ctrnn(genotype_list, num_neurons):
    """Build a CTRNN from a genotype vector."""
    decoder = GenotypeDecoder(
        num_neurons=num_neurons,
        include_gains=False,
        tau_range=(0.5, 5.0),
        weight_range=(-10.0, 10.0),
        bias_range=(-10.0, 10.0),
    )
    genotype = np.array(genotype_list)
    params = decoder.decode(genotype)

    net = CTRNN(
        num_neurons=num_neurons,
        time_constants=params['tau'],
        weights=params['weights'],
        biases=params['biases'],
        step_size=0.01,
        center_crossing=True
    )
    return net, params


def find_fixed_points_numerical(net, input_amplitude, n_starts=20, tol=1e-8):
    """
    Find fixed points by (a) long-time simulation from random ICs and
    (b) numerical root-finding on the steady-state equation.

    At a fixed point: dy/dt = 0
    => y = W * sigma(y + theta) + I

    Returns list of unique fixed points (as arrays).
    """
    n = net.num_neurons
    ext_input = np.zeros(n)
    # Sensory input goes to first 2 neurons
    ext_input[:min(2, n)] = input_amplitude

    found_fps = []

    for trial in range(n_starts):
        # Random initial condition in [-5, 5]
        y0 = np.random.uniform(-5, 5, n)

        # Define the residual: at FP, dy/dt = 0
        # dy/dt = (-y + W * sigma(y + theta) + I) / tau
        # Steady state: y = W * sigma(y + theta) + I
        def residual(y):
            activation = net.biases + y
            if net.center_crossing:
                sig = 2.0 / (1.0 + np.exp(-np.clip(activation, -500, 500))) - 1.0
            else:
                sig = 1.0 / (1.0 + np.exp(-np.clip(activation, -500, 500)))
            return -y + np.dot(net.weights, sig) + ext_input

        try:
            fp, info, ier, mesg = fsolve(residual, y0, full_output=True)
            if ier == 1:  # converged
                # Check it's actually a FP
                res_norm = np.linalg.norm(residual(fp))
                if res_norm < tol:
                    found_fps.append(fp)
        except Exception:
            pass

    # Also try simulation-based: run network for a long time
    for trial in range(n_starts):
        net.reset(np.random.uniform(-3, 3, n).astype(np.float64))
        for _ in range(5000):
            net.step(ext_input)
        # Check if it converged to FP (low velocity)
        state = net.get_state()
        activation = net.biases + state
        if net.center_crossing:
            sig = 2.0 / (1.0 + np.exp(-np.clip(activation, -500, 500))) - 1.0
        else:
            sig = 1.0 / (1.0 + np.exp(-np.clip(activation, -500, 500)))
        dy = (-state + np.dot(net.weights, sig) + ext_input) / net.tau
        velocity = np.linalg.norm(dy)
        if velocity < 0.01:
            found_fps.append(state)

    # Deduplicate: cluster FPs within tolerance
    if len(found_fps) == 0:
        return []

    unique_fps = [found_fps[0]]
    for fp in found_fps[1:]:
        is_new = True
        for ufp in unique_fps:
            if np.linalg.norm(fp - ufp) < 0.1:
                is_new = False
                break
        if is_new:
            unique_fps.append(fp)

    return unique_fps


def classify_attractor(net, input_amplitude, warmup=2000, measure_steps=3000):
    """
    Classify the attractor type by running the network and analyzing the trajectory.

    Returns dict with:
    - 'type': 'fixed_point', 'limit_cycle', 'chaotic', or 'transient'
    - 'trajectory_variance': variance of neural states (low = FP, moderate = LC, high = chaotic)
    - 'max_lyapunov': estimated max Lyapunov exponent
    - 'oscillation_frequency': dominant frequency if oscillatory
    """
    n = net.num_neurons
    ext_input = np.zeros(n)
    ext_input[:min(2, n)] = input_amplitude

    # Warmup
    net.reset(np.random.uniform(-1, 1, n).astype(np.float64))
    for _ in range(warmup):
        net.step(ext_input)

    # Record trajectory
    states = np.zeros((measure_steps, n))
    for t in range(measure_steps):
        net.step(ext_input)
        states[t] = net.get_state()

    # Compute trajectory variance (per-neuron, then average)
    traj_var = np.mean(np.var(states, axis=0))

    # Check if settled to fixed point
    late_states = states[-500:]
    late_var = np.mean(np.var(late_states, axis=0))

    # Estimate Lyapunov exponent via perturbation
    # Run two nearby trajectories
    state_ref = states[-1].copy()
    net.set_state(state_ref)
    perturb = np.random.randn(n) * 1e-6
    perturb_norm = np.linalg.norm(perturb)

    state_perturbed = state_ref + perturb

    # Run both forward
    lyap_steps = 500
    net.set_state(state_ref)
    for _ in range(lyap_steps):
        net.step(ext_input)
    state_ref_final = net.get_state()

    net.set_state(state_perturbed)
    for _ in range(lyap_steps):
        net.step(ext_input)
    state_pert_final = net.get_state()

    sep = np.linalg.norm(state_ref_final - state_pert_final)
    if perturb_norm > 0 and sep > 0:
        lyap = np.log(sep / perturb_norm) / (lyap_steps * net.step_size)
    else:
        lyap = 0.0

    # Oscillation detection via autocorrelation
    if n >= 1:
        # Use first neuron's trajectory for frequency analysis
        signal = late_states[:, 0] - np.mean(late_states[:, 0])
        if np.std(signal) > 1e-8:
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            # Find first peak after initial decay
            peaks = []
            for i in range(1, len(autocorr)-1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.3:
                    peaks.append(i)
                    break
            if peaks:
                osc_period = peaks[0] * net.step_size
                osc_freq = 1.0 / osc_period if osc_period > 0 else 0.0
            else:
                osc_freq = 0.0
        else:
            osc_freq = 0.0
    else:
        osc_freq = 0.0

    # Classify
    if late_var < 1e-6:
        attractor_type = 'fixed_point'
    elif lyap > 0.1:
        attractor_type = 'chaotic'
    elif osc_freq > 0:
        attractor_type = 'limit_cycle'
    elif late_var > 0.01:
        attractor_type = 'quasi_periodic'
    else:
        attractor_type = 'fixed_point'

    return {
        'type': attractor_type,
        'trajectory_variance': float(traj_var),
        'late_variance': float(late_var),
        'max_lyapunov': float(lyap),
        'oscillation_frequency': float(osc_freq),
    }


def compute_jacobian_eigenvalues(net, state, input_amplitude):
    """Compute Jacobian eigenvalues at a specific state with given input."""
    n = net.num_neurons
    ext_input = np.zeros(n)
    ext_input[:min(2, n)] = input_amplitude

    # Compute Jacobian of continuous-time system (NOT discrete)
    # dy/dt = (-y + W * sigma(y + theta) + I) / tau
    # J_ij = (1/tau_i) * (-delta_ij + W_ij * sigma'(y_j + theta_j))
    activation = net.biases + state
    if net.center_crossing:
        sig = 2.0 / (1.0 + np.exp(-np.clip(activation, -500, 500))) - 1.0
        # Derivative of center-crossing sigmoid: (1 - sig^2) / 2 ... wait
        # sigma(x) = 2/(1+exp(-x)) - 1
        # sigma'(x) = 2 * exp(-x) / (1+exp(-x))^2 = (1 - sigma(x)^2) / 2
        # Actually: let s = 1/(1+exp(-x)), then sigma = 2s-1
        # sigma' = 2 * s * (1-s) = 2 * ((sigma+1)/2) * ((1-sigma)/2) = (1-sigma^2)/2
        sig_deriv = (1 - sig**2) / 2.0
    else:
        sig = 1.0 / (1.0 + np.exp(-np.clip(activation, -500, 500)))
        sig_deriv = sig * (1 - sig)

    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                J[i, j] = (-1.0 + net.weights[i, j] * sig_deriv[j]) / net.tau[i]
            else:
                J[i, j] = (net.weights[i, j] * sig_deriv[j]) / net.tau[i]

    eigenvalues = np.linalg.eigvals(J)
    return eigenvalues


def measure_bifurcation_proximity(net, input_range=np.linspace(0, 1.5, 30)):
    """
    Scan input amplitude and detect qualitative changes in dynamics.

    Bifurcation proximity = distance (in input space) to nearest qualitative change.
    Returns the minimum distance to a bifurcation point.
    """
    n = net.num_neurons
    prev_type = None
    bifurcation_points = []
    attractor_sequence = []

    for amp in input_range:
        result = classify_attractor(net, amp, warmup=1000, measure_steps=1500)
        attractor_sequence.append({
            'input_amplitude': float(amp),
            'type': result['type'],
            'late_variance': result['late_variance'],
            'lyapunov': result['max_lyapunov'],
        })

        if prev_type is not None and result['type'] != prev_type:
            bifurcation_points.append(float(amp))
        prev_type = result['type']

    # Bifurcation proximity: distance from typical operating input (0.5) to nearest bifurcation
    operating_input = 0.5
    if bifurcation_points:
        distances = [abs(bp - operating_input) for bp in bifurcation_points]
        min_distance = min(distances)
    else:
        min_distance = float('inf')  # no bifurcation found in scanned range

    return {
        'bifurcation_points': bifurcation_points,
        'n_bifurcations': len(bifurcation_points),
        'min_bifurcation_distance': min_distance,
        'n_distinct_regimes': len(set(a['type'] for a in attractor_sequence)),
        'attractor_sequence': attractor_sequence,
    }


def analyze_condition(net, params, num_neurons, const_score):
    """Run full attractor geometry analysis for one condition."""
    results = {
        'num_neurons': num_neurons,
        'constitutive_score': const_score,
    }

    # 1. Fixed point analysis under multiple input conditions
    fp_counts = []
    eigenvalue_data = []

    for amp in INPUT_AMPLITUDES:
        fps = find_fixed_points_numerical(net, amp, n_starts=15)
        fp_counts.append(len(fps))

        # Eigenvalue analysis at each fixed point
        for fp in fps:
            eigs = compute_jacobian_eigenvalues(net, fp, amp)
            max_real = float(np.max(np.real(eigs)))
            has_complex = bool(np.any(np.abs(np.imag(eigs)) > 0.01))
            eigenvalue_data.append({
                'input_amplitude': float(amp),
                'max_real_eigenvalue': max_real,
                'is_stable': max_real < 0,
                'has_complex_eigenvalues': has_complex,
                'eigenvalues_real': [float(np.real(e)) for e in eigs],
                'eigenvalues_imag': [float(np.imag(e)) for e in eigs],
            })

    results['fixed_points'] = {
        'counts_by_input': {str(amp): cnt for amp, cnt in zip(INPUT_AMPLITUDES, fp_counts)},
        'mean_fp_count': float(np.mean(fp_counts)),
        'max_fp_count': int(np.max(fp_counts)),
        'varies_with_input': len(set(fp_counts)) > 1,
    }

    # Summarize eigenvalue statistics
    if eigenvalue_data:
        all_max_real = [e['max_real_eigenvalue'] for e in eigenvalue_data]
        stable_frac = sum(1 for e in eigenvalue_data if e['is_stable']) / len(eigenvalue_data)
        complex_frac = sum(1 for e in eigenvalue_data if e['has_complex_eigenvalues']) / len(eigenvalue_data)
        results['eigenvalues'] = {
            'mean_max_real': float(np.mean(all_max_real)),
            'max_max_real': float(np.max(all_max_real)),
            'stable_fp_fraction': float(stable_frac),
            'complex_eigenvalue_fraction': float(complex_frac),
            'detail': eigenvalue_data,
        }
    else:
        results['eigenvalues'] = {
            'mean_max_real': 0.0, 'max_max_real': 0.0,
            'stable_fp_fraction': 1.0, 'complex_eigenvalue_fraction': 0.0,
            'detail': [],
        }

    # 2. Attractor classification under typical input (0.5)
    attractor = classify_attractor(net, 0.5, warmup=2000, measure_steps=3000)
    results['attractor_at_operating'] = attractor

    # Also classify at zero input (autonomous dynamics)
    attractor_zero = classify_attractor(net, 0.0, warmup=2000, measure_steps=3000)
    results['attractor_at_zero'] = attractor_zero

    # 3. Bifurcation proximity
    bif = measure_bifurcation_proximity(net, input_range=np.linspace(0, 1.5, 20))
    results['bifurcation'] = {
        'n_bifurcations': bif['n_bifurcations'],
        'bifurcation_points': bif['bifurcation_points'],
        'min_distance_from_operating': bif['min_bifurcation_distance'],
        'n_distinct_regimes': bif['n_distinct_regimes'],
    }

    # 4. Input sensitivity: how much does attractor change across input range?
    variances = [a['late_variance'] for a in bif['attractor_sequence']]
    results['input_sensitivity'] = {
        'variance_range': float(np.max(variances) - np.min(variances)),
        'variance_cv': float(np.std(variances) / (np.mean(variances) + 1e-10)),
        'mean_variance': float(np.mean(variances)),
    }

    return results


def main():
    print("=" * 70)
    print("ATTRACTOR GEOMETRY ANALYSIS — Paper 2 v0.7")
    print("=" * 70)

    phase_data, dyn_data, mech_data = load_data()
    conditions = phase_data['conditions']

    all_results = []
    skipped = 0

    for ns in NETWORK_SIZES:
        for s in ALL_SEEDS:
            run_id = f"net{ns}_seed{s}"
            cond = conditions.get(run_id, {})

            if 'error' in cond or 'scores' not in cond:
                skipped += 1
                continue
            const_score = cond['scores']['constitutive']

            genotype = cond.get('evolution', {}).get('best_genotype', None)
            if genotype is None:
                skipped += 1
                continue

            print(f"\n  Analyzing {run_id} (score={const_score:.3f})...", end=" ", flush=True)

            try:
                net, params = build_ctrnn(genotype, ns)
                result = analyze_condition(net, params, ns, const_score)
                result['run_id'] = run_id
                result['seed'] = s
                all_results.append(result)
                atype = result['attractor_at_operating']['type']
                n_bif = result['bifurcation']['n_bifurcations']
                print(f"attractor={atype}, bifurcations={n_bif}")
            except Exception as e:
                print(f"ERROR: {e}")
                skipped += 1

    print(f"\n\nAnalyzed: {len(all_results)} conditions (skipped: {skipped})")

    # === STATISTICAL ANALYSIS ===
    print(f"\n{'='*70}")
    print("CORRELATIONS WITH EMBODIMENT DEPENDENCE")
    print(f"{'='*70}")

    cs = np.array([r['constitutive_score'] for r in all_results])

    # Key metrics to correlate with ED
    metrics = {
        'mean_fp_count': [r['fixed_points']['mean_fp_count'] for r in all_results],
        'max_fp_count': [float(r['fixed_points']['max_fp_count']) for r in all_results],
        'mean_max_real_eigenvalue': [r['eigenvalues']['mean_max_real'] for r in all_results],
        'stable_fp_fraction': [r['eigenvalues']['stable_fp_fraction'] for r in all_results],
        'complex_eigenvalue_fraction': [r['eigenvalues']['complex_eigenvalue_fraction'] for r in all_results],
        'trajectory_variance_at_operating': [r['attractor_at_operating']['trajectory_variance'] for r in all_results],
        'lyapunov_at_operating': [r['attractor_at_operating']['max_lyapunov'] for r in all_results],
        'n_bifurcations': [float(r['bifurcation']['n_bifurcations']) for r in all_results],
        'n_distinct_regimes': [float(r['bifurcation']['n_distinct_regimes']) for r in all_results],
        'bifurcation_proximity': [r['bifurcation']['min_distance_from_operating'] for r in all_results],
        'input_sensitivity_range': [r['input_sensitivity']['variance_range'] for r in all_results],
    }

    correlation_results = {}
    for name, vals in metrics.items():
        vals_arr = np.array(vals)
        # Handle inf values
        finite_mask = np.isfinite(vals_arr)
        if np.sum(finite_mask) < 10:
            print(f"  {name:<40} too few finite values ({np.sum(finite_mask)})")
            continue
        rho, p = spearmanr(vals_arr[finite_mask], cs[finite_mask])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:<40} rho={rho:+.3f} (p={p:.4f}) {sig}")
        correlation_results[name] = {'rho': float(rho), 'p': float(p)}

    # === ATTRACTOR TYPE DISTRIBUTION ===
    print(f"\n{'='*70}")
    print("ATTRACTOR TYPE DISTRIBUTION")
    print(f"{'='*70}")

    # At operating input (0.5)
    types = [r['attractor_at_operating']['type'] for r in all_results]
    unique_types = sorted(set(types))
    print("\n  At operating input (0.5):")
    for t in unique_types:
        subset = [r for r in all_results if r['attractor_at_operating']['type'] == t]
        scores = [r['constitutive_score'] for r in subset]
        print(f"    {t:<20} n={len(subset):>3}  mean_ED={np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # At zero input
    types_zero = [r['attractor_at_zero']['type'] for r in all_results]
    unique_types_zero = sorted(set(types_zero))
    print("\n  At zero input:")
    for t in unique_types_zero:
        subset = [r for r in all_results if r['attractor_at_zero']['type'] == t]
        scores = [r['constitutive_score'] for r in subset]
        print(f"    {t:<20} n={len(subset):>3}  mean_ED={np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # === HIGH vs LOW COMPARISON ===
    print(f"\n{'='*70}")
    print("HIGH vs LOW EMBODIMENT: ATTRACTOR GEOMETRY")
    print(f"{'='*70}")

    high = [r for r in all_results if r['constitutive_score'] >= HIGH_THRESHOLD]
    low = [r for r in all_results if r['constitutive_score'] < LOW_THRESHOLD]

    print(f"\n  High ED (n={len(high)}):")
    for t in unique_types:
        n_t = sum(1 for r in high if r['attractor_at_operating']['type'] == t)
        print(f"    {t:<20} {n_t} ({100*n_t/max(len(high),1):.0f}%)")

    print(f"\n  Low ED (n={len(low)}):")
    for t in unique_types:
        n_t = sum(1 for r in low if r['attractor_at_operating']['type'] == t)
        print(f"    {t:<20} {n_t} ({100*n_t/max(len(low),1):.0f}%)")

    # Mann-Whitney on key metrics
    print(f"\n  Mann-Whitney U comparisons (High vs Low):")
    for name in ['mean_fp_count', 'mean_max_real_eigenvalue', 'stable_fp_fraction',
                  'n_bifurcations', 'trajectory_variance_at_operating', 'lyapunov_at_operating']:
        h_vals = [metrics[name][all_results.index(r)] for r in high]
        l_vals = [metrics[name][all_results.index(r)] for r in low]
        if len(h_vals) >= 3 and len(l_vals) >= 3:
            h_finite = [v for v in h_vals if np.isfinite(v)]
            l_finite = [v for v in l_vals if np.isfinite(v)]
            if len(h_finite) >= 3 and len(l_finite) >= 3:
                stat, p = mannwhitneyu(h_finite, l_finite, alternative='two-sided')
                h_m = np.mean(h_finite)
                l_m = np.mean(l_finite)
                pooled_std = np.sqrt((np.std(h_finite)**2 + np.std(l_finite)**2)/2) if (np.std(h_finite)+np.std(l_finite)) > 0 else 1
                d = (h_m - l_m) / pooled_std if pooled_std > 0 else 0
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {name:<38} H={h_m:.3f} L={l_m:.3f} p={p:.4f} d={d:+.2f} {sig}")

    # === MECHANISTIC TYPE CLASSIFICATION ===
    print(f"\n{'='*70}")
    print("MECHANISTIC TYPE CLASSIFICATION")
    print(f"{'='*70}")

    # Load self-connection data from mechanistic analysis
    if mech_data and 'conditions' in mech_data:
        mech_conditions = {c['run_id']: c for c in mech_data['conditions']}
    else:
        mech_conditions = {}

    type_counts = {'A': 0, 'B': 0, 'C': 0}
    type_by_size = {ns: {'A': 0, 'B': 0, 'C': 0} for ns in NETWORK_SIZES}
    type_ed_scores = {'A': [], 'B': [], 'C': []}
    misclassifications = 0
    total_classified = 0

    for r in all_results:
        run_id = r['run_id']
        ed = r['constitutive_score']
        ns = r['num_neurons']
        attractor_type = r['attractor_at_operating']['type']
        max_real_eig = r['eigenvalues']['mean_max_real']

        # Get self-connection from mechanistic data
        mc = mech_conditions.get(run_id, {})
        mean_self_conn = mc.get('mean_self_connection', 0)

        # Classification criteria:
        # Type A: Positive self-connections (mean > 0), unstable/marginal eigenvalues
        # Type B: Negative self-connections (mean < -5), stable eigenvalues
        # Type C: Mixed / intermediate
        if mean_self_conn > 0 and max_real_eig > -0.5:
            mech_type = 'A'
        elif mean_self_conn < -5 and max_real_eig < -0.3:
            mech_type = 'B'
        else:
            mech_type = 'C'

        type_counts[mech_type] += 1
        type_by_size[ns][mech_type] += 1
        type_ed_scores[mech_type].append(ed)

        # Check prediction accuracy
        total_classified += 1
        if mech_type == 'A' and ed < LOW_THRESHOLD:
            misclassifications += 1
        elif mech_type == 'B' and ed >= HIGH_THRESHOLD:
            misclassifications += 1

    print(f"\n  Type A (amplifying, positive self-conn):  n={type_counts['A']}")
    if type_ed_scores['A']:
        print(f"    Mean ED: {np.mean(type_ed_scores['A']):.3f} ± {np.std(type_ed_scores['A']):.3f}")
    print(f"  Type B (stable, negative self-conn):     n={type_counts['B']}")
    if type_ed_scores['B']:
        print(f"    Mean ED: {np.mean(type_ed_scores['B']):.3f} ± {np.std(type_ed_scores['B']):.3f}")
    print(f"  Type C (mixed/intermediate):             n={type_counts['C']}")
    if type_ed_scores['C']:
        print(f"    Mean ED: {np.mean(type_ed_scores['C']):.3f} ± {np.std(type_ed_scores['C']):.3f}")

    print(f"\n  Misclassification rate: {misclassifications}/{total_classified} ({100*misclassifications/max(total_classified,1):.1f}%)")

    print(f"\n  Distribution by network size:")
    print(f"    {'Size':<6} {'Type A':>8} {'Type B':>8} {'Type C':>8} {'Total':>8}")
    for ns in NETWORK_SIZES:
        total = sum(type_by_size[ns].values())
        if total > 0:
            print(f"    n={ns:<3}  {type_by_size[ns]['A']:>6} ({100*type_by_size[ns]['A']/total:.0f}%)  "
                  f"{type_by_size[ns]['B']:>4} ({100*type_by_size[ns]['B']/total:.0f}%)  "
                  f"{type_by_size[ns]['C']:>4} ({100*type_by_size[ns]['C']/total:.0f}%)  {total}")

    # === CAUSAL CHAIN VALIDATION ===
    print(f"\n{'='*70}")
    print("CAUSAL CHAIN VALIDATION: self-connection -> eigenvalue -> attractor -> ED")
    print(f"{'='*70}")

    if mech_conditions:
        self_conns = []
        max_reals = []
        eds = []
        for r in all_results:
            mc = mech_conditions.get(r['run_id'], {})
            sc = mc.get('mean_self_connection', None)
            if sc is not None:
                self_conns.append(sc)
                max_reals.append(r['eigenvalues']['mean_max_real'])
                eds.append(r['constitutive_score'])

        if len(self_conns) > 10:
            sc_arr = np.array(self_conns)
            mr_arr = np.array(max_reals)
            ed_arr = np.array(eds)

            rho_sc_mr, p_sc_mr = spearmanr(sc_arr, mr_arr)
            rho_mr_ed, p_mr_ed = spearmanr(mr_arr, ed_arr)
            rho_sc_ed, p_sc_ed = spearmanr(sc_arr, ed_arr)

            print(f"\n  self-connection -> max_real_eigenvalue: rho={rho_sc_mr:+.3f} (p={p_sc_mr:.4f})")
            print(f"  max_real_eigenvalue -> ED:              rho={rho_mr_ed:+.3f} (p={p_mr_ed:.4f})")
            print(f"  self-connection -> ED (direct):         rho={rho_sc_ed:+.3f} (p={p_sc_ed:.4f})")

            # Partial correlation: does eigenvalue mediate self-connection -> ED?
            # Partial rho(SC, ED | MR) using rank-based residuals
            from scipy.stats import rankdata
            sc_rank = rankdata(sc_arr)
            mr_rank = rankdata(mr_arr)
            ed_rank = rankdata(ed_arr)

            # Regress SC_rank on MR_rank
            slope_sc = np.polyfit(mr_rank, sc_rank, 1)
            sc_resid = sc_rank - np.polyval(slope_sc, mr_rank)

            # Regress ED_rank on MR_rank
            slope_ed = np.polyfit(mr_rank, ed_rank, 1)
            ed_resid = ed_rank - np.polyval(slope_ed, mr_rank)

            partial_rho, partial_p = spearmanr(sc_resid, ed_resid)
            print(f"\n  Partial correlation SC->ED controlling for eigenvalue: rho={partial_rho:+.3f} (p={partial_p:.4f})")
            if abs(partial_rho) < abs(rho_sc_ed) * 0.7:
                print(f"  → Eigenvalue partially MEDIATES the self-connection -> ED relationship")
            else:
                print(f"  → Eigenvalue does NOT fully mediate; self-connection has direct pathway")

    # === SAVE ===
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def convert(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, float) and not np.isfinite(obj): return str(obj)
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [convert(i) for i in obj]
        return obj

    save_data = {
        'meta': {
            'timestamp': timestamp,
            'n_conditions': len(all_results),
            'skipped': skipped,
        },
        'correlations': convert(correlation_results),
        'type_classification': {
            'counts': type_counts,
            'by_size': convert(type_by_size),
            'ed_by_type': {k: {'mean': float(np.mean(v)) if v else 0, 'std': float(np.std(v)) if v else 0, 'n': len(v)} for k, v in type_ed_scores.items()},
            'misclassification_rate': misclassifications / max(total_classified, 1),
        },
        'conditions': convert(all_results),
    }

    outfile = os.path.join(RESULTS_DIR, f'attractor_geometry_{timestamp}.json')
    with open(outfile, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to: {outfile}")


if __name__ == "__main__":
    main()
