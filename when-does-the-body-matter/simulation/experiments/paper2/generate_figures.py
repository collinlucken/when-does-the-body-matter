"""
Generate publication-quality figures for Paper 2.

Figure 1: Scatter/violin — 60 conditions, network size vs constitutive score
Figure 2: CV reduction bar chart
Figure 3: Classification distribution stacked bars
Figure 4: Example neural trajectories (embodied vs ghost conditions)

All saved as PDF for publication quality.
"""

import sys
import os
import json
import numpy as np


def compute_ed_equal_weight(cond):
    """Compute ED using equal-weight, individually-capped formula.

    ED = (min(1, FB) + min(1, DC) + min(1, CF)) / 3
    """
    fb = min(1.0, cond['ghost_frozen_body']['neural_divergence'])
    dc = min(1.0, cond['ghost_disconnected']['neural_divergence'])
    cf = min(1.0, cond['ghost_counterfactual']['neural_divergence'])
    return (fb + dc + cf) / 3.0
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../../results/paper2')
FIGURE_DIR = os.path.join(os.path.dirname(__file__), '../../../Paper_2_Constitutive_vs_Causal/figures')

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def load_data():
    """Load the 10-seed results."""
    from pathlib import Path
    json_files = sorted(Path(RESULTS_DIR).glob('phase_a_10seeds_*.json'), reverse=True)
    with open(json_files[0], 'r') as f:
        return json.load(f)


def figure1_scatter_violin(data):
    """
    Figure 1: Network size vs embodiment dependence.
    Left: strip/swarm plot of all 60 points with box overlay.
    Shows correlation, variance reduction, and non-monotonicity at a glance.
    """
    conditions = data['conditions']

    sizes_list = []
    scores_list = []
    for k, v in conditions.items():
        if 'error' not in v:
            sizes_list.append(v['config']['num_neurons'])
            scores_list.append(compute_ed_equal_weight(v))

    sizes = np.array(sizes_list)
    scores = np.array(scores_list)
    unique_sizes = sorted(set(sizes))
    size_labels = [str(s) for s in unique_sizes]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))

    # Group data
    grouped = defaultdict(list)
    for s, sc in zip(sizes, scores):
        grouped[s].append(sc)

    # Box plot (light, behind)
    bp_data = [grouped[s] for s in unique_sizes]
    bp = ax.boxplot(bp_data, positions=range(len(unique_sizes)), widths=0.5,
                    patch_artist=True, showfliers=False, zorder=2)
    colors = ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_color('#333333')
            line.set_linewidth(0.8)

    # Strip plot (individual points with jitter)
    np.random.seed(42)
    for i, s in enumerate(unique_sizes):
        y = np.array(grouped[s])
        x = np.full_like(y, i) + np.random.uniform(-0.15, 0.15, size=len(y))
        ax.scatter(x, y, c=colors[i], edgecolors='#333333', linewidths=0.5,
                   s=35, zorder=3, alpha=0.85)

    # Mean trend line
    means = [np.mean(grouped[s]) for s in unique_sizes]
    ax.plot(range(len(unique_sizes)), means, 'k-o', markersize=5, linewidth=1.5,
            zorder=4, label=f'Mean (Spearman $\\rho$=0.39, p=0.002)')

    # 95% CI bands
    from scipy.stats import t as t_dist
    for i, s in enumerate(unique_sizes):
        vals = np.array(grouped[s])
        n = len(vals)
        se = np.std(vals, ddof=1) / np.sqrt(n)
        ci = t_dist.ppf(0.975, df=n-1) * se
        ax.fill_between([i - 0.1, i + 0.1],
                        [means[i] - ci, means[i] - ci],
                        [means[i] + ci, means[i] + ci],
                        color='black', alpha=0.1, zorder=1)

    ax.set_xticks(range(len(unique_sizes)))
    ax.set_xticklabels(size_labels)
    ax.set_xlabel('Network Size (neurons)')
    ax.set_ylabel('Embodiment Dependence Score')
    ax.set_ylim(-0.05, 1.10)
    ax.axhline(y=0.3, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.axhline(y=0.7, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate n per group
    for i, s in enumerate(unique_sizes):
        ax.text(i, -0.03, f'n={len(grouped[s])}', ha='center', va='top',
                fontsize=7, color='gray')

    fig.tight_layout()
    outpath = os.path.join(FIGURE_DIR, 'fig1_scatter_boxplot.pdf')
    fig.savefig(outpath)
    fig.savefig(outpath.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {outpath}")


def figure2_cv_reduction(data):
    """
    Figure 2: Coefficient of variation by network size.
    Shows the variance reduction finding: CV drops 84% → 43%.
    """
    conditions = data['conditions']
    unique_sizes = [2, 3, 4, 5, 6, 8]

    grouped = defaultdict(list)
    for k, v in conditions.items():
        if 'error' not in v:
            grouped[v['config']['num_neurons']].append(compute_ed_equal_weight(v))

    cvs = []
    means = []
    stds = []
    for s in unique_sizes:
        vals = np.array(grouped[s])
        m = np.mean(vals)
        sd = np.std(vals, ddof=1)
        cvs.append(sd / m * 100 if m > 0 else 0)
        means.append(m)
        stds.append(sd)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    # Left: CV bars
    colors = ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
    bars = ax1.bar(range(len(unique_sizes)), cvs, color=colors, edgecolor='#333333',
                   linewidth=0.5, width=0.6)

    # Annotate values
    for i, (bar, cv) in enumerate(zip(bars, cvs)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{cv:.0f}%', ha='center', va='bottom', fontsize=8)

    ax1.set_xticks(range(len(unique_sizes)))
    ax1.set_xticklabels([str(s) for s in unique_sizes])
    ax1.set_xlabel('Network Size (neurons)')
    ax1.set_ylabel('Coefficient of Variation (%)')
    ax1.set_ylim(0, 100)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('(a) Variance Reduction')

    # Right: Mean ± Std bars
    ax2.bar(range(len(unique_sizes)), means, yerr=stds, capsize=4,
            color=colors, edgecolor='#333333', linewidth=0.5, width=0.6,
            error_kw={'linewidth': 0.8})

    ax2.set_xticks(range(len(unique_sizes)))
    ax2.set_xticklabels([str(s) for s in unique_sizes])
    ax2.set_xlabel('Network Size (neurons)')
    ax2.set_ylabel('Embodiment Dependence')
    ax2.set_ylim(0, 1.15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('(b) Mean ± SD')

    fig.tight_layout()
    outpath = os.path.join(FIGURE_DIR, 'fig2_cv_reduction.pdf')
    fig.savefig(outpath)
    fig.savefig(outpath.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {outpath}")


def figure3_classification(data):
    """
    Figure 3: Stacked bar chart of solution classifications by network size.
    Uses 3-category scheme from paper text: Low ED (<0.30), Mixed (0.30-0.70), High ED (>=0.70).
    Shows shift from Low ED at small sizes to High ED at large.
    """
    conditions = data['conditions']
    unique_sizes = [2, 3, 4, 5, 6, 8]

    # 3-category classification from raw scores (matches paper text thresholds)
    classes = ['Causal-Dominant (<0.30)', 'Mixed (0.30\u20130.70)', 'Embodiment-Dominant (\u22650.70)']
    class_colors = ['#d73027', '#abd9e9', '#4575b4']

    counts = {s: {c: 0 for c in classes} for s in unique_sizes}
    for k, v in conditions.items():
        if 'error' not in v:
            ns = v['config']['num_neurons']
            score = compute_ed_equal_weight(v)
            if score < 0.30:
                cat = 'Causal-Dominant (<0.30)'
            elif score < 0.70:
                cat = 'Mixed (0.30\u20130.70)'
            else:
                cat = 'Embodiment-Dominant (\u22650.70)'
            if ns in counts:
                counts[ns][cat] += 1

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

    x = np.arange(len(unique_sizes))
    width = 0.6
    bottoms = np.zeros(len(unique_sizes))

    for cls, color in zip(classes, class_colors):
        heights = [counts[s][cls] / sum(counts[s].values()) * 100
                   if sum(counts[s].values()) > 0 else 0
                   for s in unique_sizes]
        ax.bar(x, heights, width, bottom=bottoms, label=cls,
               color=color, edgecolor='white', linewidth=0.5)
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in unique_sizes])
    ax.set_xlabel('Network Size (neurons)')
    ax.set_ylabel('Percentage of Runs (%)')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=0.9, fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    outpath = os.path.join(FIGURE_DIR, 'fig3_classification.pdf')
    fig.savefig(outpath)
    fig.savefig(outpath.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {outpath}")


def figure4_trajectories(data):
    """
    Figure 4: Example neural trajectories under embodied and ghost conditions.
    Uses net8_seed2048 (constitutive score = 1.0) as the example.
    Re-runs the simulation to get time-series data.
    """
    from simulation.ctrnn import CTRNN
    from simulation.evolutionary import GenotypeDecoder

    # Pick a high-scoring condition with saved genotype
    target = 'net8_seed2048'
    cond = data['conditions'].get(target, {})
    geno = cond.get('evolution', {}).get('best_genotype', None)

    if geno is None:
        print("  SKIPPING fig4: no genotype for target condition")
        return

    geno = np.array(geno)
    num_neurons = 8
    decoder = GenotypeDecoder(
        num_neurons=num_neurons,
        include_gains=False,
        tau_range=(0.5, 5.0),
        weight_range=(-10.0, 10.0),
        bias_range=(-10.0, 10.0),
    )
    params = decoder.decode(geno)

    # Simulation parameters
    dt = 0.1
    max_speed = 3.0
    sensor_range = 40.0
    arena_size = 50.0
    sim_steps = 500
    light_pos = np.array([40.0, 40.0])

    # Run embodied condition
    def run_trial(condition='embodied'):
        brain = CTRNN(num_neurons=num_neurons, step_size=dt)
        brain.tau = params['tau']
        brain.weights = params['weights']
        brain.biases = params['biases']
        brain.reset()

        agent_pos = np.array([25.0, 25.0])
        agent_angle = 0.0

        states = np.zeros((sim_steps, num_neurons))
        positions = np.zeros((sim_steps, 2))

        for t in range(sim_steps):
            # Sensor input
            if condition == 'embodied':
                to_light = light_pos - agent_pos
                dist = np.linalg.norm(to_light)
                angle_to_light = np.arctan2(to_light[1], to_light[0])
                rel_angle = angle_to_light - agent_angle
                left_activation = max(0, np.cos(rel_angle - 0.3)) * max(0, 1 - dist / sensor_range)
                right_activation = max(0, np.cos(rel_angle + 0.3)) * max(0, 1 - dist / sensor_range)
                sensor_input = np.zeros(num_neurons)
                sensor_input[0] = left_activation
                sensor_input[1] = right_activation
            elif condition == 'frozen':
                # Frozen at starting position
                to_light = light_pos - np.array([25.0, 25.0])
                dist = np.linalg.norm(to_light)
                angle_to_light = np.arctan2(to_light[1], to_light[0])
                rel_angle = angle_to_light - 0.0
                left_activation = max(0, np.cos(rel_angle - 0.3)) * max(0, 1 - dist / sensor_range)
                right_activation = max(0, np.cos(rel_angle + 0.3)) * max(0, 1 - dist / sensor_range)
                sensor_input = np.zeros(num_neurons)
                sensor_input[0] = left_activation
                sensor_input[1] = right_activation
            elif condition == 'disconnected':
                sensor_input = np.zeros(num_neurons)

            brain.step(sensor_input)
            states[t] = brain.state.copy()
            positions[t] = agent_pos.copy()

            if condition == 'embodied':
                # Motor output → movement
                outputs = brain.get_output()
                left_motor = outputs[-2] if num_neurons >= 2 else outputs[0]
                right_motor = outputs[-1] if num_neurons >= 2 else outputs[0]
                speed = (left_motor + right_motor) / 2 * max_speed
                turn = (right_motor - left_motor) * 2.0
                agent_angle += turn * dt
                agent_pos = agent_pos + speed * dt * np.array([np.cos(agent_angle), np.sin(agent_angle)])
                agent_pos = np.clip(agent_pos, 0, arena_size)

        return states, positions

    embodied_states, embodied_pos = run_trial('embodied')
    frozen_states, _ = run_trial('frozen')
    disconn_states, _ = run_trial('disconnected')

    # Compute divergence over time
    def cumulative_divergence(s1, s2):
        return np.sqrt(np.sum((s1 - s2)**2, axis=1))

    div_frozen = cumulative_divergence(embodied_states, frozen_states)
    div_disconn = cumulative_divergence(embodied_states, disconn_states)

    time_axis = np.arange(sim_steps) * dt

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))

    # Top-left: Neural state trajectories (first 3 neurons)
    ax = axes[0, 0]
    for i in range(min(3, num_neurons)):
        ax.plot(time_axis, embodied_states[:, i], '-', linewidth=0.8,
                label=f'Neuron {i+1}', alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neural State')
    ax.set_title('(a) Embodied: Neural Dynamics')
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Top-right: Ghost trajectories for neuron 0
    ax = axes[0, 1]
    ax.plot(time_axis, embodied_states[:, 0], 'b-', linewidth=1, label='Embodied', alpha=0.8)
    ax.plot(time_axis, frozen_states[:, 0], 'r--', linewidth=1, label='Frozen Body', alpha=0.8)
    ax.plot(time_axis, disconn_states[:, 0], 'g:', linewidth=1, label='Disconnected', alpha=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron 1 State')
    ax.set_title('(b) Ghost Condition Comparison')
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Bottom-left: Divergence over time
    ax = axes[1, 0]
    ax.plot(time_axis, div_frozen, 'r-', linewidth=1, label='Frozen Body', alpha=0.8)
    ax.plot(time_axis, div_disconn, 'g-', linewidth=1, label='Disconnected', alpha=0.8)
    ax.axhline(y=0.1, color='gray', linestyle=':', linewidth=0.5, label='Threshold (0.1)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('L2 Divergence')
    ax.set_title('(c) Neural Divergence')
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Bottom-right: Agent trajectory
    ax = axes[1, 1]
    ax.plot(embodied_pos[:, 0], embodied_pos[:, 1], 'b-', linewidth=0.8, alpha=0.7)
    ax.plot(embodied_pos[0, 0], embodied_pos[0, 1], 'go', markersize=6, label='Start')
    ax.plot(light_pos[0], light_pos[1], 'y*', markersize=12, markeredgecolor='orange',
            label='Light')
    ax.plot(embodied_pos[-1, 0], embodied_pos[-1, 1], 'rs', markersize=5, label='End')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('(d) Agent Trajectory')
    ax.set_xlim(0, arena_size)
    ax.set_ylim(0, arena_size)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='lower left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ed_score = compute_ed_equal_weight(cond)
    fig.suptitle(f'Example: 8-neuron controller (seed 2048, ED = {ed_score:.2f})',
                 fontsize=10, y=1.02)
    fig.tight_layout()
    outpath = os.path.join(FIGURE_DIR, 'fig4_trajectories.pdf')
    fig.savefig(outpath)
    fig.savefig(outpath.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data['conditions'])} conditions\n")

    print("Generating Figure 1: Scatter + Box plot...")
    figure1_scatter_violin(data)

    print("Generating Figure 2: CV reduction...")
    figure2_cv_reduction(data)

    print("Generating Figure 3: Classification distribution...")
    figure3_classification(data)

    print("Generating Figure 4: Example trajectories...")
    figure4_trajectories(data)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
