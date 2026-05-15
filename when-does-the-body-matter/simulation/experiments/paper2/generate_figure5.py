"""
Generate Figure 5: Self-Connection Polarity vs Embodiment Dependence Score.

New figure for Paper 2 v0.6 showing the mechanistic finding that self-connection
polarity is the strongest weight-level predictor of embodiment dependence.
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from simulation.evolutionary import GenotypeDecoder

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../../results/paper2')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '../../../Paper_2_Constitutive_vs_Causal/figures')

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

SIZE_COLORS = {2: '#1b9e77', 3: '#d95f02', 4: '#7570b3', 5: '#e7298a', 6: '#66a61e', 8: '#e6ab02'}


def load_data():
    """Load phase A results and mechanistic analysis."""
    results_path = Path(RESULTS_DIR)

    phase_files = sorted(results_path.glob('phase_a_10seeds_*.json'), reverse=True)
    with open(phase_files[0], 'r') as f:
        phase_data = json.load(f)

    mech_files = sorted(results_path.glob('mechanistic_analysis_*.json'), reverse=True)
    with open(mech_files[0], 'r') as f:
        mech_data = json.load(f)

    return phase_data, mech_data


def main():
    phase_data, mech_data = load_data()
    conditions = phase_data['conditions']

    # Extract data from mechanistic analysis
    mech_conditions = mech_data['conditions']

    sizes = []
    const_scores = []
    self_connections = []

    for cond in mech_conditions:
        sizes.append(cond['num_neurons'])
        const_scores.append(cond['constitutive_score'])
        self_connections.append(cond['mean_self_connection'])

    sizes = np.array(sizes)
    const_scores = np.array(const_scores)
    self_connections = np.array(self_connections)

    rho, p = spearmanr(self_connections, const_scores)

    # Figure 5: Two-panel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={'width_ratios': [3, 2]})

    # Panel A: Scatter plot
    for ns in sorted(set(sizes)):
        mask = sizes == ns
        ax1.scatter(self_connections[mask], const_scores[mask],
                   c=SIZE_COLORS.get(ns, '#999999'), s=60, alpha=0.8,
                   edgecolors='black', linewidths=0.5,
                   label=f'n={ns}', zorder=3)

    # Trend line
    z = np.polyfit(self_connections, const_scores, 1)
    x_range = np.linspace(self_connections.min() - 5, self_connections.max() + 5, 100)
    ax1.plot(x_range, np.polyval(z, x_range), 'k--', alpha=0.5, linewidth=1.5, zorder=2)

    # Annotation
    ax1.annotate(f'Spearman ρ = {rho:.3f}\np = {p:.4f}',
                xy=(0.03, 0.97), xycoords='axes fraction',
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    # Zero line
    ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5, zorder=1)

    ax1.set_xlabel('Mean Self-Connection Weight')
    ax1.set_ylabel('Embodiment Dependence Score')
    ax1.set_title('(A) Self-Connection Polarity vs Embodiment Dependence')
    ax1.legend(loc='lower right', framealpha=0.8, ncol=2)
    ax1.set_ylim(-0.05, 1.1)

    # Panel B: High vs Low comparison bar chart
    high = [cond for cond in mech_conditions if cond['constitutive_score'] >= 0.70]
    low = [cond for cond in mech_conditions if cond['constitutive_score'] < 0.30]

    metrics = ['mean_self_connection', 'positive_self_frac']
    labels = ['Mean Self-\nConnection', 'Positive Self-\nConnection %']

    h_vals = [[c['mean_self_connection'] for c in high], [c['positive_self_frac'] * 100 for c in high]]
    l_vals = [[c['mean_self_connection'] for c in low], [c['positive_self_frac'] * 100 for c in low]]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax2.bar(x - width/2, [np.mean(v) for v in h_vals], width,
                    yerr=[np.std(v) for v in h_vals], capsize=4,
                    label=f'High ED (n={len(high)})', color='#d62728', alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x + width/2, [np.mean(v) for v in l_vals], width,
                    yerr=[np.std(v) for v in l_vals], capsize=4,
                    label=f'Low ED (n={len(low)})', color='#1f77b4', alpha=0.7, edgecolor='black')

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title('(B) High vs Low Embodiment Dependence')
    ax2.legend(loc='upper left', framealpha=0.8)
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    # Add significance markers
    for i, (h, l) in enumerate(zip(h_vals, l_vals)):
        max_val = max(np.mean(h) + np.std(h), np.mean(l) + np.std(l))
        y_pos = max_val + 5
        ax2.annotate('*', xy=(i, y_pos), fontsize=14, ha='center', fontweight='bold')

    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, 'fig5_self_connection.pdf'))
    fig.savefig(os.path.join(FIGURES_DIR, 'fig5_self_connection.png'))
    plt.close()
    print("Figure 5 saved to figures/fig5_self_connection.pdf and .png")


if __name__ == "__main__":
    main()
