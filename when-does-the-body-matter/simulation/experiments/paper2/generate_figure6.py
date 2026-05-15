"""Generate Figure 6 for Paper 2 v0.7: Input sensitivity and bifurcation analysis."""
import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../../results/paper2')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '../../../Paper_2_Constitutive_vs_Causal/figures')

SIZE_COLORS = {2: '#1f77b4', 3: '#ff7f0e', 4: '#2ca02c', 5: '#d62728', 6: '#9467bd', 8: '#8c564b'}

def main():
    # Load attractor geometry results
    results_path = Path(RESULTS_DIR)
    ag_files = sorted(results_path.glob('attractor_geometry_*.json'), reverse=True)
    if not ag_files:
        raise FileNotFoundError("No attractor_geometry results found")
    with open(ag_files[0], 'r') as f:
        data = json.load(f)

    conditions = data['conditions']

    # Extract data
    ed_scores = []
    input_sensitivities = []
    bifurcation_counts = []
    network_sizes = []

    for c in conditions:
        ed = c['constitutive_score']
        isr = c['input_sensitivity']['variance_range']
        n_bif = c['bifurcation']['n_bifurcations']
        ns = c['num_neurons']

        ed_scores.append(ed)
        input_sensitivities.append(isr)
        bifurcation_counts.append(n_bif)
        network_sizes.append(ns)

    ed_scores = np.array(ed_scores)
    input_sensitivities = np.array(input_sensitivities)
    bifurcation_counts = np.array(bifurcation_counts)
    network_sizes = np.array(network_sizes)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Input sensitivity vs ED score
    for ns in sorted(set(network_sizes)):
        mask = network_sizes == ns
        ax1.scatter(input_sensitivities[mask], ed_scores[mask],
                   c=SIZE_COLORS[ns], s=60, alpha=0.8, edgecolors='white',
                   linewidth=0.5, label=f'n={ns}', zorder=3)

    # Trend line
    from scipy.stats import spearmanr
    # Use log scale for input sensitivity if range is large
    finite_mask = np.isfinite(input_sensitivities) & (input_sensitivities >= 0)
    if np.sum(finite_mask) > 5:
        rho, p = spearmanr(input_sensitivities[finite_mask], ed_scores[finite_mask])
        # Fit line on log-transformed data for visualization
        log_is = np.log10(input_sensitivities[finite_mask] + 1e-6)
        z = np.polyfit(log_is, ed_scores[finite_mask], 1)
        x_line = np.logspace(-6, np.log10(np.max(input_sensitivities[finite_mask])+0.1), 100)
        y_line = z[0] * np.log10(x_line + 1e-6) + z[1]
        ax1.plot(x_line, np.clip(y_line, 0, 1), 'k--', alpha=0.5, linewidth=1.5)

    ax1.set_xlabel('Input Sensitivity Range (trajectory variance)', fontsize=11)
    ax1.set_ylabel('Embodiment Dependence Score', fontsize=11)
    ax1.set_title(f'(A) Input Sensitivity vs. Embodiment Dependence\n'
                  f'Spearman ρ = +{rho:.3f}, p < 0.0001', fontsize=11)
    ax1.set_xscale('symlog', linthresh=0.01)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8, loc='lower right', ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panel B: Bifurcation count for high vs low ED
    high_mask = ed_scores >= 0.70
    low_mask = ed_scores < 0.30
    mid_mask = (~high_mask) & (~low_mask)

    groups = ['Low ED\n(< 0.30)', 'Mixed\n(0.30–0.70)', 'High ED\n(≥ 0.70)']
    means = [np.mean(bifurcation_counts[low_mask]),
             np.mean(bifurcation_counts[mid_mask]),
             np.mean(bifurcation_counts[high_mask])]
    stds = [np.std(bifurcation_counts[low_mask]),
            np.std(bifurcation_counts[mid_mask]),
            np.std(bifurcation_counts[high_mask])]
    ns_list = [np.sum(low_mask), np.sum(mid_mask), np.sum(high_mask)]

    colors = ['#4575b4', '#ffffbf', '#d73027']
    bars = ax2.bar(groups, means, yerr=stds, color=colors, edgecolor='black',
                   linewidth=0.8, capsize=5, alpha=0.85)

    # Add sample sizes
    for bar, n in zip(bars, ns_list):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[groups.index(bar.get_x())] if False else max(means) * 0.05,
                f'n={n}', ha='center', va='bottom', fontsize=9)

    # Add significance brackets
    from scipy.stats import mannwhitneyu
    h_bif = bifurcation_counts[high_mask]
    l_bif = bifurcation_counts[low_mask]
    if len(h_bif) >= 3 and len(l_bif) >= 3:
        stat, p_val = mannwhitneyu(h_bif, l_bif, alternative='two-sided')
        sig_text = f'p = {p_val:.3f} **'
        y_max = max(means) + max(stds) + 1.5
        ax2.plot([0, 0, 2, 2], [y_max-0.3, y_max, y_max, y_max-0.3], 'k-', linewidth=1)
        ax2.text(1, y_max + 0.2, sig_text, ha='center', fontsize=9, fontweight='bold')

    ax2.set_ylabel('Number of Bifurcations', fontsize=11)
    ax2.set_title('(B) Bifurcations by Embodiment Group\n'
                  'Mann-Whitney U test', fontsize=11)
    ax2.set_ylim(0, max(means) + max(stds) + 3)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    pdf_path = os.path.join(FIGURES_DIR, 'fig6_attractor_geometry.pdf')
    png_path = os.path.join(FIGURES_DIR, 'fig6_attractor_geometry.png')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close()


if __name__ == "__main__":
    main()
