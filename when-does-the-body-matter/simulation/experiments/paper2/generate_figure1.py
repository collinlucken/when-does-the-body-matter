"""
Generate Figure 1 for Paper 2: Example neural trajectories showing embodied
behavior vs ghost conditions for representative evolved controllers.

Produces a 3-panel figure:
  (A) Embodied vs Frozen-body ghost — neural state trajectories
  (B) Embodied vs Disconnected ghost — neural state trajectories
  (C) Embodied vs Counterfactual ghost — neural state trajectories

Shows two examples: one HIGH-ED agent (n=8, Type A) and one LOW-ED agent (n=2, Type B),
laid out as a 2×3 grid (6 panels total).

Shaded regions indicate L2 divergence between embodied and ghost trajectories.
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from simulation.ctrnn import CTRNN
from simulation.evolutionary import GenotypeDecoder

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../../results/paper2')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '../../../Paper_2_Constitutive_vs_Causal/figures')


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


def compute_bilateral_sensors(agent_x, agent_y, agent_heading, light_x, light_y,
                              body_radius=1.0, sensor_offset_angle=np.pi/6, max_range=40.0):
    """Compute bilateral photosensor readings given agent pose and light position."""
    # Sensor positions
    left_angle = agent_heading + sensor_offset_angle
    right_angle = agent_heading - sensor_offset_angle

    left_x = agent_x + body_radius * np.cos(left_angle)
    left_y = agent_y + body_radius * np.sin(left_angle)
    right_x = agent_x + body_radius * np.cos(right_angle)
    right_y = agent_y + body_radius * np.sin(right_angle)

    left_dist = np.sqrt((left_x - light_x)**2 + (left_y - light_y)**2)
    right_dist = np.sqrt((right_x - light_x)**2 + (right_y - light_y)**2)

    left_sensor = max(0.0, 1.0 - left_dist / max_range)
    right_sensor = max(0.0, 1.0 - right_dist / max_range)

    return left_sensor, right_sensor


def run_embodied_trial(net, num_neurons, episode_steps=500, dt=0.01,
                       light_x=40.0, light_y=40.0, rng=None):
    """Run one embodied phototaxis trial, recording neural states and sensory input."""
    if rng is None:
        rng = np.random.RandomState(42)

    net.reset()

    # Agent state
    agent_x, agent_y = 25.0, 25.0
    agent_heading = rng.uniform(0, 2*np.pi)
    max_speed = 3.0

    states = np.zeros((episode_steps, num_neurons))
    outputs = np.zeros((episode_steps, num_neurons))
    sensory_trace = np.zeros((episode_steps, 2))
    positions = np.zeros((episode_steps, 2))

    for t in range(episode_steps):
        # Compute sensors
        left_s, right_s = compute_bilateral_sensors(
            agent_x, agent_y, agent_heading, light_x, light_y
        )
        sensory_trace[t] = [left_s, right_s]

        # Neural input (pad to network size)
        ext_input = np.zeros(num_neurons)
        ext_input[0] = left_s
        if num_neurons > 1:
            ext_input[1] = right_s

        # Record state before step
        states[t] = net.get_state()
        output = net.step(ext_input)
        outputs[t] = output

        # Motor commands (differential drive)
        if num_neurons >= 2:
            left_motor = float(output[0])
            right_motor = float(output[1])
        else:
            left_motor = float(output[0])
            right_motor = float(output[0])

        # Update agent position
        forward_speed = (left_motor + right_motor) / 2.0 * max_speed
        turn_rate = (right_motor - left_motor) * 2.0

        agent_heading += turn_rate * dt
        agent_x += forward_speed * np.cos(agent_heading) * dt
        agent_y += forward_speed * np.sin(agent_heading) * dt

        # Arena boundaries (50×50)
        agent_x = np.clip(agent_x, 0, 50)
        agent_y = np.clip(agent_y, 0, 50)

        positions[t] = [agent_x, agent_y]

    return states, outputs, sensory_trace, positions


def run_ghost_frozen_body(net, num_neurons, positions, episode_steps=500,
                          light_x=40.0, light_y=40.0):
    """Frozen body ghost: body position fixed at trial start, constant sensor input."""
    net.reset()

    # Fixed position = initial position
    fixed_x, fixed_y = positions[0]
    fixed_heading = 0.0  # fixed heading

    left_s, right_s = compute_bilateral_sensors(
        fixed_x, fixed_y, fixed_heading, light_x, light_y
    )

    states = np.zeros((episode_steps, num_neurons))
    for t in range(episode_steps):
        ext_input = np.zeros(num_neurons)
        ext_input[0] = left_s  # constant
        if num_neurons > 1:
            ext_input[1] = right_s

        states[t] = net.get_state()
        net.step(ext_input)

    return states


def run_ghost_disconnected(net, num_neurons, episode_steps=500):
    """Disconnected ghost: zero sensory input throughout."""
    net.reset()

    states = np.zeros((episode_steps, num_neurons))
    for t in range(episode_steps):
        ext_input = np.zeros(num_neurons)
        states[t] = net.get_state()
        net.step(ext_input)

    return states


def run_ghost_counterfactual(net, num_neurons, episode_steps=500, rng=None):
    """Counterfactual ghost: random uncorrelated sensory input."""
    if rng is None:
        rng = np.random.RandomState(99)
    net.reset()

    states = np.zeros((episode_steps, num_neurons))
    for t in range(episode_steps):
        ext_input = np.zeros(num_neurons)
        ext_input[0] = rng.uniform(0, 1)
        if num_neurons > 1:
            ext_input[1] = rng.uniform(0, 1)

        states[t] = net.get_state()
        net.step(ext_input)

    return states


def compute_divergence(embodied_states, ghost_states):
    """Compute L2 divergence between embodied and ghost trajectories at each timestep."""
    diff = embodied_states - ghost_states
    return np.sqrt(np.sum(diff**2, axis=1))


def load_condition(data, run_id):
    """Load genotype and score for a condition."""
    cond = data['conditions'][run_id]
    genotype = cond['evolution']['best_genotype']
    score = cond['scores']['constitutive']
    num_neurons = cond['config']['num_neurons']
    return genotype, score, num_neurons


def plot_trajectories_panel(ax, time, embodied_states, ghost_states,
                            neuron_indices, title, ghost_label,
                            show_legend=True, alpha_fill=0.15,
                            global_div_max=None):
    """Plot embodied vs ghost trajectories for selected neurons with divergence shading."""
    colors_embodied = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors_ghost = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']

    for idx, ni in enumerate(neuron_indices):
        ci = idx % len(colors_embodied)
        # Embodied
        ax.plot(time, embodied_states[:, ni], color=colors_embodied[ci],
                linewidth=1.2, label=f'Embodied n{ni+1}' if idx < 2 else None, zorder=3)
        # Ghost
        ax.plot(time, ghost_states[:, ni], color=colors_ghost[ci],
                linewidth=1.0, linestyle='--',
                label=f'{ghost_label} n{ni+1}' if idx < 2 else None, zorder=2)

    # Divergence shading between embodied and ghost (all neurons)
    div = compute_divergence(embodied_states, ghost_states)

    # Use global divergence max for consistent scaling across all panels
    if global_div_max is None:
        global_div_max = np.max(div) if np.max(div) > 0 else 1

    # Plot divergence as shaded area with SHARED scale
    ax2 = ax.twinx()
    ax2.fill_between(time, 0, div, color='red', alpha=0.12, zorder=1)
    ax2.set_ylabel('L2 divergence', fontsize=8, color='red', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=7)
    ax2.set_ylim(0, global_div_max * 1.2)  # Same scale for all panels

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (dt=0.01)', fontsize=9)
    ax.set_ylabel('Neural state', fontsize=9)
    ax.grid(True, alpha=0.2)

    if show_legend:
        ax.legend(fontsize=7, loc='upper left', ncol=2)


def main():
    # Load data
    results_path = Path(RESULTS_DIR)
    phase_files = sorted(results_path.glob('phase_a_10seeds_*.json'), reverse=True)
    if not phase_files:
        raise FileNotFoundError("No phase_a_10seeds results found")
    with open(phase_files[0], 'r') as f:
        data = json.load(f)

    # Select representative conditions
    # HIGH-ED: n=8, seed=2048 (ED=1.0, Type A)
    # LOW-ED: n=2, seed=512 (ED=0.247, Type B)

    # Find best high-ED and low-ED with genotypes
    high_candidates = []
    low_candidates = []
    for run_id, cond in data['conditions'].items():
        if 'evolution' not in cond or 'best_genotype' not in cond.get('evolution', {}):
            continue
        score = cond.get('scores', {}).get('constitutive', 0)
        ns = cond.get('config', {}).get('num_neurons', 0)
        if score >= 0.80 and ns >= 6:
            high_candidates.append((run_id, score, ns))
        if score < 0.25 and ns <= 3:
            low_candidates.append((run_id, score, ns))

    # Sort by score
    high_candidates.sort(key=lambda x: -x[1])
    low_candidates.sort(key=lambda x: x[1])

    if not high_candidates:
        raise ValueError("No high-ED conditions found with genotypes")
    if not low_candidates:
        raise ValueError("No low-ED conditions found with genotypes")

    high_id, high_score, high_nn = high_candidates[0]
    low_id, low_score, low_nn = low_candidates[0]

    print(f"HIGH-ED example: {high_id} (ED={high_score:.3f}, n={high_nn})")
    print(f"LOW-ED example:  {low_id} (ED={low_score:.3f}, n={low_nn})")

    # Build networks
    high_genotype, _, _ = load_condition(data, high_id)
    low_genotype, _, _ = load_condition(data, low_id)

    high_net, _ = build_ctrnn(high_genotype, high_nn)
    low_net, _ = build_ctrnn(low_genotype, low_nn)

    episode_steps = 500
    light_x, light_y = 40.0, 40.0

    # --- Run HIGH-ED agent ---
    rng = np.random.RandomState(42)
    high_emb_states, high_emb_outputs, high_sensory, high_positions = \
        run_embodied_trial(high_net, high_nn, episode_steps, light_x=light_x, light_y=light_y, rng=rng)

    high_fb_states = run_ghost_frozen_body(high_net, high_nn, high_positions, episode_steps, light_x, light_y)
    high_dc_states = run_ghost_disconnected(high_net, high_nn, episode_steps)
    high_cf_states = run_ghost_counterfactual(high_net, high_nn, episode_steps, rng=np.random.RandomState(99))

    # --- Run LOW-ED agent ---
    rng = np.random.RandomState(42)
    low_emb_states, low_emb_outputs, low_sensory, low_positions = \
        run_embodied_trial(low_net, low_nn, episode_steps, light_x=light_x, light_y=light_y, rng=rng)

    low_fb_states = run_ghost_frozen_body(low_net, low_nn, low_positions, episode_steps, light_x, light_y)
    low_dc_states = run_ghost_disconnected(low_net, low_nn, episode_steps)
    low_cf_states = run_ghost_counterfactual(low_net, low_nn, episode_steps, rng=np.random.RandomState(99))

    # === CREATE FIGURE ===
    time = np.arange(episode_steps) * 0.01

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Select neurons to display (first 2 for visibility)
    high_neurons = [0, 1]  # show first 2 of 8
    low_neurons = [0, 1] if low_nn >= 2 else [0]

    # Compute global divergence max across ALL panels for consistent scaling
    all_divs = [
        compute_divergence(high_emb_states, high_fb_states),
        compute_divergence(high_emb_states, high_dc_states),
        compute_divergence(high_emb_states, high_cf_states),
        compute_divergence(low_emb_states, low_fb_states),
        compute_divergence(low_emb_states, low_dc_states),
        compute_divergence(low_emb_states, low_cf_states),
    ]
    global_div_max = max(np.max(d) for d in all_divs)

    # Row 1: HIGH-ED
    plot_trajectories_panel(axes[0, 0], time, high_emb_states, high_fb_states,
                           high_neurons,
                           f'(A) High-ED (n={high_nn}, ED={high_score:.2f})\nFrozen Body Ghost',
                           'FB', show_legend=True, global_div_max=global_div_max)

    plot_trajectories_panel(axes[0, 1], time, high_emb_states, high_dc_states,
                           high_neurons,
                           f'(B) High-ED (n={high_nn}, ED={high_score:.2f})\nDisconnected Ghost',
                           'DC', show_legend=False, global_div_max=global_div_max)

    plot_trajectories_panel(axes[0, 2], time, high_emb_states, high_cf_states,
                           high_neurons,
                           f'(C) High-ED (n={high_nn}, ED={high_score:.2f})\nCounterfactual Ghost',
                           'CF', show_legend=False, global_div_max=global_div_max)

    # Row 2: LOW-ED
    plot_trajectories_panel(axes[1, 0], time, low_emb_states, low_fb_states,
                           low_neurons,
                           f'(D) Low-ED (n={low_nn}, ED={low_score:.2f})\nFrozen Body Ghost',
                           'FB', show_legend=True, global_div_max=global_div_max)

    plot_trajectories_panel(axes[1, 1], time, low_emb_states, low_dc_states,
                           low_neurons,
                           f'(E) Low-ED (n={low_nn}, ED={low_score:.2f})\nDisconnected Ghost',
                           'DC', show_legend=False, global_div_max=global_div_max)

    plot_trajectories_panel(axes[1, 2], time, low_emb_states, low_cf_states,
                           low_neurons,
                           f'(F) Low-ED (n={low_nn}, ED={low_score:.2f})\nCounterfactual Ghost',
                           'CF', show_legend=False, global_div_max=global_div_max)

    # Overall title
    fig.suptitle('Figure 1: Neural State Trajectories Under Embodied and Ghost Conditions\n'
                 'Top row: High embodiment dependence (large divergence). '
                 'Bottom row: Low embodiment dependence (minimal divergence).',
                 fontsize=11, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Compute summary statistics for console output
    high_fb_div = np.mean(compute_divergence(high_emb_states, high_fb_states))
    high_dc_div = np.mean(compute_divergence(high_emb_states, high_dc_states))
    high_cf_div = np.mean(compute_divergence(high_emb_states, high_cf_states))

    low_fb_div = np.mean(compute_divergence(low_emb_states, low_fb_states))
    low_dc_div = np.mean(compute_divergence(low_emb_states, low_dc_states))
    low_cf_div = np.mean(compute_divergence(low_emb_states, low_cf_states))

    print(f"\nHigh-ED mean divergences: FB={high_fb_div:.3f}, DC={high_dc_div:.3f}, CF={high_cf_div:.3f}")
    print(f"Low-ED mean divergences:  FB={low_fb_div:.3f}, DC={low_dc_div:.3f}, CF={low_cf_div:.3f}")

    # Save
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pdf_path = os.path.join(FIGURES_DIR, 'fig1_neural_trajectories.pdf')
    png_path = os.path.join(FIGURES_DIR, 'fig1_neural_trajectories.png')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close()


if __name__ == "__main__":
    main()
