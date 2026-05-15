# Phase A Preliminary Analysis: Constitutive vs. Causal Embodiment in Evolved CTRNN Agents

**Date**: 2026-02-16
**Status**: Preliminary — based on MicrobialGA × 3 network sizes, 2000 generations

---

## 1. Experimental Design

### Task
Phototaxis with variable light positions. Agent starts at arena center (25, 25) and must navigate toward a light source placed at one of 8 locations (four corners + four edge midpoints). This forces sensor-dependent behavior: a fixed motor pattern cannot succeed across all 8 directions.

### Conditions
- **Network sizes**: 3, 5, 8 neurons (CTRNN with center-crossing sigmoid)
- **EA**: MicrobialGA (Harvey 2009), population 50, 2000 generations, mutation σ=0.2
- **Physics**: Agent dt=0.1, max_speed=3.0, sensor_range=40.0, 50×50 arena
- **Fitness**: 50% time-averaged proximity + 50% approach score (initial-final distance / initial distance)
- **Seeds**: Deterministic (42 + offset per condition)

### Ghost Conditions
1. **Frozen body**: Agent stays at starting position; sensory input computed from frozen position (constant). Tests whether varying sensory feedback matters.
2. **Disconnected**: Zero sensory input. Tests whether any sensory feedback matters.
3. **Counterfactual**: Random sensory input. Tests whether specific sensorimotor contingency matters.

### Morphology Substitutions
Six body configurations tested with evolved controller: baseline, wider sensors (π/3), narrower sensors (π/12), larger body (r=2.0), faster motors (2×), slower motors (0.5×).

---

## 2. Results

### 2.1 Evolution

| Network Size | Best Fitness | Gen @ Plateau | Plateau Note |
|---|---|---|---|
| 3 neurons | 0.456 | ~400 | No improvement after gen 400 |
| 5 neurons | 0.448 | ~400 | Slight improvement at 1600 |
| 8 neurons | 0.464 | ~400 | Slow improvement through 2000 |

All conditions plateau around 0.45 fitness by generation 400. The 8-neuron network shows the best final fitness and the most continued improvement, suggesting the larger search space is harder to explore but eventually finds better solutions.

**Limitation**: Fitness 0.45 is modest. A "perfect" phototaxer would approach 0.8-0.9. MicrobialGA with pop 50 and 2000 gen is underevolved. Beer's original work uses 10,000-100,000+ generations. Further evolution is needed for definitive claims.

### 2.2 Ghost Conditions

| Condition | Frozen Body Div | Disconnected Div | Counterfactual Div | Time-to-Div (Frozen) |
|---|---|---|---|---|
| 3 neurons | 0.006 | 0.014 | 0.003 | >1000 (never) |
| 5 neurons | 0.019 | 0.027 | 0.001 | >1000 (never) |
| 8 neurons | **0.194** | **0.692** | **0.188** | **81 steps** |

**Key finding**: Neural divergence scales with network capacity. The 8-neuron network shows ~30× the frozen-body divergence and ~50× the disconnected divergence compared to the 3-neuron network.

The 8-neuron condition shows a clear temporal pattern:
- Disconnected ghost diverges at step 9 (neural dynamics immediately differ without ANY sensory input)
- Frozen body ghost diverges at step 81 (the specific VARYING sensory feedback from body movement takes longer to matter)
- This gap (9 → 81) suggests the network uses sensory input generally but is less dependent on the specific sensorimotor coupling

### 2.3 Morphology Substitutions

| Morphology | net3 Degradation | net5 Degradation | net8 Degradation |
|---|---|---|---|
| Baseline | 0.0% | 0.0% | 0.0% |
| Wider sensors | 0.0% | 0.0% | 0.2% |
| Narrower sensors | 0.0% | 0.0% | 0.0% |
| Larger body | 0.0% | 0.0% | 0.0% |
| Faster motors | 0.0%* | 0.0%* | 0.1% |
| **Slower motors** | **15.9%** | **20.4%** | **18.1%** |

*Faster motors actually improve fitness slightly in net3/net5 conditions.

**Key finding**: Morphology degradation is entirely driven by motor speed reduction. Sensor geometry modifications have near-zero effect. This suggests evolved controllers use a motor program that benefits from speed but doesn't rely on specific sensor arrangements.

### 2.4 Constitutive vs. Causal Scores

| Condition | Constitutive Score | Causal Score | Classification |
|---|---|---|---|
| 3 neurons | 0.007 | 0.968 | CAUSAL DOMINANT |
| 5 neurons | 0.015 | 0.959 | CAUSAL DOMINANT |
| 8 neurons | 0.292 | 0.963 | WEAK CONSTITUTIVE |

---

## 3. Interpretation

### 3.1 The Capacity-Constitutivity Hypothesis

The central finding is that **network computational capacity predicts the degree of constitutive embodiment**. Small networks (3-5 neurons) evolve controllers where the body plays a purely causal role: the network computes a motor program that drives the body, but the specific sensorimotor loop is not exploited. Large networks (8 neurons) begin to exploit the sensorimotor loop, making the body partially constitutive of the cognitive process.

This aligns with Potochnik's causal pattern framework: the constitutive/causal distinction is not binary but a continuum, and the relevant pattern depends on the system's organizational complexity.

### 3.2 Why Motor Speed Matters More Than Sensor Geometry

The asymmetric morphology results (motor degradation >> sensor degradation) reveal that the evolved controllers are "motor-first" systems:
1. The network evolves a motor pattern that produces forward/turning behavior
2. Sensory input modulates this pattern but doesn't drive it
3. Reducing motor speed degrades the motor pattern's effectiveness
4. Changing sensor geometry has little effect because sensors provide only coarse directional bias

This is consistent with the dynamical systems perspective (Beer 1995, 2003): behavior emerges from the intrinsic dynamics of the network, with sensory input serving as a perturbation rather than a specification.

### 3.3 Implications for Paper 2

For the constitutive-vs-causal argument:
1. The ghost condition methodology matters — the standard sensory-replay ghost is tautological for deterministic systems (a methodological contribution)
2. The constitutive/causal distinction is not a property of the TASK but of the SOLUTION: the same phototaxis task can be solved with constitutive OR causal embodiment
3. Network capacity is a mediating variable: larger networks have the computational resources to exploit sensorimotor coupling, creating constitutive dependence

### 3.4 Limitations and Next Steps

1. **Underevolved agents**: 2000 generations is insufficient. Need 10,000+ for mature solutions.
2. **Single EA**: Only MicrobialGA tested. CMA-ES needs proper implementation (pycma library).
3. **Single task**: Only phototaxis. Need categorical perception and perceptual crossing for the full robustness matrix.
4. **No Wimsattian analysis yet**: Need to verify that the capacity-constitutivity pattern is robust across independent variations (EA types, tasks, seeds).
5. **CMA-ES implementation**: The custom CMA-ES in evolutionary.py is broken for pop>20. Need pycma.

---

## 4. Files Generated

- `results/paper2/phase_a_mga_2000gen.json` — Full numerical results
- `results/paper2/proof_of_concept.json` — Initial (flawed) proof of concept
- `simulation/experiments/paper2/phase_a_corrected.py` — Corrected experiment code
