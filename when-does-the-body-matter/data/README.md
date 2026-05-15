# Data

Canonical analysis outputs underlying the paper.

- `phase_a_10seeds_with_all_genotypes.json` — the 60-condition ground-truth file. Includes evolved genotypes, raw and capped ghost-condition divergences (FB, DC, CF), and the composite ED scores reported in Table 1 of the paper.
- `dynamical_analysis_60.json` — dynamical-regime measures (perturbation growth rate, state entropy, inter-trial trajectory distance, fraction amplifying, participation ratio, maximum Lyapunov exponent) computed on the embodied trajectories of the 60 networks.
- `attractor_geometry_60.json` — fixed-point analysis, attractor classifications at the operating input, bifurcation counts, and input-sensitivity scores from the input-amplitude scan.
- `mechanistic_analysis.json` — weight-level analyses (self-connection polarity, maximum real eigenvalue) and partial-correlation comparisons reported in §4.3.
- `statistical_corrections.json` — full 26-test Benjamini-Hochberg FDR table referenced in §4 of the paper.

Filenames preserve the original timestamps from generation where useful; the latest-timestamped JSON per analysis type is the one cited.
