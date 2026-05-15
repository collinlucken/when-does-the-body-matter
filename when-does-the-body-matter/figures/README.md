# Figures

Paper figures, in PDF and PNG.

| File                                | Appears in paper as |
|-------------------------------------|---------------------|
| `paper_fig1_scatter_boxplot.{pdf,png}`     | Figure 1 — embodiment dependence by network size (scatter + per-size distributions) |
| `paper_fig2_self_connection.{pdf,png}`     | Figure 2 — self-connection polarity vs. embodiment dependence |
| `paper_fig3_attractor_geometry.{pdf,png}`  | Figure 3 — input sensitivity vs. ED and the bifurcation-count contrast |

The `legacy/` subdirectory contains four figures from earlier drafts that
are not in the submitted paper (`fig1_neural_trajectories`, `fig2_cv_reduction`,
`fig3_classification`, `fig4_trajectories`). They are kept for archival
purposes only.

To regenerate the paper figures from the data files:

```bash
python -m simulation.experiments.paper2.generate_figure1
python -m simulation.experiments.paper2.generate_figure5
python -m simulation.experiments.paper2.generate_figure6
```

(The generator scripts retain their original numbering — `generate_figure5.py`
produces what the paper labels Figure 2, and `generate_figure6.py` produces
what the paper labels Figure 3.)
