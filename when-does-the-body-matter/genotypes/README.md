# Genotypes

Evolved CTRNN parameters as flat NumPy arrays (`.npy`), one file per
(network size, seed) pair.

Naming convention: `genotype_n{SIZE}_s{SEED}.npy`. Loading:

```python
import numpy as np
gen = np.load("genotype_n4_s42.npy")  # flat genotype vector
```

Three representative seeds (42, 137, 256) for each of six network sizes
(n = 2, 3, 4, 5, 6, 8) are provided here as standalone `.npy` files for
convenient access — 18 files total. The complete 60-condition dataset,
including the remaining seven seeds per size, is in
`../data/phase_a_10seeds_with_all_genotypes.json`.

To decode a flat genotype vector into CTRNN parameters (time constants,
weights, biases):

```python
from simulation.evolutionary import GenotypeDecoder
decoder = GenotypeDecoder(num_neurons=4)  # match the network size
params = decoder.decode(gen)
```
