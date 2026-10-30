# Toy sampler experiments

These exploratory notebooks are separate from the manuscript benchmark commands
in the repository root README. Notebook source is retained; saved cell outputs,
plots, animations, and generated sample arrays are omitted.

Run notebooks with this directory as the working directory in the installed
project environment:

| Notebook | Purpose |
| --- | --- |
| `test_GMM.ipynb` | GMMVI sampler experiment |
| `test_NF_NS.ipynb` | Neural spline flow sampler experiment |
| `test_NF_realNVP.ipynb` | RealNVP sampler experiment |
| `test_NF_zuko.ipynb` | Zuko flow sampler experiment |
| `test_autodr.ipynb` | AutoDR sampler experiment |
| `test_doraemon.ipynb` | DORAEMON sampler experiment |
| `plot_variational_inference.ipynb` | Plot previously generated sampler arrays |

The sampler notebooks write `.npy`, `.png`, and `.gif` files into the working
directory. Before running the comparison plotting notebook, run the relevant
sampler experiments for the requested beta values and place the resulting arrays
in `files/`, using the exact filenames referenced by the plotting cells. The
plotting notebook does not train the samplers or generate its input arrays.
The existing exploratory cells may require adjusting beta values and filenames
between experiments; they are not an automated reproduction pipeline.

Generated files are ignored by Git. Before committing notebook changes, clear all
cell outputs and execution counts. Keep code, Markdown, and notebook metadata
needed to run the experiment.
