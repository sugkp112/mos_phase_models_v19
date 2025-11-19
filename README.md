# mos_phase_models_v19
Python implementation of the mosquito navigation phase model (wind–vortex–thermal perturbation simulations) used to generate all results and figures in the manuscript.

Mosquito Navigation Phase Model (mos_phase_models_v19)

This repository contains the full Python implementation of the mosquito navigation phase model used in the manuscript.
The code generates all simulations and figures, including wind-only collapse, vortex-driven transitions, thermal-decoy dilution, and the combined aerodynamic invisibility phase map.

Contents

mos_phase_models_v19.py – main simulation engine

run_phase_scans.py – wind, vortex, and combined parameter scans

figures/ – example output figures (optional)

requirements.txt – Python package dependencies

Model summary

The model simulates N mosquitoes navigating a CO₂ + heat scalar field, perturbed by:

deterministic wind (advective flow),

deterministic vortex shear,

thermal decoys,

Gaussian sensory noise.

Outputs include success rates, phase diagrams, time distributions, and convergence tests.

Reproducibility

All main figures (Fig. 1–4) and supplementary figures (S1–S6) can be reproduced by running:

python run_phase_scans.py


Python ≥ 3.10 is recommended.

License

MIT License.
