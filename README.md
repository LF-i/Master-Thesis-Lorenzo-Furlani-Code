# Master's Thesis of Lorenzo Furlani: 
## Multi-Asset Spatio-Temporal Momentum Transformer

Instructions to run experiments:
1. Set up Python environment contained in environment.yml.
2. Decide which model to run: DMN, TMOM, CS-DMN, STMOM.
3. Change the name of the src folder of the model you want to run: from "src_{MODEL_NAME}" to "src".
4. Open run_experiments.ipynb.
5. Follow the instructions in the comments whose row starts with "TODO".
6. Make sure to check that you completed the TODO in all cells before running any cell.

Reproducibility:
- The results exact to the decimal are only reproducible with the exact same code, machine, environment, and setting (e.g. one thread per job).
- The used machine is presented in Section 5.2 of the thesis.
- The random seeds used for the experiments and for randomised grid search are {0,1,2,3,4} and {42}, respectively, and are to be found in run_experiments.ipynb.
