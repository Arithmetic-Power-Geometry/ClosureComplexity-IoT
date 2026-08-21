# ClosureComplexity-IoT

Reproducibility software for the manuscript **“Equally Damaged, Unequally Recoverable: Closure Complexity for Target-Relative Structural Repair of Failure-Degraded IoT Networks.”**

The software operationalizes Closure Complexity (CC) as the exact minimum admissible repair cost required to move a degraded IoT/WSN state into a declared QoS satisfaction region. It also computes CER, COG/NCOG, closure interaction, target-restricted Pareto closure, matched recoverability cases, regression diagnostics, and repeated-seed scalability results.

## One-click GitHub workflow

Upload the contents of this folder to the root of your GitHub repository, then open **Actions → Reproduce Paper → Run workflow**.

The workflow will:

1. install the pinned Python dependencies;
2. run the unit tests;
3. regenerate the full 135-network benchmark;
4. recompute all statistical and structural analyses;
5. run repeated stochastic scaling trials;
6. regenerate all publication tables and figures;
7. validate internal numerical consistency;
8. write every generated output under `results/`;
9. upload `results/` as a GitHub Actions artifact; and
10. commit the regenerated `results/` directory back to the repository when the workflow has permission to do so.

## Local reproduction

```bash
python -m pip install -r requirements.txt
pytest -q
python run_all.py
```

All generated outputs are placed under:

```text
results/
├── raw/       # benchmark and repeated-seed raw CSV outputs
├── tables/    # manuscript-ready numerical summaries
├── figures/   # PDF and PNG figures
└── manifest.json
```

`run_all.py` deletes the previous `results/` directory before rebuilding it, so stale or duplicate outputs cannot silently survive a reproduction run.

## Main robustness upgrades

The current package includes four checks intended to prevent common reviewer misinterpretations:

- aggregate cost summaries are computed only over each method's feasible solutions, while paired solver comparisons use COG/NCOG;
- the CC-versus-CER representative case exports the exact selected action masks and labels;
- nested regressions report adjusted R² and VIF diagnostics, with degraded latency excluded from the multivariable model because it is an exact affine transform of degraded reliability in the unrepaired state under the benchmark latency equation;
- scalability experiments repeat GA, PSO, NSGA-II, and Random Search across 12 independent algorithm seeds for every action-space size and report median/IQR NCOG and exact-hit rates.

## Reproducibility design

The principal benchmark uses fixed network seeds and a declared eight-action repair universe. The scaling study holds the damaged network realization fixed while varying only the candidate-action universe and algorithmic random seeds. Exact CC is always the reference quantity; heuristic algorithms are evaluated by their gap to that exact target-relative minimum.

The wireless equations are controlled benchmark abstractions and are not presented as site-specific RF calibration. The code is intended to reproduce the manuscript's computational claims exactly under the pinned environment.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.

## Citation

See `CITATION.cff` and the accompanying manuscript.
