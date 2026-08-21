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
