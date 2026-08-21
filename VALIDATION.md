# Validation

Final clean-run validation completed on 2026-08-21.

- `pytest -q`: 7 passed.
- Clean `python run_all.py`: completed successfully from an empty `results/` directory.
- Numerical validator: passed tables, paired COG/NCOG, CC/CER selected actions, regression diagnostics, and repeated-seed scaling checks.
- `run_all.py` recreates `results/raw`, `results/tables`, `results/figures`, and `results/manifest.json` from scratch.
- GitHub Actions workflow: `.github/workflows/reproduce.yml` installs dependencies, tests, reproduces the study, uploads the results artifact, and commits regenerated results when repository permissions allow.
- License: Apache-2.0.
