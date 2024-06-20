## Replication package
This repository contains the code to generate the dataset, metrics and visualizations for our experiments of randomly sampling HP protein conformations in 2D-9D.

### Quickstart
1. Install and activate the virtual environment with `python -m venv .venv && . .venv && pip install -r requirements`
2. Sample conformations using `SAMPLES=1000 MAX_LENGTH=200 ./sample_all_conformations` (Takes very, very long, so maybe reduce the number of samples; The results are stored in `./conformations`; Old results are deleted at the beginning of the script)
3. Visualize a single conformation using `./view_conformation --input <path/to/generated/csv> --conformation <index (first column) in the CSV file>`
4. Generate summary statistics using `./post_processing.py`
5. The number of samples required to obtain a valid conformation can be determined using `./count_required_samples.py` which writes the corresponding number to `./required_samples.csv`
6. The figures assuming a beta binomial distribution are created via the figures_bb_fit.ipynb and assume the results of sampling and number of samples required are in the aforementioned directories & files.

