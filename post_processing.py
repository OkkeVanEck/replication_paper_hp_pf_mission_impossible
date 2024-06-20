#! /usr/bin/env python3

if __name__ != "__main__":
    raise ImportError(f"{__name__} should be run as a script and not imported")

import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

conformations_path = Path(__file__).parent / "conformations"


def generate_summary_dimension(directory):
    output_file = directory.parent / f"summary_{directory.name}.csv"
    stats = ["min", "max", "mean", "std"]
    columns = ["length", "req_samples"] + [f"num_collisions_{s}" for s in stats]
    output_file.write_text(",".join(columns) + "\n")
    with open(output_file, "a") as f:
        for conformations_csv in directory.glob("*.csv"):
            length = int(conformations_csv.stem)
            f.write(f"{length}")
            num_collisions = pd.read_csv(conformations_csv)["num_collisions"]
            p_no_collisions = np.count_nonzero(num_collisions == 0) / len(
                num_collisions
            )
            req_samples = np.inf if p_no_collisions == 0 else 1 / p_no_collisions
            f.write(f",{req_samples}")
            for s in stats:
                f.write(f",{getattr(num_collisions, s)()}")
            f.write("\n")
    print(".", end="", flush=True)


dimension_dirs = []
for path in conformations_path.iterdir():
    if not path.is_dir():
        continue
    dimension_dirs.append(path)

print("Generating summary over lengths by dimension", end="", flush=True)
start = time.time()

# [generate_summary_dimension(d) for d in dimension_dirs]
with Pool(os.cpu_count()) as pool:
    pool.map(generate_summary_dimension, dimension_dirs)

print("Done!")
print(f"Elapsed time: {time.time() - start:.2f} s")
