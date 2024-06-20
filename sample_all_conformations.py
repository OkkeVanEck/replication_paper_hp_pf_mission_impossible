#! /usr/bin/env python3

if __name__ != "__main__":
    raise ImportError(f"{__name__} should be run as a script and not imported")

import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.chdir(Path(__file__).parent)


samples = int(os.getenv("SAMPLES", 100_000))
max_length = int(os.getenv("MAX_LENGTH", 200))
lengths = range(10, max_length + 1, 10)
dimensions = range(2, 9 + 1)
# None -> 'primitive' (only option for now)
lattices_3D = [None]
lattices_2D = [None]

conformations_dir = Path("conformations")

if conformations_dir.exists():
    shutil.rmtree(conformations_dir)
conformations_dir.mkdir(parents=True)


def sample(dimension, lattice, length):
    output_folder_name = f"{dimension}D" + ("" if lattice is None else f"_{lattice}")
    output_file = conformations_dir / output_folder_name / f"{length}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    command = (
        f"{sys.executable} ./sample_conformations.py"
        + f" --length {length}"
        + f" --dimension {dimension}"
        + ("" if lattice is None else f" --lattice {lattice}")
        + f" --samples {samples}"
        + f" > {output_file} 2> /dev/null"
    )
    os.system(command)
    print(".", end="", flush=True)


argss = []
for d in dimensions:
    if d == 2:
        lattices = lattices_2D
    elif d == 3:
        lattices = lattices_3D
    else:
        lattices = [None]
    for lattice in lattices:
        argss += [(d, lattice, n) for n in lengths]

print("Sampling conformations", end="", flush=True)
start = time.time()

with Pool(os.cpu_count()) as pool:
    pool.starmap(sample, argss)

print(" Done!")
print(f"Elapsed time: {time.time() - start:.2f} s")
