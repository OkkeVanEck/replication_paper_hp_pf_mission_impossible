#! /usr/bin/env python3

if __name__ != "__main__":
    raise ImportError(f"{__name__} should be run as a script and not imported")

import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import sample_conformations as pf

os.chdir(Path(__file__).parent)

output_file = Path("required_samples.csv")


trials = int(os.getenv("TRIALS", 32))
max_samples = int(os.getenv("MAX_SAMPLES", 1_000_000))
max_length = int(os.getenv("MAX_LENGTH", 200))
lengths = range(10, max_length + 1, 10)
dimensions = range(2, 9 + 1)
# None -> 'primitive' (only option for now)
lattices_3D = [None]
lattices_2D = [None]


def sample(args):
    args = list(args)
    if args[1] is None:
        args[1] = "primitive"
    dimension, lattice, length = args
    results = dict()
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        num_samples = 0
        has_collisions = True
        while num_samples < max_samples and has_collisions:
            num_samples += 1
            conformation = pf.generate_conformation(length, dimension, lattice, rng)
            positions = pf.get_positions(conformation, dimension, lattice)
            has_collisions = 0 < len(pf.get_collisions(positions))
        if has_collisions:
            results[seed] = np.nan, np.nan
            # It appears that the maximum number of samples is not sufficient -> stop trying (saves CPU hours)
            break
        else:
            results[seed] = (num_samples, str(conformation).replace("\n", ""))
    return args, results


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

output_file.write_text("dimension,lattice,length,seed,samples_required,conformation\n")
with Pool(os.cpu_count()) as pool:
    results = pool.imap_unordered(sample, argss)
    try:
        for args, trial_results in results:
            print(".", end="", flush=True)
            for seed in trial_results:
                csv_cells = args + [seed] + list(trial_results[seed])
                with open(output_file, "a") as f:
                    f.write(",".join([str(c) for c in csv_cells]) + "\n")
    except KeyboardInterrupt:
        pass

print(" Done!")
print(f"Elapsed time: {time.time() - start:.2f} s")
