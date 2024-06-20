#! /usr/bin/env python3

import argparse
import os
import random

import numpy as np

np.set_printoptions(formatter={"int": lambda i: format(i, "x")})


def generate_conformation(length, dimension, lattice, rng=random):
    # The number of moves is equal to the number of connections in the chain (n-1)
    conformation = np.zeros(length - 1, dtype=int)
    for i in range(0, len(conformation)):
        last_move = 0 if i == 0 else conformation[i - 1]
        move = -last_move
        if i == 0 and lattice == "primitive":
            # Since the orientation of the conformation does not matter, the first move can be fixed
            move = 1
        while move == -last_move:  # Avoid trivial collision by "going back"
            if lattice == "primitive":
                move = (
                    # Support both built-in and NumPy generator
                    rng.integers(1, dimension + 1)
                    if hasattr(rng, "integers")
                    else rng.randint(1, dimension)
                )
            else:
                raise ValueError(f'Unknown lattice "{lattice}"')
            move *= rng.choice([-1, 1])
        conformation[i] = move
    return conformation


def _get_position_hash(position):
    return str((np.round(position * 100)).astype(int))[:-1]


def get_positions(conformation, dimension, lattice):
    positions = np.zeros(shape=(len(conformation) + 1, dimension))
    for i, move in enumerate(conformation):
        direction = move // abs(move)
        move = abs(move)
        offset = np.zeros(dimension)
        offset[move - 1] = 1
        positions[i + 1] = positions[i].copy() + offset * direction
    return positions


def get_num_occupied_lattice_points(positions):
    return len(set(_get_position_hash(p) for p in positions))


def get_bounding_volume(positions):
    axes_min = np.min(positions, axis=0)
    axes_range = np.max(positions, axis=0) - axes_min
    axes_center = axes_min + axes_range / 2
    return np.prod(axes_range), axes_range, axes_center


def get_collisions(positions):
    positions_to_indices = dict()
    for i in range(len(positions)):
        k = _get_position_hash(positions[i])
        if k not in positions_to_indices:
            positions_to_indices[k] = []
        positions_to_indices[k].append(i)
    collisions = [v for v in positions_to_indices.values() if len(v) > 1]
    return collisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--length", type=int, required=True, help="Length of the amino acid sequence"
    )
    parser.add_argument(
        "--samples", type=int, required=True, help="Number of samples to generate"
    )
    parser.add_argument(
        "--dimension", type=int, required=True, help="Dimension of the folding space"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the random number generator"
    )

    parser.description = """
    Generates random conformations of chains (for HP model protein folding) and count the number of collisions.
    The resulting conformations is represented by a sequence of moves that make up the complete fold.
    Each digit of a move corresponds to an axis and the sign of the move indicates the direction.
    The output is a CSV with the following columns: index, conformation, collisions.
    The integer coordinates of the conformation are printed to stderr.
    """.strip()

    args = parser.parse_args()

    random.seed(args.seed)

    if args.dimension < 2 and args.dimension >= 0xA:
        raise ValueError("Dimension must be between 2 and 9")
    print(
        ",conformation,num_collisions,max_aminos_per_collision,bounding_volume,num_occupied_lattice_points"
    )
    for i in range(args.samples):
        conformation = generate_conformation(args.length, args.dimension, "primitive")
        positions = get_positions(conformation, args.dimension, "primitive")
        collisions = get_collisions(positions)
        max_aminos_per_collision = 0
        if len(collisions) > 0:
            max_aminos_per_collision = max(len(c) for c in collisions)
        bounding_volume, _, _ = get_bounding_volume(positions)
        num_occupied_lattice_points = get_num_occupied_lattice_points(positions)
        print(
            f"{i},{str(conformation).replace(os.linesep, '')},{len(collisions)},{max_aminos_per_collision},{bounding_volume},{num_occupied_lattice_points}"
        )


if __name__ == "__main__":
    main()
