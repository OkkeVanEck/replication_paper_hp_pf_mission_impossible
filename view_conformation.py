#! /usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

from sample_conformations import get_collisions, get_positions


def view_conformation(conformation, lattice, plotter):
    positions = get_positions(conformation, 3, lattice)
    collisions = get_collisions(positions)
    for i in range(len(positions)):
        position = positions[i]
        if i > 0:
            cylinder = pv.Cylinder(
                center=positions[i - 1] - (positions[i - 1] - position) / 2,
                direction=positions[i - 1] - position,
                radius=0.1,
                height=1,
            )
            plotter.add_mesh(
                cylinder,
                color="lightgray",
                smooth_shading=True,
                split_sharp_edges=True,
            )
        color = "lightblue"
        for collision in collisions:
            if i in collision:
                # Indicate collision, but do not draw it multiple times (to avoid z-fighting)
                color = "red" if i == collision[0] else None
        if color is None:
            continue
        sphere = pv.Icosphere(radius=0.2, center=position)
        plotter.add_mesh(
            sphere,
            color=color,
            smooth_shading=True,
            split_sharp_edges=True,
        )
    # plotter.add_bounding_box(line_width=5, color="green")
    if np.sum(np.absolute(positions[:, 2])) < 1:
        # View 2D conformations from the front
        plotter.view_xy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV file containing the conformations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output file of the visualization",
    )
    parser.add_argument(
        "--conformation",
        type=int,
        required=True,
        help="Index of the conformation in the input CSV file",
    )

    parser.description = "Visualizes a selected conformation in 3D."

    args = parser.parse_args()

    csv_row = pd.read_csv(args.input).iloc[args.conformation]
    conformation = np.array(
        [int(s, 16) for s in csv_row["conformation"][1:-1].split(" ")]
    )
    pl = pv.Plotter()
    pl.enable_anti_aliasing()
    view_conformation(conformation, "primitive", pl)
    if args.output is not None:
        pl.show(
            auto_close=False,
            interactive_update=True,
            screenshot=args.output,
            window_size=(800, 600),
        )
        pl.close()
    else:
        pl.show()


if __name__ == "__main__":
    main()
