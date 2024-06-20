
import seaborn as sns
import os
from collections import Counter
from random import random
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as optimize
from scipy.stats import betabinom
from IPython.display import display, HTML
from colorsys import rgb_to_hls, hls_to_rgb
from collections import defaultdict
from matplotlib.gridspec import GridSpec
from scipy.stats import betabinom
from scipy.special import comb


def read_conformations_file(length, dimension):
    return pd.read_csv(os.path.join("conformations", dimension, str(length)+".csv"))


def make_bb_individual_fit(conflicts, protein_lengths, length, dimension, fname, data_limit_per_len):
    def func_to_minimize(x):
        a, b = x[0], x[1]
        return sum(-betabinom.logpmf(conflicts, protein_lengths, a, b))

    def process_fit(result):
        conformation_df = read_conformations_file(length, dimension)
        mu_data, sigma_data = conformation_df["num_collisions"].mean(
        ), conformation_df["num_collisions"].std()
        a = result.x[0]
        b = result.x[1]
        mu_fit = betabinom.mean(length, a, b)
        sigma_fit = betabinom.std(length, a, b)

        resdict = {"length": length, "dimension": dimension, "a": a, "b": b, "mu_data": mu_data, "mu_fit": mu_fit, "std_data": sigma_data, "std_fit": sigma_fit
                   }

        with open(fname, 'wb') as f:
            pickle.dump(resdict, f)
        return resdict

    return func_to_minimize, process_fit


def make_bb_direct_fit(conflicts, protein_lengths, length, dimension, fname, data_limit_per_len):
    def func_to_minimize(x):
        a, b = x[0], x[1]
        return sum(-betabinom.logpmf(conflicts, protein_lengths, a, b))

    def process_fit(result):
        all_resdicts = []
        for length in [(i+1)*10 for i in range(20)]:
            conformation_df = read_conformations_file(length, dimension)
            mu_data, sigma_data = conformation_df["num_collisions"].mean(
            ), conformation_df["num_collisions"].std()
            a = result.x[0]
            b = result.x[1]
            mu_fit = betabinom.mean(length, a, b)
            sigma_fit = betabinom.std(length, a, b)

            resdict = {"length": length, "dimension": dimension, "a": a, "b": b, "mu_data": mu_data, "mu_fit": mu_fit, "std_data": sigma_data, "std_fit": sigma_fit
                       }

            lengthwise_fname = fname_length_fit(
                "direct", dimension, length, data_limit_per_len)
            with open(lengthwise_fname, 'wb') as f:
                pickle.dump(resdict, f)
        with open(fname, 'wb') as f:
            pickle.dump(all_resdicts, f)
        return resdict
    return func_to_minimize, process_fit


def make_bb_reparameterize_fit(conflicts, protein_lengths, length, dimension, fname, data_limit_per_len):
    def fit_param_to_ab(x, _length):
        a = x[0]*_length + x[1]
        b = x[2]*_length + x[3]
        return a, b

    def func_to_minimize(x):
        a, b = fit_param_to_ab(x, protein_lengths)
        return sum(-betabinom.logpmf(conflicts, protein_lengths, a, b))

    def process_fit(result):
        all_resdicts = []
        for length in [(i+1)*10 for i in range(20)]:
            conformation_df = read_conformations_file(length, dimension)
            mu_data, sigma_data = conformation_df["num_collisions"].mean(
            ), conformation_df["num_collisions"].std()
            a, b = fit_param_to_ab(result.x, length)
            mu_fit = betabinom.mean(length, a, b)
            sigma_fit = betabinom.std(length, a, b)

            resdict = {"length": length,
                       "dimension": dimension,
                       "a": a,
                       "b": b,
                       "mu_data": mu_data,
                       "mu_fit": mu_fit,
                       "std_data": sigma_data,
                       "std_fit": sigma_fit,
                       "x0": result.x[0],
                       "x1": result.x[1],
                       "x2": result.x[2],
                       "x3": result.x[3],
                       }
            lengthwise_fname = fname_length_fit(
                "reparameterize", dimension, length, data_limit_per_len)
            with open(lengthwise_fname, 'wb') as f:
                pickle.dump(resdict, f)
            all_resdicts.append(resdict)
        with open(fname, "wb") as f:
            pickle.dump(all_resdicts, f)
        return resdict

    return func_to_minimize, process_fit


def get_min_max(length, dimension):
    conformation_df = read_conformations_file(length, dimension)
    PLs = conformation_df["num_collisions"].apply(lambda x: length)
    conflicts = conformation_df["num_collisions"]
    return min(conflicts), max(conflicts)


def make_bb_fit_individual(length, dimension, num_tries):
    bounds = [
        (0, None),
        (0, None)
    ]


def make_bb_fit_reparameterize(length, dimension, num_tries):
    bounds = [
        (0, None),
        (0, None),
        (0, None),
        (0, None)
    ]


def get_fit_result_with_mix_max(fname, length, dimension):
    with open(fname, 'rb') as f:
        results = pickle.load(f)
    if not ("min" in results):
        _min, _max = get_min_max(length, dimension)
        results["min"] = _min
        results["max"] = _max
        with open(fname, 'wb') as f:
            pickle.dump(results, f)
    return results


def fname_length_fit(kind, dimension, length, data_limit_per_len):
    return os.path.join("fits", f"{kind}_{dimension}_{length}_data-lim={str(data_limit_per_len)}.pkl")


def fname_full_fit(kind, dimension, data_limit_per_len):
    return os.path.join("fits", f"{kind}_{dimension}_data-lim={str(data_limit_per_len)}.pkl")


def get_fit(length, dimension, kind, overwrite=False, num_tries=100, data_limit_per_len=np.inf, verbose=False):
    """ get fits for beta binomial

    Args:
        length (int): 10,20,30,...200
        dimension (str): eg: '2D', '3D'
        kind (str): how to fit the binomial, options: ["reparameterize", "direct", "individual"].
        overwrite (bool, optional): whether to redo the fit. Defaults to True.
        num_tries (int, optional): how often the fitting is re-initialized before giving up. Defaults to 50.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    fname = {
        "direct": fname_full_fit(kind, dimension, data_limit_per_len),
        "individual": os.path.join("fits", f"{kind}_{dimension}_{length}_data-lim={str(data_limit_per_len)}.pkl"),
        "reparameterize": fname_full_fit(kind, dimension, data_limit_per_len)
    }[kind]

    if verbose:
        print("fname =", fname, end="\r")

    if not os.path.exists(fname) or overwrite:
        par_dir = os.path.join(*(fname.split(os.sep)[:-1]))
        if not os.path.exists(par_dir):
            if verbose:
                print("par_dir=", par_dir, end="\r")
            os.makedirs(par_dir, exist_ok=True)

        if kind == "direct":

            conformation_dfs = []
            for _length in [(i+1)*10 for i in range(20)]:
                conformation_df = read_conformations_file(_length, dimension)
                conformation_df["length"] = conformation_df["num_collisions"].apply(
                    lambda x: _length)
                conformation_df = conformation_df.head(
                    min(len(conformation_df), data_limit_per_len))

                conformation_dfs.append(conformation_df)

            conformation_df = pd.concat(conformation_dfs)
            func_to_minimize, process_fit = make_bb_direct_fit(
                conformation_df["num_collisions"], conformation_df["length"], length, dimension, fname, data_limit_per_len)

        if kind == "individual":

            initial_guess = [random(), random()]
            conformation_df = read_conformations_file(length, dimension)
            conformation_df["length"] = conformation_df["num_collisions"].apply(
                lambda x: length)
            conformation_df = conformation_df.head(
                min(len(conformation_df), data_limit_per_len))
            func_to_minimize, process_fit = make_bb_individual_fit(
                conformation_df["num_collisions"], conformation_df["length"], length, dimension, fname, data_limit_per_len)

        if kind == "reparameterize":

            conformation_dfs = []
            for _length in [(i+1)*10 for i in range(20)]:
                conformation_df = read_conformations_file(_length, dimension)
                conformation_df["length"] = conformation_df["num_collisions"].apply(
                    lambda x: _length)
                conformation_df = conformation_df.head(
                    min(len(conformation_df), data_limit_per_len))
                conformation_dfs.append(conformation_df)

            conformation_df = pd.concat(conformation_dfs)
            func_to_minimize, process_fit = make_bb_reparameterize_fit(
                conformation_df["num_collisions"], conformation_df["length"], length, dimension, fname, data_limit_per_len)

        for i in range(num_tries):
            if verbose:
                print(
                    f"fitting... {kind=}, {length=}, {dimension=} trial {i}/{num_tries}")

            if kind in ["individual", "direct"]:
                bounds = [
                    (0, None),
                    (0, None)
                ]
                initial_guess = [random(), random()]
            elif kind == "reparameterize":
                bounds = [
                    (0, None),
                    (0, None),
                    (0, None),
                    (0, None)
                ]
                initial_guess = [random(), random(), random(), random()]
            else:
                raise ValueError

            result = optimize.minimize(
                func_to_minimize, initial_guess, bounds=bounds)
            if result.success:
                process_fit(result)
                break
        else:
            raise ValueError(
                f"couldn't find fit result for {kind=}, {length=}, {dimension=}")
    else:
        if verbose:
            print("fit exists")

    if kind in ["reparameterize", "direct"]:
        fname = fname_length_fit(kind, dimension, length, data_limit_per_len)

    return get_fit_result_with_mix_max(fname, length, dimension)


def unpack_fitfunc(length, dimension_str, fit_kind, overwrite=False, verbose=True):

    # fit_kind, (enum_d, d, dimension_str), length = arg
    if verbose:
        print("starting", fit_kind, dimension_str, length)
    return get_fit(length, dimension_str, fit_kind, overwrite=overwrite, verbose=verbose)


def lighten_color(color):
    """Lighten a color by half the difference between the current color and white."""
    # Convert color to RGB
    r, g, b = [
        x / 255.0 for x in tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))]

    # Convert RGB to HSL
    h, l, s = rgb_to_hls(r, g, b)

    # Calculate the difference between current color and white
    l_diff = 1.0 - l

    # Lighten the color by half of the difference
    new_l = min(1, l + l_diff / 2)

    # Convert the new HSL back to RGB
    new_r, new_g, new_b = hls_to_rgb(h, new_l, s)

    # Convert RGB back to hexadecimal color code
    return "#{:02x}{:02x}{:02x}".format(int(new_r * 255), int(new_g * 255), int(new_b * 255))


def cmap_from_colorblind(dimension, brighten_n_times):
    """returns hex color for dimension & amount of times brightened
    brightening only relevant for multiple plots of same dimension, only relevant for distribution plot.

    Args:
        dimension int: which dimension
        n_amino_idx : _description_

    Returns:
        str: hex

    >>>cmap_from_colorblind(2, 0) # 2D
    >>>'#f4fbfe'
    """
    dim_idx = dimension-2
    colorblind_colors = list(sns.color_palette("colorblind").as_hex())
    color = colorblind_colors[dim_idx]
    for _ in range(brighten_n_times):
        color = lighten_color(color)

    return color


if __name__ == "__main__":
    import itertools
    from multiprocessing import Process
    import multiprocessing

    dimensions = [str(d)+"D" for d in range(2, 10)]
    lengths = [(i+1)*10 for i in range(20)]

    fit_everything = False
    if fit_everything:
        combined_iter = [e for e in itertools.product(
            lengths, dimensions, ["individual"])]

        combined_iter_2 = [e for e in itertools.product(
            [10], dimensions, ["reparameterize", "direct"])]

        combined_iter = combined_iter + combined_iter_2
        pool = multiprocessing.Pool()

        # Run the function with multiple arguments in parallel
        results = pool.starmap(unpack_fitfunc, combined_iter)

    else:
        fit_reparameterized = True
        if fit_reparameterized:
            combined_iter = [e for e in itertools.product(
                [10], dimensions, ["reparameterize"])]

            pool = multiprocessing.Pool()

            # Run the function with multiple arguments in parallel
            results = pool.starmap(unpack_fitfunc, combined_iter)

    # Close the pool to free resources
    pool.close()
    pool.join()
