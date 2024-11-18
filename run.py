#!/usr/bin/env python3
import argparse
import os
import pickle
import warnings

import torch
from botorch.exceptions import BadInitialCandidatesWarning, OptimizationWarning
from botorch.test_functions import Ackley, Griewank, Levy, Michalewicz, Rosenbrock
from gpytorch.utils.warnings import NumericalWarning

from algorithms import bo_unconstrained, bo_randomffc, ffc_bo, ffc_bo_nested, eipu
# from algorithms_lookahead import eipu_lookahead, bo_batch
from cost import SetupCostModel
from test_functions import Salomon, Schwefel

# ignore initial candidates warnings - BadInitialCandidatesWarning
# ignore small noise warnings - NumericalWarning
# ignore Optimization Warnings from optimize_acqf - OptimizationWarning
# ignore Optimization failed in `gen_candidates_scipy` - RuntimeWarning
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=NumericalWarning)
warnings.filterwarnings("ignore", category=OptimizationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main(problem, dim, algo, k, seed, switching_cost, xcs, outdir, p=None):
    """
    :param problem: Name of the problem to solve
    :param dim: Dimensionality of the problem
    :param algo: Algorithm to use
    :param k: Fixed Feature Cycle (FFC) length for FFC algorithms
    :param seed: Random seed
    :param switching_cost: Setup cost
    :param xcs: Number of expensive dimensions in the problem 1 <= xcs < dim
    :param outdir: Output directory
    :param p: probability of selecting a fixed feature evaluation
    :return:
    """
    try:
        obj = problems[problem](dim=dim, negate=True)
    except TypeError:
        obj = problems[problem](negate=True)
        assert obj.dim == dim, f"Problem {problem} is not defined for dim={dim}"

    # Define the cost model
    xc_rng = torch.Generator()
    xc_rng.manual_seed(seed)

    XC_DIMS = torch.randperm(dim, generator=xc_rng)[:xcs].sort().values
    cost_model = SetupCostModel(switching_cost=switching_cost, xc_dims=XC_DIMS)

    # File naming and existence check
    # Determine the suffix based on the algorithm
    suffix = ""
    if p is not None:
        suffix += "_p" + f"{p * 100:.0f}"
    elif k is not None:
        suffix += "_k" + str(k)

    # Construct the file name using the determined suffix
    file_name = f"{outdir}/{problem}{dim}d_{algo}{suffix}_sc{switching_cost}_xc{xcs}_{','.join(XC_DIMS.numpy().astype(str))}_r{seed}.pkl".lower()
    print(file_name)

    # if file exists, skip the experiment
    if os.path.exists(file_name):
        print(f"File already exists... Skip experiment")
        return

    # Run the experiment
    if algo == "bo_random" and p is not None:
        X, Y_noisy, Y_exact = bo_randomffc(obj, p=p, cost_model=cost_model, seed=seed)
        C = cost_model(X)
    elif algo == "bo":
        X, Y_noisy, Y_exact = bo_unconstrained(obj, cost_model=cost_model, seed=seed)
        C = cost_model(X)
    elif algo == "pbo":
        X, Y_noisy, Y_exact = ffc_bo(obj, k=k, cost_model=cost_model, ff_acq='ei', seed=seed)
        C = cost_model(X)
    elif algo == "pbonested":
        X, Y_noisy, Y_exact = ffc_bo_nested(obj, k=k, cost_model=cost_model, ff_acq='ei', seed=seed)
        C = cost_model(X)
    elif algo == "eipu":
        X, Y_noisy, Y_exact = eipu(obj, cost_model=cost_model, seed=seed)
        C = cost_model(X)
    # elif algo == "eilo":
    #     X, Y_noisy, Y_exact = eipu_lookahead(obj, k=k, cost_model=cost_model, seed=seed)
    #     C = cost_model(X)
    # elif algo == "eipubatch":
    #     X, Y_noisy, Y_exact = bo_batch(obj, k=k, cost_model=cost_model, seed=seed)
    #     C = cost_model(X)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    results = { # Save results to a dictionary
        'X': X,
        'Y_noisy': Y_noisy,
        'Y_exact': Y_exact,
        'C': C,
        'obj': obj,
        'problem': problem,
        'dim': obj.dim,
        'switching_cost': switching_cost,
        'xcs': xcs,
        'xc_dims': XC_DIMS,
        'algo': algo,
        'k': k,
        'p': p,
        'seed': seed,
    }

    with open(file_name, 'wb') as f:
        pickle.dump(results, f)


# Command line arguments
parser = argparse.ArgumentParser()
# Define your problem
problems = {
    'ackley': lambda dim, negate: Ackley(negate=negate, dim=dim, bounds=[(-15, 30)] * dim),
    'griewank': lambda dim, negate: Griewank(negate=negate, dim=dim, bounds=[(-300, 600)] * dim),
    'levy': Levy,
    'michalewicz': Michalewicz,
    'rosenbrock': Rosenbrock,
    'salomon': lambda dim, negate: Salomon(negate=negate, dim=dim, bounds=[(-50, 100)] * dim),
    'schwefel': Schwefel,
}
parser.add_argument('--problem', type=str, choices=problems.keys(), required=True, help='Problem to solve')
parser.add_argument('--dim', type=int, default=2, help='Dimensionality of the problem')
parser.add_argument('--algo', type=str, choices=['bo', 'bo_random', 'pbo', 'pbonested', 'eipu'],
                    required=True, help='Algorithm to use')
parser.add_argument('--k', type=int, default=None, help='Fixed Feature Cycle (FFC) length')
parser.add_argument('--p', type=float, default=None, help='Number of fixed feature function evaluations')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--switching_cost', type=int, default=5, help='Setup cost')
parser.add_argument('--xcs', type=int, default=1, help='Iterable of dimension indices Xc')
parser.add_argument('--outdir', type=str, help='Output directory')
args = parser.parse_args()

# Main execution
if __name__ == "__main__":
    # Check if p is greater than 1 for pBO and pBOnested algorithms
    if args.algo.startswith('ffc') and args.k <= 1:
        raise ValueError("k must be greater than 1 for pBO and pBOPE algorithms.")

    if not 1 <= args.xcs < args.dim:
        raise ValueError(f"Invalid xcs: {args.xcs}. Must be in [1, {args.dim})")

    main(problem=args.problem, dim=args.dim, algo=args.algo, k=args.k, seed=args.seed,
         switching_cost=args.switching_cost, xcs=args.xcs, p=args.p, outdir=args.outdir)
