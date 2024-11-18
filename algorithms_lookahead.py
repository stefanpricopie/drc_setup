import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction, qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch import ExactMarginalLogLikelihood

from acquisitions import ExpectedImprovement, ExpectedImprovementwithLookahead
from algorithms import generate_initial_data, random_sample_bounds
from functools import partial
from utils import gen_batch_initial_conditions, gen_candidates_scipy

NOISE_SE = 1e-03
dtype = torch.double    # use double precision for numerical stability


def eipu_lookahead(obj, cost_model, seed, noise_se=NOISE_SE):
    """
    Expected Improvement with Cost (EIpu) algorithm.
    :param obj: Objective function
    :param cost_model: Cost model
    :param seed: Seed for random number generator
    :param noise_se: Noise standard deviation
    :return:
    """
    # Set random seed
    torch.manual_seed(seed)
    n_init = 2 * obj.dim

    # Generate initial data with initial obj.dim points
    train_X, train_Y, exact_Y = generate_initial_data(obj, n=n_init)

    X_cost = torch.tensor(0, dtype=dtype)   # Ignore cost of initial points

    budget = 10 * obj.dim * (cost_model.switching_cost + 1)
    while X_cost.sum() < budget:
        # Train GP - Matérn 5/2 kernels with Automatic Relevance Discovery (ARD)
        gp = SingleTaskGP(train_X=train_X, train_Y=train_Y,
                          train_Yvar=torch.full_like(train_Y, noise_se**2),
                          input_transform=Normalize(d=train_X.shape[-1]),
                          outcome_transform=Standardize(m=1))

        mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
        fit_gpytorch_mll(mll)

        # Get 100 xuc samples
        xuc_bounds = obj.bounds.T[~torch.isin(torch.arange(obj.dim), cost_model.xc_dims), :]
        xuc_samples = random_sample_bounds(bounds=xuc_bounds, n=100)

        # Expected Improvement acquisition function
        eilu = ExpectedImprovementwithLookahead(gp, best_f=train_Y.max().item(),
                                                xc_dims=cost_model.xc_dims, d=obj.dim, xuc_samples=xuc_samples)


        # Get the best point on the entire input space
        full_x, full_x_acq = optimize_acqf(   # returns a 2D tensor of shape (q, d)
            acq_function=eilu,
            bounds=obj.bounds,
            q=1,
            num_restarts=10,
            raw_samples=2048,
        )

        # TODO: Make fixed feature lookahead faster
        # Fixed Feature acquisition function
        ff_eilu = FixedFeatureAcquisitionFunction(
            acq_function=eilu,
            d=obj.dim,
            columns=cost_model.xc_dims,    # indices of fixed features
            values=train_X[-1, cost_model.xc_dims],
        )

        ff_x, ff_x_acq = optimize_acqf(   # returns a 2D tensor of shape (q, d)
            acq_function=ff_eilu,
            bounds=obj.bounds[:, ~torch.isin(torch.arange(obj.dim), cost_model.xc_dims)],    # remove bounds of fixed features
            q=1,
            num_restarts=10,
            raw_samples=2048,
        )

        if ff_x_acq > full_x_acq/cost_model.switching_cost:
            new_x = ff_eilu._construct_X_full(ff_x)
        else:
            new_x = full_x

        # Get objective value and cost
        exact_y = obj(new_x).unsqueeze(-1)  # add output dimension
        train_y = exact_y + noise_se * torch.randn_like(exact_y)

        # Update training points
        train_X = torch.cat([train_X, new_x])
        train_Y = torch.cat([train_Y, torch.tensor([train_y], dtype=train_Y.dtype, device=train_Y.device).unsqueeze(1)])
        exact_Y = torch.cat([exact_Y, torch.tensor([exact_y], dtype=exact_Y.dtype, device=exact_Y.device).unsqueeze(1)])

        # Update cost. n_init-1 because we need the last point to calculate
        # the cost but calculate cost from n_init onwards
        X_cost = cost_model(train_X[n_init-1:])[1:]   # Ignore cost of initial points

    return train_X, train_Y, exact_Y


def bo_batch(obj, cost_model, seed, noise_se=NOISE_SE):
    """
    Bayesian optimization with Expected Improvement acquisition function

    :param obj: objective function
    :param xc_dims: indices of fixed features - used only in the initialization
    :param cost_model: cost model
    :param seed: seed for random number generator
    :param noise_se: noise standard deviation
    :return:
    """
    # Set random seed
    torch.manual_seed(seed)
    n_init = 2 * obj.dim

    # Generate initial data with initial obj.dim points
    train_X, train_Y, exact_Y = generate_initial_data(obj, n=n_init)

    X_cost = torch.tensor(0, dtype=dtype)   # Ignore cost of initial points

    budget = 10 * obj.dim * (cost_model.switching_cost + 1)
    while X_cost.sum() < budget:
        # Train GP - Matérn 5/2 kernels with Automatic Relevance Discovery (ARD)
        gp = SingleTaskGP(train_X=train_X, train_Y=train_Y,
                          train_Yvar=torch.full_like(train_Y, noise_se**2),
                          input_transform=Normalize(d=train_X.shape[-1]),
                          outcome_transform=Standardize(m=1))

        mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
        fit_gpytorch_mll(mll)

        # Expected Improvement acquisition function
        qei = qExpectedImprovement(gp, best_f=train_Y.max().item())

        # Get the best point on the entire input space
        full_x, full_x_acq = optimize_acqf(
            acq_function=qei,
            bounds=obj.bounds,
            q=2,
            num_restarts=10,
            raw_samples=2048,
            ic_generator=partial(gen_batch_initial_conditions,
                                 xc_dims=cost_model.xc_dims),
            gen_candidates=partial(gen_candidates_scipy,
                                   xc_dims=cost_model.xc_dims),
        )

        # Fixed Feature acquisition function
        ff_qei = FixedFeatureAcquisitionFunction(
            acq_function=qei,
            d=obj.dim,
            columns=cost_model.xc_dims,    # indices of fixed features
            values=train_X[-1, cost_model.xc_dims],
        )

        try:
            ff_x, ff_x_acq = optimize_acqf(   # returns a 2D tensor of shape (q, d)
                acq_function=ff_qei,
                bounds=obj.bounds[:, ~torch.isin(torch.arange(obj.dim), cost_model.xc_dims)],    # remove bounds of fixed features
                q=2,
                num_restarts=10,
                raw_samples=2048,
            )
        except RuntimeError as e:
            print(f"Runtime error during optimization: {e}")
            # Optionally, log the error to a file or logging service

            # Example: Save the current state for debugging
            # Define the filename for saving
            filename = "gp_optimization_state.pt"

            # Collect the current state data
            state_data = {
                "train_X": train_X,
                "train_Y": train_Y,
                "noise_se": noise_se,
                "best_point_so_far": train_X[-1]  # Assuming the last point is the best so far
            }

            # Save the state using torch.save
            torch.save(state_data, filename)
            print(f"Saved the optimization state to '{filename}' for debugging.")

            # Decide how to handle the exception
            # For re-raising, simply use:
            raise
            # Or, for a more graceful exit, you might skip the current iteration,
            # log the issue, or attempt a fallback strategy.

        if ff_x_acq > full_x_acq/cost_model.switching_cost:
            new_x = ff_qei._construct_X_full(ff_x)
        else:
            new_x = full_x

        # select from batch the best point
        ei = ExpectedImprovement(gp, best_f=train_Y.max().item())
        ei_value = ei(new_x.unsqueeze(1))
        _, max_index = ei_value.max(0)
        new_x = new_x[max_index].unsqueeze(0)

        # Get objective value and cost
        exact_y = obj(new_x).unsqueeze(-1)  # add output dimension
        train_y = exact_y + noise_se * torch.randn_like(exact_y)

        # Update training points
        train_X = torch.cat([train_X, new_x])
        train_Y = torch.cat([train_Y, torch.tensor([train_y], dtype=train_Y.dtype, device=train_Y.device).unsqueeze(1)])
        exact_Y = torch.cat([exact_Y, torch.tensor([exact_y], dtype=exact_Y.dtype, device=exact_Y.device).unsqueeze(1)])

        # Update cost. n_init-1 because we need the last point to calculate
        # the cost but calculate cost from n_init onwards
        X_cost = cost_model(train_X[n_init-1:])[1:]   # Ignore cost of initial points

    return train_X, train_Y, exact_Y
