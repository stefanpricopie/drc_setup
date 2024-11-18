import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction, PosteriorStandardDeviation
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch import ExactMarginalLogLikelihood

from acquisitions import ExpectedImprovement

NOISE_SE = 1e-03
dtype = torch.double    # use double precision for numerical stability


def random_sample_bounds(bounds, n):
    """
    Generate n random samples from the bounds
    :param bounds: Tensor of shape (d, 2) containing the lower and upper bounds for each dimension
    :param n: Number of samples to generate
    :return: Tensor of shape (n, d) containing the n samples
    """
    x_list = []
    for lb, ub in bounds:
        # Generate a tensor with values uniformly distributed between 0 and 1
        # Scale and shift to match the bounds [lb, ub]
        tensor = torch.rand(n, dtype=dtype) * (ub - lb) + lb
        tensor = tensor.unsqueeze(-1)
        x_list.append(tensor)

    return torch.cat(x_list, dim=-1) if x_list else torch.empty(0, dtype=dtype)


def generate_initial_data(obj, n, xc_dims=None, k=1):
    """
    Generate initial data with multiple evaluations per expensive xc

    :param obj: objective function
    :param n: Number of (initial) points to generate.
    :param k: the number of cheap evaluations per expensive xc. default is 1 for normal BO
    :param xc_dims: Indices of fixed features
    :return: Tensor of dimension (n, d) containing the initial points,
            Tensor of dimension (n, 1) containing the objective values at the initial points,
            Tensor of dimension (n, 1) containing the cost values at the initial points
    """
    # if k is not 1 then xc_dims must be provided and vice versa
    if k != 1 and xc_dims is None:
        raise ValueError("If k is not 1, then xc_dims must be provided.")
    elif k == 1 and xc_dims is not None:
        raise ValueError("If xc_dims is provided, then k must not be 1.")
    
    xc_dims = xc_dims if xc_dims is not None else torch.tensor([], dtype=torch.long)

    # Check divisibility of n by k
    if n % k != 0:
        raise ValueError(f"Number of initial points {n} must be divisible by k = {k}.")

    # Get empty tensor train_x of shape (n, d)
    train_x = torch.empty((n, obj.dim), dtype=dtype)

    # Create a boolean mask for the entire tensor
    mask = torch.zeros(obj.dim, dtype=torch.bool)

    # Update the mask to True for the subset of indices
    mask[xc_dims] = True

    # Get xc_bounds and xuc_bounds
    xc_bounds = obj.bounds.T[mask, :]
    xuc_bounds = obj.bounds.T[~mask, :]

    # Get how many xuc sampled from n
    nxc = n // k

    # Generate xc and repeat along the first dimension for k times
    xc = random_sample_bounds(xc_bounds, nxc)
    xc = xc.repeat_interleave(k, dim=0)

    # Update train_x
    train_x[:, mask] = xc

    # Generate xuc
    xuc = random_sample_bounds(xuc_bounds, n)

    # Update train_x
    train_x[:, ~mask] = xuc

    exact_obj = obj(train_x).unsqueeze(-1)  # add output dimension

    train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)

    return train_x, train_obj, exact_obj


def bo_unconstrained(obj, cost_model, seed, noise_se=NOISE_SE):
    """
    Bayesian optimization with Expected Improvement acquisition function

    :param obj: objective function
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

        new_x, _ = optimize_acqf(
            acq_function=ExpectedImprovement(gp, best_f=train_Y.max().item()),
            bounds=obj.bounds,
            q=1,
            num_restarts=10,
            raw_samples=2048,
        )

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


def bo_randomffc(obj, p, cost_model, seed, noise_se=NOISE_SE):
    """
    Bayesian optimization with Expected Improvement acquisition function

    :param obj: objective function
    :param p: probability of using Fixed Feature acquisition function
    :param cost_model: cost model
    :param seed: seed for random number generator
    :param noise_se: noise standard deviation
    :return:
    """
    xc_dims = cost_model.xc_dims

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

        acq_function = ExpectedImprovement(gp, best_f=train_Y.max().item())

        # get random boolean with probability pff to be True
        if torch.rand(1) < p:
            ff_acq = FixedFeatureAcquisitionFunction(
                acq_function=acq_function,
                d=obj.dim,
                columns=xc_dims,    # indices of fixed features
                values=train_X[-1, xc_dims],
            )

            new_xuc, _ = optimize_acqf(   # returns a 2D tensor of shape (q, d)
                acq_function=ff_acq,
                bounds=obj.bounds[:, ~torch.isin(torch.arange(obj.dim), xc_dims)],    # remove bounds of fixed features
                q=1,
                num_restarts=10,
                raw_samples=2048,
            )

            # Construct full input tensor
            new_x = ff_acq._construct_X_full(new_xuc)
        else:
            new_x, _ = optimize_acqf(
                acq_function=acq_function,
                bounds=obj.bounds,
                q=1,
                num_restarts=10,
                raw_samples=2048,
            )

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


def ffc_bo(obj, k, cost_model, ff_acq, seed, noise_se=NOISE_SE):
    """
    Fixed Feature Cycle (FFC) BO algorithm.

    :param obj: Objective function
    :param k: Periodicity length
    :param cost_model: cost model
    :param ff_acq: Fixed Feature acquisition function - ExpectedImprovement or PosteriorStandardDeviation
    :param seed: Random seed
    :param noise_se: Noise standard deviation
    :return:
    """
    xc_dims = cost_model.xc_dims

    # Set random seed
    torch.manual_seed(seed)
    n_init = 2 * obj.dim

    # Generate initial data with initial obj.dim points
    train_X, train_Y, exact_Y = generate_initial_data(obj, n=n_init)

    T = n_init
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
        ei = ExpectedImprovement(gp, best_f=train_Y.max().item())

        if T % k == n_init % k:
            new_x, _ = optimize_acqf(   # returns a 2D tensor of shape (q, d)
                acq_function=ei,
                bounds=obj.bounds,
                q=1,
                num_restarts=10,
                raw_samples=2048,
            )
        else:
            if ff_acq == 'ei':
                acq_function = ei
            elif ff_acq == 'pstd':
                acq_function = PosteriorStandardDeviation(gp)
            else:
                raise NotImplementedError(f'Fixed Feature acquisition function {ff_acq} not implemented.')

            ff_ei = FixedFeatureAcquisitionFunction(
                acq_function=acq_function,
                d=obj.dim,
                columns=xc_dims,    # indices of fixed features
                values=train_X[-1, xc_dims],
            )

            new_xuc, _ = optimize_acqf(   # returns a 2D tensor of shape (q, d)
                acq_function=ff_ei,
                bounds=obj.bounds[:, ~torch.isin(torch.arange(obj.dim), xc_dims)],    # remove bounds of fixed features
                q=1,
                num_restarts=10,
                raw_samples=2048,
            )

            # Construct full input tensor
            new_x = ff_ei._construct_X_full(new_xuc)

        # Get objective value and cost
        exact_y = obj(new_x).unsqueeze(-1)  # add output dimension
        train_y = exact_y + noise_se * torch.randn_like(exact_y)

        # Update training points
        train_X = torch.cat([train_X, new_x])
        train_Y = torch.cat([train_Y, torch.tensor([train_y], dtype=train_Y.dtype, device=train_Y.device).unsqueeze(1)])
        exact_Y = torch.cat([exact_Y, torch.tensor([exact_y], dtype=exact_Y.dtype, device=exact_Y.device).unsqueeze(1)])

        # Update time
        T += 1
        # Update cost. n_init-1 because we need the last point to calculate
        # the cost but calculate cost from n_init onwards
        X_cost = cost_model(train_X[n_init-1:])[1:]   # Ignore cost of initial points

    return train_X, train_Y, exact_Y


def ffc_bo_nested(obj, k, cost_model, ff_acq, seed, noise_se=NOISE_SE):
    """
    Fixed Feature Cycle (FFC) BO algorithm with nested optimization.

    :param obj: Objective function
    :param k: Periodicity length
    :param cost_model: cost model
    :param ff_acq: Fixed Feature acquisition function - ExpectedImprovement or PosteriorStandardDeviation
    :param seed: Random seed
    :param noise_se: Noise standard deviation
    :return:
    """
    xc_dims = cost_model.xc_dims

    # Set random seed
    torch.manual_seed(seed)
    n_init = 2 * k * obj.dim

    # Generate initial data with initial obj.dim points
    train_X, train_Y, exact_Y = generate_initial_data(obj, k=k, xc_dims=xc_dims, n=n_init)

    T = n_init
    X_cost = torch.tensor(0, dtype=dtype)   # Ignore cost of initial points

    budget = 10 * obj.dim * (cost_model.switching_cost + 1)
    #FIXME: This is not correct. The budget should include the cost of the initial points for the special nested case
    while X_cost.sum() < budget:
        # Train GP - Matérn 5/2 kernels with Automatic Relevance Discovery (ARD)
        gp = SingleTaskGP(train_X=train_X, train_Y=train_Y,
                          train_Yvar=torch.full_like(train_Y, noise_se**2),
                          input_transform=Normalize(d=train_X.shape[-1]),
                          outcome_transform=Standardize(m=1))

        mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
        fit_gpytorch_mll(mll)

        if T % k == n_init % k:
            ### Update Xc and Yc ##
            # Get the best point for each unique Xc value - TODO: can make this faster by only considering the last p points
            train_Xc, train_Yc = _get_best_by_xc(train_X=train_X, train_Y=train_Y, xc_dims=xc_dims)


            ### Optimize xc ###
            # Train GP - Matérn 5/2 kernels with Automatic Relevance Discovery (ARD) - do not specify noise variance
            gp_costly = SingleTaskGP(train_X=train_Xc, train_Y=train_Yc,
                                     # train_Yvar=torch.full_like(train_Yc, noise_se**2),
                                      input_transform=Normalize(d=train_Xc.shape[-1]),
                                      outcome_transform=Standardize(m=1))

            mll_costly = ExactMarginalLogLikelihood(gp_costly.likelihood, gp_costly)
            fit_gpytorch_mll(mll_costly)

            new_xc, _ = optimize_acqf(   # returns a 2D tensor of shape (q, d)
                acq_function=ExpectedImprovement(gp_costly, best_f=train_Yc.max().item(),
                                                observation_noise=True),
                bounds=obj.bounds[:, xc_dims],
                q=1,
                num_restarts=10,
                raw_samples=2048,
            )

            ### Optimize xuc ###
            ff_ei = FixedFeatureAcquisitionFunction(
                acq_function=ExpectedImprovement(gp, best_f=train_Y.max().item()),
                d=obj.dim,
                columns=xc_dims,    # indices of fixed features
                values=new_xc,                  # use the new xc as fixed features
            )

            new_xuc, _ = optimize_acqf(   # returns a 2D tensor of shape (q, d)
                acq_function=ff_ei,
                bounds=obj.bounds[:, ~torch.isin(torch.arange(obj.dim), xc_dims)],    # remove bounds of fixed features
                q=1,
                num_restarts=10,
                raw_samples=2048,
            )

            # Construct full input tensor
            new_x = ff_ei._construct_X_full(new_xuc)

        else:
            if ff_acq == 'ei':
                acq_function = ExpectedImprovement(gp, best_f=train_Y.max().item())
            elif ff_acq == 'pstd':
                acq_function = PosteriorStandardDeviation(gp)
            else:
                raise NotImplementedError(f'Fixed Feature acquisition function {ff_acq} not implemented.')

            ff_pstd = FixedFeatureAcquisitionFunction(
                acq_function=acq_function,
                d=obj.dim,
                columns=xc_dims,    # indices of fixed features
                values=train_X[-1, xc_dims],
            )

            new_xuc, _ = optimize_acqf(   # returns a 2D tensor of shape (q, d)
                acq_function=ff_pstd,
                bounds=obj.bounds[:, ~torch.isin(torch.arange(obj.dim), xc_dims)],    # remove bounds of fixed features
                q=1,
                num_restarts=10,
                raw_samples=2048,
            )

            # Construct full input tensor
            new_x = ff_pstd._construct_X_full(new_xuc)

        # Get objective value and cost
        exact_y = obj(new_x).unsqueeze(-1)  # add output dimension
        train_y = exact_y + noise_se * torch.randn_like(exact_y)

        # Update training points
        train_X = torch.cat([train_X, new_x])
        train_Y = torch.cat([train_Y, torch.tensor([train_y], dtype=train_Y.dtype, device=train_Y.device).unsqueeze(1)])
        exact_Y = torch.cat([exact_Y, torch.tensor([exact_y], dtype=exact_Y.dtype, device=exact_Y.device).unsqueeze(1)])

        # Update time
        T += 1
        # Update cost. n_init-1 because we need the last point to calculate
        # the cost but calculate cost from n_init onwards
        X_cost = cost_model(train_X[n_init-1:])[1:]   # Ignore cost of initial points

    return train_X, train_Y, exact_Y


def _get_best_by_xc(train_X, train_Y, xc_dims):
    # Get unique values in Xc and their first occurrence
    unique_values, unique_indices = torch.unique(train_X[:, xc_dims], dim=0, return_inverse=True)

    # Initialize a list to store the results
    best_indices_per_unique_xc = []

    # Iterate through unique Xc values
    for idx, _ in enumerate(unique_values):
        # Get the indices of the current unique Xc value
        indices = torch.where(unique_indices.eq(idx))[0]

        # Get the best index for the current unique Xc value
        best_index = torch.argmax(train_Y[indices])

        # Store the best index
        best_indices_per_unique_xc.append(indices[best_index].item())

    # sort best indices in place
    best_indices_per_unique_xc = torch.tensor(best_indices_per_unique_xc, dtype=torch.int)
    best_indices_per_unique_xc.sort()

    # Get the best indices - Xc has fewer dimensions than X
    train_Xc = torch.stack([train_X[best_indices_per_unique_xc[i], xc_dims]
                            for i in range(len(best_indices_per_unique_xc))], dim=0)
    train_Yc = train_Y[best_indices_per_unique_xc, :]

    return train_Xc, train_Yc


def eipu(obj, cost_model, seed, noise_se=NOISE_SE):
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

        # Expected Improvement acquisition function
        ei = ExpectedImprovement(gp, best_f=train_Y.max().item())

        # Get the best point on the entire input space
        full_x, full_x_acq = optimize_acqf(   # returns a 2D tensor of shape (q, d)
            acq_function=ei,
            bounds=obj.bounds,
            q=1,
            num_restarts=10,
            raw_samples=2048,
        )

        # Fixed Feature acquisition function
        ff_ei = FixedFeatureAcquisitionFunction(
            acq_function=ei,
            d=obj.dim,
            columns=cost_model.xc_dims,    # indices of fixed features
            values=train_X[-1, cost_model.xc_dims],
        )

        ff_x, ff_x_acq = optimize_acqf(   # returns a 2D tensor of shape (q, d)
            acq_function=ff_ei,
            bounds=obj.bounds[:, ~torch.isin(torch.arange(obj.dim), cost_model.xc_dims)],    # remove bounds of fixed features
            q=1,
            num_restarts=10,
            raw_samples=2048,
        )

        cost_cool = (budget - X_cost.sum()) / budget
        if ff_x_acq > full_x_acq/(cost_model.switching_cost+1)**cost_cool:
            new_x = ff_ei._construct_X_full(ff_x)
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


