from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch import settings
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch.exceptions.warnings import (
    BadInitialCandidatesWarning,
    SamplingWarning,
)
from botorch.generation.gen import (_process_scipy_result,
                                    _convert_nonlinear_inequality_constraints,
                                    make_scipy_nonlinear_inequality_constraints,
                                    _arrayify,
                                    make_scipy_bounds,
                                    make_scipy_linear_constraints,
                                    minimize_with_timeout,
                                    nonlinear_constraint_is_feasible)
from botorch.generation.utils import (
    _remove_fixed_features_from_optimization,
)
from botorch.logging import _get_logger
from botorch.optim.initializers import (is_nonnegative, initialize_q_batch, initialize_q_batch_nonneg,
                                        sample_points_around_best, sample_q_batches_from_polytope)
from botorch.optim.utils import columnwise_clamp
from botorch.optim.utils import fix_features
from botorch.utils.sampling import (
    draw_sobol_samples,
    manual_seed,
)
from torch import Tensor
from torch.quasirandom import SobolEngine

logger = _get_logger()


def gen_batch_initial_conditions(
        acq_function: AcquisitionFunction,
        bounds: Tensor,
        q: int,
        num_restarts: int,
        raw_samples: int,
        fixed_features: Optional[Dict[int, float]] = None,
        options: Optional[Dict[str, Union[bool, float, int]]] = None,
        inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        generator: Optional[Callable[[int, int, int], Tensor]] = None,
        fixed_X_fantasies: Optional[Tensor] = None,
        xc_dims: Optional[List[int]] = None,
) -> Tensor:
    r"""Generate a batch of initial conditions for random-restart optimziation.

    TODO: Support t-batches of initial conditions.

    Args:
        acq_function: The acquisition function to be optimized.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates to consider.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic. Note: if `sample_around_best` is True (the default is False),
            then `2 * raw_samples` samples are used.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        options: Options for initial condition generation. For valid options see
            `initialize_q_batch` and `initialize_q_batch_nonneg`. If `options`
            contains a `nonnegative=True` entry, then `acq_function` is
            assumed to be non-negative (useful when using custom acquisition
            functions). In addition, an "init_batch_limit" option can be passed
            to specify the batch limit for the initialization. This is useful
            for avoiding memory limits when computing the batch posterior over
            raw samples.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.
        generator: Callable for generating samples that are then further
            processed. It receives `n`, `q` and `seed` as arguments and
            returns a tensor of shape `n x q x d`.
        fixed_X_fantasies: A fixed set of fantasy points to concatenate to
            the `q` candidates being initialized along the `-2` dimension. The
            shape should be `num_pseudo_points x d`. E.g., this should be
            `num_fantasies x d` for KG and `num_fantasies*num_pareto x d`
            for HVKG.
        xc_dims: indices of fixed features for the batch initialisation

    Returns:
        A `num_restarts x q x d` tensor of initial conditions.

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0.], [1.]])
        >>> Xinit = gen_batch_initial_conditions(
        >>>     qEI, bounds, q=3, num_restarts=25, raw_samples=500
        >>> )
    """
    if bounds.isinf().any():
        raise NotImplementedError(
            "Currently only finite values in `bounds` are supported "
            "for generating initial conditions for optimization."
        )
    options = options or {}
    sample_around_best = options.get("sample_around_best", False)
    if sample_around_best and equality_constraints:
        raise UnsupportedError(
            "Option 'sample_around_best' is not supported when equality"
            "constraints are present."
        )
    if sample_around_best and generator:
        raise UnsupportedError(
            "Option 'sample_around_best' is not supported when custom "
            "generator is be used."
        )
    seed: Optional[int] = options.get("seed")
    batch_limit: Optional[int] = options.get(
        "init_batch_limit", options.get("batch_limit")
    )
    factor, max_factor = 1, 5
    init_kwargs = {}
    device = bounds.device
    bounds_cpu = bounds.cpu()
    if "eta" in options:
        init_kwargs["eta"] = options.get("eta")
    if options.get("nonnegative") or is_nonnegative(acq_function):
        init_func = initialize_q_batch_nonneg
        if "alpha" in options:
            init_kwargs["alpha"] = options.get("alpha")
    else:
        init_func = initialize_q_batch

    q = 1 if q is None else q
    # the dimension the samples are drawn from
    effective_dim = bounds.shape[-1] * q
    if effective_dim > SobolEngine.MAXDIM and settings.debug.on():
        warnings.warn(
            f"Sample dimension q*d={effective_dim} exceeding Sobol max dimension "
            f"({SobolEngine.MAXDIM}). Using iid samples instead.",
            SamplingWarning,
        )

    while factor < max_factor:
        with warnings.catch_warnings(record=True) as ws:
            n = raw_samples * factor
            if generator is not None:
                X_rnd = generator(n, q, seed)
            # check if no constraints are provided
            elif not (inequality_constraints or equality_constraints):
                if effective_dim <= SobolEngine.MAXDIM:
                    X_rnd = draw_sobol_samples(bounds=bounds_cpu, n=n, q=q, seed=seed)
                else:
                    with manual_seed(seed):
                        # load on cpu
                        X_rnd_nlzd = torch.rand(
                            n, q, bounds_cpu.shape[-1], dtype=bounds.dtype
                        )
                    X_rnd = bounds_cpu[0] + (bounds_cpu[1] - bounds_cpu[0]) * X_rnd_nlzd
            else:
                X_rnd = sample_q_batches_from_polytope(
                    n=n,
                    q=q,
                    bounds=bounds,
                    n_burnin=options.get("n_burnin", 10000),
                    thinning=options.get("thinning", 32),
                    seed=seed,
                    equality_constraints=equality_constraints,
                    inequality_constraints=inequality_constraints,
                )
            # sample points around best
            if sample_around_best:
                X_best_rnd = sample_points_around_best(
                    acq_function=acq_function,
                    n_discrete_points=n * q,
                    sigma=options.get("sample_around_best_sigma", 1e-3),
                    bounds=bounds,
                    subset_sigma=options.get("sample_around_best_subset_sigma", 1e-1),
                    prob_perturb=options.get("sample_around_best_prob_perturb"),
                )
                if X_best_rnd is not None:
                    X_rnd = torch.cat(
                        [
                            X_rnd,
                            X_best_rnd.view(n, q, bounds.shape[-1]).cpu(),
                        ],
                        dim=0,
                    )

            # set fixed features for each batch equal to the first value
            X_rnd[..., xc_dims] = X_rnd[:, 0, xc_dims].unsqueeze(1)

            if fixed_X_fantasies is not None:
                if (d_f := fixed_X_fantasies.shape[-1]) != (d_r := X_rnd.shape[-1]):
                    raise BotorchTensorDimensionError(
                        "`fixed_X_fantasies` and `bounds` must both have the same "
                        f"trailing dimension `d`, but have {d_f} and {d_r}, "
                        "respectively."
                    )
                X_rnd = torch.cat(
                    [
                        X_rnd,
                        fixed_X_fantasies.cpu()
                        .unsqueeze(0)
                        .expand(X_rnd.shape[0], *fixed_X_fantasies.shape),
                    ],
                    dim=-2,
                        )
            with torch.no_grad():
                if batch_limit is None:
                    batch_limit = X_rnd.shape[0]
                Y_rnd_list = []
                start_idx = 0
                while start_idx < X_rnd.shape[0]:
                    end_idx = min(start_idx + batch_limit, X_rnd.shape[0])
                    Y_rnd_curr = acq_function(
                        X_rnd[start_idx:end_idx].to(device=device)
                    ).cpu()
                    Y_rnd_list.append(Y_rnd_curr)
                    start_idx += batch_limit
                Y_rnd = torch.cat(Y_rnd_list)
            batch_initial_conditions = init_func(
                X=X_rnd, Y=Y_rnd, n=num_restarts, **init_kwargs
            ).to(device=device)
            if not any(issubclass(w.category, BadInitialCandidatesWarning) for w in ws):
                return batch_initial_conditions
            if factor < max_factor:
                factor += 1
                if seed is not None:
                    seed += 1  # make sure to sample different X_rnd
    warnings.warn(
        "Unable to find non-zero acquisition function values - initial conditions "
        "are being selected randomly.",
        BadInitialCandidatesWarning,
    )
    return batch_initial_conditions


def gen_candidates_scipy(
        initial_conditions: Tensor,
        acquisition_function: AcquisitionFunction,
        lower_bounds: Optional[Union[float, Tensor]] = None,
        upper_bounds: Optional[Union[float, Tensor]] = None,
        inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
        nonlinear_inequality_constraints: Optional[List[Tuple[Callable, bool]]] = None,
        options: Optional[Dict[str, Any]] = None,
        fixed_features: Optional[Dict[int, Optional[float]]] = None,
        timeout_sec: Optional[float] = None,
        xc_dims: Optional[List[int]] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a set of candidates using `scipy.optimize.minimize`.

    Optimizes an acquisition function starting from a set of initial candidates
    using `scipy.optimize.minimize` via a numpy converter.

    Args:
        initial_conditions: Starting points for optimization, with shape
            (b) x q x d.
        acquisition_function: Acquisition function to be used.
        lower_bounds: Minimum values for each column of initial_conditions.
        upper_bounds: Maximum values for each column of initial_conditions.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.
        nonlinear_inequality_constraints: A list of tuples representing the nonlinear
            inequality constraints. The first element in the tuple is a callable
            representing a constraint of the form `callable(x) >= 0`. In case of an
            intra-point constraint, `callable()`takes in an one-dimensional tensor of
            shape `d` and returns a scalar. In case of an inter-point constraint,
            `callable()` takes a two dimensional tensor of shape `q x d` and again
            returns a scalar. The second element is a boolean, indicating if it is an
            intra-point or inter-point constraint (`True` for intra-point. `False` for
            inter-point). For more information on intra-point vs inter-point
            constraints, see the docstring of the `inequality_constraints` argument to
            `optimize_acqf()`. The constraints will later be passed to the scipy
            solver.
        options: Options used to control the optimization including "method"
            and "maxiter". Select method for `scipy.minimize` using the
            "method" key. By default uses L-BFGS-B for box-constrained problems
            and SLSQP if inequality or equality constraints are present. If
            `with_grad=False`, then we use a two-point finite difference estimate
            of the gradient.
        fixed_features: This is a dictionary of feature indices to values, where
            all generated candidates will have features fixed to these values.
            If the dictionary value is None, then that feature will just be
            fixed to the clamped value and not optimized. Assumes values to be
            compatible with lower_bounds and upper_bounds!
        timeout_sec: Timeout (in seconds) for `scipy.optimize.minimize` routine -
            if provided, optimization will stop after this many seconds and return
            the best solution found so far.
        xc_dims: indices of fixed features for the batch initialisation

    Returns:
        2-element tuple containing

        - The set of generated candidates.
        - The acquisition value for each t-batch.

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0., 0.], [1., 2.]])
        >>> Xinit = gen_batch_initial_conditions(
        >>>     qEI, bounds, q=3, num_restarts=25, raw_samples=500
        >>> )
        >>> batch_candidates, batch_acq_values = gen_candidates_scipy(
                initial_conditions=Xinit,
                acquisition_function=qEI,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
            )
    """
    options = options or {}
    options = {**options, "maxiter": options.get("maxiter", 2000)}

    # if there are fixed features we may optimize over a domain of lower dimension
    reduced_domain = False
    if fixed_features:
        # if there are no constraints, things are straightforward
        if not (
                inequality_constraints
                or equality_constraints
                or nonlinear_inequality_constraints
        ):
            reduced_domain = True
        # if there are we need to make sure features are fixed to specific values
        else:
            reduced_domain = None not in fixed_features.values()

    if nonlinear_inequality_constraints:
        if not isinstance(nonlinear_inequality_constraints, list):
            raise ValueError(
                "`nonlinear_inequality_constraints` must be a list of tuples, "
                f"got {type(nonlinear_inequality_constraints)}."
            )
        nonlinear_inequality_constraints = _convert_nonlinear_inequality_constraints(
            nonlinear_inequality_constraints
        )

    if reduced_domain:
        _no_fixed_features = _remove_fixed_features_from_optimization(
            fixed_features=fixed_features,
            acquisition_function=acquisition_function,
            initial_conditions=initial_conditions,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        )
        # call the routine with no fixed_features
        clamped_candidates, batch_acquisition = gen_candidates_scipy(
            initial_conditions=_no_fixed_features.initial_conditions,
            acquisition_function=_no_fixed_features.acquisition_function,
            lower_bounds=_no_fixed_features.lower_bounds,
            upper_bounds=_no_fixed_features.upper_bounds,
            inequality_constraints=_no_fixed_features.inequality_constraints,
            equality_constraints=_no_fixed_features.equality_constraints,
            nonlinear_inequality_constraints=_no_fixed_features.nonlinear_inequality_constraints,  # noqa: E501
            options=options,
            fixed_features=None,
            timeout_sec=timeout_sec,
        )
        clamped_candidates = _no_fixed_features.acquisition_function._construct_X_full(
            clamped_candidates
        )
        return clamped_candidates, batch_acquisition
    clamped_candidates = columnwise_clamp(
        X=initial_conditions, lower=lower_bounds, upper=upper_bounds
    )

    shapeX = clamped_candidates.shape
    x0 = clamped_candidates.view(-1)
    bounds = make_scipy_bounds(
        X=initial_conditions, lower_bounds=lower_bounds, upper_bounds=upper_bounds
    )
    constraints = make_scipy_linear_constraints(
        shapeX=shapeX,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
    )

    with_grad = options.get("with_grad", True)
    if with_grad:

        def f_np_wrapper(x: np.ndarray, f: Callable):
            """Given a torch callable, compute value + grad given a numpy array."""
            if np.isnan(x).any():
                raise RuntimeError(
                    f"{np.isnan(x).sum()} elements of the {x.size} element array "
                    f"`x` are NaN."
                )
            X = (
                torch.from_numpy(x)
                .to(initial_conditions)
                .view(shapeX)
                .contiguous()
                .requires_grad_(True)
            )
            X_fix = fix_features(X, fixed_features=fixed_features)
            loss = f(X_fix).sum()
            # compute gradient w.r.t. the inputs (does not accumulate in leaves)
            grad = torch.autograd.grad(loss, X)[0].contiguous()
            grad[..., xc_dims] = grad[..., xc_dims].mean(axis=1, keepdim=True)
            gradf = _arrayify(grad.view(-1))
            if np.isnan(gradf).any():
                msg = (
                    f"{np.isnan(gradf).sum()} elements of the {x.size} element "
                    "gradient array `gradf` are NaN. "
                    "This often indicates numerical issues."
                )
                if initial_conditions.dtype != torch.double:
                    msg += " Consider using `dtype=torch.double`."
                raise RuntimeError(msg)
            fval = loss.item()
            return fval, gradf

    else:

        def f_np_wrapper(x: np.ndarray, f: Callable):
            X = torch.from_numpy(x).to(initial_conditions).view(shapeX).contiguous()
            with torch.no_grad():
                X_fix = fix_features(X=X, fixed_features=fixed_features)
                loss = f(X_fix).sum()
            fval = loss.item()
            return fval

    if nonlinear_inequality_constraints:
        # Make sure `batch_limit` is 1 for now.
        if not (len(shapeX) == 3 and shapeX[:2] == torch.Size([1, 1])):
            raise ValueError(
                "`batch_limit` must be 1 when non-linear inequality constraints "
                "are given."
            )
        constraints += make_scipy_nonlinear_inequality_constraints(
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            f_np_wrapper=f_np_wrapper,
            x0=x0,
            shapeX=shapeX,
        )
    x0 = _arrayify(x0)

    def f(x):
        return -acquisition_function(x)

    res = minimize_with_timeout(
        fun=f_np_wrapper,
        args=(f,),
        x0=x0,
        method=options.get("method", "SLSQP" if constraints else "L-BFGS-B"),
        jac=with_grad,
        bounds=bounds,
        constraints=constraints,
        callback=options.get("callback", None),
        options={
            k: v
            for k, v in options.items()
            if k not in ["method", "callback", "with_grad"]
        },
        timeout_sec=timeout_sec,
    )
    _process_scipy_result(res=res, options=options)

    candidates = fix_features(
        X=torch.from_numpy(res.x).to(initial_conditions).reshape(shapeX),
        fixed_features=fixed_features,
    )

    # SLSQP sometimes fails in the line search or may just fail to find a feasible
    # candidate in which case we just return the starting point. This happens rarely,
    # so it shouldn't be an issue given enough restarts.
    if nonlinear_inequality_constraints:
        for con, is_intrapoint in nonlinear_inequality_constraints:
            if not nonlinear_constraint_is_feasible(
                    con, is_intrapoint=is_intrapoint, x=candidates
            ):
                candidates = torch.from_numpy(x0).to(candidates).reshape(shapeX)
                warnings.warn(
                    "SLSQP failed to converge to a solution the satisfies the "
                    "non-linear constraints. Returning the feasible starting point."
                )
                break

    clamped_candidates = columnwise_clamp(
        X=candidates, lower=lower_bounds, upper=upper_bounds, raise_on_violation=True
    )
    with torch.no_grad():
        batch_acquisition = acquisition_function(clamped_candidates)

    return clamped_candidates, batch_acquisition

