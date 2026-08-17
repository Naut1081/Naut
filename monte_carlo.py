#!/usr/bin/env python3
"""
Monte Carlo Simulator - monte_carlo.py

Beginner's guide (printed at startup unless --skip-intro):
--------------------------------------------------------------------------------
What is Monte Carlo simulation and how it works (short, approachable)

Core idea
- Monte Carlo simulation uses random sampling to estimate numerical quantities
  that are difficult or impossible to compute analytically (expectations,
  probabilities, integrals).
- Instead of solving an equation exactly, we draw many random samples from the
  model and compute the statistic of interest on each sample, then average.

Key mathematical principles
- Law of Large Numbers (LLN): sample averages converge to the true expectation
  as the number of samples grows.
- Central Limit Theorem (CLT): for large sample counts, the Monte Carlo
  estimator is approximately normally distributed around the true value; its
  standard error shrinks like 1/sqrt(N).

Basic procedure (practical steps)
1. Define the model: specify the distribution or stochastic process
   (e.g., standard normal, uniform, geometric Brownian motion).
2. Define the quantity of interest: a function of the sampled random variables
   (f(X)), a payoff, or a probability indicator.
3. Draw N independent samples (X1, X2, ..., XN) from the model.
4. Compute values Yi = f(Xi) and the Monte Carlo estimate
   E[f(X)] ≈ (1/N) Σ Yi.
5. Estimate standard error: se ≈ s / sqrt(N) where s is the sample standard
   deviation of the Yi.
6. (Optional) apply variance-reduction methods (antithetic variates, control
   variates, importance sampling, stratified sampling) to reduce variance for
   the same N.

Interpretation and diagnostics
- A 95% confidence interval is approximately estimate ± 1.96 * se.
- If se is large, increase sample size or use variance reduction.
- For path-dependent processes (e.g., options on GBM), ensure time
  discretization is fine enough for the payoff.

Examples included in this script: estimating π, simulating GBM, European call
pricing with antithetic variates and chunking, and an optimized high-resolution
GBM plot sized to 1980×1770 pixels.

--------------------------------------------------------------------------------
"""

import argparse
import math
import numpy as np
import sys

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
except Exception:
    plt = None
    LineCollection = None

# The same beginner guide is available programmatically so we can print or skip it.
BEGINNER_GUIDE = """
Beginner's guide — Monte Carlo simulation (short)
------------------------------------------------
Core idea:
- Monte Carlo uses random sampling to estimate expectations, probabilities, and
  integrals that are hard or impossible to compute analytically.

LLN & CLT:
- Law of Large Numbers (LLN): sample averages converge to the true expectation.
- Central Limit Theorem (CLT): estimator ≈ Normal(true_value, var/N) for large N.

Practical steps:
1) Specify the stochastic model (distribution or process).
2) Define f(X) — the quantity you measure on each sample.
3) Draw N samples Xi, compute Yi = f(Xi).
4) Estimate = mean(Yi); se ≈ sd(Yi) / sqrt(N).
5) Use variance-reduction if needed: antithetic, control variates, importance sampling.

Examples in this repo: pi estimation, GBM simulation, European call pricing.
"""

# -------------------------
# Generic Monte Carlo utils
# -------------------------

def estimate_expectation(sampler, func, n_samples, rng=None):
    """
    Estimate E[func(X)] where X ~ sampler(rng).
    sampler: callable(rng, size) -> samples (vectorized if possible)
    func: callable(samples) -> values
    n_samples: int
    rng: numpy.random.Generator
    Returns: (estimate, std_error)
    """
    rng = rng or np.random.default_rng()
    samples = sampler(rng, n_samples)
    values = func(samples)
    mean = np.mean(values)
    std_error = np.std(values, ddof=1) / math.sqrt(n_samples)
    return mean, std_error


# -------------------------
# Pi estimation demo
# -------------------------

def estimate_pi(n_samples, rng=None):
    """
    Estimate pi by sampling uniform points in the unit square and checking inside quarter-circle.
    """
    rng = rng or np.random.default_rng()
    x = rng.random(n_samples)
    y = rng.random(n_samples)
    inside = (x * x + y * y) <= 1.0
    fraction = inside.mean()
    pi_est = 4.0 * fraction
    se = 4.0 * math.sqrt(fraction * (1 - fraction) / n_samples)
    return pi_est, se


# -------------------------
# Geometric Brownian Motion
# -------------------------

def simulate_gbm(S0, mu, sigma, T, steps, n_paths, rng=None, dtype=np.float32):
    """
    Simulate n_paths of GBM on [0, T] with `steps` time steps (including endpoint).
    Memory-optimized: uses dtype (float32 by default) to reduce memory footprint.
    Returns: time grid (length steps+1), array of shape (n_paths, steps+1) with dtype
    """
    rng = rng or np.random.default_rng()
    dt = T / steps
    times = np.linspace(0, T, steps + 1, dtype=dtype)
    normals = rng.standard_normal(size=(n_paths, steps)).astype(dtype)
    increments = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt).astype(dtype) * normals
    log_paths = np.empty((n_paths, steps + 1), dtype=dtype)
    log_paths[:, 0] = 0.0
    log_paths[:, 1:] = np.cumsum(increments, axis=1, dtype=dtype)
    S = (S0 * np.exp(log_paths)).astype(dtype)
    return times, S


# -------------------------
# Option pricing examples
# -------------------------

def price_european_call_mc(S0, K, r, sigma, T, n_paths, steps=1, antithetic=False, rng=None, dtype=np.float32):
    """
    Price a European call option by Monte Carlo on underlying following GBM.
    - steps: number of time steps (1 = sample terminal only)
    - antithetic: use antithetic variates for variance reduction
    - dtype: numeric type, float32 reduces memory
    Returns: (price_estimate, std_error)
    """
    rng = rng or np.random.default_rng()
    max_chunk = int(5_000_000)
    if n_paths > max_chunk:
        sum_payoff = 0.0
        sum_sq = 0.0
        processed = 0
        while processed < n_paths:
            chunk = min(max_chunk, n_paths - processed)
            price_chunk, se_chunk, total_chunk = _price_european_call_mc_chunk(
                S0, K, r, sigma, T, chunk, steps=steps, antithetic=antithetic, rng=rng, dtype=dtype
            )
            sum_payoff += price_chunk * total_chunk
            sum_sq += (se_chunk * math.sqrt(total_chunk))**2
            processed += total_chunk
        mean = sum_payoff / n_paths
        std_error = math.sqrt(sum_sq) / n_paths
        return mean, std_error
    else:
        price, se, _ = _price_european_call_mc_chunk(S0, K, r, sigma, T, n_paths, steps=steps, antithetic=antithetic, rng=rng, dtype=dtype)
        return price, se


def _price_european_call_mc_chunk(S0, K, r, sigma, T, n_paths, steps=1, antithetic=False, rng=None, dtype=np.float32):
    rng = rng or np.random.default_rng()
    if antithetic:
        half = n_paths // 2
        normals = rng.standard_normal(size=(half, steps)).astype(dtype)
        normals = np.vstack([normals, -normals])
        if normals.shape[0] < n_paths:
            extra = rng.standard_normal(size=(1, steps)).astype(dtype)
            normals = np.vstack([normals, extra])
    else:
        normals = rng.standard_normal(size=(n_paths, steps)).astype(dtype)

    dt = np.array(T / steps, dtype=dtype)
    increments = (r - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * normals
    log_ST = increments.sum(axis=1, dtype=dtype)
    ST = S0 * np.exp(log_ST)
    payoffs = np.maximum(ST - K, 0.0).astype(dtype)
    discounted = np.exp(-r * T).astype(dtype) * payoffs
    price = float(discounted.mean())
    se = float(discounted.std(ddof=1) / math.sqrt(n_paths))
    return price, se, n_paths


# -------------------------
# Plotting utilities
# -------------------------

def plot_gbm_paths(times, paths, n_plot=None, figsize_pixels=(1980, 1770), show_mean=True, dpi=100):
    """
    Plot GBM paths using LineCollection for speed.
    - figsize_pixels: (width_px, height_px) of the figure to fill (maps to figsize via dpi)
    - n_plot: number of sample paths to draw (if None draw up to 200 or min(n_paths,200))
    - dpi: dots per inch to control sizing when saving or showing
    """
    if plt is None or LineCollection is None:
        raise RuntimeError("matplotlib is required for plotting")

    width_px, height_px = figsize_pixels
    figsize = (width_px / dpi, height_px / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    n_paths = paths.shape[0]
    if n_plot is None:
        n_plot = min(200, n_paths)
    n_plot = min(n_plot, n_paths)

    segments = []
    for i in range(n_plot):
        segments.append(np.column_stack([times, paths[i]]))
    lc = LineCollection(segments, colors='gray', linewidths=0.6, alpha=0.5)
    ax.add_collection(lc)
    ax.autoscale()
    if show_mean:
        mean_path = paths.mean(axis=0)
        ax.plot(times, mean_path, color="red", lw=2, label="mean")

    ax.set_xlabel("Time")
    ax.set_ylabel("S")
    ax.set_title("GBM sample paths")
    if show_mean:
        ax.legend()

    plt.tight_layout()
    plt.show()


# -------------------------
# CLI and examples
# -------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Monte Carlo simulator examples")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Global skip-intro flag (print the beginner guide unless this is set)
    parser.add_argument("--skip-intro", action="store_true", help="Skip printing the beginner's guide at startup")

    # pi
    p_pi = sub.add_parser("pi", help="Estimate pi by Monte Carlo")
    p_pi.add_argument("--n", type=int, default=1000000, help="number of samples")

    # expectation
    p_exp = sub.add_parser("expectation", help="Estimate expectation E[f(X)]")
    p_exp.add_argument("--n", type=int, default=200000, help="samples")
    p_exp.add_argument("--dist", choices=["normal", "uniform"], default="normal")
    p_exp.add_argument("--function", choices=["x", "x2", "exp"], default="x2",
                       help="function of X to take expectation of")

    # gbm
    p_gbm = sub.add_parser("gbm", help="Simulate Geometric Brownian Motion")
    p_gbm.add_argument("--S0", type=float, default=100.0)
    p_gbm.add_argument("--mu", type=float, default=0.05)
    p_gbm.add_argument("--sigma", type=float, default=0.2)
    p_gbm.add_argument("--T", type=float, default=1.0)
    p_gbm.add_argument("--steps", type=int, default=252)
    p_gbm.add_argument("--n_paths", type=int, default=500)
    p_gbm.add_argument("--plot", action="store_true")
    p_gbm.add_argument("--plot_paths", type=int, default=None, help="how many sample paths to draw")
    p_gbm.add_argument("--dpi", type=int, default=100, help="dpi for plotting (pixel mapping)")

    # option
    p_opt = sub.add_parser("option", help="Price a European call option")
    p_opt.add_argument("--S0", type=float, default=100.0)
    p_opt.add_argument("--K", type=float, default=100.0)
    p_opt.add_argument("--r", type=float, default=0.05)
    p_opt.add_argument("--sigma", type=float, default=0.2)
    p_opt.add_argument("--T", type=float, default=1.0)
    p_opt.add_argument("--n", type=int, default=200000)
    p_opt.add_argument("--steps", type=int, default=1)
    p_opt.add_argument("--antithetic", action="store_true")
    p_opt.add_argument("--dtype", choices=["float32", "float64"], default="float32")

    args = parser.parse_args(argv)

    # Print the beginner's guide unless the user asked to skip
    if not args.skip_intro:
        print(BEGINNER_GUIDE)

    rng = np.random.default_rng()

    if args.mode == "pi":
        est, se = estimate_pi(args.n, rng=rng)
        print(f"pi estimate = {est:.6f} (SE = {se:.6f}, n = {args.n})")
        return

    if args.mode == "expectation":
        n = args.n
        if args.dist == "normal":
            sampler = lambda rng, size: rng.standard_normal(size)
        else:
            sampler = lambda rng, size: rng.random(size)

        if args.function == "x":
            func = lambda x: x
        elif args.function == "x2":
            func = lambda x: x * x
        else:
            func = lambda x: np.exp(x)

        est, se = estimate_expectation(sampler, func, n, rng=rng)
        print(f"estimate = {est:.6f} (SE = {se:.6f}, n = {n})")
        return

    if args.mode == "gbm":
        dtype = np.float32
        times, paths = simulate_gbm(args.S0, args.mu, args.sigma, args.T, args.steps, args.n_paths, rng=rng, dtype=dtype)
        mean_path = paths.mean(axis=0)
        print(f"Simulated {args.n_paths} GBM paths. Final S mean = {mean_path[-1]:.4f}")
        if args.plot:
            if plt is None:
                print("matplotlib not available; cannot plot.")
                return
            dpi = args.dpi or 100
            figsize_pixels = (1980, 1770)
            plot_gbm_paths(times, paths, n_plot=args.plot_paths, figsize_pixels=figsize_pixels, dpi=dpi)
        return

    if args.mode == "option":
        dtype = np.float32 if args.dtype == "float32" else np.float64
        price, se = price_european_call_mc(
            args.S0, args.K, args.r, args.sigma, args.T, args.n, steps=args.steps,
            antithetic=args.antithetic, rng=rng, dtype=dtype
        )
        print(f"European call MC price = {price:.6f} (SE = {se:.6f}, n = {args.n}, antithetic={args.antithetic}, dtype={args.dtype})")
        return


if __name__ == "__main__":
    main()
