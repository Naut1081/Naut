# Monte Carlo Simulator

Beginner's guide — What is Monte Carlo simulation and how it works
-----------------------------------------------------------------

Core idea
- Monte Carlo simulation uses random sampling to estimate numerical quantities
  that are difficult or impossible to compute analytically (expectations,
  probabilities, integrals).
- Instead of solving an equation exactly, draw many random samples from the
  model and compute the statistic of interest on each sample, then average.

Key mathematical principles
- Law of Large Numbers (LLN): sample averages converge to the true expectation.
- Central Limit Theorem (CLT): for large sample counts, the Monte Carlo estimator
  is approximately normally distributed; the standard error shrinks like 1/sqrt(N).

Basic procedure (practical steps)
1. Define the model (distribution or process).
2. Define the quantity of interest f(X).
3. Draw N samples Xi and compute Yi = f(Xi).
4. Estimate = mean(Yi); se ≈ sd(Yi) / sqrt(N).
5. Optionally apply variance reduction (antithetic, control variates, importance sampling).

Repository contents
- monte_carlo.py — optimized, self-contained script implementing:
  - estimate_expectation(sampler, func)
  - estimate_pi
  - simulate_gbm (float32 default, memory-optimized)
  - price_european_call_mc (antithetic + chunking)
  - plotting optimized to produce a 1980×1770 px figure (LineCollection)
- requirements.txt — numpy, matplotlib
- .gitignore — Python common ignores
- LICENSE — MIT (if you want a different license, tell me)

Quick usage examples
- Estimate pi:
  python monte_carlo.py pi --n 1000000
- Price an at-the-money European call:
  python monte_carlo.py option --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1 --n 200000 --antithetic
- Simulate GBM and plot (fills 1980×1770 pixels by default):
  python monte_carlo.py gbm --S0 100 --mu 0.05 --sigma 0.2 --T 1 --steps 252 --n_paths 2000 --plot

Notes
- The script prints a brief beginner's guide at startup unless you pass --skip-intro.
- Plotting uses matplotlib; to save the high-resolution plot directly, modify the script
  or save from the plot window (plt.savefig(..., dpi=100)).
