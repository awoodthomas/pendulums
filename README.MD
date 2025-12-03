# Simulations

Project for n-pendulum and pendulum-like simulations and visualization. I have a few interests with this:

- Refresh myself on dynamics
- Learn a bit about python GPU libraries (Jax)
- Learn a bit about performance profiling (used xprof with Jax)
- Learn some py5 (Python's version of the popular p5js)
- Use all of the above to make some cool visualizations


## Contents
- `n_pendulum_numpy.py`: NumPy animation with trails/projections. Explores two options for simulating future perturbed paths (very cool for visualizing the chaotic nature of pendulums): one with brute force simulation, the other using some calculus of variations by simulating the state transition matrix.
- `k_n_pendulum_jax.py`: Simulate many pendulums at once on GPU using Jax. 
- `pendulums.py`: Library for shared functions and data structures.

### Less interesting contents
- `generate_trajectory.py` and `display_trajectory.py`: Seperates trajectory generation from animation.
- `spring_cart_pendulum.py`: Just a fun different system.
- `math.ipynb`: Scratch space for derivations using sympy.


## Requirements
- Python 3.11
- NumPy, SciPy, JAX (optional), py5

I used poetry to manage and install dependencies. Getting Jax to work on my Surface Laptop + WSL took some doing - in the end I just needed to update my NVIDIA driver, which I did with NVIDIA App (on Windows).

```bash
# From the repository root
poetry install
poetry run python -m pip install "jax[cuda13]"
```

## Notes
Coming from a last project in Rust, I used typing throughout - maybe a bit overboard. 