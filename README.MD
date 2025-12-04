# Simulations

Project for n-pendulum and pendulum-like simulations and visualization. I have a few interests with this:

- Refresh myself on dynamics
- Learn a bit about python GPU libraries (`jax`)
- Learn a bit about performance profiling (used `xprof` with `jax`)
- Learn some `py5` (Python's version of the popular p5js, very popular for generative art)
- Use all of the above to make some cool visualizations


## Contents
- `n_pendulum_numpy.py`: `numpy` based integration. Explores two options for simulating future perturbed paths (very cool for visualizing the chaotic nature of pendulums): one with brute force simulation, the other using some calculus of variations by simulating the state transition matrix. Uses `py5` for display.
- `k_n_pendulum_jax_vispy.py`: Simulate many pendulums at once on GPU using `jax` and display with `vispy` (utilizing OpenGL under the hood). Can hit 30 fps for 10,000 4-link pendulums, or 100,000 double pendulums! Graphics cards are cool.
- `k_n_pendulum_jax.py`: Original attempt at many pendulum simulation using `py5`. Can hit 30 fps for only ~200 double pendulums, `py5` drawing simply takes too long. 
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

# Learnings
I was optimistic about improving performance with a 2nd order velocity verlet integrator instead of rk4 - first time trying a sympletic integrator. But it did conserve energy as well as rk4, and reducing the timestep to improve that would eliminate benefits.