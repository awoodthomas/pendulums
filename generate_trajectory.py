"""
Generate n-pendulum trajectory and save to file.
"""
from typing import Any, Dict, List, Tuple

import numpy as np
import math
import time
from scipy.integrate import solve_ivp
import pickle
from pathlib import Path
import pendulums
from pendulums import PendulumMetadata

# Parameters
n: int = 3
theta0: List[float] = [math.pi * 0.75, math.pi + 0.1, 0]
omega0: List[float] = [0.0 for _ in range(n)]
m: np.ndarray = np.ones(n)
r: np.ndarray = np.ones(n)
t_final: float = 2.0
max_step: float = 0.02
num_perturbations: int = 5


def generate_main_trajectory(
    theta_init: np.ndarray,
    omega_init: np.ndarray,
    metadata: PendulumMetadata,
):
    """Generate main pendulum trajectory."""
    print("Generating main trajectory...")
    n = len(theta_init)
    y0 = np.concatenate((theta_init, omega_init))

    sol = solve_ivp(
        pendulums.n_pendulum_ode_np,
        (0, t_final),
        y0,
        max_step=max_step,
        args=(metadata,),
    )

    # Compute energy
    energy = np.zeros(sol.y.shape[1])
    for x in range(sol.y.shape[1]):
        theta = sol.y[0:n, x]
        omega = sol.y[n:, x]
        energy[x] = pendulums.n_pendulum_energy(theta, omega, metadata)

    print(f"  Generated {sol.y.shape[1]} time steps")
    print(f"  Energy range: [{energy.min():.4f}, {energy.max():.4f}]")
    print(
        f"  Energy conservation (ptp/mean): {np.ptp(energy)/np.mean(energy):.2e}")

    return sol, energy


def generate_perturbed_trajectories(sol: Any, metadata: PendulumMetadata):
    """Generate perturbed trajectories for visualization of phase space."""
    print("Generating perturbed trajectories...")
    n = sol.y.shape[0]
    step_count = sol.y.shape[1]
    future_paths = []

    start_time = time.time()
    progress = 0

    for x in range(step_count):
        theta = sol.y[0:n, x]
        omega = sol.y[n:, x]

        perturbed_paths = []
        for i in range(num_perturbations):
            perturbation = (i - num_perturbations // 2) * math.radians(3.0)
            omega_perturbed = omega.copy()
            omega_perturbed += perturbation
            y0 = np.concatenate((theta, omega_perturbed))
            perturb_sol = solve_ivp(
                pendulums.n_pendulum_ode_np,
                (0, 1.5),
                y0,
                args=(metadata,),
            )
            perturbed_paths.append(perturb_sol.y)

        future_paths.append(perturbed_paths)

        current_time = time.time()
        elapsed_time = current_time - start_time
        estimated_time_remaining = (
            elapsed_time / (x + 1)) * (step_count - x - 1)
        pct = (x + 1) / step_count * 100
        if pct >= progress + 10:
            progress = pct
            print(
                f"  Perturbed step {x} of {step_count}. Estimated time remaining: {estimated_time_remaining:.0f}s")

    print(f"  Generated {len(future_paths)} sets of perturbed trajectories")
    return future_paths


def save_trajectory(
    sol: Any,
    energy: np.ndarray,
    future_paths: List[np.ndarray] | List[List[np.ndarray]],
    output_file: str,
) -> None:
    """Save trajectory data to file using pickle."""
    print(f"Saving trajectory to {output_file}...")

    trajectory_data = {
        'sol_t': sol.t,
        'sol_y': sol.y,
        'energy': energy,
        'future_paths': future_paths,
        'parameters': {
            'n_pendulums': n,
            'theta_init': theta0,
            'omega_init': omega0,
            'masses': m,
            'lengths': r,
            't_final': t_final,
            'num_perturbations': num_perturbations,
        }
    }

    # Save as pickle (binary, preserves numpy arrays exactly)
    with open(output_file, 'wb') as f:
        pickle.dump(trajectory_data, f)

    file_size_mb = Path(output_file).stat().st_size / (1024**2)
    print(f"  Saved {file_size_mb:.2f} MB")


def main() -> None:
    output_file = 'trajectory.pkl'

    print(f"N-Pendulum Trajectory Generator")
    print(f"================================")
    print(f"Parameters:")
    print(f"  n_pendulums: {n}")
    print(f"  Initial theta: {theta0}")
    print(f"  Initial omega: {omega0}")
    print(f"  Masses: {m}")
    print(f"  Lengths: {r}")
    print(f"  Integration time: 0 to {t_final}s")
    print()

    # Generate trajectories
    metadata = PendulumMetadata(masses=m, lengths=r)
    sol, energy = generate_main_trajectory(
        np.array(theta0), np.array(omega0), metadata)
    future_paths = generate_perturbed_trajectories(sol, metadata)

    # Save to file
    save_trajectory(sol, energy, future_paths, output_file)

    print(f"\n✓ Trajectory generation complete!")
    print(f"  Output file: {output_file}")


if __name__ == "__main__":
    main()
