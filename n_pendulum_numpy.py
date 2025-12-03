from typing import List, Optional, Tuple
from enum import Enum

import numpy as np
import math
import py5
import time
import os
import scipy.integrate
import pendulums
from pendulums import PendulumMetadata, PendulumAnimation


class TrailProjectionAnimation(PendulumAnimation):
    """Extended animation with trail tracking and future perturbed path projections."""

    class PerturbationSettings():
        class PerturbationMode(Enum):
            NONE = 0
            INTEGRATOR = 1
            VARIATIONAL = 2
            BOTH = 3

        def __init__(self,
                     n: int,
                     mode: PerturbationMode = PerturbationMode.NONE,
                     num_perturbations: int = 7,
                     epsilon: float = math.radians(7.0),
                     time: float = 1.5,
                     dt: float = 0.1):
            self.mode = mode
            perturbation_amts: np.ndarray = np.linspace(
                -epsilon, epsilon, num_perturbations)
            self.perturbations: np.ndarray = np.zeros(
                shape=(n*2, num_perturbations))
            self.perturbations[n:2*n, :] = np.tile(perturbation_amts, (n, 1))
            self.time = time
            self.dt = dt

    def __init__(
        self,
        metadata: PendulumMetadata,
        state0: np.ndarray,
        perturbation_settings: 'TrailProjectionAnimation.PerturbationSettings',
        fps: int = 30,
        m_to_px: int = 200,
        keep_trail: bool = False,
    ):
        super().__init__(metadata, fps, m_to_px)

        # Simulation state
        self.state = state0

        # Trail and projection settings
        self.p_settings = perturbation_settings
        self.keep_trail = keep_trail

        # Animation state
        self.last_bob_coords: List[Tuple[float, float]] = []
        self.last_step_time = time.time()
        self.f_paths: Optional[np.ndarray] = None
        self.f_paths_var: Optional[np.ndarray] = None

    def generate_perturbed_trajectories(self) -> np.ndarray:
        """Generate perturbed trajectories using direct integration. 
        Not efficient for many perturbations."""
        eval_times = np.arange(0, self.p_settings.time, self.p_settings.dt)
        num_perturbations = self.p_settings.perturbations.shape[1]
        perturbed_paths = np.zeros(
            shape=(num_perturbations, 2*self.metadata.n_pendulums, len(eval_times)))
        for i in range(num_perturbations):
            y0 = self.state + self.p_settings.perturbations[:, i]
            perturb_sol = scipy.integrate.solve_ivp(
                pendulums.n_pendulum_ode_np, (0, self.p_settings.time), y0, t_eval=eval_times, args=(self.metadata,))
            perturbed_paths[i, :, :] = perturb_sol.y

        return perturbed_paths

    def generate_perturbed_trajectories_var(self) -> np.ndarray:
        """Generate perturbed trajectories using variational equations. 
        This has a higher initial cost, because you need to integrate the 2n x 2n
         state transition matrix, but is more efficient for many perturbations.
         Of note: phi, the state transition matrix, needs to be instantiated as
         the identity matrix at the particular time that you want to start perturbing 
         from. See https://home.cs.colorado.edu/~lizb/chaos/variational-notes.pdf"""
        num_perturbations = self.p_settings.perturbations.shape[1]
        n = self.metadata.n_pendulums
        phi = np.eye(n*n, n*n)
        y0 = np.concatenate([self.state, phi.ravel()])
        perturb_sol = scipy.integrate.solve_ivp(
            pendulums.n_pendulum_ode_var_np,
            (0, self.p_settings.time),
            y0,
            t_eval=np.arange(0, self.p_settings.time, self.p_settings.dt),
            args=(self.metadata,))
        perturbed_paths = np.zeros(
            shape=(num_perturbations, 2*n, perturb_sol.y.shape[1]))
        for i, s in enumerate(perturb_sol.y.T):
            phi = s[2*n:].reshape(n*n, n*n)
            deltas = np.dot(phi, self.p_settings.perturbations)
            # Add the base state (2*n,1) broadcast to all perturbations
            base = s[:2*n, None]     # shape (2*n, 1)
            perturbed_paths[:, :, i] = (base + deltas).T

        return perturbed_paths

    def predraw_update(self) -> None:
        """Update phase space projections before drawing. This runs before
        each draw call, and makes uses of some idle time in between draws."""
        match self.p_settings.mode:
            case self.PerturbationSettings.PerturbationMode.BOTH:
                self.f_paths = self.generate_perturbed_trajectories()
                self.f_paths_var = self.generate_perturbed_trajectories_var()
            case self.PerturbationSettings.PerturbationMode.INTEGRATOR:
                self.f_paths = self.generate_perturbed_trajectories()
            case self.PerturbationSettings.PerturbationMode.VARIATIONAL:
                self.f_paths_var = self.generate_perturbed_trajectories_var()
            case _:
                pass

    def draw(self) -> None:
        """Main draw loop."""

        # Integrate physics
        self.integrate_step()

        # Extract state
        n = self.metadata.n_pendulums
        theta = self.state[0:n]

        # Draw background
        self.draw_background()

        # Draw pendulum and get last bob position
        last_bob_pos = self.draw_pendulum(theta)

        # Draw trail
        self.draw_trail(last_bob_pos)

        # Draw projections
        self.draw_projections()

        # Draw energy display
        # self.draw_energy_display(theta, omega)

    # TODO: consider using a sympletic integrator (Verlet?) for better energy conservation
    # with lower cost
    def integrate_step(self, dt: float = 0.01) -> None:
        """Integrate the pendulum state forward in time."""
        t = time.time()
        n_steps = int((t - self.last_step_time) / dt)

        if n_steps > 0:
            for _ in range(n_steps):
                self.state = pendulums.rk4_step_np(
                    pendulums.n_pendulum_ode_np,
                    self.state,
                    self.last_step_time,
                    dt,
                    self.metadata
                )
            self.last_step_time += n_steps * dt

    def draw_trail(self, last_bob_pos: Tuple[float, float]) -> None:
        """Draw the trajectory trail of the last bob."""
        self.last_bob_coords.append(last_bob_pos)

        # Determine which points to draw. If we keeping the whole trail, we only need to
        # draw the latest point, since the others already persist in the trail graphics.
        pts = self.last_bob_coords[-2:] if self.keep_trail else self.last_bob_coords[-400:]
        n_pts = len(pts)

        if n_pts < 2:
            return

        full_stroke = 4
        alpha = 200

        # For a permanent trail, saving and imaging a PGraphics is more efficient
        self.trail_graphics.begin_draw()
        self.trail_graphics.no_fill()

        if not self.keep_trail:
            self.trail_graphics.clear()

        # Draw trail in three layers for glow effect
        for stroke_weight, alpha_factor in [(full_stroke * 3, alpha / 4),
                                            (full_stroke * 1.5, alpha / 3),
                                            (full_stroke, alpha)]:
            self.trail_graphics.stroke_weight(stroke_weight)

            for i in range(n_pts - 1):
                if not self.keep_trail:
                    t_frac = i / max(1, n_pts - 2)
                    self.trail_graphics.stroke(
                        3, 244, 252, alpha_factor * t_frac)
                else:
                    self.trail_graphics.stroke(3, 244, 252, alpha_factor)

                x, y = pts[i]
                x1, y1 = pts[i + 1]
                self.trail_graphics.line(x, y, x1, y1)

        self.trail_graphics.end_draw()
        py5.image(self.trail_graphics, 0, 0)

    def draw_projections(self) -> None:
        """Draw the perturbed future path projections.
        Integrated vs. variational paths are shown in different colors"""
        if self.f_paths is not None:
            perturbations = len(self.f_paths)
            py5.color_mode(py5.CMAP, py5.mpl_cmaps.PLASMA, perturbations, 1)

            for i in range(perturbations):
                s_fp = self.f_paths[i, :, :]

                with py5.begin_shape():  # type: ignore
                    py5.stroke_weight(3)
                    py5.stroke(i, 0.5)
                    py5.no_fill()

                    for s in s_fp.T:
                        x, y = self.origin_x, self.origin_y
                        for j in range(self.metadata.n_pendulums):
                            x = x + \
                                self.metadata.lengths[j] * \
                                self.m_to_px * np.sin(s[j])
                            y = y + \
                                self.metadata.lengths[j] * \
                                self.m_to_px * np.cos(s[j])
                        py5.vertex(x, y)

        if self.f_paths_var is not None:
            perturbations = len(self.f_paths_var)
            py5.color_mode(py5.CMAP, py5.mpl_cmaps.OCEAN, perturbations, 1)

            for i in range(perturbations):
                s_fp = self.f_paths_var[i, :, :]

                with py5.begin_shape():  # type: ignore
                    py5.stroke_weight(3)
                    py5.stroke(i, 0.5)
                    py5.no_fill()

                    for s in s_fp.T:
                        x, y = self.origin_x, self.origin_y
                        for j in range(self.metadata.n_pendulums):
                            x = x + \
                                self.metadata.lengths[j] * \
                                self.m_to_px * np.sin(s[j])
                            y = y + \
                                self.metadata.lengths[j] * \
                                self.m_to_px * np.cos(s[j])
                        py5.vertex(x, y)

    def draw_energy_display(self, theta: np.ndarray, omega: np.ndarray) -> None:
        """Draw the energy display overlay."""
        energy = pendulums.n_pendulum_energy(theta, omega, self.metadata)
        py5.fill(255, 255, 255)
        py5.text_size(16)
        py5.text(f"Total Energy: {energy:.2f} J", 10, 20)


# Module-level instance for py5 callbacks
_animation: TrailProjectionAnimation


def settings() -> None:
    _animation.settings()


def setup() -> None:
    """py5 setup callback - initializes the animation instance."""
    _animation.setup()


def predraw_update() -> None:
    """py5 predraw callback - updates projections before draw."""
    _animation.predraw_update()


def draw() -> None:
    """py5 draw callback - main animation loop."""
    _animation.draw()


if __name__ == "__main__":
    # Read settings from environment
    mode_setting = os.environ.get(
        "PROJECTION", "NONE").upper()
    projection_mode = TrailProjectionAnimation.PerturbationSettings.PerturbationMode[
        mode_setting]

    keep_trail = os.environ.get(
        "KEEP_TRAIL", "0") not in ("0", "False", "false")

    # Create initial state
    theta = np.array([math.pi * 0.75, math.pi / 2])
    n = len(theta)
    omega = np.zeros(n)
    omega[0] = math.pi
    state0 = np.concatenate((theta, omega))

    # Create pendulum metadata
    masses = np.ones(n)
    lengths = np.ones(n)
    lengths[1] = 1.5
    metadata = PendulumMetadata(masses=masses, lengths=lengths)

    # Create animation
    _animation = TrailProjectionAnimation(
        metadata=metadata,
        state0=state0,
        fps=30,
        m_to_px=200,
        perturbation_settings=TrailProjectionAnimation.PerturbationSettings(
            n=n,
            mode=projection_mode,
            num_perturbations=3,
            epsilon=math.radians(9.0),
        ),
        keep_trail=keep_trail,
    )
    py5.run_sketch()
