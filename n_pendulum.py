from typing import List, Optional, Tuple
from enum import Enum
from datetime import datetime
import line_profiler


import numpy as np
import math
import py5
import time
import os
import scipy.integrate
import pendulums
from pendulums import PendulumMetadata, Py5PendulumAnimation


class TrailProjectionAnimation(Py5PendulumAnimation):
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
        window_size_px: int = 1000,
        keep_trail: bool = False,
        export_frames: bool = False,
    ):
        super().__init__(metadata, fps, window_size_px)

        # Simulation state
        self.state = state0

        # Trail and projection settings
        self.p_settings = perturbation_settings
        self.keep_trail = keep_trail
        self.export_frames = export_frames
        self.frame_count = 0
        self.frames_folder: Optional[str] = None

        # Animation state
        self.last_bob_coords: List[Tuple[float, float]] = []
        self.f_paths: Optional[np.ndarray] = None
        self.f_paths_var: Optional[np.ndarray] = None

    @line_profiler.profile
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

    @line_profiler.profile
    def generate_perturbed_trajectories_var(self) -> np.ndarray:
        """Generate perturbed trajectories using variational equations.
        This has a higher initial cost, because you need to integrate the 2n x 2n
         state transition matrix, but is more efficient for many perturbations.
         Of note: phi, the state transition matrix, needs to be instantiated as
         the identity matrix at the particular time that you want to start perturbing
         from. See https://home.cs.colorado.edu/~lizb/chaos/variational-notes.pdf"""
        num_perturbations = self.p_settings.perturbations.shape[1]
        n = self.metadata.n_pendulums
        phi = np.eye(2*n, 2*n)
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
            phi = s[2*n:].reshape(2*n, 2*n)
            deltas = np.dot(phi, self.p_settings.perturbations)
            # Add the base state (2*n,1) broadcast to all perturbations
            base = s[:2*n, None]     # shape (2*n, 1)
            perturbed_paths[:, :, i] = (base + deltas).T

        return perturbed_paths

    @line_profiler.profile
    def integration_step(self) -> None:
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

        # Integrate physics
        self.integrate_step()

    @line_profiler.profile
    def draw(self) -> None:
        """Main draw loop."""

        # Extract state
        n = self.metadata.n_pendulums
        theta = self.state[0:n]
        omega = self.state[n:2*n]

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

        # Save frame if export enabled
        if self.export_frames and self.frames_folder:
            py5.save_frame(os.path.join(self.frames_folder,
                           f'frame{self.frame_count}.png'))
            self.frame_count += 1
            if self.frame_count % self.fps == 0:
                print(
                    f"Exported {self.frame_count} frames ({self.frame_count / self.fps} s)...")

    # Consider using sympletic Verlet integrator, did not conserve energy well
    def integrate_step(self) -> None:
        """Integrate the pendulum state forward in time."""
        t = self.steps * self.STEP_SIZE
        n_steps = max(min(round((1 / self.fps) / self.STEP_SIZE), 5), 1)

        if n_steps > 0:
            for _ in range(n_steps):
                self.state = pendulums.rk4_step_np(
                    pendulums.n_pendulum_ode_np,
                    self.state,
                    t,
                    self.STEP_SIZE,
                    self.metadata
                )
        self.steps += 1

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
        alpha = 255

        # For a permanent trail, saving and imaging a PGraphics is more efficient
        self.trail_graphics.begin_draw()
        self.trail_graphics.no_fill()

        if not self.keep_trail:
            self.trail_graphics.clear()

        # Draw trail in three layers for glow effect
        # for stroke_weight, alpha_factor in [(full_stroke * 3, alpha / 4),
        #                                     (full_stroke * 1.5, alpha / 3),
        #                                     (full_stroke, alpha)]:
        for stroke_weight, alpha_factor in [(full_stroke, alpha)]:
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
                        py5.curve_vertex(x, y)

        if self.f_paths_var is not None:
            perturbations = len(self.f_paths_var)
            if self.p_settings.mode == self.PerturbationSettings.PerturbationMode.BOTH:
                py5.color_mode(py5.CMAP, py5.mpl_cmaps.OCEAN, perturbations, 1)
            else:
                py5.color_mode(py5.CMAP, py5.mpl_cmaps.PLASMA,
                               perturbations, 1)

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
                        py5.curve_vertex(x, y)

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
    py5.frame_rate(round(_animation.fps*speed))

    # Create frames directory if export is enabled
    if _animation.export_frames:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        _animation.frames_folder = f'py5frames-{timestamp}'
        os.makedirs(_animation.frames_folder, exist_ok=True)
        print(f"Exporting frames to: {_animation.frames_folder}")


def predraw_update() -> None:
    """py5 predraw callback - updates projections before draw."""
    _animation.integration_step()


def draw() -> None:
    """py5 draw callback - main animation loop."""
    _animation.draw()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='n-pendulum simulation with trail tracking and future perturbed path projections',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Example: python n_pendulum.py --theta 135 90 --omega 180 0 --projection VARIATIONAL --num-perturbations 5'
    )

    # Simulation parameters
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for visualization')
    parser.add_argument('--window-size', type=int, default=1200,
                        help='Window size in pixels (square window)')

    # Initial conditions (in degrees)
    parser.add_argument('--theta', type=float, nargs='+',
                        default=[126.0, 72.0],
                        help='Initial angles in degrees (one per pendulum segment)')
    parser.add_argument('--omega', type=float, nargs='+',
                        default=[],
                        help='Initial angular velocities in degrees/s (one per pendulum segment, defaults to zeros)')

    # Physical parameters
    parser.add_argument('--mass', type=float, nargs='+',
                        default=[],
                        help='Masses for each pendulum segment (defaults to ones)')
    parser.add_argument('--radius', type=float, nargs='+',
                        default=[],
                        help='Lengths/radii for each pendulum segment (defaults to ones)')

    # Perturbation settings
    parser.add_argument('--projection', type=str, default='NONE',
                        choices=['NONE', 'INTEGRATOR',
                                 'VARIATIONAL', 'BOTH'],
                        help='Perturbation projection mode')
    parser.add_argument('--num-perturbations', type=int, default=5,
                        help='Number of perturbed trajectories to project')
    parser.add_argument('--epsilon', type=float, default=9.0,
                        help='Perturbation magnitude in degrees')
    parser.add_argument('--projection-time', type=float, default=1.5,
                        help='Time horizon for projections in seconds')
    parser.add_argument('--projection-dt', type=float, default=0.1,
                        help='Time step for projection integration')

    # Visualization options
    parser.add_argument('--keep-trail', action='store_true',
                        help='Keep trail of pendulum motion')
    parser.add_argument('--speed', type=float, default=1.0,)
    parser.add_argument('--export-frames', action='store_true',
                        help='Export each frame as PNG to a py5frames-date-time folder. To generate a video:ffmpeg -framerate 60 -i frame%d.png -c:v libx264 -crf 15 -preset slow -pix_fmt yuv420p output.mp4 ')

    args = parser.parse_args()

    speed = args.speed

    # Convert theta from degrees to radians
    theta = np.array(args.theta) * math.pi / 180.0
    n = theta.shape[0]

    # Handle omega (default to zeros if not specified)
    if len(args.omega) == 0:
        omega = np.zeros(n)
    else:
        omega = np.array(args.omega) * math.pi / 180.0
        if len(omega) != n:
            parser.error(
                f"Number of omega values ({len(omega)}) must match number of theta values ({n})")

    # Handle masses (default to ones if not specified)
    if len(args.mass) == 0:
        masses = np.ones(n)
    else:
        masses = np.array(args.mass)
        if len(masses) != n:
            parser.error(
                f"Number of mass values ({len(masses)}) must match number of theta values ({n})")

    # Handle radii/lengths (default to ones if not specified)
    if len(args.radius) == 0:
        lengths = np.ones(n)
    else:
        lengths = np.array(args.radius)
        if len(lengths) != n:
            parser.error(
                f"Number of radius values ({len(lengths)}) must match number of theta values ({n})")

    # Create initial state
    state0 = np.concatenate((theta, omega))

    # Create pendulum metadata
    metadata = PendulumMetadata(masses=masses, lengths=lengths)

    # Get projection mode
    projection_mode = TrailProjectionAnimation.PerturbationSettings.PerturbationMode[
        args.projection]

    # Create animation
    _animation = TrailProjectionAnimation(
        metadata=metadata,
        state0=state0,
        fps=args.fps,
        window_size_px=args.window_size,
        perturbation_settings=TrailProjectionAnimation.PerturbationSettings(
            n=n,
            mode=projection_mode,
            num_perturbations=args.num_perturbations,
            epsilon=math.radians(args.epsilon),
            time=args.projection_time,
            dt=args.projection_dt,
        ),
        keep_trail=args.keep_trail,
        export_frames=args.export_frames,
    )
    py5.run_sketch()
