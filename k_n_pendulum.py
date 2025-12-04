import jax
import jax.numpy as jnp
import numpy as np
import math
import time
import os
import vispy
from vispy import app, scene
from vispy.color import get_colormap
from vispy.color.colormap import Colormap
from vispy.gloo.util import _screenshot

import pendulums
from pendulums import PendulumMetadata, PendulumAnimation


class KN_Pendulum_JAX_vispy(PendulumAnimation):
    """n-pendulum simulation using JAX for acceleration and batching."""
    P_MAGNITUDE = 1e-5
    STEP_SIZE = 1 / 60.0  # 60 Hz simulation step size - 0.02 is good enough

    def __init__(self,
                 pendulum_metadata: PendulumMetadata,
                 state0: np.ndarray,
                 k: int = 10,
                 fps: int = 30,
                 window_size_px: int = 1400,
                 colormap: Colormap = get_colormap('plasma')) -> None:
        super().__init__(pendulum_metadata, fps, window_size_px)
        self.k = k  # Number of perturbed trajectories
        # TODO: can we pass this in as some sort of dict for metadata?
        self.r_jax = jnp.array(pendulum_metadata.lengths)
        self.m_jax = jnp.array(pendulum_metadata.masses)

        self.states = jnp.tile(state0, (self.k, 1))
        key = jax.random.PRNGKey(0)
        n = self.metadata.n_pendulums
        perturbations = jax.random.uniform(key, shape=(self.k, state0.shape[0]),
                                           minval=-self.P_MAGNITUDE,
                                           maxval=self.P_MAGNITUDE)
        self.states = self.states.at[:, :n].add(perturbations[:, :n])
        self.colormap = colormap
        self.steps = 0

    def setup(self) -> None:
        """Set up vispy graphics objects for drawing pendulums."""
        print("JIT-compiling vectorized RK4 stepper...")
        self.rk4_step_batch = jax.jit(jax.vmap(pendulums.n_pendulum_rk4_step_jax,
                                               in_axes=(0, None, None, None, None, None)),
                                      static_argnums=(5,))
        print("Warming up JIT...")
        t = time.time()
        if ENABLE_PROFILING:
            jax.profiler.start_trace("/tmp/profile-data")
        self.rk4_step_batch(self.states, 0.0, self.STEP_SIZE,
                            self.m_jax, self.r_jax, self.metadata.n_pendulums).block_until_ready()
        print(f"JIT warmup complete after {time.time()-t:.1f}s.")

        self.canvas = scene.SceneCanvas(
            keys='interactive', size=(self.window_size_px, self.window_size_px))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.view.camera.set_range(x=(0, self.window_size_px),  # pyright: ignore[reportAttributeAccessIssue]
                                   y=(0, self.window_size_px))

        thetas_np = np.asarray(self.states)[:, 0:self.metadata.n_pendulums]
        per_pendulum = self.metadata.n_pendulums*2
        pos = np.zeros(shape=(per_pendulum*self.k, 2))
        colors = np.repeat(self.colormap[np.linspace(
            0, 1, self.k)], per_pendulum, axis=0)
        for i in range(self.k):
            x0, y0 = self.origin_x, self.origin_y
            pos[0, :] = [x0, y0]
            for j in range(self.metadata.n_pendulums):
                x = x0 + self.metadata.lengths[j] * self.m_to_px * \
                    math.sin(thetas_np[i, j])
                y = y0 - self.metadata.lengths[j] * self.m_to_px * \
                    math.cos(thetas_np[i, j])
                pos[i*per_pendulum + j*2, :] = [x0, y0]
                pos[i*per_pendulum + j*2+1, :] = [x, y]
                x0, y0 = x, y
        # Using individual line segments for each pendulum is slow, you can't get
        # more than about 200 pendulums in real time. A single line defined by
        # segments is much faster.
        # method = 'agg' would let us do lines with width>1, but isn't working for me.
        line = scene.visuals.Line(  # pyright: ignore[reportAttributeAccessIssue]
            pos=pos, color=colors, width=1, connect='segments', parent=self.view.scene, method='gl')
        self.pendulum_lines = line

        print("Finished vispy setup, starting to draw.")

    # TODO: can we JIT this function? Though it's not the bottleneck right now.
    # 10,000 4-link pendulums: ~4.4 ms
    def _compute_positions_vectorized(self, states):
        """Vectorized computation of pendulum joint positions per system.

        Returns an array of shape (k, n_pendulums, 2) with absolute
        screen-space coordinates for each joint (bob) for each of the k systems.
        """
        thetas = states[:, :self.metadata.n_pendulums]
        # thetas shape: (k, n_pendulums)
        sin_thetas = jnp.sin(thetas)
        cos_thetas = jnp.cos(thetas)

        # Cumulative positions from origin for each link
        x = jnp.cumsum(self.r_jax * sin_thetas * self.m_to_px, axis=1)
        # Screen-space Y increases upward in panzoom; subtract to go "down" from origin
        y = -jnp.cumsum(self.r_jax * cos_thetas * self.m_to_px, axis=1)

        # Add origin offset
        x = x + self.origin_x
        y = y + self.origin_y

        # Stack into last-dim=2 coordinates
        return jnp.stack((x, y), axis=2)

    # About 15 ms per frame for 10,000 4-link pendulums
    def integration_step(self) -> None:
        with jax.profiler.StepTraceAnnotation("integration_step"):
            self.steps += 1

            # Number of fixed-size steps to take to catch up to realtime
            # I tested making a JITd function to take multiple steps at once on GPU.
            # Did not see any benefit, and has a high compile cost since number of steps
            # is a static variable (compiled per value). So keeping it simple
            n_steps = min(round((1 / self.fps) / self.STEP_SIZE), 5)
            if n_steps > 0:
                for _ in range(n_steps):
                    with jax.profiler.StepTraceAnnotation("substep"):
                        self.states = self.rk4_step_batch(
                            self.states, self.steps*self.fps, self.STEP_SIZE, self.m_jax, self.r_jax, self.metadata.n_pendulums).block_until_ready()

            if ENABLE_PROFILING and self.steps == 30:
                jax.profiler.stop_trace()
                print("Stopped profiling trace.")

    # About 7 ms per frame for 10,000 4-link pendulums
    # The actual drawing is then another 7-8 ms
    def update_pendulums(self) -> None:
        with jax.profiler.TraceAnnotation("updating pendulums"):
            # Joint positions per system: (k, n, 2)
            # TODO: move this array stuff into JAX? Can we pass a jax array to set_data?
            # np asarray: 1.75 ms
            joints = np.asarray(
                self._compute_positions_vectorized(self.states))
            k = self.k
            n = self.metadata.n_pendulums

            # The rest: <1.75 ms

            # Build segment endpoints by pairing origin->joint0, joint0->joint1, ...
            # previous joints: origin for first, then shift joints by one (excluding last)
            origin = np.array(
                [self.origin_x, self.origin_y], dtype=np.float32)
            prev = np.empty_like(joints)
            prev[:, 0, :] = origin
            if n > 1:
                prev[:, 1:, :] = joints[:, :-1, :]

            # Interleave prev and current to form segments for all k systems
            # Result shape: (k*n*2, 2)
            segments = np.empty((k * n * 2, 2), dtype=np.float32)
            segments[0::2] = prev.reshape(k * n, 2)
            segments[1::2] = joints.reshape(k * n, 2)

            self.pendulum_lines.set_data(pos=segments)

    def frame_step(self, _event) -> None:
        self.integration_step()
        self.update_pendulums()
        self.steps = self.steps + 1
        if ENABLE_PROFILING and self.steps == 20:
            jax.profiler.stop_trace()
            print("Stopped profiling trace.")

    def frame_step_and_capture(self, t) -> np.ndarray:
        self.frame_step(None)
        # Read the pixels from the vispy canvas
        # self.canvas.update()
        app.process_events()  # Process any pending events to ensure rendering is complete
        img = _screenshot((0, 0, self.window_size_px, self.window_size_px))[
            :, :, :3]
        return img

    def start_animation(self) -> None:
        self.setup()

        self.canvas.show()
        self.timer = app.Timer(
            1.0/args.fps, connect=self.frame_step, start=True)

        app.run()


ENABLE_PROFILING = os.environ.get(
    "PROFILE", "0") not in ("0", "False", "false")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='k-n pendulum simulation using JAX and VisPy',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Example: python k_n_pendulum.py --k 20000 --theta 120 60 30 --radius 1 1.5 1'
    )

    # Simulation parameters
    parser.add_argument('--k', type=int, default=100,
                        help='Number of perturbed trajectories to simulate')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for visualization')
    parser.add_argument('--window-size', type=int, default=1400,
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

    parser.add_argument('--color_map', type=str, default='plasma',
                        help='Colormap to use for pendulums. Favorite are plasma, hot, managua')

    args = parser.parse_args()

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
        m = np.ones(n)
    else:
        m = np.array(args.mass)
        if len(m) != n:
            parser.error(
                f"Number of mass values ({len(m)}) must match number of theta values ({n})")

    # Handle radii/lengths (default to ones if not specified)
    if len(args.radius) == 0:
        r = np.ones(n)
    else:
        r = np.array(args.radius)
        if len(r) != n:
            parser.error(
                f"Number of radius values ({len(r)}) must match number of theta values ({n})")

    # Create initial state
    state0 = np.concatenate((theta, omega))

    colormap = get_colormap(args.color_map)

    # Create animation
    animation = KN_Pendulum_JAX_vispy(
        pendulum_metadata=PendulumMetadata(
            masses=m,
            lengths=r,
        ),
        state0=state0,
        k=args.k,
        fps=args.fps,
        window_size_px=args.window_size,
        colormap=colormap
    )
    animation.start_animation()
