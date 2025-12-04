import jax
import jax.numpy as jnp
import numpy as np
import math
import time
import os
from vispy import app, scene
from vispy.color import get_colormap
from typing import Optional

import pendulums
from pendulums import PendulumMetadata, PendulumAnimation


class KN_Pendulum_JAX(PendulumAnimation):
    """n-pendulum simulation using JAX for acceleration and batching."""
    P_MAGNITUDE = 1e-5
    STEP_SIZE = 0.02

    def __init__(self,
                 pendulum_metadata: PendulumMetadata,
                 state0: np.ndarray,
                 k: int = 10,
                 fps: int = 30,
                 window_size_px: int = 1400,
                 draw_trails: bool = False) -> None:
        super().__init__(pendulum_metadata, fps, window_size_px)
        self.k = k  # Number of perturbed trajectories
        # TODO: can we pass this in as some sort of dict for metadata?
        self.r_jax = jnp.array(pendulum_metadata.lengths)
        self.m_jax = jnp.array(pendulum_metadata.masses)

        self.states = jnp.tile(state0, (self.k, 1))
        key = jax.random.PRNGKey(0)
        perturbations = jax.random.uniform(key, shape=(self.k, state0.shape[0]),
                                           minval=-self.P_MAGNITUDE,
                                           maxval=self.P_MAGNITUDE)
        self.states = self.states.at[:, :n].add(perturbations[:, :n])
        self.steps = 0
        self.last_step_time: Optional[float] = None

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

    def setup(self) -> None:
        """Set up vispy graphics objects for drawing pendulums."""
        thetas_np = np.asarray(self.states)[:, 0:self.metadata.n_pendulums]
        per_pendulum = self.metadata.n_pendulums*2
        pos = np.zeros(shape=(per_pendulum*self.k, 2))
        colors = np.repeat(colormap[np.linspace(
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
            pos=pos, color=colors, width=1, connect='segments', parent=view.scene, method='gl')
        self.pendulum_lines = line

        print("Finished vispy setup, starting to draw.")

    # TODO: can we JIT this function?
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
    def predraw_update(self) -> None:
        with jax.profiler.StepTraceAnnotation("integration_step"):
            t = time.time()
            self.steps += 1
            if self.last_step_time is None:
                self.last_step_time = t

            # Number of fixed-size steps to take to catch up to realtime
            # I tested making a JITd function to take multiple steps at once on GPU.
            # Did not see any benefit, and has a high compile cost since number of steps
            # is a static variable (compiled per value). So keeping it simple
            n_steps = int((t - self.last_step_time) / self.STEP_SIZE)
            if n_steps > 0:
                for _ in range(n_steps):
                    with jax.profiler.StepTraceAnnotation("substep"):
                        self.states = self.rk4_step_batch(
                            self.states, t, self.STEP_SIZE, self.m_jax, self.r_jax, self.metadata.n_pendulums).block_until_ready()
                self.last_step_time += n_steps * self.STEP_SIZE

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


def on_timer(event):
    _animation.predraw_update()
    _animation.update_pendulums()
    _animation.steps = _animation.steps + 1
    if ENABLE_PROFILING and _animation.steps == 20:
        jax.profiler.stop_trace()
        print("Stopped profiling trace.")


ENABLE_PROFILING = os.environ.get(
    "PROFILE", "0") not in ("0", "False", "false")

_animation: KN_Pendulum_JAX
canvas = scene.SceneCanvas(keys='interactive', size=(1000, 1000), show=True)
colormap = get_colormap('plasma')
timer = app.Timer(1.0/30, connect=on_timer, start=True)

if __name__ == "__main__":
    # Pendulum initial conditions
    theta = np.array([math.pi*0.7, math.pi*0.4])
    n = theta.shape[0]
    omega = np.zeros(n)
    omega[0:5] = -math.pi/2
    state0 = np.concatenate((theta, omega))
    m = np.ones(n)  # masses
    r = np.ones(n)
    r[0] = 1.5
    _animation = KN_Pendulum_JAX(
        pendulum_metadata=PendulumMetadata(
            masses=m,
            lengths=r,
        ),
        state0=state0,
        k=100000,
        fps=30,
        window_size_px=2000,
        draw_trails=True
    )
    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'
    view.camera.set_range(x=(0, _animation.window_size_px),  # pyright: ignore[reportAttributeAccessIssue]
                          y=(0, _animation.window_size_px))
    _animation.setup()
    canvas.show()
    app.run()
