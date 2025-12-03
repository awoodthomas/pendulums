from typing import Optional

import py5
import jax
import jax.numpy as jnp
import numpy as np
import math
import time
import os
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
                 m_to_px: int = 150) -> None:
        super().__init__(pendulum_metadata, fps, m_to_px)
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
                                               in_axes=(0, None, None, None, None)))
        print("Warming up JIT...")
        if ENABLE_PROFILING:
            jax.profiler.start_trace("/tmp/profile-data")
        self.rk4_step_batch(self.states, 0.0, self.STEP_SIZE,
                            self.m_jax, self.r_jax)

    def predraw_update(self) -> None:
        with jax.profiler.StepTraceAnnotation("integration_step"):
            t = time.time()
            self.steps += 1
            if self.last_step_time is None:
                self.last_step_time = t

            # Number of fixed-size steps to take to catch up to realtime
            # TODO: consider making JITd function to take multiple steps at once
            n_steps = int((t - self.last_step_time) / self.STEP_SIZE)
            if n_steps > 0:
                for _ in range(n_steps):
                    self.states = self.rk4_step_batch(
                        self.states, t, self.STEP_SIZE, m, r).block_until_ready()
                self.last_step_time += n_steps * self.STEP_SIZE

            if ENABLE_PROFILING and self.steps == 30:
                jax.profiler.stop_trace()
                print("Stopped profiling trace.")

    def draw(self) -> None:
        with jax.profiler.StepTraceAnnotation("draw_step", step_num=self.steps):
            with jax.profiler.TraceAnnotation("background draw"):
                # Gradient background
                py5.image(self.background_graphics, 0, 0)  # type: ignore

            # Convert the JAX state angles to numpy floats for drawing (do once, not per pendulum)
            with jax.profiler.TraceAnnotation("numpy conversion"):
                thetas_np = np.asarray(self.states)[
                    :, 0:self.metadata.n_pendulums]

            # Draw rods and bobs with shadows and gradients
            with jax.profiler.TraceAnnotation("drawing pendulums"):
                py5.stroke_weight(6)
                py5.color_mode(py5.CMAP, py5.mpl_cmaps.PLASMA, self.k, 1)
                for i in range(self.k):
                    x0, y0 = self.origin_x, self.origin_y
                    py5.stroke(i, 0.5)
                    for j in range(self.metadata.n_pendulums):
                        x = x0 + self.metadata.lengths[j] * self.m_to_px * \
                            math.sin(thetas_np[i, j])
                        y = y0 + self.metadata.lengths[j] * self.m_to_px * \
                            math.cos(thetas_np[i, j])
                        py5.line(x0, y0, x, y)
                        x0, y0 = x, y

            # TODO: consider drawing trails?


def settings() -> None:
    _animation.settings()


def setup() -> None:
    _animation.setup()


def predraw_update() -> None:
    _animation.predraw_update()


def draw() -> None:
    _animation.draw()


ENABLE_PROFILING = os.environ.get(
    "PROFILE", "0") not in ("0", "False", "false")

_animation: KN_Pendulum_JAX


if __name__ == "__main__":
    # Pendulum initial conditions
    theta = np.array([math.pi*0.8, math.pi*0.5])
    n = theta.shape[0]
    omega = np.array([0.0 for _ in range(n)])
    state0 = np.concatenate((theta, omega))
    m = np.ones(n)  # masses
    r = np.ones(n)
    _animation = KN_Pendulum_JAX(
        pendulum_metadata=PendulumMetadata(
            masses=m,
            lengths=r,
        ),
        state0=state0,
        k=20,
        fps=60,
        m_to_px=200,
    )

    py5.run_sketch()
