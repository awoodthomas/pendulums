from typing import Any, Callable, List, Sequence, Tuple, cast
from dataclasses import dataclass

from cycler import V
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import py5


NDArrayFloat = npt.NDArray[np.floating]

g: float = 9.81

# Old total step time: 32-33 ms
# Draw loop: 14 ms
# Dead time: 17.5 ms
# Integration time: 3 ms
# Background draw: 2 ms
# Numpy conversion: 5.4 ms -> 1-2 ms by slicing numpy array instead of jit array
# Draw pendulums: 3.5 ms


@dataclass
class PendulumMetadata:
    """Physical parameters for an n-pendulum system."""
    masses: np.ndarray  # Mass of each bob
    lengths: np.ndarray  # Length of each rod

    @property
    def n_pendulums(self) -> int:
        """Number of pendulums in the system."""
        return len(self.masses)


class PendulumAnimation:
    """Base class for pendulum animation, managing display and graphics."""

    def __init__(
        self,
        metadata: PendulumMetadata,
        fps: int = 30,
        m_to_px: int = 200,
    ):
        self.metadata = metadata
        self.fps = fps
        self.m_to_px = m_to_px

        # Calculate canvas size from pendulum lengths
        self.extent_px = round(np.sum(metadata.lengths) * m_to_px * 2) + 75
        self.origin_x = self.extent_px / 2.0
        self.origin_y = self.extent_px / 2.0

        # Graphics objects (initialized in setup)
        self.background_graphics: Any = None
        self.trail_graphics: Any = None

    def settings(self) -> None:
        """Initialize py5 canvas and graphics objects."""
        py5.size(self.extent_px, self.extent_px)

    def setup(self) -> None:
        py5.frame_rate(self.fps)
        py5.rect_mode(py5.CENTER)

        # Create graphics buffers
        self.trail_graphics = py5.create_graphics(
            self.extent_px, self.extent_px)
        self.background_graphics = py5.create_graphics(
            self.extent_px, self.extent_px)

        # Pre-render background with gradient and pivot
        self._render_background()

    def _render_background(self) -> None:
        """Pre-render the background gradient and pivot point."""
        self.background_graphics.begin_draw()
        self.background_graphics.background('#0a0e27')

        # Gradient
        for i in range(py5.height):
            inter = i / py5.height
            r_val = int(26 + (45 - 26) * inter)
            g_val = int(31 + (53 - 31) * inter)
            b_val = int(58 + (97 - 58) * inter)
            self.background_graphics.stroke(r_val, g_val, b_val)
            self.background_graphics.line(0, i, py5.width, i)

        # Pivot point with glow
        self.background_graphics.no_stroke()
        self.background_graphics.fill(100, 150, 255, 80)
        self.background_graphics.ellipse(self.origin_x, self.origin_y, 40, 40)
        self.background_graphics.fill(100, 150, 255, 120)
        self.background_graphics.ellipse(self.origin_x, self.origin_y, 25, 25)
        self.background_graphics.fill(255, 200, 0)
        self.background_graphics.ellipse(self.origin_x, self.origin_y, 8, 8)
        self.background_graphics.end_draw()

    def draw_background(self) -> None:
        """Draw the pre-rendered background."""
        py5.image(self.background_graphics, 0, 0)

    def draw_pendulum(self, theta: np.ndarray) -> Tuple[float, float]:
        """Draw the pendulum rods and bobs, return last bob position."""
        x0, y0 = self.origin_x, self.origin_y
        py5.color_mode(py5.RGB)

        for i in range(self.metadata.n_pendulums):
            x = x0 + self.metadata.lengths[i] * self.m_to_px * np.sin(theta[i])
            y = y0 + self.metadata.lengths[i] * self.m_to_px * np.cos(theta[i])

            # Rod shadow
            py5.stroke_weight(6)
            py5.stroke(0, 0, 0, 60)
            py5.line(x0 + 2, y0 + 2, x + 2, y + 2)

            # Rod
            py5.stroke_weight(6)
            py5.stroke("#222222")  # type: ignore
            py5.line(x0, y0, x, y)

            # Bob with glow effect
            py5.no_stroke()
            py5.fill(255, 100, 100, 100)
            py5.ellipse(x, y, 20, 20)
            py5.fill(255, 150, 100)
            py5.ellipse(x, y, 12, 12)
            py5.fill(255, 200, 100)
            py5.ellipse(x, y, 6, 6)

            x0, y0 = x, y

        return x0, y0


@jax.jit
def double_pendulum_ode_jax(
    t: float,
    y: jax.Array,
    m: jax.Array,
    r: jax.Array,
) -> jax.Array:
    """Explicitly coded double pendulum equations of motion. No obvious benefit.
    %timeit -n 10000 pendulums.n_pendulum_ode_jax(0.0, state_jax, r_jax, m_jax, n)
    218 μs ± 16.1 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    %timeit -n 10000 pendulums.double_pendulum_ode_jax(0.0, state_jax, r_jax, m_jax)
    212 μs ± 28.7 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    """
    n = 2
    theta = y[0:n]
    omega = y[n:]
    dtheta = omega

    mass_matrix = jnp.array([[(m[0] + m[1])*r[0]**2, jnp.cos(theta[0] - theta[1])
                              * m[1]*r[0]*r[1]], [jnp.cos(theta[0] - theta[1])*m[1]*r[0]*r[1], m[1]*r[1]**2]])

    b = jnp.array([-g*(m[0] + m[1])*jnp.sin(theta[0])*r[0] - jnp.sin(theta[0] - theta[1])*m[1]*omega[1] **
                   2*r[0]*r[1], -g*jnp.sin(theta[1])*m[1]*r[1] + jnp.sin(theta[0] - theta[1])*m[1]*omega[0]**2*r[0]*r[1]])

    domega = jnp.linalg.solve(mass_matrix, b)

    dy = jnp.concatenate((dtheta, domega))
    return dy


@jax.jit(static_argnames=['n'])
def n_pendulum_ode_jax(
    t: float,
    y: jax.Array,
    m: jax.Array,
    r: jax.Array,
    n: int,
) -> jax.Array:
    """N-pendulum equations of motion. JAX seems to handle the list comprehensions at max function well
    enough to JIT, %timeit shows very similar performance to the explicit double pendulum equations."""

    theta = y[0:n]
    omega = y[n:]
    dtheta = omega

    mass_matrix = jnp.array([
        [jnp.sum(m[max(i, j):]) * r[i] * r[j] * jnp.cos(theta[i] - theta[j])
         for j in range(n)]
        for i in range(n)
    ])

    b = -jnp.array([
        jnp.sum(jnp.array([jnp.sum(m[max(i, j):]) * r[i] * r[j] * omega[j] ** 2 * jnp.sin(theta[i] - theta[j])
                           for j in range(n)])) + g * jnp.sum(m[i:]) * r[i] * jnp.sin(theta[i])
        for i in range(n)
    ])

    domega = jnp.linalg.solve(mass_matrix, b)

    dy = jnp.concatenate((dtheta, domega))
    return dy


@jax.jit
def double_pendulum_rk4_step_jax(
    state: jax.Array,
    t: float,
    dt: float,
    m: jax.Array,
    r: jax.Array,
) -> jax.Array:
    k1 = double_pendulum_ode_jax(t, state, m, r)
    k2 = double_pendulum_ode_jax(t + 0.5*dt, state + 0.5*dt*k1, m, r)
    k3 = double_pendulum_ode_jax(t + 0.5*dt, state + 0.5*dt*k2, m, r)
    k4 = double_pendulum_ode_jax(t + dt, state + dt*k3, m, r)
    return cast(jax.Array, state + (dt/6)*(k1+2*k2+2*k3+k4))


@jax.jit(static_argnames=['n'])
def n_pendulum_rk4_step_jax(
    state: jax.Array,
    t: float,
    dt: float,
    m: jax.Array,
    r: jax.Array,
    n: int,
) -> jax.Array:
    k1 = n_pendulum_ode_jax(t, state, m, r, n)
    k2 = n_pendulum_ode_jax(t + 0.5*dt, state + 0.5*dt*k1, m, r, n)
    k3 = n_pendulum_ode_jax(t + 0.5*dt, state + 0.5*dt*k2, m, r, n)
    k4 = n_pendulum_ode_jax(t + dt, state + dt*k3, m, r, n)
    return cast(jax.Array, state + (dt/6)*(k1+2*k2+2*k3+k4))


# Numpy
def n_pendulum_ode_np(
    t: float,
    y: NDArrayFloat,
    metadata: PendulumMetadata
) -> NDArrayFloat:
    """N-pendulum equations of motion."""
    n = len(y)//2
    theta = y[0:n]
    omega = y[n:]
    dtheta = omega
    m = metadata.masses
    r = metadata.lengths

    mass_matrix = np.array([[np.sum(m[max(i, j):]) * r[i] * r[j] * np.cos(theta[i]-theta[j])
                           for j in range(n)] for i in range(n)])

    b = -np.array([np.sum([np.sum(m[max(i, j):]) * r[i] * r[j] * omega[j]**2 * np.sin(theta[i]-theta[j])
                          for j in range(n)]) + g * np.sum(m[i:]) * r[i] * np.sin(theta[i]) for i in range(n)])
    domega = np.linalg.solve(mass_matrix, b)

    dy = np.concatenate((dtheta, domega))
    return dy


def rk4_step_np(
    f: Callable[..., NDArrayFloat],
    state_in: NDArrayFloat,
    t: float,
    dt: float,
    *args: Any,
) -> NDArrayFloat:
    k1 = f(t, state_in, *args)
    k2 = f(t + 0.5*dt, state_in + 0.5*dt*k1, *args)
    k3 = f(t + 0.5*dt, state_in + 0.5*dt*k2, *args)
    k4 = f(t + dt, state_in + dt*k3, *args)
    return cast(NDArrayFloat, state_in + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4))


def velocity_verlet_step_np(
    f: Callable[..., NDArrayFloat],
    state_in: NDArrayFloat,
    t: float,
    dt: float,
    *args: Any,
) -> NDArrayFloat:
    n = len(state_in) // 2
    r = state_in[0:n]
    v = state_in[n:]
    at = f(t, state_in, *args)[n:]
    r_next = r + v * dt + 0.5 * at * dt**2
    a_next = f(t + dt, np.concatenate((r_next, v)), *args)[n:]
    v_next = v + 0.5 * (at + a_next) * dt
    return np.concatenate((r_next, v_next))


def jacobian_fd(
    f: Callable[..., NDArrayFloat],
    t: float,
    y: NDArrayFloat,
    *args: Any,
) -> NDArrayFloat:
    eps = 1e-6
    n = len(y)
    A = np.zeros((n, n))
    fx = f(t, y, *args)
    for i in range(n):
        y2 = y.copy()
        y2[i] += eps
        A[:, i] = (f(t, y2, *args) - fx) / eps
    return A


def n_pendulum_ode_var_np(
    t: float,
    y: NDArrayFloat,
    metadata: PendulumMetadata,
) -> NDArrayFloat:
    """N-pendulum equations of motion."""
    # L = (2 n) + (2 n)^2
    n = round(np.sqrt(1+4*len(y))/4)
    phi = y[n*2:].reshape((2*n, 2*n))  # transition matrix
    physical_state = y[0:n*2]

    dy = n_pendulum_ode_np(t, physical_state, metadata)
    jacobian = jacobian_fd(n_pendulum_ode_np, t, physical_state, metadata)
    dPhi = np.dot(jacobian, phi)

    return np.concatenate([dy, dPhi.ravel()])


# Supporting functions
def n_pendulum_energy(
    theta: NDArrayFloat,
    omega: NDArrayFloat,
    metadata: PendulumMetadata,
) -> float:
    """Compute total energy for a state."""
    n = len(theta)
    k = 0.0
    u = 0.0
    m = metadata.masses
    r = metadata.lengths
    for i in range(n):
        v_i_sq = 0.0
        y_i = 0.0
        for j in range(i+1):
            for k in range(i+1):
                v_i_sq += r[j] * r[k] * omega[j] * \
                    omega[k] * np.cos(theta[j]-theta[k])
            y_i += r[j] * np.cos(theta[j])
        k += 0.5 * m[i] * v_i_sq
        u -= m[i] * g * y_i
    return k + u


def pendulum_coords_for_ang(
    theta: Sequence[float] | NDArrayFloat,
    metadata: PendulumMetadata,
) -> List[NDArrayFloat]:
    coords: List[NDArrayFloat] = []
    x0, y0 = 0, 0
    r = metadata.lengths
    for i, th in enumerate(theta):
        x = x0 + r[i] * np.sin(th)
        y = y0 + r[i] * np.cos(th)
        x0, y0 = x, y
        coords.append(np.array([x, y]))
    return coords
