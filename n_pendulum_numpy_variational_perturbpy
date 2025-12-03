"""
This was a bit of a failed experiment. It is very similar to n_pendulum_numpy.py,
and also uses variational equations to estimate perturbed trajectories. Rather than
integrating the seconds of future path + state transition matrix at every time point,
it instead integrates only the main trajectory + STM. It integrates ahead by a couple
of seconds - creating an effective fixed path. 

The challenge is that the state transition matrix maps the perturbations only from t=0,
and we want to map a perturbation at each time t. We can estimate intermediate STMs
by inverting the STM at time t and multiplying by the STM at time t+dt. This is tricky
because it requires a matrix inversion at every time step for the perturbed trajectory,
and it can fail if the STM is ill conditioned. The results also seem to be less accurate.
See "estimation of the state transition matrix": https://en.wikipedia.org/wiki/State-transition_matrix
"""
from turtle import pen
from typing import Any, List, Tuple

import numpy as np
import math
import py5
import time
import os
import pendulums

FPS: int = 30
g: float = 9.81
m_to_px: int = 150

# Initial conditions
theta0: np.ndarray = np.array([math.pi*0.5, math.pi*0.2])
n = len(theta0)
omega0: np.ndarray = np.zeros(n)
omega0[0] = math.pi
m: np.ndarray = np.ones(n)
r: np.ndarray = np.ones(n)
metadata = pendulums.PendulumMetadata(masses=m, lengths=r)
# r[0] = 1

# Transition matrix
phi0: np.ndarray = np.eye(n*2)
state: np.ndarray = np.concatenate([theta0, omega0, phi0.ravel()])

# Display
extent_px: int = round(np.sum(r)*m_to_px*2) + 10
num_perturbations: int = 7
epsilon: float = math.radians(9.0)
perturbation_amts: np.ndarray = np.linspace(
    -epsilon, epsilon, num_perturbations)
perturbations: np.ndarray = np.zeros(shape=(n*2, num_perturbations))
perturbations[n:2*n, :] = np.tile(perturbation_amts, (n, 1))
x_center: float = extent_px / 2
y_center: float = extent_px / 2
origin: np.ndarray = np.array([x_center, y_center])
ahead_steps: int = 1 * FPS

# Settings
PROJECTION: bool = os.environ.get(
    "PROJECTION", "0") not in ("0", "False", "false")
KEEP_TRAIL: bool = os.environ.get(
    "KEEP_TRAIL", "0") not in ("0", "False", "false")
# TODO : when keeping trail, draw to a persistent graphics object and then use py5.image()

trail_graphics: Any = None
background_graphics: Any = None


def generate_perturbed_var(step: int) -> List[List[np.ndarray]]:
    phi_now_inv = np.linalg.inv(state[n*2:].reshape(2*n, 2*n))

    perturbed_paths: List[List[np.ndarray]] = [[]
                                               for _ in range(num_perturbations)]
    ahead_path_len: int = max(min(ahead_steps, len(main_path) - step), 0)
    for i in range(ahead_path_len):
        phi_future = main_path[step+i][n*2:].reshape(2*n, 2*n)
        theta_future = main_path[step+i][0:n]
        for j in range(num_perturbations):
            delta0 = perturbations[:, j]
            delta = np.dot(phi_future, np.dot(phi_now_inv, delta0))
            theta_perturbed = theta_future + delta[0:n]
            # if (i == 0):
            #     print(np.linalg.norm(theta_future - theta_perturbed))
            coord = pendulums.pendulum_coords_for_ang(
                theta_perturbed, metadata)[-1] * m_to_px + origin
            perturbed_paths[j].append(coord)
    print(len(perturbed_paths[0]))
    return perturbed_paths


# def generate_perturbed_trajectories_var(state):
#     phi = state[n*2:].reshape(2*n, 2*n)

#     coords = []
#     for i in range(num_perturbations):
#         delta0 = perturbations[:, i]
#         delta = np.dot(phi, delta0)
#         theta_perturbed = state[0:n] + delta[0:n]
#         coords.append(pendulums.rel_pendulum_coords_for_ang(
#             theta_perturbed, r)[-1] * m_to_px + origin)

#     return coords


def setup() -> None:
    global trail_graphics, background_graphics
    py5.size(extent_px, extent_px, py5.P2D)
    py5.frame_rate(FPS)
    py5.rect_mode(py5.CENTER)
    trail_graphics = py5.create_graphics(extent_px, extent_px, py5.P2D)
    background_graphics = py5.create_graphics(extent_px, extent_px, py5.P2D)
    background_graphics.begin_draw()
    background_graphics.background('#0a0e27')
    for i in range(py5.height):
        inter = i / py5.height
        # Interpolate between two colors
        r_val = int(26 + (45 - 26) * inter)
        g_val = int(31 + (53 - 31) * inter)
        b_val = int(58 + (97 - 58) * inter)
        background_graphics.stroke(r_val, g_val, b_val)
        background_graphics.line(0, i, py5.width, i)
    # Draw pivot point with glow
    background_graphics.no_stroke()
    background_graphics.fill(100, 150, 255, 80)
    background_graphics.ellipse(x_center, y_center, 40, 40)
    background_graphics.fill(100, 150, 255, 120)
    background_graphics.ellipse(x_center, y_center, 25, 25)
    background_graphics.fill(255, 200, 0)
    background_graphics.ellipse(x_center, y_center, 8, 8)
    background_graphics.end_draw()

    for _ in range(ahead_steps):
        predraw_update()
        time.sleep(1/FPS)


main_path: List[np.ndarray] = []
last_step_time: float = time.time()
f_paths: List[List[Tuple[float, float]]] = [
    [(0.0, 0.0)] for _ in range(num_perturbations)]
paused: bool = False


def predraw_update() -> None:
    global state, last_step_time, f_paths, PROJECTION
    t = time.time()
    STEP_SIZE = 0.01  # s
    # Number of fixed-size steps to take to catch up to realtime
    n_steps = int((t - last_step_time) / STEP_SIZE)
    if n_steps > 0:
        for _ in range(n_steps):
            state = pendulums.rk4_step_np(pendulums.n_pendulum_ode_var_np,
                                          state, last_step_time, STEP_SIZE, metadata)

        last_step_time += n_steps * STEP_SIZE
    # if PROJECTION:
    #     f_paths = generate_perturbed_trajectories(state)  # Function not defined
    main_path.append(state)


step: int = 0


def draw() -> None:
    global main_path, background_graphics, step

    # Gradient background and center pivot point
    py5.color_mode(py5.RGB)
    py5.image(background_graphics, 0, 0)

    x0, y0 = x_center, y_center
    coords = [coord * m_to_px +
              origin for coord in pendulums.pendulum_coords_for_ang(main_path[step][0:n], metadata)]
    f_paths = generate_perturbed_var(step)
    # Draw rods and bobs with shadows and gradients
    print(step)
    for (x, y) in coords:
        # Rod shadow
        py5.stroke_weight(6)
        py5.stroke(0, 0, 0, 60)
        py5.line(x0 + 2, y0 + 2, x + 2, y + 2)

        # Rod with gradient-like effect
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

    # Draw trajectory trail and emit particles at tail
    global trail_graphics
    states = main_path[max(step-200, 0):step]
    if KEEP_TRAIL:
        states = main_path[max(step-2, 0):step]
    n_pts = len(states)
    full_stroke = 4
    alpha = 200
    # print(n_pts, len(main_path), step-200, step)
    if n_pts > 1:
        trail_graphics.begin_draw()
        trail_graphics.no_fill()
        trail_graphics.stroke_weight(full_stroke*3)
        trail_graphics.stroke(3, 244, 252, alpha/4)
        pts = [pendulums.pendulum_coords_for_ang(
            state[0:n], metadata)[-1] * m_to_px + origin for state in states]
        if not KEEP_TRAIL:
            trail_graphics.clear()
        for i in range(n_pts-1):
            if not KEEP_TRAIL:
                t_frac = i / max(1, n_pts - 2)
                trail_graphics.stroke(3, 244, 252, alpha/4*t_frac)
            x, y = pts[i]
            x1, y1 = pts[i+1]
            trail_graphics.line(x, y, x1, y1)
        trail_graphics.stroke_weight(full_stroke*1.5)
        trail_graphics.stroke(3, 244, 252, alpha/3)
        for i in range(n_pts-1):
            if not KEEP_TRAIL:
                t_frac = i / max(1, n_pts - 2)
                trail_graphics.stroke(3, 244, 252, alpha/4*t_frac)
            x, y = pts[i]
            x1, y1 = pts[i+1]
            trail_graphics.line(x, y, x1, y1)
        trail_graphics.stroke_weight(full_stroke)
        trail_graphics.stroke(3, 244, 252, alpha)
        for i in range(n_pts-1):
            if not KEEP_TRAIL:
                t_frac = i / max(1, n_pts - 2)
                trail_graphics.stroke(3, 244, 252, alpha/4*t_frac)
            x, y = pts[i]
            x1, y1 = pts[i+1]
            trail_graphics.line(x, y, x1, y1)
        trail_graphics.end_draw()
        py5.image(trail_graphics, 0, 0)

    py5.color_mode(py5.CMAP, py5.mpl_cmaps.PLASMA, num_perturbations)
    if len(f_paths[0]) > 0:
        for i, set in enumerate(f_paths):
            # print(i, set[step:])
            py5.stroke_weight(2)
            py5.stroke(i)
            py5.no_fill()
            with py5.begin_shape():  # type: ignore
                for (x, y) in set:
                    py5.vertex(x, y)

    step += 1


def key_pressed() -> None:
    py5.print_line_profiler_stats()


if __name__ == "__main__":
    py5.profile_draw()
    py5.run_sketch()
