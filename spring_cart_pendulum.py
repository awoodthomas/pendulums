from typing import List, Tuple, cast

import numpy as np
import math
import py5
import time
import os

FPS: int = 30
g: float = 9.81
m_to_px: int = 150

n: int = 2
x: float = 1
xd: float = 0
theta: float = math.pi*0.75
omega: float = 5*math.pi/2
state: np.ndarray = np.array([x, xd, theta, omega])
mc: float = 1
mp: float = 1
l: float = 1.0
k: float = 30.0
extent_px: int = 800


def cart_pendulum_ode(t: float, y: np.ndarray) -> np.ndarray:
    x = y[0]
    xd = y[1]
    theta = y[2]
    omega = y[3]

    A = np.array([[mc + mp, mp * l * np.cos(theta)],
                  [mp * l * np.cos(theta), mp * l**2]])

    b = np.array([mp * l * omega**2 * np.sin(theta) - k * x,
                  -mp * g * l * np.sin(theta)])

    dds = np.linalg.solve(A, b)
    dy = np.array([xd, dds[0], omega, dds[1]])
    return dy


def rk4_step_np(state_in: np.ndarray, t: float, dt: float) -> np.ndarray:
    k1 = cart_pendulum_ode(t, state_in)
    k2 = cart_pendulum_ode(t + 0.5*dt, state_in + 0.5*dt*k1)
    k3 = cart_pendulum_ode(t + 0.5*dt, state_in + 0.5*dt*k2)
    k4 = cart_pendulum_ode(t + dt, state_in + dt*k3)
    return cast(np.ndarray, state_in + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4))


def energy_from_state(s: np.ndarray) -> float:
    x = s[0]
    xd = s[1]
    theta = s[2]
    omega = s[3]

    K_cart = 0.5 * mc * xd**2
    K_pend = 0.5 * mp * ((xd + l * omega * np.cos(theta))**2 +
                         (l * omega * np.sin(theta))**2)
    U_spring = 0.5 * k * x**2
    U_grav = -mp * g * l * np.cos(theta)

    return cast(float, K_cart + K_pend + U_spring + U_grav)


def setup() -> None:
    py5.size(extent_px, extent_px)
    py5.frame_rate(FPS)
    py5.rect_mode(py5.CENTER)


last_bob_coords: List[Tuple[float, float]] = []
last_step_time: float = time.time()


def draw() -> None:
    t = time.time()
    STEP_SIZE = 0.005  # s

    global state, last_bob_coords, last_step_time

    # Number of fixed-size steps to take to catch up to realtime
    n_steps = int((t - last_step_time) / STEP_SIZE)
    if n_steps > 0:
        for _ in range(n_steps):
            state = rk4_step_np(state, last_step_time, STEP_SIZE)

        last_step_time += n_steps * STEP_SIZE

    # Gradient background
    py5.background('#0a0e27')  # type: ignore
    for i in range(py5.height):
        inter = i / py5.height
        # Interpolate between two colors
        r_val = int(26 + (45 - 26) * inter)
        g_val = int(31 + (53 - 31) * inter)
        b_val = int(58 + (97 - 58) * inter)
        py5.stroke(r_val, g_val, b_val)
        py5.line(0, i, py5.width, i)

    # Print text with energy
    e = energy_from_state(state)
    py5.fill(255, 255, 255)
    py5.text_size(16)
    py5.text(f"Total Energy: {e:.2f} J", 10, 20)

    x_origin = py5.width / 2
    y_origin = py5.height / 2 + 100

    # Draw cart
    py5.no_stroke()
    py5.fill(100, 150, 255, 80)
    py5.rect_mode(py5.CENTER)
    xc = x_origin + state[0] * m_to_px
    yc = y_origin / 2 + 100
    py5.rect(xc, yc, 80, 40)

    # Draw rods and bobs with shadows and gradients
    xp = xc + l * m_to_px * np.sin(state[2])
    yp = yc + l * m_to_px * np.cos(state[2])

    py5.stroke_weight(6)
    py5.stroke("#222222")  # type: ignore
    py5.line(xc, yc, xp, yp)

    # Bob with glow effect
    py5.no_stroke()
    py5.fill(255, 100, 100, 100)
    py5.ellipse(xp, yp, 20, 20)
    py5.fill(255, 150, 100)
    py5.ellipse(xp, yp, 12, 12)
    py5.fill(255, 200, 100)
    py5.ellipse(xp, yp, 6, 6)

    # Draw spring from origin to center of the cart
    py5.stroke_weight(4)
    py5.no_fill()
    py5.stroke(150, 150, 200)
    n_coils = 10
    spring_length = xc - (x_origin - 40)
    coil_spacing = spring_length / n_coils
    spring_y_offset = 10
    spring_points = []
    for i in range(n_coils + 1):
        x_s = x_origin - 40 + i * coil_spacing
        if i == 0 or i == n_coils:
            y_s = yc
        elif i % 2 == 0:
            y_s = yc - spring_y_offset
        else:
            y_s = yc + spring_y_offset
        spring_points.append((x_s, y_s))
    with py5.begin_shape():  # type: ignore
        for (x_s, y_s) in spring_points:
            py5.vertex(x_s, y_s)

    # Draw trajectory trail and emit particles at tail
    last_bob_coords.append((xp, yp))
    pts = last_bob_coords[-200:]
    n_pts = len(pts)
    if n_pts > 1:
        for i in range(n_pts - 1):
            t_frac = i / max(1, n_pts - 2)
            alpha = int(30 + 170 * t_frac)
            sw = 1.0 + 1.5 * t_frac
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            py5.stroke_weight(sw)
            py5.stroke(190, 190, 195, alpha)
            py5.line(x1, y1, x2, y2)


if __name__ == "__main__":
    py5.run_sketch()
