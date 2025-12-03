"""
Display and animate n-pendulum trajectory (class-based, no globals).
"""
from typing import Any, Dict, List, Tuple, cast

import pickle
import numpy as np
import py5
from pendulums import PendulumMetadata, PendulumAnimation


class DisplayTrajectoryAnimation(PendulumAnimation):
    def __init__(
        self,
        sol_t: np.ndarray,
        sol_y: np.ndarray,
        energy: np.ndarray,
        future_paths: List[List[np.ndarray]],
        metadata: PendulumMetadata,
        num_perturbations: int,
        fps: int = 30,
        m_to_px: int = 200,
    ) -> None:
        super().__init__(metadata=metadata, fps=fps, m_to_px=m_to_px)
        self.sol_t = sol_t
        self.sol_y = sol_y
        self.energy = energy
        self.future_paths = future_paths
        self.num_perturbations = num_perturbations

        self.last_bob_coords: List[Tuple[float, float]] = []
        self.repetitions: int = 0

    def draw(self) -> None:
        step_count = self.sol_y.shape[1]
        t = (py5.frame_count / self.fps) % (self.sol_t[-1])

        if (py5.frame_count / self.fps) > self.sol_t[-1] * self.repetitions:
            self.last_bob_coords.clear()
            self.repetitions += 1

        theta = [np.interp(t, self.sol_t, self.sol_y[i, :])
                 for i in range(self.metadata.n_pendulums)]
        e = np.interp(t, self.sol_t, self.energy)

        self.draw_background()
        last_pos = self.draw_pendulum(np.array(theta))

        # Trail
        self.last_bob_coords.append(last_pos)
        py5.begin_shape()
        py5.stroke_weight(2.5)
        py5.stroke(102, 0, 0)
        py5.no_fill()
        for (x, y) in self.last_bob_coords:
            py5.curve_vertex(x, y)
        py5.end_shape()

        # Future paths
        time_index = min(range(step_count),
                         key=lambda i: abs(self.sol_t[i] - t))
        for i in range(self.num_perturbations):
            s_fp = self.future_paths[time_index][i]
            py5.color_mode(py5.CMAP, py5.mpl_cmaps.PLASMA,
                           self.num_perturbations)

            py5.begin_shape()
            py5.stroke_weight(1.5)
            py5.stroke(i)
            py5.no_fill()
            for s in s_fp.T:
                x, y = self.origin_x, self.origin_y
                for j in range(self.metadata.n_pendulums):
                    x = x + self.metadata.lengths[j] * \
                        self.m_to_px * np.sin(s[j])
                    y = y + self.metadata.lengths[j] * \
                        self.m_to_px * np.cos(s[j])
                py5.curve_vertex(x, y)
            py5.end_shape()

        # HUD
        py5.fill(255, 255, 255)
        py5.text_size(16)
        py5.text(f"Total Energy: {e:.2f} J", 10, 20)


def load_trajectory(filename: str) -> Dict[str, Any]:
    with open(filename, 'rb') as f:
        data = cast(Dict[str, Any], pickle.load(f))
    return data


# Module-level animation instance for py5 callbacks
_animation: DisplayTrajectoryAnimation


def settings() -> None:
    _animation.settings()


def setup() -> None:
    _animation.setup()


def draw() -> None:
    _animation.draw()


if __name__ == "__main__":
    data = load_trajectory('trajectory_triple.pkl')
    sol_t = data['sol_t']
    sol_y = data['sol_y']
    energy = data['energy']
    future_paths = data['future_paths']
    params = data['parameters']

    metadata = PendulumMetadata(
        masses=params['masses'], lengths=params['lengths'])
    _animation = DisplayTrajectoryAnimation(
        sol_t=sol_t,
        sol_y=sol_y,
        energy=energy,
        future_paths=future_paths,
        metadata=metadata,
        num_perturbations=params['num_perturbations'],
        fps=30,
        m_to_px=200,
    )
    py5.run_sketch()
    py5.run_sketch()
