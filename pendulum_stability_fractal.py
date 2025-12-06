import jax
import time
import os
import threading
import queue
from jax import numpy as jnp
import numpy as np
import vispy
from vispy import app, scene
from vispy.scene.visuals import Image, Text
from vispy.geometry import Rect
import pendulums
from pendulums import PendulumMetadata, PendulumAnimation
import colorstamps


class DoublePendulumStability(PendulumAnimation):

    def __init__(self, metadata: PendulumMetadata, n_states: int = 1000, fps: int = 30, cmap: str = 'teuling0f') -> None:
        if metadata.n_pendulums != 2:
            raise ValueError(
                "DoublePendulumStability only supports 2-pendulums.")
        super().__init__(metadata, fps, window_size_px=n_states)
        self.m_jax = jnp.array(metadata.masses)
        self.r_jax = jnp.array(metadata.lengths)
        # State format: 3D JNP array, shape (n_states, n_states, 4)
        self.states = jnp.zeros(shape=(n_states, n_states, 4))
        theta1 = jnp.linspace(-jnp.pi, jnp.pi, n_states)
        theta2 = jnp.linspace(-jnp.pi, jnp.pi, n_states)
        theta1_grid, theta2_grid = jnp.meshgrid(theta1, theta2, indexing='xy')
        self.states = self.states.at[:, :, 0].set(theta1_grid)
        self.states = self.states.at[:, :, 1].set(theta2_grid)
        # Store as uint8 for faster data transfer to vispy
        self.cmap = jnp.array(
            (colorstamps.stamps.get_cmap(cmap, n_states)*255).astype(jnp.uint8))

        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False

    def setup_animation(self) -> None:
        """Set up vispy graphics objects for drawing pendulums."""
        print("JIT-compiling vectorized RK4 stepper...")
        if ENABLE_PROFILING:
            jax.profiler.start_trace("/tmp/profile-data")

        # Create a JIT function for stepping all the pendulums in parallel
        self.rk4_step_batch = jax.jit(jax.vmap(jax.vmap(pendulums.n_pendulum_rk4_step_jax,
                                               in_axes=(0, None, None, None, None, None)),
                                               in_axes=(0, None, None, None, None, None)),
                                      static_argnums=(5,))
        self.states_to_RGB = jax.jit(
            jax.vmap(jax.vmap(self.state_to_RGB,
                     in_axes=(0, None)), in_axes=(0, None)))
        print("Warming up JIT...")
        t = time.time()
        self.rk4_step_batch(self.states, 0.0, self.STEP_SIZE,
                            self.m_jax, self.r_jax, self.metadata.n_pendulums).block_until_ready()
        print(f"JIT warmup complete after {time.time()-t:.1f}s.")

        self.canvas = scene.SceneCanvas(
            keys='interactive', size=(self.window_size_px, self.window_size_px))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.view.camera.set_range(x=(0, self.window_size_px),  # pyright: ignore[reportAttributeAccessIssue]
                                   y=(0, self.window_size_px),
                                   margin=0)
        self.canvas.connect(self.on_mouse_release)

        self.canvas.connect(self.on_mouse_move)

        self.image = Image(
            np.zeros((self.window_size_px, self.window_size_px, 3),
                     dtype=jnp.uint8),
            parent=self.view.scene, method='subdivide')
        self.info_text = Text(text="", parent=self.view.scene)

    def simulation_loop(self) -> None:
        """Runs in a separate thread to compute frames."""
        n_steps = max(round((1.0/self.fps) / self.STEP_SIZE), 1)

        while self.running:
            # 1. Integration
            for _ in range(n_steps):
                with jax.profiler.StepTraceAnnotation("integration_step"):
                    self.states = self.rk4_step_batch(self.states, 0.0, self.STEP_SIZE,
                                                      self.m_jax, self.r_jax, self.metadata.n_pendulums).block_until_ready()

            # 2. RGB Conversion
            rgb = self.states_to_RGB(self.states, self.cmap)

            # 3. Put in queue (blocks if queue is full, throttling the sim)
            self.frame_queue.put(rgb)

    def animation_step(self, _event) -> None:
        """Perform a single animation step: get a frame from the queue and render."""
        try:
            with jax.profiler.StepTraceAnnotation("animation_step"):
                # rgb, _ = colorstamps.apply_stamp( # 51 ms !!!
                #     normXY[:, :, 0], normXY[:, :, 1], self.cmap)
                rgb = self.frame_queue.get()
                self.image.set_data(rgb)  # 7.5 ms (jax->np array)
                self.canvas.update()
                self.steps += 1
            if ENABLE_PROFILING and self.steps == 30:
                jax.profiler.stop_trace()
                print("Stopped profiling trace.")
        except queue.Empty:
            pass
    # print(f"Completed frame {self.steps}, {self.fps} FPS")

    def state_to_RGB(self, state: jnp.ndarray, cmap: jnp.ndarray) -> jnp.ndarray:
        """Convert a single pendulum state to an RGB color."""
        assert state.shape == (4,)
        normalized = jnp.uint16((state + jnp.pi) / (2 * jnp.pi)
                                * self.window_size_px)
        # normalized = jnp.uint8((state + jnp.pi) / (2 * jnp.pi)
        #                        * 255)
        return cmap[normalized[0], normalized[1]]

    def start_animation(self) -> None:
        self.setup_animation()

        # Start simulation thread
        self.running = True
        self.simulation_thread = threading.Thread(
            target=self.simulation_loop, daemon=True)
        self.simulation_thread.start()

        self.canvas.show()
        self.timer = app.Timer(
            1.0/self.fps, connect=self.animation_step, start=True)
        app.run()
        # Make sure the sim thread shuts down
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join()

    def on_mouse_move(self, event):
        # print(event.pos, self.view.camera.get_state('rect'))
        rect: Rect = self.view.camera.get_state()['rect']
        x_index = event.pos[0] / self.window_size_px * rect.width + rect.left
        y_index = -event.pos[1] / self.window_size_px * rect.height + rect.top

        # If valid index, update the text display to show the initial thetas at the mouse position
        # Update text position to follow mouse
        if 0 <= x_index < self.window_size_px and 0 <= y_index < self.window_size_px:
            theta1 = np.rad2deg(
                x_index / self.window_size_px * 2 * jnp.pi - jnp.pi)
            theta2 = np.rad2deg(
                y_index / self.window_size_px * 2 * jnp.pi - jnp.pi)
            self.info_text.text = f"Theta1: {theta1:.2f} deg\nTheta2: {theta2:.2f} deg\n[{event.pos[0]}, {event.pos[1]}]"
            self.info_text.pos = (x_index + 10, y_index + 10)
        else:
            self.info_text.text = ""

    def on_mouse_release(self, event):
        time.sleep(0.5)
        rect: Rect = self.view.camera.get_state()['rect']
        print(rect.left, rect.width, rect.bottom, rect.height)
        theta1_min = rect.left * 2 * jnp.pi / self.window_size_px - jnp.pi
        theta1_max = (rect.left + rect.width) * 2 * \
            jnp.pi / self.window_size_px - jnp.pi
        theta2_min = rect.bottom * 2 * jnp.pi / self.window_size_px - jnp.pi
        theta2_max = (rect.bottom + rect.height) * 2 * \
            jnp.pi / self.window_size_px - jnp.pi
        print(
            f"Theta1 range: {jnp.rad2deg(theta1_min):.2f} deg to {jnp.rad2deg(theta1_max):.2f} deg")
        print(
            f"Theta2 range: {jnp.rad2deg(theta2_min):.2f} deg to {jnp.rad2deg(theta2_max):.2f} deg")


ENABLE_PROFILING = os.environ.get(
    "PROFILE", "0") not in ("0", "False", "false")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Double Pendulum Stability Visualization")
    parser.add_argument('--n_states', type=int, default=1400,
                        help='Number of samples to divide the initial condition space')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for the animation')
    parser.add_argument('--cmap', type=str, default='fourCorners',
                        help='Colorstamp map to use for visualization')
    args = parser.parse_args()
    metadata = PendulumMetadata(
        lengths=np.ones(2),
        masses=np.ones(2),
    )
    animation = DoublePendulumStability(
        metadata, n_states=args.n_states, fps=args.fps, cmap=args.cmap)
    animation.start_animation()
