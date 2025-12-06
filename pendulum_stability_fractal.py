import jax
import time
import os
import threading
import queue
from jax import numpy as jnp
import numpy as np
import vispy
from vispy import app, scene
from vispy.scene.visuals import Image, Text, Line, Rectangle
import pendulums
from pendulums import PendulumMetadata, PendulumAnimation
import colorstamps


class DoublePendulumStability(PendulumAnimation):

    def __init__(self, metadata: PendulumMetadata, n_states: int = 1000, fps: int = 30, cmap: str = 'teuling0f', trippy: bool = False) -> None:
        if metadata.n_pendulums != 2:
            raise ValueError(
                "DoublePendulumStability only supports 2-pendulums.")
        super().__init__(metadata, fps, window_size_px=n_states)
        self.m_jax = jnp.array(metadata.masses)
        self.r_jax = jnp.array(metadata.lengths)

        self.theta1_min, self.theta1_max = -jnp.pi, jnp.pi
        self.theta2_min, self.theta2_max = -jnp.pi, jnp.pi
        self.init_states()

        # Store as uint8 for faster data transfer to vispy
        if 'pendulum' in cmap:
            match cmap:
                case 'pendulum1':
                    self.cmap = pendulums.cmap_pendulum(n_states)
                case 'pendulum2':
                    self.cmap = pendulums.cmap_pendulum1(n_states)
                case 'pendulum3':
                    self.cmap = pendulums.cmap_pendulum2(n_states)
                case _:
                    raise ValueError(f"Unknown pendulum colormap: {cmap}")
        elif 'radial_' in cmap:
            # MPL name is the part after 'radial_'
            mpl_name = cmap[len('radial_'):]
            self.cmap = pendulums.radial_sym_matplotlib(
                name=mpl_name, k=n_states)
        else:
            self.cmap = jnp.array(
                (colorstamps.stamps.get_cmap(cmap, n_states)*255).astype(jnp.uint8))
        self.cmap = jnp.asarray(self.cmap, dtype=jnp.uint8)

        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.reset_event = threading.Event()
        self.new_bounds = None
        self.selection_start = None
        self.trippy = trippy

    def init_states(self, t1_min=-jnp.pi, t1_max=jnp.pi, t2_min=-jnp.pi, t2_max=jnp.pi):
        theta1 = jnp.linspace(t1_min, t1_max, self.window_size_px)
        theta2 = jnp.linspace(t2_min, t2_max, self.window_size_px)
        theta1_grid, theta2_grid = jnp.meshgrid(theta1, theta2, indexing='xy')
        self.states = jnp.zeros(
            shape=(self.window_size_px, self.window_size_px, 4))
        self.states = self.states.at[:, :, 0].set(theta1_grid)
        self.states = self.states.at[:, :, 1].set(theta2_grid)

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
                              in_axes=(0, None, None)), in_axes=(0, None, None)))
        print("Warming up JIT...")
        t = time.time()
        self.rk4_step_batch(self.states, 0.0, self.step_size,
                            self.m_jax, self.r_jax, self.metadata.n_pendulums).block_until_ready()
        print(f"JIT warmup complete after {time.time()-t:.1f}s.")

        self.canvas = scene.SceneCanvas(
            keys='interactive', size=(self.window_size_px, self.window_size_px))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(interactive=False)
        self.view.camera.set_range(x=(0, self.window_size_px),  # pyright: ignore[reportAttributeAccessIssue]
                                   y=(0, self.window_size_px),
                                   margin=0)
        self.canvas.connect(self.on_mouse_press)
        self.canvas.connect(self.on_mouse_release)
        self.canvas.connect(self.on_mouse_move)
        self.canvas.connect(self.on_key_release)

        self.selection_line = Line(
            color='white', width=2, parent=self.view.scene)
        self.selection_line.visible = False

        self.image = Image(
            np.zeros((self.window_size_px, self.window_size_px, 3),
                     dtype=jnp.uint8),
            parent=self.view.scene, method='subdivide')
        self.image.order = 0

        self.info_text_bg = Rectangle(
            color=(1, 1, 1, 1), border_color='black', center=(0, 0, -0.1), parent=self.view.scene)
        self.info_text_bg.order = 10
        self.info_text_bg.visible = False

        self.info_text = Text(text="", color='black', font_size=10,
                              anchor_x='left', anchor_y='top', parent=self.view.scene)
        self.info_text.order = 11

    def simulation_loop(self) -> None:
        """Runs in a separate thread to compute frames."""
        n_steps = max(round((1.0/self.fps) / self.step_size), 1)

        while self.running:
            # Handle reset of simulation range
            if self.reset_event.is_set():
                if self.new_bounds:
                    self.theta1_min, self.theta1_max, self.theta2_min, self.theta2_max = self.new_bounds
                    self.init_states(*self.new_bounds)
                    self.new_bounds = None
                    n_steps = max(round((1.0/self.fps) / self.step_size), 1)
                self.reset_event.clear()
                with self.frame_queue.mutex:
                    self.frame_queue.queue.clear()
                self.steps = 0

            # 1. Integration
            for _ in range(n_steps):
                with jax.profiler.StepTraceAnnotation("integration_step"):
                    self.states = self.rk4_step_batch(self.states, 0.0, self.step_size,
                                                      self.m_jax, self.r_jax, self.metadata.n_pendulums).block_until_ready()

            # 2. RGB Conversion
            if self.trippy:
                rgb = self.states_to_RGB(self.states, self.cmap, jnp.pi)
            else:
                rgb = self.states_to_RGB(self.states, self.cmap, 2*jnp.pi)

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

    def state_to_RGB(self, state: jnp.ndarray, cmap: jnp.ndarray, extramod=2*jnp.pi) -> jnp.ndarray:
        """Convert a single pendulum state to an RGB color."""
        assert state.shape == (4,)
        normalized = jnp.uint16(
            jnp.mod(jnp.mod(state, extramod) + jnp.pi, 2*jnp.pi) / (2 * jnp.pi)
            * self.window_size_px)
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
        self.on_close()

    def on_close(self, _event=None):
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join()
        if hasattr(self, 'timer'):
            self.timer.stop()

    def on_mouse_press(self, event):
        if event.button == 1:
            # Convert screen y to scene y
            w, h = self.canvas.size
            scale_x = self.window_size_px / w
            scale_y = self.window_size_px / h
            scene_pos = (event.pos[0] * scale_x, (h - event.pos[1]) * scale_y)

            self.selection_start = scene_pos
            self.selection_line.visible = True
            self.update_selection_line(scene_pos)

    def update_selection_line(self, current_pos):
        if self.selection_start is None:
            return
        x0, y0 = self.selection_start
        x1, y1 = current_pos

        # Draw rectangle
        pos = np.array([
            [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]
        ])
        self.selection_line.set_data(pos=pos)

    def on_mouse_move(self, event):
        w, h = self.canvas.size
        scale_x = self.window_size_px / w
        scale_y = self.window_size_px / h
        x_screen, y_screen = event.pos
        x_scene = x_screen * scale_x
        y_scene = (h - y_screen) * scale_y

        if self.selection_start is not None:
            self.update_selection_line((x_scene, y_scene))

        # Update text display
        if 0 <= x_screen < w and 0 <= y_screen < h:
            t1_range = self.theta1_max - self.theta1_min
            t2_range = self.theta2_max - self.theta2_min

            theta1 = self.theta1_min + \
                (x_scene / self.window_size_px) * t1_range
            theta2 = self.theta2_min + \
                (y_scene / self.window_size_px) * t2_range

            self.info_text.text = f"Theta1: {np.rad2deg(theta1):.2f} deg\nTheta2: {np.rad2deg(theta2):.2f} deg"
            self.info_text.visible = True
            self.info_text_bg.visible = True

            # Calculate box position and size
            # Estimate dimensions in screen pixels
            bw_px = 160
            bh_px = 40
            offset_px = 15

            bw_scene = bw_px * scale_x
            bh_scene = bh_px * scale_y
            offset_scene_x = offset_px * scale_x
            offset_scene_y = offset_px * scale_y

            # Default position: Bottom-Right of mouse
            # Text anchor is Top-Left, so pos is the top-left corner of the box
            pos_x = x_scene + offset_scene_x
            pos_y = y_scene - offset_scene_y

            # Check Right edge
            if pos_x + bw_scene > self.window_size_px:
                # Flip to Left
                pos_x = x_scene - offset_scene_x - bw_scene

            # Check Bottom edge
            # Box extends down from pos_y to pos_y - bh_scene
            if pos_y - bh_scene < 0:
                # Flip to Top
                # Box should be above mouse. Bottom of box at y_scene + offset
                # Top of box (pos_y) at y_scene + offset + bh_scene
                pos_y = y_scene + offset_scene_y + bh_scene

            self.info_text.pos = (pos_x, pos_y)

            # Update background rectangle
            # Rectangle is defined by center, width, height
            center_x = pos_x + bw_scene / 2
            center_y = pos_y + bh_scene / 2
            self.info_text_bg.center = (center_x, center_y)
            self.info_text_bg.width = bw_scene
            self.info_text_bg.height = bh_scene
        else:
            self.info_text.visible = False
            self.info_text_bg.visible = False

    def on_mouse_release(self, event):
        if self.selection_start is not None:
            x0, y0 = self.selection_start

            w, h = self.canvas.size
            scale_x = self.window_size_px / w
            scale_y = self.window_size_px / h
            x1 = event.pos[0] * scale_x
            y1 = (h - event.pos[1]) * scale_y

            self.selection_start = None
            self.selection_line.visible = False

            # Ensure min < max
            x_min_px, x_max_px = sorted([x0, x1])
            y_min_px, y_max_px = sorted([y0, y1])

            # Check if selection is too small (just a click)
            if (x_max_px - x_min_px) < 5 or (y_max_px - y_min_px) < 5:
                return

            # Map pixels to theta
            t1_range = self.theta1_max - self.theta1_min
            new_t1_min = self.theta1_min + \
                (x_min_px / self.window_size_px) * t1_range
            new_t1_max = self.theta1_min + \
                (x_max_px / self.window_size_px) * t1_range

            t2_range = self.theta2_max - self.theta2_min
            new_t2_min = self.theta2_min + \
                (y_min_px / self.window_size_px) * t2_range
            new_t2_max = self.theta2_min + \
                (y_max_px / self.window_size_px) * t2_range
            # Decrease step size when we zoom in to see smaller features
            zoom_factor = (new_t1_max - new_t1_min) / (2 * jnp.pi)
            alpha = 0.2  # non-linear time step scaling

            self.step_size = 0.001 + \
                (self.DEFAULT_STEP_SIZE - 0.001) * (zoom_factor ** alpha)

            print(
                f"Zooming to: T1=[{new_t1_min:.2f}, {new_t1_max:.2f}], T2=[{new_t2_min:.2f}, {new_t2_max:.2f}], dt={self.step_size:.4f}s")

            self.new_bounds = (new_t1_min, new_t1_max, new_t2_min, new_t2_max)
            self.reset_event.set()

    def on_key_release(self, event):
        if event.key == ' ':
            print("Resetting to full view")
            self.new_bounds = (-jnp.pi, jnp.pi, -jnp.pi, jnp.pi)
            self.step_size = self.DEFAULT_STEP_SIZE
            self.reset_event.set()


ENABLE_PROFILING = os.environ.get(
    "PROFILE", "0") not in ("0", "False", "false")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Double Pendulum Stability Visualization")
    parser.add_argument('--n_states', type=int, default=1000,
                        help='Number of samples to divide the initial condition space')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for the animation')
    parser.add_argument('--cmap', type=str, default='fourCorners',
                        help='Colormap to use. Accepts colorstamps (e.g. teuling0f), and any matplotlib sequential in the form radial_viridis. Recommended cmaps: teuling0f, fourCorners, radial_inferno, radial_ocean, pendulum1, pendulum2, pendulum3')
    parser.add_argument('--trippy', action='store_true',
                        help='Use trippier color mapping (bad modulus)')
    args = parser.parse_args()
    metadata = PendulumMetadata(
        lengths=np.ones(2),
        masses=np.ones(2),
    )
    animation = DoublePendulumStability(
        metadata, n_states=args.n_states, fps=args.fps, cmap=args.cmap, trippy=args.trippy)
    animation.start_animation()
