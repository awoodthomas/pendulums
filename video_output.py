import numpy as np
from pendulums import PendulumMetadata
from k_n_pendulum import KN_Pendulum_JAX_vispy
from pendulum_stability_fractal import DoublePendulumStabilityFractal
from moviepy.video.VideoClip import VideoClip

# theta = np.deg2rad(np.array([126, 72]))  # Initial angles in degrees
# n = len(theta)
# omega = np.zeros(n)
# omega[0] = np.deg2rad(-180)
# state0 = np.array(np.concatenate([theta, omega]))
# m = np.ones(n)
# r = np.ones(n)
# r[1] = 1.5
# metadata = PendulumMetadata(masses=m, lengths=r)
# k = 100000
# fps = 60

# animation = KN_Pendulum_JAX_vispy(metadata, state0, k=k, fps=fps)
# animation.setup()
# animation.canvas.show()

# animation = VideoClip(animation.frame_step_and_capture, duration=20)
# animation.write_videofile('100k_double.mp4', threads=6, fps=fps)

n = 2
m = np.ones(n)
r = np.ones(n)
metadata = PendulumMetadata(masses=m, lengths=r)
k = 1400
fps = 60

animation = DoublePendulumStabilityFractal(
    metadata=metadata,
    n_states=k,
    fps=fps,
    cmap='teuling0f')
animation.step_size = 0.01
animation.setup()
animation.canvas.show()
animation.start_simulation()

animation = VideoClip(animation.frame_step_and_capture, duration=30)
animation.write_videofile('teuling_heart.mp4', threads=6,
                          fps=fps, codec='libx264', audio=False, bitrate='20m')
