import numpy as np
from pendulums import PendulumMetadata
from k_n_pendulum import KN_Pendulum_JAX_vispy
from moviepy.video.VideoClip import VideoClip

theta = np.deg2rad(np.array([126, 72]))  # Initial angles in degrees
n = len(theta)
omega = np.zeros(n)
omega[0] = np.deg2rad(-180)
state0 = np.array(np.concatenate([theta, omega]))
m = np.ones(n)
r = np.ones(n)
r[1] = 1.5
metadata = PendulumMetadata(masses=m, lengths=r)
k = 100000
fps = 60

animation = KN_Pendulum_JAX_vispy(metadata, state0, k=k, fps=fps)
animation.setup()
animation.canvas.show()

animation = VideoClip(animation.frame_step_and_capture, duration=20)
animation.write_videofile('100k_double.mp4', threads=6, fps=fps)
