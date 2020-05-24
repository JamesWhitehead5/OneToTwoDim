# The purpose of this file is to perform a simple, otherwise verifiable optimization using angular propagation of
# optical fields.
# The setup will be to optimize a SLM to focus to a point

import tensorflow as tf
import AngularPropagateTensorflow.AngularPropagateTensorflow as ap_tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Generator, List, Tuple
import time
from datetime import datetime

# Code written using Tensorflow 2.1.0
print("tf version:", tf.__version__)
print("GPU:", tf.config.experimental.list_physical_devices('GPU'))

# Some helper functions
get_intensity = lambda real, imag: real**2 + imag**2
complex_split = lambda c: (np.real(c), np.imag(c))
field_initializer_random = lambda:  np.sqrt(np.random.rand(*slm_shape))*np.exp(1j*np.random.rand(*slm_shape)*np.pi*2)
field_initializer_uniform = lambda: np.ones(slm_shape, dtype=np.complex128)


# define a target intensity with a maximum in the center
def generate_2d_gaussian():
    sigma = wavelength*20.
    x = np.linspace(-0.5, 0.5, slm_shape[0]) * dd * slm_shape[0]
    y = np.linspace(-0.5, 0.5, slm_shape[1]) * dd * slm_shape[1]
    Y, X = tf.meshgrid(y, x, indexing='xy')
    return 1/np.sqrt(2*np.pi)*np.exp(-0.5 * ((X / sigma) ** 2 + (Y / sigma) ** 2))


def generate_center_point():
    field = np.zeros(shape=slm_shape)
    field[slm_shape[0]//2, slm_shape[1]//2] = 1.
    return field


@tf.function
def loss():
    complex_e_field = tf.dtypes.complex(e_field_real, e_field_imag)
    resultant_field = ap_tf.propagate_angular_padded(field=complex_e_field, k=k, z_list=[target_focal_length, ], dx=dd, dy=dd)[0, :, :]
    intensity = tf.abs(resultant_field)**2
    return -intensity[slm_n//2, slm_n//2]


def train_step():
    with tf.GradientTape() as tape:
        tape.watch(e_field_imag)
        tape.watch(e_field_real)
        current_loss = loss()
    grads = tape.gradient(target=current_loss, sources=[e_field_imag, e_field_real, ])

    # update weights
    optimizer.apply_gradients(zip(grads, [e_field_imag, e_field_real, ]))

    # Clip magnitude of phasors to remove gain. Keep angle the same.
    phase = tf.math.atan2(e_field_imag, e_field_real)
    mag = tf.sqrt(e_field_real**2 + e_field_imag**2)
    mag = tf.clip_by_value(mag, clip_value_min=0., clip_value_max=1.)
    e_field_imag.assign(mag*tf.sin(phase))
    e_field_real.assign(mag*tf.cos(phase))
    return tf.reduce_mean(current_loss).numpy() # return loss for logging


# Set up logging.
# stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# logdir = 'logs\\simple_angular_prop_tf\\%s' % stamp
# writer = tf.summary.create_file_writer(logdir)
# tf.summary.trace_on(graph=True, profiler=True)

wavelength = 633e-9 # HeNe
k = 2.*np.pi/wavelength
lens_aperture = 50.*wavelength
target_focal_length = 50.*wavelength
dd = wavelength/2.
slm_n = int(lens_aperture/dd)
slm_shape = (slm_n, slm_n, )

# initialize and define training variables
e_field_real, e_field_imag = complex_split(field_initializer_random())
e_field_real = tf.Variable(e_field_real, trainable=True)
e_field_imag = tf.Variable(e_field_imag, trainable=True)

optimizer = tf.optimizers.Adam(learning_rate=0.05)
# Training loop
iterations = 2 ** 10
n_update = 2 ** 6  # Updates information every * iterations

# log to plot parameter convergence
field_log = []

# Training loop
t = time.time()
for i in range(iterations):
    error = train_step()
    if i % n_update == 0:
        field_log.append(e_field_real.numpy() + 1j*e_field_imag.numpy())
        t_now = time.time()
        print("Error: {}\tTimePerUpdate(s): {}".format(error, t_now - t))
        t = t_now


# with writer.as_default():
#   tf.summary.trace_export(
#       name="loss",
#       step=0,
#       profiler_outdir=logdir)

# plot convergence of a small sample of parameters
field_log = np.array(field_log)
plt.plot(np.imag(field_log[:, :, 0]))
plt.show()

# plot intensity
plt.imshow(np.sqrt(get_intensity(e_field_real, e_field_imag)))
plt.colorbar()
plt.show()

# plot phase
plt.imshow(np.arctan2(e_field_imag.numpy(), e_field_real.numpy()))
plt.colorbar()
plt.show()

# plot propagation
complex_e_field = tf.dtypes.complex(e_field_real, e_field_imag)
resultant_field = ap_tf.propagate_angular_padded(field=complex_e_field, k=k, z_list=[target_focal_length, ], dx=dd, dy=dd)[0, :, :]
intensity = tf.abs(resultant_field)**2
plt.imshow(intensity)
plt.colorbar()
plt.show()

