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
print("TPU:", tf.config.experimental.list_physical_devices('TPU'))


def complex_matmul(a_real, a_imag, b_real, b_imag):
    real = a_real*b_real - a_imag*b_imag
    imag = a_real*b_imag + a_imag*b_real
    return real, imag


def loss():

    # cannot use complex numbers in matrix multiplication. WTF?
    if True:

        a_real = tf.constant(np.random.rand(*slm_shape), dtype=tf.float64)
        a_imag = tf.constant(np.random.rand(*slm_shape), dtype=tf.float64)
        complex_e_field = tf.complex(e_field_real, e_field_imag)

        resultant_field_real, resultant_field_imag = complex_matmul(a_real, a_imag, e_field_real, e_field_imag)
        resultant_field = tf.complex(resultant_field_real, resultant_field_imag)
    else:

        a = tf.constant(np.random.rand(*slm_shape) + 1j*np.random.rand(*slm_shape), dtype=tf.complex128)
        complex_e_field = tf.complex(e_field_real, e_field_imag)
        resultant_field = a*complex_e_field


    intensity = tf.abs(resultant_field)**2
    # the loss is the correlation between the resultant intensity and the target intensity
    return tf.reduce_mean(intensity**2)


slm_n = 10
slm_shape = (slm_n, slm_n, )

complex_split = lambda c: (np.real(c), np.imag(c))
field_initializer = lambda:  np.sqrt(np.random.rand(*slm_shape))*np.exp(1j*np.random.rand(*slm_shape)*np.pi*2)
e_field_real, e_field_imag = complex_split(field_initializer())
# np.max(np.abs(e_field_real + 1j*e_field_imag)) should be <1.

e_field_real = tf.Variable(e_field_real, trainable=True)
e_field_imag = tf.Variable(e_field_imag, trainable=True)


get_intensity = lambda real, imag: real**2 + imag**2





optimizer = optimizer = tf.optimizers.SGD()

def train_step():
    with tf.GradientTape() as tape:
        tape.watch(e_field_imag)
        tape.watch(e_field_real)
        current_loss = loss()
    grads = tape.gradient(target=current_loss, sources=[e_field_imag, e_field_real, ])
    # update weights
    optimizer.apply_gradients(zip(grads, [e_field_imag, e_field_real, ]))

    return tf.reduce_mean(current_loss).numpy()


# %%

# Training loop
iterations = 2 ** 15
n_update = 2 ** 6  # Updates information every * iterations

# log to plot parameter convergence
field_log = []

t = time.time()
for i in range(iterations):
    error = train_step()
    if i % n_update == 0:
        field_log.append(e_field_real.numpy() + 1j*e_field_imag.numpy())
        t_now = time.time()
        print("Error: {}\tTimePerUpdate(s): {}".format(error, t_now - t))
        t = t_now

# %%

# for field in field_log:
#     plt.imshow(np.abs(field)**2)
#     plt.show()

# with writer.as_default():
#   tf.summary.trace_export(
#       name="loss",
#       step=0,
#       profiler_outdir=logdir)


plt.imshow(get_intensity(e_field_real, e_field_imag))
plt.show()



