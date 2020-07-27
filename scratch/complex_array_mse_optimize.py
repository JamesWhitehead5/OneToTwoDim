

import tensorflow as tf
import AngularPropagateTensorflow.AngularPropagateTensorflow as ap_tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Generator, List, Tuple
import time

# Code written using Tensorflow 2.1.0
print("tf version:", tf.__version__)
print("GPU:", tf.config.experimental.list_physical_devices('GPU'))
print("TPU:", tf.config.experimental.list_physical_devices('TPU'))


x_real = tf.Variable(np.random.rand(100, 100), trainable=True)
x_imag = tf.Variable(np.random.rand(100, 100), trainable=True)

target_real = np.random.rand(100, 100)
target_imag = np.random.rand(100, 100)

def loss():
    # sum squared error
    return tf.reduce_sum((x_real-target_real)**2 + (x_imag-target_imag)**2)


# optimizer = tf.optimizers.Adam(learning_rate=0.005)
optimizer = tf.optimizers.SGD()

x_real_log = []
x_imag_log = []

for i in range(1000):
    with tf.GradientTape() as tape:
        tape.watch(x_imag)
        tape.watch(x_real)
        current_loss = loss()
    grads = tape.gradient(target=current_loss, sources=[x_real, x_imag, ])
    optimizer.apply_gradients(zip(grads, [x_real, x_imag,]))
    print(current_loss.numpy())

    x_real_log.append(x_real.numpy())
    x_imag_log.append(x_imag.numpy())

x_real_log = np.array(x_real_log)
x_imag_log = np.array(x_imag_log)

# plot a column of each matrix
plt.plot(x_real_log[:, 0])
plt.plot(x_imag_log[:, 0])
plt.show()

