import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from typing import Generator, List, Tuple
import time

def upper_triangular_indicies(size):
    indicies = []
    i = 0
    while i < size:
        j = 0
        while j < i:
            indicies.append((i, j))
            j += 1
        i += 1
    return indicies

def triangular_indicies(size):
    indicies = []
    i = 0
    while i < size:
        j = 0
        while j <= i:
            indicies.append((i, j))
            j += 1
        i += 1
    return indicies

def correlation(m1, m2):
    return tf.reduce_sum(tf.math.multiply(m1, m2))

def correlation_matrix(basis_set):
    basis_n, basis_size, _ = np.shape(basis_set)
    indicies = triangular_indicies(basis_n)

    correlation_matrix = np.zeros(shape=(basis_n, basis_n))

    for i, j in indicies:
        coor = tf.abs(correlation(basis_set[i, :, :], basis_set[j, :, :]))
        correlation_matrix[i, j] = coor

    return correlation_matrix

def crosscorrelation(basis_set):
    basis_n, basis_size, _ = np.shape(basis_set)
    indicies = upper_triangular_indicies(basis_n)

    sum = 0.
    for i, j in indicies:
        sum += tf.abs(correlation(basis_set[i, :, :], basis_set[j, :, :]))

    return sum


def test_orthogonality():
    basis_size = 4
    basis_n = basis_size * basis_size
    basis_set = np.random.rand(basis_n, basis_size, basis_size)
    print(crosscorrelation(basis_set))

    basis_set = np.array([
        [[1., 0], [0, 0]],
        [[0., 1], [0, 0]],
        [[0., 0], [1, 0]],
        [[0., 0], [0, 1]],
    ])

    print(crosscorrelation(basis_set))



# Code written using Tensorflow 2.1.0
print("tf version:", tf.__version__)
print("GPU:", tf.config.experimental.list_physical_devices('GPU'))
print("TPU:", tf.config.experimental.list_physical_devices('TPU'))

# @tf.function
def loss():
    return crosscorrelation(basis_set)

basis_size = 4
basis_n = 6
basis_set = np.random.rand(basis_n, basis_size, basis_size)
basis_set = tf.Variable(basis_set, trainable=True)

optimizer = tf.optimizers.Adam(learning_rate=0.005)

basis_set_log = []

for i in range(1000):
    with tf.GradientTape() as tape:
        tape.watch(basis_set)
        current_loss = loss()
    grads = tape.gradient(target=current_loss, sources=[basis_set, ])
    optimizer.apply_gradients(zip(grads, [basis_set, ]))


    # normalize bases in energy
    basis_set.assign(
        tf.transpose(
            tf.transpose(basis_set, perm=[1,2, 0]) / tf.sqrt(tf.reduce_mean(basis_set**2, axis=[1, 2])),
            perm=[2, 0, 1]
        )
    )

    print(current_loss.numpy())

    basis_set_log.append(basis_set.numpy())

basis_set_log_arr = np.array(basis_set_log)

# plot a column of each matrix
plt.plot(basis_set_log_arr[:, 0, 0])
plt.show()

fig, axs = plt.subplots(basis_size, basis_size)
for i in range(basis_size):
    for j in range(basis_size):
        axs[i, j].imshow(basis_set[i*basis_size + j, :, :])
plt.show()

plt.imshow(correlation_matrix(basis_set))


