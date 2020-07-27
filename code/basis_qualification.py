


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


split_complex = lambda c: (tf.math.real(c), tf.math.imag(c))

complex_mat_mul = lambda m1_real, m1_imag, m2_real, m2_imag: (
    m1_real * m2_real - m1_imag * m2_imag,
    m1_real * m2_imag + m1_imag * m2_real,
)

# complex_mat_mul = lambda m1, m2: tf.complex(
#     *complex_mat_mul(*split_complex(m1), *split_complex(m2))
# )


def correlation(m1, m2):
    m1_real, m1_imag = split_complex(m1)
    m2_real, m2_imag = split_complex(m2)

    corr_real, corr_imag = complex_mat_mul(m1_real, -m1_imag, m2_real, m2_imag)

    return tf.reduce_sum(tf.complex(corr_real, corr_imag))


def correlation_matrix(basis_set):
    basis_n, basis_size, basis_size = np.shape(basis_set)
    indicies = triangular_indicies(basis_n)

    correlation_matrix = np.zeros(shape=(basis_n, basis_n), dtype=np.complex128)

    for i, j in indicies:
        correlation_matrix[i, j] = correlation(basis_set[i, :, :], basis_set[j, :, :])

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



def apply_low_pass_filter(basis_set_real, basis_set_imag, cutoff_ratio):
    # basis_set = tf.complex(basis_set_real, basis_set_imag)
    # return  split_complex(basis_set)

    # n, nx, ny = tf.shape(basis_set_real)
    # identity = tf.constant(np.identity(nx), dtype=tf.complex128)
    # #ones = tf.ones(shape=(nx, ny), dtype=tf.complex128)
    # basis_set = tf.complex(basis_set_real, basis_set_imag)
    #
    # return complex_mat_mul(*split_complex(basis_set), *split_complex(identity))


    basis_set = tf.complex(basis_set_real, basis_set_imag)
    basis_n, basis_size, _ = np.shape(basis_set)

    # generate kernel
    pad = int((1.-cutoff_ratio)*basis_size/2.)
    pass_hole_size = basis_size - 2*pad
    kernel_fd = tf.ones(shape=(pass_hole_size, pass_hole_size), dtype=tf.complex128)
    kernel_fd = tf.pad(kernel_fd, [[pad, pad], [pad, pad]])
    kernel_fd = tf.signal.ifftshift(kernel_fd, axes=[0, 1])
    basis_set_fd = tf.signal.fft2d(basis_set)

    filtered_basis_fd = tf.complex(
        *complex_mat_mul(
            *split_complex(kernel_fd), *split_complex(basis_set_fd)
        )
    )

    filtered_basis = tf.signal.ifft2d(filtered_basis_fd)

    return split_complex(filtered_basis)


@tf.function
def loss():
    return crosscorrelation(tf.complex(basis_set_real, basis_set_imag))

# Code written using Tensorflow 2.1.0
print("tf version:", tf.__version__)
print("GPU:", tf.config.experimental.list_physical_devices('GPU'))
print("TPU:", tf.config.experimental.list_physical_devices('TPU'))




basis_size = 6
basis_n = 16
basis_set_real = tf.Variable(np.random.rand(basis_n, basis_size, basis_size), trainable=True)
basis_set_imag = tf.Variable(np.random.rand(basis_n, basis_size, basis_size), trainable=True)

optimizer = tf.optimizers.Adam(learning_rate=0.005)



basis_set_log = []

for i in range(10000):
    with tf.GradientTape() as tape:
        tape.watch(basis_set_real)
        tape.watch(basis_set_imag)
        current_loss = loss()
    grads = tape.gradient(target=current_loss, sources=[basis_set_real, basis_set_imag, ])
    optimizer.apply_gradients(zip(grads, [basis_set_real, basis_set_imag, ]))

    # normalize bases in energy
    scale_factor = tf.sqrt(tf.reduce_mean(basis_set_real**2 + basis_set_imag**2, axis=[1, 2]))

    basis_set_real.assign(
        tf.transpose(
            tf.transpose(basis_set_real, perm=[1, 2, 0]) / scale_factor,
            perm=[2, 0, 1]
        )
    )

    basis_set_imag.assign(
        tf.transpose(
            tf.transpose(basis_set_imag, perm=[1, 2, 0]) / scale_factor,
            perm=[2, 0, 1]
        )
    )

    # low pass filter

    filtered_basis_set_real, filtered_basis_set_imag = apply_low_pass_filter(basis_set_real, basis_set_imag, cutoff_ratio=(2.1 / 6.))
    basis_set_real.assign(filtered_basis_set_real)
    basis_set_imag.assign(filtered_basis_set_imag)

    print(current_loss.numpy())

    basis_set_log.append(basis_set_real.numpy() + 1j*basis_set_imag.numpy())





basis_set_log_arr = np.array(basis_set_log)

# plot a column of each matrix
plt.plot(basis_set_log_arr[:, 0, 0])
plt.show()



fig, axs = plt.subplots(basis_size, basis_size)
for i in range(basis_size):
    for j in range(basis_size):

        basis = basis_set_real[i * basis_size + j, :, :].numpy() + 1j * basis_set_real[i * basis_size + j, :,
                                                                            :].numpy()
        axs[i, j].imshow(np.abs(basis))
        axs[i, j].axis('off')
plt.show()



# plot abs correlation matrix
plt.imshow(
    np.abs(
        correlation_matrix(tf.complex(basis_set_real, basis_set_imag))
    )
)





# basis_set = apply_low_pass_filter(basis_set, cutoff_ratio=0.5)