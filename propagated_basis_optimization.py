


import tensorflow as tf
import numpy as np
import AngularPropagateTensorflow.AngularPropagateTensorflow as ap_tf

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


def plot_modes(set):
    n_modes = set.shape[0]
    fig, axs = plt.subplots(1, n_modes)
    for i in range(n_modes):
        axs[i].imshow(np.abs(set[i, :, :]))
        axs[i].axis('off')
    plt.show()

def plot_modes_fft(set):
    n_modes = set.shape[0]
    fig, axs = plt.subplots(1, n_modes)
    for i in range(n_modes):
        fft = np.fft.fftshift(np.fft.fft2(set[i, :, :]))
        axs[i].imshow(np.abs(fft))
        axs[i].axis('off')
    plt.show()



def apply_low_pass_filter(basis_set_real, basis_set_imag, pad):
    basis_set = tf.complex(basis_set_real, basis_set_imag)
    basis_n, basis_size, _ = np.shape(basis_set)

    # generate kernel
    # pad = int((1.-cutoff_ratio)*basis_size/2.)
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

def complex_initializer_random(shape):
    return np.sqrt(np.random.rand(*shape)) * np.exp(1j * np.random.rand(*shape) * np.pi * 2)

# field_set_initializer_random = lambda:  np.sqrt(np.random.rand(n_modes, *slm_shape))*np.exp(1j*np.random.rand(n_modes, *slm_shape)*np.pi*2)
# field_set_initializer_uniform = lambda: np.ones(shape=(n_modes, *slm_shape), dtype=np.complex128)

# plt.imshow(generate_1d_slm_basis_set([1.,2.,3.], pixel_width=2, pixel_height=3, pixel_spacing=4))


def generate_1d_slm_basis_set(weights):
    """

    :param weights: List of complex coeffients that coorespond to the phasor at each pixel in the 1d slm
    :return: Single phasor field that represents the near field after the 1D SLM
    """

    from oneD_slm_field_generator import generate_1d_slm

    n_basis = len(weights)

    basis_set = []

    for i, weight in enumerate(weights):
        input_vec = [0.]*n_basis
        input_vec[i] = weight
        field = generate_1d_slm(input_vec, pixel_width=50, pixel_height=50, pixel_spacing=60)
        basis_set.append(field)

    return tf.stack(basis_set, axis=0)

# basis_set = generate_1d_slm_basis_set([1., 2., 3., 4.,])

@tf.function
def forward():
    def prop(x):
        """Propagates fields `prop_distantce`"""
        return ap_tf.propagate_angular_padded(field=x, k=k, z_list=[prop_distance, ], dx=dd,
                                                       dy=dd)[0, :, :]
    def lens_f(A, f):
        """Apply lens phase profile to field with focal length `prop_distance`"""
        x = tf.cast(tf.linspace(-dd * slm_size / 2., dd * slm_size / 2., slm_size), dtype=tf.float64)
        y = tf.cast(tf.linspace(-dd * slm_size / 2., dd * slm_size / 2., slm_size), dtype=tf.float64)
        Y, X = tf.meshgrid(y, x, indexing='xy')
        phi = -k*tf.sqrt(f**2 + X**2 + Y**2)
        field = tf.complex(tf.cos(phi), tf.sin(phi))
        lensed_field = tf.complex(
            *complex_mat_mul(
                *split_complex(A),
                *split_complex(field),
            )
        )
        return lensed_field

    def lens_full(A):
        return lens_f(A, prop_distance)

    def lens_half(A):
        return lens_f(A, prop_distance/2.)

    def meta(A):
        field_after = tf.complex(
            *complex_mat_mul(
                *split_complex(A),
                metasurface_real, metasurface_imag,
            )
        )
        return field_after

    field_set = tf.complex(slm_basis_set_real, slm_basis_set_imag)

    field_set = tf.map_fn(fn=prop, elems=field_set)
    field_set = tf.map_fn(fn=lens_full, elems=field_set)
    field_set = tf.map_fn(fn=prop, elems=field_set)

    field_set = tf.map_fn(fn=meta, elems=field_set)

    field_set = tf.map_fn(fn=prop, elems=field_set)
    field_set = tf.map_fn(fn=lens_full, elems=field_set)
    field_set = tf.map_fn(fn=prop, elems=field_set)
    return field_set


# @tf.function
def loss():
    propagated_fields = forward()
    propagated_fields_real, propagated_fields_imag = split_complex(propagated_fields)

    # low pass filter
    filtered_prop_fields_real, filtered_prop_fields_imag = apply_low_pass_filter(propagated_fields_real, propagated_fields_imag, pad=(slm_size-filter_hole_size)//2)

    mode_overlap = crosscorrelation(tf.complex(propagated_fields_real, propagated_fields_imag))

    intensity = filtered_prop_fields_real**2 + filtered_prop_fields_imag**2
    # energy is low frequency modes
    energy_set = tf.reduce_sum(intensity, axis=[1, 2])
    total_energy = tf.reduce_sum(tf.sqrt(energy_set))

    return mode_overlap
    #return mode_overlap #- total_energy


def train_step():
    with tf.GradientTape() as tape:
        tape.watch(metasurface_real)
        tape.watch(metasurface_imag)
        current_loss = loss()
    grads = tape.gradient(target=current_loss, sources=[metasurface_real, metasurface_imag, ])

    # update weights
    optimizer.apply_gradients(zip(grads, [metasurface_real, metasurface_imag, ]))

    # Clip magnitude of phasors to remove gain. Keep angle the same.
    phase = tf.math.atan2(metasurface_imag, metasurface_real)
    mag = tf.sqrt(metasurface_real**2 + metasurface_imag**2)
    mag = tf.clip_by_value(mag, clip_value_min=0., clip_value_max=1.)
    metasurface_imag.assign(mag*tf.sin(phase))
    metasurface_real.assign(mag*tf.cos(phase))

    return tf.reduce_mean(current_loss).numpy() # return loss for logging


# Set up logging.
# stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# logdir = 'logs\\simple_angular_prop_tf\\%s' % stamp
# writer = tf.summary.create_file_writer(logdir)
# tf.summary.trace_on(graph=True, profiler=True)

wavelength = 633e-9 # HeNe
k = 2.*np.pi/wavelength
lens_aperture = 250.*wavelength
prop_distance = 400.*wavelength

slm_size = 500 # side length of the array
dd = lens_aperture/slm_size # array element spacing
slm_shape = (slm_size, slm_size, )
filter_hole_size = 2 # allows `filter_hole_size`**2 number of orthogonal modes through
n_modes = 4


# initialize the SLM basis set.
slm_basis_set = generate_1d_slm_basis_set(weights=tf.ones(n_modes, dtype=tf.complex128))
slm_basis_set_real, slm_basis_set_imag = split_complex(slm_basis_set)

# initialize and define training variables
metasurface_real, metasurface_imag = split_complex(complex_initializer_random(shape=slm_shape))
metasurface_real = tf.Variable(metasurface_real, dtype=tf.float64, trainable=True)
metasurface_imag = tf.Variable(metasurface_imag, dtype=tf.float64, trainable=True)

# f = forward(); plot_modes(f)
# plot_modes(slm_basis_set)
#
# plot_modes_fft(f)

# start the leanring loop


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
        field_log.append(metasurface_real.numpy() + 1j*metasurface_imag.numpy())
        t_now = time.time()
        print("Error: {}\tTimePerUpdate(s): {}".format(error, t_now - t))
        t = t_now




plot_modes(metasurface_real.numpy() + 1j * metasurface_imag.numpy())
plt.show()

plot_modes(forward().numpy())
plt.show()


plot_modes_fft(forward())
plt.show()

# plot abs correlation matrix
plt.imshow(
    np.abs(
        correlation_matrix(forward())
    )
)
plt.show()

propagation = forward()
filtered_prop_fields = tf.complex(*apply_low_pass_filter(*split_complex(propagation), pad=(slm_size-filter_hole_size)//2))
plt.imshow(
    plot_modes(filtered_prop_fields.numpy())
)
plt.show()

plot_modes_fft(filtered_prop_fields.numpy())
plt.show()
