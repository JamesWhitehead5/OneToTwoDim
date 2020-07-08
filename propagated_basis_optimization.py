import tensorflow as tf
import numpy as np
import sys
import AngularPropagateTensorflow as ap

import oneD_slm_field_generator

import matplotlib.pyplot as plt
from typing import Generator, List, Tuple
import time


def strictly_triangular_indices(size: int) -> List[Tuple[int, int]]:
    """Returns a list of tuples that contain the indices of strictly non-zero elements of a _strictly_ triangular matrix
    of dimension `size` x `size` """

    indices = []
    i = 0
    while i < size:
        j = 0
        while j < i:
            indices.append((i, j))
            j += 1
        i += 1
    return indices


def triangular_indices(size: int) -> List[Tuple[int, int]]:
    """Returns a list of tuples that contain the indices of strictly non-zero elements of a triangular matrix
    of dimension `size` x `size` """
    indices = []
    i = 0
    while i < size:
        j = 0
        while j <= i:  # `<=` to include diagonal elements
            indices.append((i, j))
            j += 1
        i += 1
    return indices


# split complex tensor into real and imaginary parts
split_complex = lambda c: (tf.math.real(c), tf.math.imag(c))

# element-wise multiply complex tensors `m1` and `m2`
complex_mul = lambda m1_real, m1_imag, m2_real, m2_imag: (
    m1_real * m2_real - m1_imag * m2_imag,
    m1_real * m2_imag + m1_imag * m2_real,
)


def correlation(m1, m2):
    """
    Calculates the discrete cross-correlation between complex tensors `m1` and `m2`.
    :param m1: complex tensor
    :param m2: complex tensor with same shape as m2
    :return: complex scalar
    """

    m1_real, m1_imag = split_complex(m1)
    m2_real, m2_imag = split_complex(m2)

    # complex_conjugate(m1)*m2
    corr_real, corr_imag = complex_mul(m1_real, -m1_imag, m2_real, m2_imag)

    return tf.reduce_sum(tf.complex(corr_real, corr_imag))


def correlation_matrix(basis_set):
    """
    Takes a list of 2D complex matrices calculates the correlation between each matrix. It then generates a correlation
    matrix where elements contain correlations. Since correlation(a, b) = complex_conjugate(correlation(b, a)), the
    output can be completely represented using a triangular matrix

    :param basis_set: Rank 3 tensor. The first dimension enumerates the set of function. The second and third dimensions
    are the axes of the matrix
    :return: complex triangular matrix containing the correlations between matrices
    """

    basis_n, basis_size, basis_size = np.shape(basis_set)
    indices = triangular_indices(basis_n)

    correlation_matrix = np.zeros(shape=(basis_n, basis_n), dtype=np.complex128)

    for i, j in indices:
        correlation_matrix[i, j] = correlation(basis_set[i, :, :], basis_set[j, :, :])

    return correlation_matrix


def crosscorrelation(basis_set):
    """
    Calculates the sum of the norm(correlation) between the matrices in `basis_set`

    :param basis_set: Rank 3 tensor. The first dimension enumerates the set of function. The second and third dimensions
    are the axes of the matrix
    :return: Real valued scalar
    """

    basis_n, basis_size, _ = np.shape(basis_set)

    cum_sum = 0.
    for i, j in strictly_triangular_indices(basis_n):
        cum_sum += tf.math.abs(correlation(basis_set[i, :, :], basis_set[j, :, :]))

    return cum_sum


# TODO: Put this in a test file. Write more tests!
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


def nearest_rectangle(a: int) -> Tuple[int, int]:
    """
    Helper method for plotting `a` figures on a `n` by `m` grid. Searches of an `n` and `m` where `m`*`n` = `a` while
    minimizing abs(`m`-`n`)
    :param a: `a`
    :return: the tuple (`n`, `m`)
    """
    a = int(a)
    upper = int(np.sqrt(a))
    lower = int(np.sqrt(a))

    factor = None
    while True:
        if a / lower == a // lower:
            factor = lower
            break
        if a / upper == a // upper:
            factor = upper
            break
        lower -= 1
        upper += 1

    factor_pair = factor, a // factor
    return min(factor_pair), max(factor_pair)


def plot_modes(set):
    n_modes = set.shape[0]
    plot_shape = nearest_rectangle(n_modes)
    fig, axs = plt.subplots(*plot_shape)
    for i in range(n_modes):
        ax = axs[np.unravel_index(i, shape=plot_shape)]
        im = ax.imshow(np.abs(set[i, :, :]))
        plt.axis('off')
        ax.axis('off')
        fig.colorbar(im, ax=ax)
    plt.show()


def plot_modes_fft(set):
    fft = np.fft.fftshift(np.fft.fft2(set, axes=[1, 2]))
    plot_modes(fft)


def plot_slice(image, title):
    f = plt.figure()
    half_w = sim_args['lens_aperture'] / 2. / 1e-6
    plt.imshow(image,
               extent=[-half_w, half_w, -half_w, half_w])
    plt.xlabel(r'x ($\mu m$)')
    plt.ylabel(r'y ($\mu m$)')
    plt.title(title)
    plt.colorbar()
    plt.show()


def apply_low_pass_filter(basis_set_real, basis_set_imag, pad_x, pad_y):
    basis_set = tf.complex(basis_set_real, basis_set_imag)
    basis_n, basis_size, _ = np.shape(basis_set)

    # generate kernel
    pass_hole_size_x = basis_size - 2 * pad_x
    pass_hole_size_y = basis_size - 2 * pad_y

    kernel_fd = tf.ones(shape=(pass_hole_size_y, pass_hole_size_x), dtype=dtype['comp'])
    kernel_fd = tf.pad(kernel_fd, [[pad_y, pad_y], [pad_x, pad_x]])
    kernel_fd = tf.signal.ifftshift(kernel_fd, axes=[0, 1])
    basis_set_fd = tf.signal.fft2d(basis_set)

    filtered_basis_fd = tf.complex(
        *complex_mul(
            *split_complex(kernel_fd), *split_complex(basis_set_fd)
        )
    )

    filtered_basis = tf.signal.ifft2d(filtered_basis_fd)
    return split_complex(filtered_basis)


def complex_initializer_random(shape):
    """
    Generates a complex valued tensor with values that are distributed with uniform density in the unit circle of the
    complex plane
    """
    return np.sqrt(np.random.rand(*shape)) * np.exp(1j * np.random.rand(*shape) * np.pi * 2)


@tf.function
def forward(weights):
    def prop(x):
        """Propagates fields `prop_distantce`"""
        # return ap.propagate_angular_bw_limited(field=x, k=sim_args['k'], z_list=[sim_args['prop_distance'], ],
        #                                       dx=sim_args['dd'],
        #                                       dy=sim_args['dd'],
        #                                       )[0, :, :]

        return ap.propagate_padded(
            propagator=ap.propagate_angular_bw_limited,
            field=x, k=sim_args['k'], z_list=[sim_args['prop_distance'], ],
            dx=sim_args['dd'],
            dy=sim_args['dd'],
            pad_factor=1.,
        )[0, :, :]

    def lens_f(A, f):
        """Apply lens phase profile to field with focal length `prop_distance`"""
        dd = sim_args['dd']
        slm_size = sim_args['slm_size']
        k = sim_args['k']

        x = tf.linspace(-dd * slm_size / 2., dd * slm_size / 2., slm_size)
        y = tf.linspace(-dd * slm_size / 2., dd * slm_size / 2., slm_size)
        Y, X = tf.meshgrid(y, x, indexing='xy')

        Y = tf.cast(Y, dtype=dtype['real'])
        X = tf.cast(X, dtype=dtype['real'])

        phi = -k * tf.sqrt(f ** 2 + X ** 2 + Y ** 2)
        field = tf.complex(tf.cos(phi), tf.sin(phi))
        lensed_field = tf.complex(
            *complex_mul(
                *split_complex(A),
                *split_complex(field),
            )
        )
        return lensed_field

    def lens_full(A):
        return lens_f(A, sim_args['prop_distance'])

    def lens_half(A):
        return lens_f(A, sim_args['prop_distance'] / 2.)

    def meta1(A):
        field_after = tf.complex(
            *complex_mul(
                *split_complex(A),
                metasurface1_real, metasurface1_imag,
            )
        )
        return field_after

    def meta2(A):
        field_after = tf.complex(
            *complex_mul(
                *split_complex(A),
                metasurface2_real, metasurface2_imag,
            )
        )
        return field_after

    field_set = fg(weights)

    field_set = tf.map_fn(fn=prop, elems=field_set)
    # field_set = tf.map_fn(fn=lens_full, elems=field_set)
    #field_set = tf.map_fn(fn=prop, elems=field_set)

    field_set = tf.map_fn(fn=meta1, elems=field_set)

    field_set = tf.map_fn(fn=prop, elems=field_set)
    field_set = tf.map_fn(fn=meta2, elems=field_set)
    # field_set = tf.map_fn(fn=lens_full, elems=field_set)
    field_set = tf.map_fn(fn=prop, elems=field_set)
    return field_set


# @tf.function
def loss_orthogonal():
    """Loss function to maximize orthogonality between output modes...TODO: Add docs."""
    propagated_fields = forward(weights)
    propagated_fields_real, propagated_fields_imag = split_complex(propagated_fields)

    slm_size = sim_args['slm_size']
    filter_width = sim_args['filter_width']
    filter_height = sim_args['filter_height']

    # low pass filter
    filtered_prop_fields_real, filtered_prop_fields_imag = apply_low_pass_filter(
        propagated_fields_real,
        propagated_fields_imag,
        pad_x=(slm_size - filter_width) // 2,
        pad_y=(slm_size - filter_height) // 2,

    )

    mode_overlap = crosscorrelation(tf.complex(propagated_fields_real, propagated_fields_imag))

    intensity = filtered_prop_fields_real ** 2 + filtered_prop_fields_imag ** 2
    # energy is low frequency modes
    energy_set = tf.reduce_sum(intensity, axis=[1, 2])
    total_energy = tf.reduce_sum(tf.sqrt(energy_set))

    return mode_overlap / total_energy  # mode_overlap - 100.*total_energy
    # return mode_overlap #- total_energy


def loss_binned():
    """Loss function to map power from each input mode to a bin in the output field"""
    propagated_fields = forward(weights)
    propagated_intensities = tf.math.abs(propagated_fields) ** 2
    n_i_bins = sim_args['n_i_bins']
    n_j_bins = sim_args['n_j_bins']

    n_modes, ni, nj = propagated_fields.shape

    i_bin_size = ni // n_i_bins
    j_bin_size = nj // n_j_bins

    def sums(i):
        ibin = i // n_i_bins
        jbin = i % n_j_bins
        return tf.reduce_sum(propagated_intensities[
                             i,
                             ibin * i_bin_size:(ibin + 1) * i_bin_size,
                             jbin * j_bin_size:(jbin + 1) * j_bin_size,
                             ])

    i = tf.range(n_modes, dtype=tf.int64)

    return -tf.reduce_prod(
        tf.math.pow(
            tf.map_fn(sums, i, dtype=propagated_intensities.dtype),
            1. / n_modes  # TODO: use a different normalization.
            # Normalization is necessary or the sum exponent will
            # underflow for larger number of modes. Roots, however, are expensive to calculate. I could perhaps fix
            # this by changing the input power based on the number of modes/bins so each sum is approxmatly 1. This
            # should prevent float exponent underflow because the products of many ~1 should remain close to 1.
        )
    )


# @tf.function
def train_step():
    # Variables to be trained
    train_vars = [metasurface1_real, metasurface1_imag, metasurface2_real, metasurface2_imag, ]

    with tf.GradientTape() as tape:
        # current_loss = orthogonal_loss()
        current_loss = loss_binned()
    grads = tape.gradient(target=current_loss, sources=train_vars)

    # update weights
    optimizer.apply_gradients(zip(grads, train_vars))

    def clip_magnitudes(real, imag):
        """
        If the value of any element is outside the unit circle on the complex plane, the value is moved onto the unit
        circle while keeping the same angle.
        """
        phase = tf.math.atan2(imag, real)
        mag = tf.sqrt(real ** 2 + imag ** 2)
        mag = tf.clip_by_value(mag, clip_value_min=0., clip_value_max=1.)
        imag.assign(mag * tf.sin(phase))
        real.assign(mag * tf.cos(phase))

    clip_magnitudes(metasurface1_real, metasurface1_imag)
    clip_magnitudes(metasurface2_real, metasurface2_imag)

    def low_pass(real, imag):
        """Applies a spatial low pass filter to an array"""
        filter_width = 20
        n, _ = real.shape
        pad = (n - filter_width) // 2
        real_filtered, imag_filtered = apply_low_pass_filter(tf.expand_dims(real, axis=0), tf.expand_dims(imag, axis=0),
                                                             pad_x=pad, pad_y=pad)
        real.assign(real_filtered[0, ...])
        imag.assign(imag_filtered[0, ...])

    # low_pass(metasurface1_real, metasurface1_imag)
    # low_pass(metasurface2_real, metasurface2_imag)

    return tf.reduce_mean(current_loss)  # return loss for logging


if __name__ == '__main__':
    # Set up logging.
    # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # logdir = 'logs\\simple_angular_prop_tf\\%s' % stamp
    # writer = tf.summary.create_file_writer(logdir)
    # tf.summary.trace_on(graph=True, profiler=True)

    # define the datatype to be used in this simulation
    # The complex datatype must have double the bits of the float or you'll get casting errors (since a complex is
    # just 2 floats)
    # using float32/complex64 was a little bit faster than float64/complex128
    dtype = {'real': tf.float32, 'comp': tf.complex64, }

    sim_args = {
        'wavelength': 633e-9,  # HeNe
        'slm_size': 282,  # side length of the array
    }

    sim_args = {
        **sim_args,
        'lens_aperture': 250. * sim_args['wavelength'],
    }

    sim_args = {
        **sim_args,
        **{
            'k': 2. * np.pi / sim_args['wavelength'],
            'prop_distance': 250. * sim_args['wavelength'],

            'slm_shape': (sim_args['slm_size'], sim_args['slm_size'],),
            'dd': sim_args['lens_aperture'] / sim_args['slm_size'],  # array element spacing
            # allows `filter_height` * `filter_width` number of orthogonal modes through
            # 'filter_height': 7,
            # 'filter_width': 7,
            'n_i_bins': 7,
            'n_j_bins': 7,

            'n_modes': 49,
        }
    }

    # initialize the 1D SLM basis set.
    weights = tf.Variable(tf.ones(sim_args['n_modes'], dtype=dtype['comp']))

    slm_args = {'n_weights': sim_args['n_modes'], 'pixel_width': 2, 'pixel_height': 2, 'pixel_spacing': 3,
                'end_spacing': 20, 'dtype': dtype['comp']}
    fg = oneD_slm_field_generator.OneDPhasorField(**slm_args)

    # Make sure that this input modes shape match the simulation shape
    assert fg.n == sim_args['slm_size'], "SLM field and simulation field mismatch. Adjust the 1D Slm structure. ({} " \
                                         "vs {})".format(fg.n, sim_args['slm_size'])

    # initialize and define first metasurface
    metasurface1_real, metasurface1_imag = split_complex(complex_initializer_random(shape=sim_args['slm_shape']))
    metasurface1_real = tf.Variable(tf.cast(metasurface1_real, dtype=dtype['real']), dtype=dtype['real'],
                                    trainable=True)
    metasurface1_imag = tf.Variable(tf.cast(metasurface1_imag, dtype=dtype['real']), dtype=dtype['real'],
                                    trainable=True)

    # initialize and define second metasurface
    metasurface2_real, metasurface2_imag = split_complex(complex_initializer_random(shape=sim_args['slm_shape']))
    metasurface2_real = tf.Variable(tf.cast(metasurface2_real, dtype=dtype['real']), dtype=dtype['real'],
                                    trainable=True)
    metasurface2_imag = tf.Variable(tf.cast(metasurface2_imag, dtype=dtype['real']), dtype=dtype['real'],
                                    trainable=True)


    optimizer = tf.optimizers.Adam(learning_rate=0.1)
    # Training loop
    iterations = 2 ** 14
    n_update = 2 ** 4  # Updates information every `n_update` iterations

    # Training loop
    t = time.time()
    for i in range(iterations):
        error = train_step()
        if i % n_update == 0:
            # field_log.append(metasurface_real.numpy() + 1j*metasurface_imag.numpy())
            t_now = time.time()
            print("Error: {}\tTimePerUpdate(s): {}\t {}/{}".format(error, t_now - t, i + 1, iterations))
            t = t_now


    # Simulation complete. Now plotting results.
    fields = fg(weights)
    plt.pcolormesh(tf.abs(tf.reduce_sum(fields, axis=0)).numpy(), edgecolors='k')
    plt.show()

    plot_modes(fields)
    plt.show()

    plot_slice(np.angle(metasurface1_real.numpy() + 1j * metasurface1_imag.numpy()), "Optimized metasurface 1 phase")
    plot_slice(np.abs(metasurface1_real.numpy() + 1j * metasurface1_imag.numpy()), "Optimized metasurface 1 magnitude")

    plot_slice(np.angle(metasurface2_real.numpy() + 1j * metasurface2_imag.numpy()), "Optimized metasurface 2 phase")
    plot_slice(np.abs(metasurface2_real.numpy() + 1j * metasurface2_imag.numpy()), "Optimized metasurface 2 magnitude")

    plot_modes(forward(weights).numpy())
    plt.show()

    # plot_modes_fft(forward(weights))
    # plt.show()

    # plot abs correlation matrix
    plt.figure()
    plt.imshow(
        np.abs(
            correlation_matrix(forward(weights))
        )
    )
    plt.show()

    propagation = forward(weights)
    # filtered_prop_fields = tf.complex(
    #     *apply_low_pass_filter(
    #         *split_complex(propagation),
    #         pad_x=(sim_args['slm_size']-sim_args['filter_width'])//2,
    #         pad_y=(sim_args['slm_size']-sim_args['filter_width'])//2,
    #
    #     )
    # )
    # plot_modes(filtered_prop_fields.numpy())
    # plot_modes_fft(filtered_prop_fields.numpy())

    # # plot abs correlation matrix
    # plt.figure()
    # plt.imshow(
    #     np.abs(
    #         correlation_matrix(filtered_prop_fields)
    #     )
    # )
    # plt.show()



    #save fields
    data = {
        'oneDSLMArgs': slm_args,
        'sim_args': sim_args,
        'metasurfaces': {
            'real1': metasurface1_real,
            'imag1': metasurface1_imag,
            'real2': metasurface2_real,
            'imag2': metasurface2_imag,
        },
        'forward': propagation.numpy(),
        # 'forward_filtered': filtered_prop_fields.numpy(),
        'weights': weights.numpy(),
        'inputs_modes': fg(weights).numpy(),
    }

    import pickle
    pickle.dump(data, open("two_ms.p", "wb"))
