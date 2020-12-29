import logging
import os
import pickle
import time

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import src.oneD_slm_field_generator as oneD_slm_field_generator
from src.tools import split_complex, complex_mul, triangular_indices, strictly_triangular_indices
from AngularPropagateTensorflow import AngularPropagateTensorflow as ap

import tensorflow as tf

# tf.debugging.set_log_device_placement(True)

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


def plot_modes(set, scale=1., **kwargs):
    set = np.abs(set)
    n_modes = set.shape[0]
    plot_shape = nearest_rectangle(n_modes)
    fig, axs = plt.subplots(*plot_shape, figsize=(scale*plot_shape[0], scale*plot_shape[1]), **kwargs)
    vmax = np.max(set)
    vmin = min(np.min(set), 0.)
    for i in range(n_modes):
        ax = axs[np.unravel_index(i, shape=plot_shape)]
        im = ax.imshow(set[i, :, :], vmin=vmin, vmax=vmax)
        plt.axis('off')
        ax.axis('off')
        # fig.colorbar(im, cax=ax)
    plt.tight_layout()


    # fig.tight_layout(pad=0.01)
    # fig.colorbar(im, ax=axs[:, :], shrink=1.)
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7]) # [left bottom width height]
    #fig.colorbar(im, cax=cbar_ax)


def plot_modes_fft(set):
    fft = np.fft.fftshift(np.fft.fft2(set, axes=[1, 2]))
    plot_modes(fft)


def plot_slice(image, title, sim_args):
    f = plt.figure()
    half_w = sim_args['lens_aperture'] / 2. / 1e-6
    plt.imshow(
        image,
        extent=[-half_w, half_w, -half_w, half_w],
        interpolation='none'
    )

    ###
    # plt.pcolormesh(image)
    # ax = plt.gca()
    # # ax.pcolorfast(image)
    # ax.set_aspect('equal')
    ###

    plt.xlabel(r'x ($\mu m$)')
    plt.ylabel(r'y ($\mu m$)')
    plt.title(title)


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
def forward(weights, field_generator, sim_args, metasurface1_phase, metasurface2_phase):
    def prop(x, distance):
        """Propagates fields `prop_distantce`"""
        return ap.propagate_padded(
            propagator=ap.propagate_angular_bw_limited,
            field=x, k=sim_args['k'], z_list=[distance, ],
            dx=sim_args['dd'],
            dy=sim_args['dd'],
            pad_factor=1.,
        )[0, :, :]

    def prop_1d_to_ms1(weights):
        return prop(weights, sim_args['spacing_1d_to_ms1'])

    def prop_ms1_to_ms2(weights):
        return prop(weights, sim_args['spacing_ms1_to_ms2'])

    def prop_ms1_to_detector(weights):
        return prop(weights, sim_args['spacing_ms2_to_detector'])


    # def lens_f(A, f):
    #     """Apply lens phase profile to field with focal length `prop_distance`"""
    #     dd = sim_args['dd']
    #     slm_size = sim_args['slm_size']
    #     k = sim_args['k']
    #
    #     x = tf.linspace(-dd * slm_size / 2., dd * slm_size / 2., slm_size)
    #     y = tf.linspace(-dd * slm_size / 2., dd * slm_size / 2., slm_size)
    #     Y, X = tf.meshgrid(y, x, indexing='xy')
    #
    #     Y = tf.cast(Y, dtype=dtype['real'])
    #     X = tf.cast(X, dtype=dtype['real'])
    #
    #     phi = -k * tf.sqrt(f ** 2 + X ** 2 + Y ** 2)
    #     field = tf.complex(tf.cos(phi), tf.sin(phi))
    #     lensed_field = tf.complex(
    #         *complex_mul(
    #             *split_complex(A),
    #             *split_complex(field),
    #         )
    #     )
    #     return lensed_field

    def meta1(A):
        # field_after = tf.complex(
        #     *complex_mul(
        #         *split_complex(A),
        #         metasurface1_real, metasurface1_imag,
        #     )
        # )
        field_after = tf.complex(
            *complex_mul(
                *split_complex(A),
                tf.math.cos(metasurface1_phase), tf.math.sin(metasurface1_phase),
            )
        )
        return field_after

    def meta2(A):
        # field_after = tf.complex(
        #     *complex_mul(
        #         *split_complex(A),
        #         metasurface2_real, metasurface2_imag,
        #     )
        # )
        field_after = tf.complex(
            *complex_mul(
                *split_complex(A),
                tf.math.cos(metasurface2_phase), tf.math.sin(metasurface2_phase),
            )
        )
        return field_after

    field_set = field_generator(weights)

    field_set = tf.map_fn(fn=prop_1d_to_ms1, elems=field_set)
    # field_set = tf.map_fn(fn=lens_full, elems=field_set)
    #field_set = tf.map_fn(fn=prop, elems=field_set)

    field_set = tf.map_fn(fn=meta1, elems=field_set)

    field_set = tf.map_fn(fn=prop_ms1_to_ms2, elems=field_set)
    field_set = tf.map_fn(fn=meta2, elems=field_set)
    # field_set = tf.map_fn(fn=lens_full, elems=field_set)
    field_set = tf.map_fn(fn=prop_ms1_to_detector, elems=field_set)
    return field_set


# @tf.function
def loss_orthogonal(field_generator, metasurface1_phase, metasurface2_phase):
    """Loss function to maximize orthogonality between output modes...TODO: Add docs."""
    propagated_fields = forward(weights, field_generator, sim_args, metasurface1_phase, metasurface2_phase)
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


def loss_binned(weights, field_generator, sim_args, metasurface1_phase, metasurface2_phase):
    """Loss function to map power from each input mode to a bin in the output field"""
    propagated_fields = forward(
        weights, field_generator, sim_args, metasurface1_phase, metasurface2_phase
    )
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
def train_step(metasurface1_phase, metasurface2_phase, weights, field_generator, sim_args, optimizer):
    # Variables to be trained
    train_vars = [metasurface1_phase, metasurface2_phase, ] #  [metasurface1_real, metasurface1_imag, metasurface2_real, metasurface2_imag, ]
    with tf.GradientTape() as tape:
        # current_loss = orthogonal_loss()
        current_loss = loss_binned(
            weights, field_generator, sim_args, metasurface1_phase, metasurface2_phase
        )
    grads = tape.gradient(target=current_loss, sources=train_vars)
    # update weights
    optimizer.apply_gradients(zip(grads, train_vars))

    # def clip_magnitudes(real, imag):
    #     """
    #     If the value of any element is outside the unit circle on the complex plane, the value is moved onto the unit
    #     circle while keeping the same angle.
    #     """
    #     phase = tf.math.atan2(imag, real)
    #     mag = tf.sqrt(real ** 2 + imag ** 2)
    #     mag = tf.clip_by_value(mag, clip_value_min=0., clip_value_max=1.)
    #     imag.assign(mag * tf.sin(phase))
    #     real.assign(mag * tf.cos(phase))

    # clip_magnitudes(metasurface1_real, metasurface1_imag)
    # clip_magnitudes(metasurface2_real, metasurface2_imag)


    # @tf.function
    # def low_pass(real, imag):
    #     """Applies a spatial low pass filter to an array"""
    #     filter_width = 150
    #     n, _ = real.shape
    #     pad = (n - filter_width) // 2
    #     real_filtered, imag_filtered = apply_low_pass_filter(tf.expand_dims(real, axis=0), tf.expand_dims(imag, axis=0),
    #                                                          pad_x=pad, pad_y=pad)
    #     real.assign(real_filtered[0, ...])
    #     imag.assign(imag_filtered[0, ...])
    #
    # metasurface1_real, metasurface1_imag = tf.math.cos(metasurface1_phase), tf.math.sin(metasurface1_phase)
    # metasurface2_real, metasurface2_imag = tf.math.cos(metasurface2_phase), tf.math.sin(metasurface2_phase)
    #
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
        # 'slm_size': 473,  # defines a simulation region of `slm_size` by `slm_size`
        'slm_size': 400,  # defines a simulation region of `slm_size` by `slm_size`
    }

    sim_args = {
        **sim_args,
        'lens_aperture': 2.0e-3,
    }

    sim_args = {
        **sim_args,
        **{
            'k': 2. * np.pi / sim_args['wavelength'],

            #'spacing_1d_to_ms1': sim_args['lens_aperture']/sim_args['wavelength']*8e-6, #NA so d_min is lambda
            'spacing_1d_to_ms1': 2.0e-3,
            'spacing_ms1_to_ms2': 1.5 * 1e-3, # thickness of glass substrate
            'spacing_ms2_to_detector': 20e-3,


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
    # 1D SLM pixel coeffients
    # weighs is a constant TODO: Make constants
    weights = tf.Variable(tf.ones(sim_args['n_modes'], dtype=dtype['comp']))


    slm_args = {
        'n_weights': sim_args['n_modes'],
        'pixel_width': 2,
        'pixel_height': 2,
        'pixel_spacing': 5,
        'dtype': dtype['comp']}
    slm_args['end_spacing'] = (sim_args['slm_size'] - (
            sim_args['n_modes']*slm_args['pixel_height'] + (sim_args['n_modes'] - 1) * slm_args['pixel_spacing']
    )) // 2

    assert slm_args['end_spacing'] >= 0, "Bounds error"

    logging.warning(slm_args)
    logging.warning(sim_args)

    field_generator = oneD_slm_field_generator.OneDPhasorField(**slm_args)

    # Make sure that this input modes shape match the simulation shape
    assert field_generator.n == sim_args['slm_size'], "SLM field and simulation field mismatch. Adjust the 1D Slm structure. ({} " \
                                         "vs {})".format(field_generator.n, sim_args['slm_size'])


    metasurface1_phase = tf.Variable(
        tf.random.uniform(shape=sim_args['slm_shape'], maxval=np.pi*2., dtype=dtype['real']),
        trainable=True,
    )
    metasurface2_phase = tf.Variable(
        tf.random.uniform(shape=sim_args['slm_shape'], maxval=np.pi*2., dtype=dtype['real']),
        trainable=True,
    )

    #
    # # initialize and define first metasurface
    # metasurface1_real, metasurface1_imag = split_complex(complex_initializer_random(shape=sim_args['slm_shape']))
    # metasurface1_real = tf.Variable(tf.cast(metasurface1_real, dtype=dtype['real']), dtype=dtype['real'],
    #                                 trainable=True)
    # metasurface1_imag = tf.Variable(tf.cast(metasurface1_imag, dtype=dtype['real']), dtype=dtype['real'],
    #                                 trainable=True)
    #
    # # initialize and define second metasurface
    # metasurface2_real, metasurface2_imag = split_complex(complex_initializer_random(shape=sim_args['slm_shape']))
    # metasurface2_real = tf.Variable(tf.cast(metasurface2_real, dtype=dtype['real']), dtype=dtype['real'],
    #                                 trainable=True)
    # metasurface2_imag = tf.Variable(tf.cast(metasurface2_imag, dtype=dtype['real']), dtype=dtype['real'],
    #                                 trainable=True)


    optimizer = tf.optimizers.Adam(learning_rate=0.1)
    # Training loop
    iterations = 2 ** 8
    n_update = 2 ** 4  # Prints information every `n_update` iterations

    # Training loop
    t = time.time()
    for i in range(iterations):
        error = train_step(
            metasurface1_phase, metasurface2_phase, weights, field_generator, sim_args, optimizer
        )
        if i % n_update == 0:
            t_now = time.time()
            print("Error: {}\tTimePerUpdate(s): {}\t {}/{}".format(error, t_now - t, i + 1, iterations))
            t = t_now

    metasurface1_real, metasurface1_imag = tf.math.cos(metasurface1_phase), tf.math.sin(metasurface1_phase)
    metasurface2_real, metasurface2_imag = tf.math.cos(metasurface2_phase), tf.math.sin(metasurface2_phase)


    plotting=True
    if plotting:
        figure_dir = os.path.join("..", "figures")
        assert os.path.isdir(figure_dir), "Folder '{}' doesn't exist".format(figure_dir)

        # Phase plot should use a cyclic color map: ['twilight', 'twilight_shifted', 'hsv']


        # Simulation complete. Now plotting results.
        fields = field_generator(weights)
        plot_slice(
            tf.abs(tf.reduce_sum(fields, axis=0)).numpy(),
            title="",
            sim_args=sim_args
        )
        plt.colorbar()
        plt.set_cmap('magma')
        plt.savefig(os.path.join(figure_dir, "input_SLM.pdf"), format='pdf', dpi=1000)
        plt.show()
        #
        # # plot individual incoming modes
        # plot_modes(fields, 5)
        # plt.set_cmap('jet')
        # plt.savefig(os.path.join(figure_dir, "incoming_modes.pdf"), format='pdf', dpi=1000)
        # plt.show()

        # plot ms1 phase
        plot_slice(
            np.angle(metasurface1_real.numpy() + 1j * metasurface1_imag.numpy()),
            "",
            sim_args,
        )
        plt.set_cmap('twilight')
        plt.colorbar()
        plt.savefig(os.path.join(figure_dir, "ms1.pdf"), format='pdf', dpi=1000)
        plt.show()

        # plot ms2 phase
        plot_slice(
            np.angle(metasurface2_real.numpy() + 1j * metasurface2_imag.numpy()),
            "",
            sim_args,
        )
        plt.set_cmap('twilight')
        plt.colorbar()
        plt.savefig(os.path.join(figure_dir, "ms2.pdf"), format='pdf', dpi=1000)
        plt.show()

        plot_modes(forward(
            weights, field_generator, sim_args, metasurface1_phase, metasurface2_phase
        ).numpy(), 5)
        plt.set_cmap('magma')
        plt.savefig(os.path.join(figure_dir, "propagated_modes.pdf"), format='pdf', dpi=1000)
        plt.show()
        plt.set_cmap('magma')

        # plot_modes_fft(forward(
        #   weights, field_generator, sim_args, metasurface1_phase, metasurface2_phase
        # ))
        # plt.show()

        # plot abs correlation matrix
        plt.figure()
        plt.imshow(
            np.abs(
                correlation_matrix(forward(
                    weights, field_generator, sim_args, metasurface1_phase, metasurface2_phase
                ))
            )
        )
        plt.savefig(os.path.join(figure_dir, "correlation.pdf"), format='pdf', dpi=1000)
        plt.colorbar()
        plt.show()

        p_weights = np.zeros(49)
        # p_indicies = [8, 9, 11, 12, 15, 16, 18, 19, 29, 33, 37, 38, 39, ]
        p_indicies = [2, 4, 9, 11, 16, 17, 18, 29, 31, 33, 36, 38, 40, 43, 44, 45, 46, 47, ]
        for i in p_indicies:
            p_weights[i] = 1
        plot_slice(
            tf.math.abs(tf.reduce_sum(forward(
                tf.constant(p_weights), field_generator, sim_args, metasurface1_phase, metasurface2_phase
            ), axis=0)) ** 2,
            title="",
            sim_args=sim_args
        )
        plt.savefig(os.path.join(figure_dir, "uw_input.pdf"), format='pdf', dpi=1000)
        plt.show()
        p_fields = field_generator(p_weights)
        plot_slice(
            tf.abs(tf.reduce_sum(p_fields, axis=0)).numpy(),
            title="",
            sim_args=sim_args,
        )
        plt.savefig(os.path.join(figure_dir, "uw_propagated.pdf"), format='pdf', dpi=1000)
        plt.show()
        a = np.zeros(shape=(7, 7))
        for i in p_indicies:
            a[i % 7, i // 7] = 1.
        # plt.imshow(np.transpose(a))
        plot_slice(
            np.transpose(a),
            title="",
            sim_args=sim_args,
        )
        plt.colorbar()
        plt.savefig(os.path.join(figure_dir, "uw_target.pdf"), format='pdf', dpi=1000)
        plt.show()


        propagation = forward(
            weights, field_generator, sim_args, metasurface1_phase, metasurface2_phase
        )
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
        'forward': forward(
            weights, field_generator, sim_args, metasurface1_phase, metasurface2_phase
        ).numpy(),
        # 'forward_filtered': filtered_prop_fields.numpy(),
        'weights': weights.numpy(),
        'inputs_modes': field_generator(weights).numpy(),
    }

    dir = "./data"
    if not os.path.exists(dir):
        os.mkdir(dir)
    pickle.dump(data, open(os.path.join(dir, "two_ms.p"), "wb"))

    #
