


import tensorflow as tf
import numpy as np
import sys

from AngularPropagateTensorflow.AngularPropagateTensorflow import propagate_angular_padded

import oneD_slm_field_generator


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

complex_mul = lambda m1_real, m1_imag, m2_real, m2_imag: (
    m1_real * m2_real - m1_imag * m2_imag,
    m1_real * m2_imag + m1_imag * m2_real,
)


def correlation(m1, m2):
    m1_real, m1_imag = split_complex(m1)
    m2_real, m2_imag = split_complex(m2)

    corr_real, corr_imag = complex_mul(m1_real, -m1_imag, m2_real, m2_imag)

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

def nearest_rectangle(a):
    a = int(a)
    upper = int(np.sqrt(a))
    lower = int(np.sqrt(a))

    factor = None
    while True:
        if a/lower == a//lower:
            factor = lower
            break
        if a/upper == a//upper:
            factor = upper
            break
        lower -= 1
        upper += 1

    factor_pair = factor, a//factor
    return min(factor_pair), max(factor_pair)


def plot_modes(set):
    n_modes = set.shape[0]
    plot_shape = nearest_rectangle(n_modes)
    fig, axs = plt.subplots(*plot_shape)
    for i in range(n_modes):
        ax = axs[np.unravel_index(i, shape=plot_shape)]
        im = ax.imshow(np.abs(set[i, :, :]))
        plt.axis('off')
        fig.colorbar(im, ax=ax)
    plt.show()


def plot_modes_fft(set):
    fft = np.fft.fftshift(np.fft.fft2(set, axes=[1, 2]))
    plot_modes(fft)

def plot_slice(image, title):
    f = plt.figure()
    half_w = sim_args['lens_aperture']/2./1e-6
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
    pass_hole_size_x = basis_size - 2*pad_x
    pass_hole_size_y = basis_size - 2*pad_y

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
    return np.sqrt(np.random.rand(*shape)) * np.exp(1j * np.random.rand(*shape) * np.pi * 2)


# field_set_initializer_random = lambda:  np.sqrt(np.random.rand(n_modes, *slm_shape))*np.exp(1j*np.random.rand(n_modes, *slm_shape)*np.pi*2)
# field_set_initializer_uniform = lambda: np.ones(shape=(n_modes, *slm_shape), dtype=np.complex128)
# plt.imshow(generate_1d_slm_basis_set([1.,2.,3.], pixel_width=2, pixel_height=3, pixel_spacing=4))

# def generate_1d_slm_basis_set(coefs):
#     """
#     :param weights: List of complex coeffients that coorespond to the phasor at each pixel in the 1d slm
#     :return: Single phasor field that represents the near field after the 1D SLM
#     """
#
#     @tf.function
#     def gen_field(a):
#         return fg(a)
#
#     input_bases = tf.linalg.diag(coefs)
#     slm_fields = tf.map_fn(
#         fn=gen_field,
#         elems=input_bases
#     )
#
#     return slm_fields

@tf.function
def forward(weights):
    def prop(x):
        """Propagates fields `prop_distantce`"""
        return propagate_angular_padded(field=x, k=sim_args['k'], z_list=[sim_args['prop_distance'], ],
                                              dx=sim_args['dd'],
                                              dy=sim_args['dd'],
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

        phi = -k*tf.sqrt(f**2 + X**2 + Y**2)
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
        return lens_f(A, sim_args['prop_distance']/2.)

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

    field_set = tf.map_fn(fn=prop,      elems=field_set)
    # field_set = tf.map_fn(fn=lens_full, elems=field_set)
    field_set = tf.map_fn(fn=prop,      elems=field_set)

    field_set = tf.map_fn(fn=meta1,      elems=field_set)

    field_set = tf.map_fn(fn=prop,      elems=field_set)
    field_set = tf.map_fn(fn=meta2,      elems=field_set)
    # field_set = tf.map_fn(fn=lens_full, elems=field_set)
    field_set = tf.map_fn(fn=prop,      elems=field_set)
    return field_set

# @tf.function
def loss():
    propagated_fields = forward(weights)
    propagated_fields_real, propagated_fields_imag = split_complex(propagated_fields)

    slm_size = sim_args['slm_size']
    filter_width = sim_args['filter_width']
    filter_height = sim_args['filter_height']

    # low pass filter
    filtered_prop_fields_real, filtered_prop_fields_imag = apply_low_pass_filter(
        propagated_fields_real,
        propagated_fields_imag,
        pad_x=(slm_size-filter_width)//2,
        pad_y=(slm_size-filter_height)//2,

    )

    mode_overlap = crosscorrelation(tf.complex(propagated_fields_real, propagated_fields_imag))

    intensity = filtered_prop_fields_real**2 + filtered_prop_fields_imag**2
    # energy is low frequency modes
    energy_set = tf.reduce_sum(intensity, axis=[1, 2])
    total_energy = tf.reduce_sum(tf.sqrt(energy_set))

    return mode_overlap/total_energy #mode_overlap - 100.*total_energy
    #return mode_overlap #- total_energy

# @tf.function
def train_step():
    vars = [metasurface1_real, metasurface1_imag, metasurface2_real, metasurface2_imag,]

    with tf.GradientTape() as tape:
        # tape.watch(metasurface_real)
        # tape.watch(metasurface_imag)
        current_loss = loss()
    grads = tape.gradient(target=current_loss, sources=vars)

    # update weights
    optimizer.apply_gradients(zip(grads, vars))

    def clip_mags(real, imag):
        phase = tf.math.atan2(imag, real)
        mag = tf.sqrt(real ** 2 + imag ** 2)
        mag = tf.clip_by_value(mag, clip_value_min=0., clip_value_max=1.)
        imag.assign(mag * tf.sin(phase))
        real.assign(mag * tf.cos(phase))

    clip_mags(metasurface1_real, metasurface1_imag)
    clip_mags(metasurface2_real, metasurface2_imag)

    def low_pass(real, imag):
        filter_width = 20
        n, _ = real.shape
        pad = (n-filter_width)//2
        real_filtered, imag_filtered = apply_low_pass_filter(tf.expand_dims(real, axis=0), tf.expand_dims(imag, axis=0), pad_x=pad, pad_y=pad)
        real.assign(real_filtered[0, ...])
        imag.assign(imag_filtered[0, ...])

    # low_pass(metasurface1_real, metasurface1_imag)
    # low_pass(metasurface2_real, metasurface2_imag)

    return tf.reduce_mean(current_loss) # return loss for logging

if __name__=='__main__':
    # Set up logging.
    # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # logdir = 'logs\\simple_angular_prop_tf\\%s' % stamp
    # writer = tf.summary.create_file_writer(logdir)
    # tf.summary.trace_on(graph=True, profiler=True)

    dtype = {'real': tf.float32, 'comp': tf.complex64, }

    sim_args = {
        'wavelength': 633e-9, # HeNe
        'slm_size': 282,  # side length of the array
    }

    sim_args = {
        **sim_args,
        'lens_aperture': 250.*sim_args['wavelength'],

    }

    sim_args = {**sim_args,
        **{
            'k': 2.*np.pi/sim_args['wavelength'],
            'prop_distance': 250.*sim_args['wavelength'],

            'slm_shape': (sim_args['slm_size'], sim_args['slm_size'], ),
            'dd': sim_args['lens_aperture'] / sim_args['slm_size'],  # array element spacing
            # allows `filter_height` * `filter_width` number of orthogonal modes through
            'filter_height': 7,
            'filter_width': 7,
            'n_modes': 49,
        }
    }



    # initialize the SLM basis set.
    weights = tf.Variable(tf.ones(sim_args['n_modes'], dtype=dtype['comp']))

    #from oneD_slm_field_generator import OneDPhasorField

    slm_args = {'n_weights': sim_args['n_modes'], 'pixel_width': 2, 'pixel_height': 2, 'pixel_spacing': 3,
                'end_spacing': 20, 'dtype': dtype['comp']}
    fg = oneD_slm_field_generator.OneDPhasorField(**slm_args)


    assert fg.n == sim_args['slm_size'], "SLM field and simulation field mismatch. Adjust the 1D Slm structure. ({} vs {})".format(fg.n, sim_args['slm_size'])

    # initialize and define training variables
    metasurface1_real, metasurface1_imag = split_complex(complex_initializer_random(shape=sim_args['slm_shape']))
    metasurface1_real = tf.Variable(tf.cast(metasurface1_real, dtype=dtype['real']), dtype=dtype['real'], trainable=True)
    metasurface1_imag = tf.Variable(tf.cast(metasurface1_imag, dtype=dtype['real']), dtype=dtype['real'], trainable=True)
    # initialize and define training variables
    metasurface2_real, metasurface2_imag = split_complex(complex_initializer_random(shape=sim_args['slm_shape']))
    metasurface2_real = tf.Variable(tf.cast(metasurface2_real, dtype=dtype['real']), dtype=dtype['real'], trainable=True)
    metasurface2_imag = tf.Variable(tf.cast(metasurface2_imag, dtype=dtype['real']), dtype=dtype['real'], trainable=True)


    optimizer = tf.optimizers.Adam(learning_rate=0.1)
    # Training loop
    iterations = 2 ** 8
    n_update = 2 ** 1  # Updates information every * iterations

    # log to plot parameter convergence
    field_log = []

    # Training loop
    t = time.time()
    for i in range(iterations):
        error = train_step()
        if i % n_update == 0:
            # field_log.append(metasurface_real.numpy() + 1j*metasurface_imag.numpy())
            t_now = time.time()
            print("Error: {}\tTimePerUpdate(s): {}".format(error, t_now - t))
            t = t_now





    plot_slice(np.angle(metasurface1_real.numpy() + 1j * metasurface1_imag.numpy()), "Optimized metasurface 1 phase")
    plot_slice(np.abs(metasurface1_real.numpy() + 1j * metasurface1_imag.numpy()), "Optimized metasurface 1 magnitude")

    plot_slice(np.angle(metasurface2_real.numpy() + 1j * metasurface2_imag.numpy()), "Optimized metasurface 2 phase")
    plot_slice(np.abs(metasurface2_real.numpy() + 1j * metasurface2_imag.numpy()), "Optimized metasurface 2 magnitude")


    plot_modes(forward(weights).numpy())
    plt.show()


    plot_modes_fft(forward(weights))
    plt.show()

    # plot abs correlation matrix
    plt.figure()
    plt.imshow(
        np.abs(
            correlation_matrix(forward(weights))
        )
    )
    plt.show()




    propagation = forward(weights)
    filtered_prop_fields = tf.complex(
        *apply_low_pass_filter(
            *split_complex(propagation),
            pad_x=(sim_args['slm_size']-sim_args['filter_width'])//2,
            pad_y=(sim_args['slm_size']-sim_args['filter_width'])//2,

        )
    )
    plot_modes(filtered_prop_fields.numpy())
    plot_modes_fft(filtered_prop_fields.numpy())

    # plot abs correlation matrix
    plt.figure()
    plt.imshow(
        np.abs(
            correlation_matrix(filtered_prop_fields)
        )
    )
    plt.show()



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
        'forward_filtered': filtered_prop_fields.numpy(),
        'weights': weights.numpy(),
        'inputs_modes': fg(weights).numpy(),
    }


    import pickle
    pickle.dump(data, open( "two_ms.p", "wb" ) )

