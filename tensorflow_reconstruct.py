

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from propagated_basis_optimization import
from propagated_basis_optimization import complex_initializer_random, complex_mul, split_complex, plot_modes

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


def dummy_image(slm_size):
    imag = np.zeros(shape=(slm_size, slm_size))
    hw = slm_size*7//16
    imag[hw:-hw, :] = 1.
    imag[:, hw:-hw] = 1.
    x = np.linspace(0., slm_size, slm_size) - slm_size/2
    y = x
    X, Y= np.meshgrid(x, y)
    imag[np.sqrt(X**2 + Y**2) < slm_size/4]=1.
    return imag

def plot_complex(image):
    fig, axs = plt.subplots(2)
    axs[0].imshow(np.abs(image))
    axs[0].axis('off')
    # plt.colorbar(ax=axs[0])
    axs[1].imshow(np.angle(image))
    axs[1].axis('off')
    # plt.colorbar()
    plt.show()

def plot_filter(slm_size):
    image = dummy_image()
    image_fft = np.fft.fftshift(np.fft.fft2(image))

    filter = np.zeros(shape=image.shape)
    filter_width = 5
    filter[
        int(slm_size/2 - filter_width//2): int(slm_size/2 + filter_width//2 + 1),
        int(slm_size/2 - filter_width//2): int(slm_size/2 + filter_width//2 + 1),
        ] = 1.
    image_filtered_fft = image_fft * filter

    plot_complex(image)
    plot_complex(image_fft)
    plot_complex(image_filtered_fft)
    plot_complex(np.fft.ifft2(np.fft.ifftshift(image_filtered_fft)))

def generate_fourier_modes(slm_size):
    n = slm_size
    pad_n = 80
    hole_size = (n - 2 * pad_n)

    print("generateing {} modes".format(hole_size**2))


    filter = np.ones(shape=(hole_size, hole_size))
    filter = np.pad(filter, pad_width=pad_n, mode='constant')

    modes = []

    i_list, j_list = np.nonzero(filter)
    for i, j in zip(i_list, j_list):
        fd = np.zeros(shape=(n, n), dtype=np.complex128)
        fd[i, j] = 1.
        mode = np.fft.ifft2(np.fft.ifftshift(fd))
        modes.append(mode)

    modes = np.array(modes)
    # plot_modes(np.real(modes))
    # plot_modes(np.imag(modes))

    return np.transpose(modes, axes=(1, 2, 0))

def load_composed_modes():
    import pickle
    data = pickle.load(open("two_ms.p", "rb"))
    # return tf.transpose(data['forward'], perm=(1, 2, 0))
    return tf.transpose(data['forward'], perm=(1, 2, 0))

if __name__=='__main__':
    @tf.function
    def forward():
        image_modes_real, image_modes_imag = complex_mul(
            modes_real, modes_imag,
            weights_real, weights_imag
        )

        image_field_real = tf.reduce_sum(image_modes_real, axis=-1)
        image_field_imag = tf.reduce_sum(image_modes_imag, axis=-1)

        image_intensity = image_field_real**2 + image_field_imag**2
        return image_intensity

    def loss():
        f = forward()
        ax1.clear()
        ax1.imshow(f)

        frames.append(f.numpy())

        #return tf.reduce_sum(tf.abs(image-f)**2) #
        return -tf.reduce_sum(image * f)/n_weights


    def update(_):
        with tf.GradientTape() as tape:
            tape.watch(weights_real)
            tape.watch(weights_real)
            current_loss = loss()
        grads = tape.gradient(target=current_loss, sources=[weights_real, weights_imag, ])

        # print("Grads: {}".format(grads))

        optimizer.apply_gradients(zip(grads, [weights_real, weights_imag, ]))


        mode = 'phase'
        if mode == 'intensity_unbounded': # phase is fixed, mag is arbitrary
            # make pixels intensity only, unbounded
            mag = tf.math.sqrt(weights_real ** 2 + weights_imag ** 2)
            weights_real.assign(mag)
            weights_imag.assign(weights_imag * 0.)
        elif mode == 'intensity_bounded':
            weights_real.assign(tf.clip_by_value(weights_real, 0., 1.))
            weights_imag.assign(weights_imag * 0.)
        elif mode == 'phase': # mag is fixed at 1, phase is arbitrary
            angle = tf.math.atan2(weights_imag, weights_real)
            weights_real.assign(tf.math.cos(angle))
            weights_imag.assign(tf.math.sin(angle))
        elif mode == 'arb_bound': # mag is <= 1, phase is free
            angle = tf.math.atan2(weights_imag, weights_real)
            mag = tf.math.sqrt(weights_real ** 2 + weights_imag ** 2)
            mag = tf.clip_by_value(mag, 0., 1.)
            weights_real.assign(mag*tf.math.cos(angle))
            weights_imag.assign(mag*tf.math.sin(angle))

        print(current_loss.numpy())



    modes = generate_fourier_modes(slm_size=165)
    # modes = tf.concat(7*[modes, ], axis=-1) # apply degeneracy
    # modes = load_composed_modes()


    modes_real = np.real(modes)
    modes_imag = np.imag(modes)

    _, slm_size, _ = modes.shape

    image = dummy_image(slm_size)
    image = np.real(image)



    n_weights = modes.shape[-1]
    weights_real = tf.Variable(initial_value=np.random.rand(n_weights), dtype=tf.float64, trainable=True)
    weights_imag = tf.Variable(initial_value=np.random.rand(n_weights), dtype=tf.float64, trainable=True)


    optimizer = tf.optimizers.Adam(learning_rate=100)
    #optimizer = tf.optimizers.SGD()

    frames = []

    # style.use('fivethirtyeight')
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.axis('off')

    ani = animation.FuncAnimation(fig, update)
    plt.show()

    m = np.max(frames)
    frames = [np.array([frame, frame, frame])/m*255 for frame in frames]
    from array2gif import write_gif
    write_gif(frames, 'opt.gif', fps=10)
