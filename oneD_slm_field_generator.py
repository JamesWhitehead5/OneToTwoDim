import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def generate_1d_slm(weights, pixel_width, pixel_height, pixel_spacing):
    """

    :param weights: List of complex coeffients that coorespond to the phasor at each pixel in the 1d slm
    :return: Single phasor field that represents the near field after the 1D SLM
    """

    assert isinstance(pixel_width, int)
    assert isinstance(pixel_height, int)
    assert isinstance(pixel_spacing, int)

    n = len(weights)*pixel_height + (len(weights) + 1) * pixel_spacing # array side length
    field = tf.Variable(tf.zeros(shape=(n, n), dtype=tf.complex128))

    for i, weight in enumerate(weights):
        i_x = (n + 1)//2
        i_y = i*(pixel_spacing + pixel_height) + pixel_spacing + (pixel_height+1)//2
        weight_cell = tf.ones(shape=(pixel_height, pixel_width, ), dtype=tf.complex128) * weight
        field[
            int(i_y - pixel_height/2.): int(i_y + pixel_height/2.),
            int(i_x - pixel_width/2.): int(i_x + pixel_width/2.),
        ].assign(weight_cell)

    return field

if __name__=='__main__':

    field = generate_1d_slm([1., 1., 1., 1.], pixel_width=10, pixel_height=4, pixel_spacing=4)

    plt.pcolormesh(tf.abs(field).numpy(), edgecolors='k', linewidth=2)
    # ax = plt.gca()
    # ax.set_aspect('equal')
    plt.show()
