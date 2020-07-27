import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from propagated_basis_optimization import complex_mul, split_complex

class OneDPhasorField(tf.Module):
    def __init__(self, n_weights, pixel_width, pixel_height, pixel_spacing, end_spacing, dtype):
        assert isinstance(pixel_width, int)

        assert isinstance(pixel_height, int)
        assert isinstance(pixel_spacing, int)
        assert dtype == tf.complex64 or dtype == tf.complex128, "Dtype must be a complex datatype"

        self.dtype = dtype
        self.n_weights = n_weights

        n = n_weights * pixel_height + (n_weights - 1) * pixel_spacing + 2*end_spacing# array side length
        self.n = n
        self.field = tf.Variable(tf.zeros(shape=(n, n, n_weights,), dtype=dtype), trainable=False) # Variable supports slice assign

        for i in range(n_weights):
            i_x = (n + 1) // 2
            i_y = i * (pixel_spacing + pixel_height) + end_spacing + (pixel_height + 1) // 2
            weight_cell = tf.ones(shape=(pixel_height, pixel_width,), dtype=dtype)
            self.field[
                int(i_y - pixel_height / 2.): int(i_y + pixel_height / 2.),
                int(i_x - pixel_width / 2.): int(i_x + pixel_width / 2.),
                i,
            ].assign(weight_cell)


    @tf.function
    def __call__(self, weights):
        weights = tf.cast(weights, self.dtype)

        real, imag = complex_mul(
            *split_complex(self.field),
            *split_complex(weights),
        )

        real = tf.transpose(real, perm=(2, 0, 1))
        imag = tf.transpose(imag, perm=(2, 0, 1))

        return tf.complex(real, imag)
        #return tf.matmul(self.field, tf.expand_dims(weights, axis=-1))[..., 0]

if __name__=='__main__':

    fg = OneDPhasorField(n_weights=4, pixel_width=10, pixel_height=4, pixel_spacing=5, end_spacing=6, dtype=tf.complex128)
    fields = fg(tf.constant([1.0, 1.3, 1.6, 1.9,], dtype=tf.complex128))

    plt.pcolormesh(tf.abs(tf.reduce_sum(fields, axis=0)).numpy(), edgecolors='k', linewidth=2)
    # ax = plt.gca()
    # ax.set_aspect('equal')
    plt.show()
