"""
 Tools for common use
"""

from typing import List, Tuple

import tensorflow as tf

# split complex tensor into real and imaginary parts
def split_complex(c):
    return tf.math.real(c), tf.math.imag(c)

# element-wise multiply complex tensors `m1` and `m2`
def complex_mul(m1_real, m1_imag, m2_real, m2_imag):
    return m1_real * m2_real - m1_imag * m2_imag, m1_real * m2_imag + m1_imag * m2_real


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

