import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../code'))
from AngularPropagateTensorflow import AngularPropagateTensorflow as ap_tf

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fname = 'data/two_ms.p'
import pickle
slm = pickle.load(open(fname, "rb"))

propagation_distance = slm['sim_args']['prop_distance']
k = slm['sim_args']['k']
dd = slm['sim_args']['dd']

n_slices = 100


def prop_distance_helper(field, zs, slices):
    for i in range(n_slices):
        slices[i, :, :].assign(ap_tf.propagate_padded(
            propagator=ap_tf.propagate_angular,
            field=field,
            k=k,
            z_list=zs[i: i+1],
            dx=dd,
            dy=dd,
            pad_factor=2.,
        )[0, :, :])

    #sections = slices[:, :, field.shape[1]//2]
    #sections = tf.math.log(tf.abs(sections)**2)
    #section_list.extend([sections[i, :] for i in range(sections.shape[0])])

    return slices

def prop_distance(field, distance):
    zs = tf.cast(tf.linspace(0., distance, n_slices), dtype=tf.float32)
    slices = tf.Variable(tf.zeros(shape=(n_slices, *field.shape), dtype=field.dtype),  dtype=field.dtype)
    return prop_distance_helper(field, zs, slices)

input_modes = slm['inputs_modes']
propagations = []

for i in range(0, 33): #len(input_modes)):
    print("Propagating to field {} of {}".format(i+1, len(input_modes)))
    # z, x, y
    slices = np.zeros(shape=(3*n_slices, *input_modes[i].shape,), dtype=input_modes[i].dtype)

    field = input_modes[i]
    slices[:n_slices, :, :] = prop_distance(field, propagation_distance).numpy()
    field = slices[n_slices-1]
    field *= tf.complex(slm['metasurfaces']['real1'], slm['metasurfaces']['imag1'])
    slices[n_slices:2*n_slices, :, :] = prop_distance(field, propagation_distance).numpy()
    field = slices[2*n_slices - 1]
    field *= tf.complex(slm['metasurfaces']['real2'], slm['metasurfaces']['imag2'])
    slices[2*n_slices:3*n_slices, :, :] = prop_distance(field, propagation_distance).numpy()

    propagations.append(np.abs(slices)**2)

#pickle.dump({'propagations': propagations}, open("data/two_ms_propagation.p", "wb"), protocol=2)
# np.save(file=open("data/two_ms_propagation.p", "wb"), arr=propagations)

print("Saving output")

import h5py
with h5py.File("data/two_ms_propagation.p", "w") as f:
    for i, propagation in enumerate(propagations):
        f.create_dataset("propagation_{}".format(i), data=propagation)


print("Finished Saving output")
