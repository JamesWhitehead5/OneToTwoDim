import tensorflow as tf
import AngularPropagateTensorflow as ap_tf
import numpy as np
import matplotlib.pyplot as plt

fname = 'data/two_ms.p'
import pickle
slm = pickle.load(open(fname, "rb"))

propagation_distance = slm['sim_args']['prop_distance']
k = slm['sim_args']['k']
dd = slm['sim_args']['dd']

n_slices = 100

def prop_distance(field, distance):
    zs = np.linspace(0., distance, n_slices)
    slices = ap_tf.propagate_padded(
        propagator=ap_tf.propagate_angular,
        field=field,
        k=k,
        z_list=zs,
        dx=dd,
        dy=dd,
        pad_factor=2.,
    )

    #sections = slices[:, :, field.shape[1]//2]
    #sections = tf.math.log(tf.abs(sections)**2)
    #section_list.extend([sections[i, :] for i in range(sections.shape[0])])

    return slices

input_modes = slm['inputs_modes']
propagations = []

for i in range(0, len(input_modes)):
    # z, x, y
    slices = np.zeros(shape=(3*n_slices, *input_modes[i].shape,), dtype=input_modes[i].dtype)

    field = input_modes[i]
    slices[:n_slices, :, :] = prop_distance(field, propagation_distance)
    field = slices[n_slices-1]
    field *= tf.complex(slm['metasurfaces']['real1'], slm['metasurfaces']['imag1'])
    slices[n_slices:2*n_slices, :, :] = prop_distance(field, propagation_distance)
    field = slices[2*n_slices - 1]
    field *= tf.complex(slm['metasurfaces']['real2'], slm['metasurfaces']['imag2'])
    slices[2*n_slices:3*n_slices, :, :] = prop_distance(field, propagation_distance)

    propagations.append(np.abs(slices)**2)

#pickle.dump({'propagations': propagations}, open("data/two_ms_propagation.p", "wb"), protocol=2)
# np.save(file=open("data/two_ms_propagation.p", "wb"), arr=propagations)

import h5py
with h5py.File("data/two_ms_propagation.p", "w") as f:
    for i, propagation in enumerate(propagations):
        f.create_dataset("propagation_{}".format(i), data=propagation)



