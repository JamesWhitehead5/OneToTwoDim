# from simple_angular_prop_tf import complex_split
import tensorflow as tf
# import AngularPropagateTensorflow.AngularPropagateTensorflow as ap_tf
# import AngularPropagateTensorflow as ap_tf
from AngularPropagateTensorflow import AngularPropagateTensorflow as ap_tf
import numpy as np
import matplotlib.pyplot as plt


fname = 'slm.p'
import pickle
slm = pickle.load(open(fname, "rb"))

target_focal_length = slm['target_focal_length']
k = slm['wavenumber']
dd = slm['array_element_spacing']
slm_field = slm['slm']

z_list = np.linspace(0., 200e-6, 200)
section_list = []
for z in z_list:
    # section = ap_tf.propagate_angular_bw_limited(
    #     field=slm_field,
    #     k=k,
    #     z_list=[z, ],
    #     dx=dd,
    #     dy=dd,)[0, slm_field.shape[1]//2, :]

    section = ap_tf.propagate_padded(
        propagator=ap_tf.propagate_angular,
        field=slm_field,
        k=k,
        z_list=[z, ],
        dx=dd,
        dy=dd,
        pad_factor=1.,
    )[0, slm_field.shape[1]//2, :]

    section = tf.abs(section)**2
    section_list.append(section)

plt.imshow(tf.transpose(section_list))
plt.colorbar()
plt.show()