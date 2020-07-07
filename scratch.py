from simple_angular_prop_tf import complex_split
import tensorflow as tf
# import AngularPropagateTensorflow.AngularPropagateTensorflow as ap_tf
import AngularPropagateTensorflow as ap_tf
import numpy as np
import matplotlib.pyplot as plt


fname = 'slm.p'
import pickle
slm = pickle.load(open(fname, "rb"))

target_focal_length = slm['target_focal_length']
k = slm['wavenumber']
dd = slm['array_element_spacing']
slm_field = slm['slm']

field = ap_tf.propagate_padded(
    propagator=ap_tf.propagate_angular_bw_limited,
    field=slm_field,
    k=k,
    z_list=[target_focal_length, ],
    dx=dd,
    dy=dd,
    pad_factor=1.
)

plt.imshow(tf.math.abs(field[0, :, :]))
plt.show()
