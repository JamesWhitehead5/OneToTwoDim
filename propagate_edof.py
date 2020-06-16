from simple_angular_prop_tf import complex_split
import tensorflow as tf
import AngularPropagateTensorflow.AngularPropagateTensorflow as ap_tf
import numpy as np
import matplotlib.pyplot as plt

wavelength = 633e-9 # HeNe
k = 2.*np.pi/wavelength
lens_aperture = 50e-6
dd = wavelength/2. # array element spacing
slm_n = int(lens_aperture/dd)
slm_shape = (slm_n, slm_n, )

fname = 'slm.p'
import pickle
slm = pickle.load(open(fname, "rb"))


z_list = np.linspace(0., 300e-6, 200)
section_list = []
for z in z_list:
    section = ap_tf.propagate_angular_padded(field=slm['slm'], k=k, z_list=[z, ], dx=dd, dy=dd, pad_factor=5.)[0, slm_n//2, :]
    section = tf.abs(section)**2
    section_list.append(section)

plt.imshow(tf.transpose(section_list))
plt.colorbar()
plt.show()