import matplotlib.pyplot as plt
import numpy as np
import h5py

with h5py.File("data/two_ms_propagation.p", "r") as f:
    propagations = f.get('propagation').value

# fname = "data/two_ms_propagation.p"
# import pickle
# propagations = pickle.load(open(fname, "rb"))['propagations']


for propagation in propagations:
    plt.imshow(np.log(np.transpose(propagation[:, :, propagation.shape[2]//2])))
    plt.colorbar()
    plt.show()