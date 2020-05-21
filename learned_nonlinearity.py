# -*- coding: utf-8 -*-
"""
Created on Tue May 12 20:44:03 2020

@author: james
"""
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from typing import Generator, List, Tuple
import time

# Code written using Tensorflow 2.1.0
print("tf version:", tf.__version__)
print("GPU:", tf.config.experimental.list_physical_devices('GPU'))
print("TPU:", tf.config.experimental.list_physical_devices('TPU'))

#%%

# Nnumber of Taylor series terms to use in the model
taylor_bases = 7

# function that we are trying to fit to
# In this case, we are using a single period of a sine wave
forward = lambda x: np.sin(x*2*np.pi)

# Generates input-output pairs to train the network
# inputs range from [0, 1)
def data_generator():
    while True:
        x = np.random.rand(1)
        yield x, forward(x)

# Creates a Tensorflow dataset (I recommend using tf.dataset)
def generate_dataset(batch_size):
    return tf.data.Dataset.from_generator(data_generator,
        output_types= (tf.float32, tf.float32), 
        output_shapes=( (1, ), (1, ), )).batch(batch_size)       

# Instantiates the dataset. There is no logic behind this batch size
dataset_train = generate_dataset(batch_size=2**14)


#%%

shapes = [(1, ) for _ in range(taylor_bases)]
initializer = tf.initializers.glorot_uniform()
weights = [tf.Variable(np.random.rand(*shape), trainable=True, dtype=tf.float32) for shape in shapes]

def model(x):
    # defines a finite Taylor series (without the constants). Axis=1 is the batch axis.
    out = tf.reduce_sum([weight*x**i for i, weight in enumerate(weights)], axis=0)
    return out


def loss(pred, target):
    return tf.losses.mse(target, pred)

#%%
optimizer = tf.optimizers.Adam(learning_rate=1.)



def train_step( model, inputs , outputs ):
    with tf.GradientTape() as tape:
        current_loss = loss( model( inputs ), outputs)
    grads = tape.gradient( current_loss , weights )
    #update weights
    optimizer.apply_gradients( zip( grads , weights ) )
    
    return tf.reduce_mean( current_loss ).numpy()
    
#%%

# Training loop

iterations = 2**7
n_update = 2**0 # Updates information every * iterations

# log to plot parameter convergence
weights_log = []

t = time.time()
for i, (inputs, outputs) in enumerate(dataset_train.take(iterations)):
    error = train_step( model , inputs , outputs )
    if i % n_update == 0:
        weights_log.append([weight.numpy() for weight in weights])
        t_now = time.time()
        print("Error: {}\tTimePerUpdate(s): {}".format(error, t_now-t))
        t = t_now
    
#%%
weights_log = np.array(weights_log)

for i in range(taylor_bases):
    plt.plot(weights_log[:, i])
plt.show()

t = np.linspace(0., 1., 100)
plt.plot(t, forward(t))
plt.plot(t, model(t))
plt.show()
    
