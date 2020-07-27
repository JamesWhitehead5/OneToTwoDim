from datetime import datetime

import tensorflow as tf
import AngularPropagateTensorflow.AngularPropagateTensorflow as ap_tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Generator, List, Tuple
import time

# Code written using Tensorflow 2.1.0
print("tf version:", tf.__version__)
print("GPU:", tf.config.experimental.list_physical_devices('GPU'))
print("TPU:", tf.config.experimental.list_physical_devices('TPU'))

# The function to be traced.
@tf.function
def my_func(x, y):
  # A simple hand-rolled layer.
  return tf.nn.relu(tf.matmul(x, y))

# Set up logging.
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs\\func\\%s' % stamp
writer = tf.summary.create_file_writer(logdir)

# Sample data for your function.
x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)
# Call only one tf.function when tracing.
z = my_func(x, y)
with writer.as_default():
  tf.summary.trace_export(
      name="my_func_trace",
      step=0,
      profiler_outdir=logdir)

print(logdir)