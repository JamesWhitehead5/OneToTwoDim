import tensorflow as tf

# Build a graph.
x = tf.Variable(initial_value=5.0)
y = tf.Variable(initial_value=6.0)

# Take the gradient
with tf.GradientTape() as tape:
    f = x**2*y + y + 2. # Define the function

# Take the gradient W.R.T. [x, y]
print(tape.gradient(f, [x, y, ])) # Returns 60.0, 26.0
