import tensorflow as tf

@tf.function
def test_while():
    i = tf.constant(0)
    c = lambda i: tf.less(i, 10)
    # b = lambda i: (tf.add(i, 1),)
    def b(i):
        return tf.add(i, 1)
    r = tf.while_loop(c, b, [i])

    return r

if __name__=='__main__':
    print(test_while())
