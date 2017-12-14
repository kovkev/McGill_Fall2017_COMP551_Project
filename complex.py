# This file is very similar to deep.py

import tensorflow as tf
import argparse

from tensorflow.examples.tutorials.mnist import input_data

import sys
import tempfile

FLAGS = None

def define_neuron(x):
    """
    x is input tensor
    """

    x = tf.cast(x, tf.complex64)

    mnist_x = mnist_y = 28
    n = mnist_x * mnist_y
    c = 10
    m = 10  # m needs to be calculated

    with tf.name_scope("linear_combination"):
        complex_weight = weight_complex_variable([n,m])
        complex_bias = bias_complex_variable([m])
        h_1 = x @ complex_weight + complex_bias

    return h_1

def main(_):
    mnist = input_data.read_data_sets(
        FLAGS.data_dir,
        one_hot=True,
    )

    # `None` for the first dimension in this shape means that it is variable.
    x_shape = [None, 784]
    x = tf.placeholder(tf.float32, x_shape)
    y_ = tf.placeholder(tf.float32, [None, 10])

    yz = h_1 = define_neuron(x)

    y = tf.nn.softmax(tf.abs(yz))

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_,
            logits=y,
        )

    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        optimizer = tf.train.AdamOptimizer(1e-4)
        optimizer = tf.train.GradientDescentOptimizer(1e-4)
        train_step = optimizer.minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(
            tf.argmax(y, 1),
            tf.argmax(y_, 1),
        )
        correct_prediction = tf.cast(
            correct_prediction,
            tf.float32,
        )
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    batch_size = 100
    num_epochs = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_epochs):
            batch = mnist.train.next_batch(batch_size)
            if i % 10 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={
                        x: batch[0],
                        y_: batch[1],
                    }
                )
            train_step.run(feed_dict={
                x: batch[0],
                y_: batch[1],
            })

        print('test_accuracy %g' % accuract.eval(
            feed_dict={
                x: mnist.test.images,
                y_: mnist.test.labels,
            }
        ))



def get_m(p_r, n, c, k):
    pass

def get_m_r(p_r, n, c, k):
    if k == 0:
        return p_r / (n + c)

    return -1 * (n + c) / (2 * k) + tf.sqrt(
        tf.square(
            (n + c) / (2 * k)
        )
        +
        p_r / k
    )

def get_m_c(p_r, n, c, k):
    if k == 0:
        return p_r / (2 * (n + c))

    return -1 * (n + c) / (2 * k) + tf.sqrt(
        tf.square(
            (n + c) / (2 * k)
        )
        +
        p_r / (2 * k)
    )

def get_p_r(n,m):
    return 2*(n*m) + 2*m

def get_p_c(n,m):
    return n*m + m

def weight_complex_variable(shape):
    real = tf.truncated_normal(shape, stddev=0.1)
    imaginary = tf.truncated_normal(shape, stddev=0.1)
    complex = tf.complex(real, imaginary)
    return tf.Variable(complex)

def bias_complex_variable(shape):
    real = tf.constant(0.1, shape=shape)
    imaginary = tf.constant(0.1, shape=shape)
    complex = tf.complex(real, imaginary)
    return tf.Variable(complex)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  unparsed = []
  FLAGS, unparsed = parser.parse_known_args()
  print(sys.argv)
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
