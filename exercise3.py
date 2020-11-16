from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import argparse
import os
import sys
import tempfile
import numpy as np


FLAGS = None

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride and padding."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def conv2dtransp(x, W, o1, o2):
     # conv2dtransp returns a 2d transposed convolution layer with stride of 2 and padding
     temp_batch_size = tf.shape(x)[0]
     output_shape = [temp_batch_size, o1, o1, o2]
     return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding='SAME')

def deep(x):
    """ deep builts a graph for the autoencoder """

    # Reshape to use within a convolutional neural net.
    with tf.name_scope('reshape'):
      x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 8 feature maps.
    # Output size :28x28x8
    with tf.name_scope('conv1'):
      W_conv1 = weight_variable([3, 3, 1, 8])
      b_conv1 = bias_variable([8])
      h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    # Output size: 14x14x8
    with tf.name_scope('pool1'):
     h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps  8 feature maps to 4.
    # Output size : 14x14x4
     with tf.name_scope('conv2'):
       W_conv2 = weight_variable([3, 3, 8, 4])
       b_conv2 = bias_variable([4])
       h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    # Output size : 7x7x4
    with tf.name_scope('pool2'):
       h_pool2 = max_pool_2x2(h_conv2)

    # Third convolutional layer -- maps  4 feature maps to 2.
    # Output size : 7x7x2
    with tf.name_scope('conv3'):
      W_conv3 = weight_variable([3, 3, 4, 2])
      b_conv3 = bias_variable([2])
      h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    # First transposed convolutional layer -- maps  2 feature maps to 4.
    # Output size : 14x14x4
    with tf.name_scope('transpconv1'):
      W_trconv1 = weight_variable([2, 2, 4, 2])
      b_trconv1 = bias_variable([4])
      h_trconv1 = tf.nn.relu(conv2dtransp(h_conv3, W_trconv1, 14, 4) + b_trconv1)

    # Fourth convolutional layer -- maps  4 feature maps to 4.
    # Output size : 14x14x4
    with tf.name_scope('conv4'):
      W_conv4 = weight_variable([3, 3, 4, 4])
      b_conv4 = bias_variable([4])
      h_conv4 = tf.nn.relu(conv2d(h_trconv1, W_conv4) + b_conv4)

    # Second transposed convolutional layer -- maps  4 feature maps to 8.
    # Output size : 28x28x8
    with tf.name_scope('transpconv2'):
      W_trconv2 = weight_variable([2, 2, 8, 4])
      b_trconv2 = bias_variable([8])
      h_trconv2 = tf.nn.relu(conv2dtransp(h_conv4, W_trconv2, 28, 8) + b_trconv2)

    # Fifth convolutional layer -- maps  8 feature maps to 8.
    # Output size : 28x28x8
    with tf.name_scope('conv5'):
      W_conv5 = weight_variable([3, 3, 8, 8])
      b_conv5 = bias_variable([8])
      h_conv5 = tf.nn.relu(conv2d(h_trconv2, W_conv5) + b_conv5)

    # Output convolutional layer -- maps  8 feature maps to 1
    # Output size : 28x28x1
    with tf.name_scope('outputconv'):
       W_out = weight_variable([3, 3, 8, 1])
       b_out = bias_variable([1])
       h_out = tf.nn.relu(conv2d(h_conv5, W_out) + b_out)
    return h_out

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    display_step = 1000
    # Create the model
    learning_rates = tf.placeholder(tf.float32)
    # scale every image to values between 0 and 1
    x_image = tf.placeholder(tf.float32, [None, 784])/ 255
    new_xs = deep(x_image)
    # for the reconstructed images
    img_autoenc = tf.summary.image("autoencoder", new_xs, max_outputs=2)

    new_x = tf.reshape(new_xs, [64, 784])

    with tf.name_scope('loss'):
      loss = tf.reduce_mean(tf.square(new_x - x_image))

    learning_rates = [1e-1, 1e-2, 1e-3]
    for rate in learning_rates:
        with tf.name_scope('Adam_optimizer'):
            optimizer = tf.train.AdamOptimizer(rate).minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(FLAGS.log_dir + '/learningRate' 
                                   + str(rate), sess.graph)
            num_steps = 20000
            epochs = []
            x_loss = []
            for i in range(num_steps + 1):
               batch, _ = mnist.train.next_batch(64)
               # Run optimization op (backprop) and cost op (to get loss value)
               _, l = sess.run([optimizer, loss], feed_dict={x_image: batch})
               # Display logs per step
               if i % display_step == 0:
                   # Displaying the images in tensorboard
                   train_loss, img = sess.run([loss, img_autoenc],
                                          feed_dict={x_image: batch})
                   writer.add_summary(img, i)
                   print('Step %i: Minibatch Loss: %f' % (i, l))
                   epochs.append(i)
                   x_loss.append(l)
            writer.close()
            plt.plot(epochs, x_loss, label=rate)

        

    # for plotting the learning curves
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str,
                      default=os.path.join(os.getenv('HOME', '/home'),
                                           'lab/tensorflow'),
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
        
