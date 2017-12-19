from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import music
from numpy import random as npr
import os
import sys
import tempfile
import tensorflow as tf

# Data sets
TRAIN_FILE = "history_train1.json"
ID_LENGTH = None
TOTAL = None

FLAGS = None


def deepnn(x):
    # Fully connected layer 1
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([TOTAL, 1024])
        b_fc1 = bias_variable([1024])

    h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)

    # Fully connected layer 2
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 1024])
        b_fc2 = bias_variable([1024])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # Fully connected layer 3
    with tf.name_scope('fc3'):
        W_fc3 = weight_variable([1024, ID_LENGTH])
        b_fc3 = bias_variable([ID_LENGTH])

    y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3
    return y_conv


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    global TOTAL, ID_LENGTH
    if not os.path.exists(TRAIN_FILE):
        music.main()

    # Import data
    with open(TRAIN_FILE, 'r') as file:
        data = json.load(file)
        data_train = data["train"]
        data_test = data["test"]
        iddict = data["keys"]

    # Length of input
    TOTAL = len(data_train[0][0])
    # Length of output
    ID_LENGTH = len(data_train[0][1])

    # Create the model
    x = tf.placeholder(tf.float32, [None, TOTAL])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, ID_LENGTH])

    # Build the graph for the deep net
    y_conv = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            inds = npr.randint(len(data_train), size=10)
            batch = [data_train[i] for i in inds]
            left = [entry[0] for entry in batch]
            right = [entry[1] for entry in batch]
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: left, y_: right})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict={x: left, y_: right})

        left = [entry[0] for entry in data_test]
        right = [entry[1] for entry in data_test]
        print('test accuracy: ', accuracy.eval(feed_dict={
            x: left, y_: right}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
