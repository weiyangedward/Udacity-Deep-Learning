# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os, math, time

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
num_channels = 1  # grayscale

import numpy as np


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def inception_cnn(train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels):
    """
    Problem 2
    Try to get the best performance you can using a convolutional net.
    Look for example at the classic LeNet5 architecture, adding Dropout, and/or adding learning rate decay.
    """
    log_dir = './saved_models/'
    start_learning_rate = 0.1
    batch_size = 128
    patch_size = 5
    conv_depth0, conv_depth1, conv_depth2, conv_depth3 = 16, 16, 48, 48
    # num_hidden_1 = image_size // 4 * image_size // 4 * depth_2
    num_hidden_1 = 128
    keep_prob_hidden = 0.5
    fully_connected_layer_num = 2
    filter_count, conv_count = 3, 1
    conv_layer_count = 2
    max_pool_strides = 2

    graph = tf.Graph()

    with graph.as_default():

        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        def conv_weights(in_channel, out_channel):
            conv_weights_five_by_five = tf.Variable(tf.truncated_normal([5, 5, in_channel, out_channel], stddev=0.1))
            depth = out_channel * conv_count
            conv_biases = tf.Variable(tf.zeros([depth]))
            return conv_weights_five_by_five, conv_biases, depth

        # convolution layer weights
        conv1_weights_five_by_five, conv1_biases, conv1_depth = conv_weights(num_channels, conv_depth0)
        conv2_weights_five_by_five, conv2_biases, conv2_depth = conv_weights(conv_depth0, conv_depth1)

        def inception_weights(in_channel, out_channel):
            layer_weights_one_by_one = tf.Variable(tf.truncated_normal([1, 1, in_channel, out_channel], stddev=0.1))
            layer_weights_one_by_one_prefil = tf.Variable(tf.truncated_normal([1, 1, in_channel, out_channel], stddev=0.1))
            layer_weights_one_by_one_pool = tf.Variable(tf.truncated_normal([1, 1, in_channel, out_channel], stddev=0.1))
            layer_weights_three_by_three = tf.Variable(tf.truncated_normal([3, 3, out_channel, out_channel], stddev=0.1))
            layer_weights_five_by_five = tf.Variable(tf.truncated_normal([5, 5, out_channel, out_channel], stddev=0.1))
            depth = out_channel * filter_count + out_channel
            layer_biases = tf.Variable(tf.zeros([depth]))
            return layer_weights_one_by_one, layer_weights_one_by_one_prefil, layer_weights_one_by_one_pool, layer_weights_three_by_three, layer_weights_five_by_five, layer_biases, depth

        # inception layer weights
        layer1_weights_one_by_one, layer1_weights_one_by_one_prefil, layer1_weights_one_by_one_pool, \
        layer1_weights_three_by_three, \
        layer1_weights_five_by_five, \
        layer1_biases, layer1_depth = inception_weights(conv2_depth, conv_depth2)

        layer2_weights_one_by_one, layer2_weights_one_by_one_prefil, layer2_weights_one_by_one_pool, \
        layer2_weights_three_by_three, \
        layer2_weights_five_by_five, \
        layer2_biases, layer2_depth = inception_weights(layer1_depth, conv_depth3)

        def hidden_weights(in_weight_num, out_weight_num, layer_num):
            num_hiddens = [in_weight_num] + [int(num_hidden_1 * np.power(0.5, l)) for l in range(layer_num)] + [out_weight_num]
            hidden_layer_weights = [tf.Variable(tf.truncated_normal([num_hiddens[k], num_hiddens[k+1]], stddev=0.1)) for k in range(len(num_hiddens)-1)]
            hidden_layer_biases = [tf.Variable(tf.constant(0.0, shape=[num_hiddens[k]])) for k in range(1,len(num_hiddens))]
            return num_hiddens, hidden_layer_weights, hidden_layer_biases

        # fully connected layer weights
        num_hiddens, \
        hidden_layer_weights, \
        hidden_layer_biases = hidden_weights(image_size // (max_pool_strides ** conv_layer_count) * image_size // (max_pool_strides ** conv_layer_count) * layer2_depth, num_labels, fully_connected_layer_num)


        def inception_module(data, layer_weights_one_by_one, layer_weights_one_by_one_prefil, layer_weights_one_by_one_pool, layer_weights_three_by_three, layer_weights_five_by_five, layer_biases):
            inception_one_by_one = tf.nn.conv2d(data, layer_weights_one_by_one, [1, 1, 1, 1], padding='SAME')
            inception_one_by_one_prefil = tf.nn.conv2d(data, layer_weights_one_by_one_prefil, [1, 1, 1, 1], padding='SAME')
            inception_three_by_three = tf.nn.conv2d(inception_one_by_one_prefil, layer_weights_three_by_three, [1, 1, 1, 1], padding='SAME')
            inception_five_by_five = tf.nn.conv2d(inception_one_by_one_prefil, layer_weights_five_by_five, [1, 1, 1, 1], padding='SAME')
            pool = tf.nn.avg_pool(data, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')
            inception_pool_one_by_one = tf.nn.conv2d(pool, layer_weights_one_by_one_pool, [1, 1, 1, 1], padding='SAME')
            inception_concat = tf.concat([inception_one_by_one, inception_three_by_three, inception_five_by_five, inception_pool_one_by_one], axis=3)
            hidden = tf.nn.relu(inception_concat + layer_biases)
            # pool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
            return hidden

        def conv_layer(data, conv_weights_five_by_five, conv_biases):
            conv_five_by_five = tf.nn.conv2d(data, conv_weights_five_by_five, [1, 1, 1, 1], padding='SAME')
            conv_five_by_five_relu = tf.nn.relu(conv_five_by_five + conv_biases)
            return conv_five_by_five_relu

        def max_pool(data):
            pool = tf.nn.max_pool(data, ksize=[1, 2, 2, 1],
                                  strides=[1, max_pool_strides, max_pool_strides, 1], padding='SAME')
            return pool

        # Model.
        def model_train(data):
            conv_1 = conv_layer(data, conv1_weights_five_by_five, conv1_biases)
            conv_2 = conv_layer(conv_1, conv2_weights_five_by_five, conv2_biases)
            conv_2_pool = max_pool(conv_2)

            inception_1 = inception_module(conv_2_pool, layer1_weights_one_by_one, layer1_weights_one_by_one_prefil, layer1_weights_one_by_one_pool, layer1_weights_three_by_three, layer1_weights_five_by_five, layer1_biases)
            inception_2 = inception_module(inception_1, layer2_weights_one_by_one, layer2_weights_one_by_one_prefil, layer2_weights_one_by_one_pool, layer2_weights_three_by_three, layer2_weights_five_by_five, layer2_biases)

            inception_2_pool = max_pool(inception_2)

            shape = inception_2_pool.get_shape().as_list()
            reshape = tf.reshape(inception_2_pool, [shape[0], shape[1] * shape[2] * shape[3]])

            hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(reshape, hidden_layer_weights[0]) + hidden_layer_biases[0]), keep_prob_hidden)
            for k in range(1,len(hidden_layer_weights)-1):
                hidden = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden, hidden_layer_weights[k]) + hidden_layer_biases[k]), keep_prob_hidden)
            hidden = tf.matmul(hidden, hidden_layer_weights[len(hidden_layer_weights) - 1]) + hidden_layer_biases[
                len(hidden_layer_weights) - 1]
            return hidden


        def model_pred(data):
            conv_1 = conv_layer(data, conv1_weights_five_by_five, conv1_biases)
            conv_2 = conv_layer(conv_1, conv2_weights_five_by_five, conv2_biases)
            conv_2_pool = max_pool(conv_2)

            inception_1 = inception_module(conv_2_pool, layer1_weights_one_by_one, layer1_weights_one_by_one_prefil,
                                           layer1_weights_one_by_one_pool, layer1_weights_three_by_three,
                                           layer1_weights_five_by_five, layer1_biases)
            inception_2 = inception_module(inception_1, layer2_weights_one_by_one, layer2_weights_one_by_one_prefil,
                                           layer2_weights_one_by_one_pool, layer2_weights_three_by_three,
                                           layer2_weights_five_by_five, layer2_biases)

            inception_2_pool = max_pool(inception_2)

            shape = inception_2_pool.get_shape().as_list()
            reshape = tf.reshape(inception_2_pool, [shape[0], shape[1] * shape[2] * shape[3]])

            hidden = tf.nn.relu(tf.matmul(reshape, hidden_layer_weights[0]) + hidden_layer_biases[0])
            for k in range(1, len(hidden_layer_weights) - 1):
                hidden = tf.nn.relu(tf.matmul(hidden, hidden_layer_weights[k]) + hidden_layer_biases[k])
            hidden = tf.matmul(hidden, hidden_layer_weights[len(hidden_layer_weights) - 1]) + hidden_layer_biases[
                len(hidden_layer_weights) - 1]
            return hidden

        # Training computation.
        logits = model_train(tf_train_dataset)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

        # decay learning rate
        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 1000, 0.96, staircase=True)

        # Optimizer.
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(model_pred(tf_train_dataset))
        valid_prediction = tf.nn.softmax(model_pred(tf_valid_dataset))
        test_prediction = tf.nn.softmax(model_pred(tf_test_dataset))


    num_steps = 400001

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        start_time = time.time()
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 200 == 0):
                duration = time.time() - start_time
                print('Minibatch loss at step %d: %f (%.3f sec)' % (step, l, duration))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
                print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
                start_time = time.time()

inception_cnn(train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels)
