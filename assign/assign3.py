# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import math

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


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
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


def prob1():
    """
    Introduce and tune L2 regularization for both logistic and neural network models. Remember that L2 amounts to
    adding a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor
    t using nn.l2_loss(t). The right amount of regularization should improve your validation / test accuracy.
    :return:
    """
    # Parameters
    learning_rate = 0.5
    training_epochs = 3
    batch_size = 128
    display_step = 1
    beta = 0.01

    hidden_nodes_1 = 1024

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, image_size * image_size])  # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, num_labels])  # 0-9 digits recognition => 10 classes

    # Set model weights
    W_1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_nodes_1]))
    b_1 = tf.Variable(tf.zeros([hidden_nodes_1]))
    W_2 = tf.Variable(tf.truncated_normal([hidden_nodes_1, num_labels]))
    b_2 = tf.Variable(tf.zeros([num_labels]))

    # Minimize error using cross entropy
    # cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    logits_1 = tf.matmul(x, W_1) + b_1
    relu_layer = tf.nn.relu(logits_1)
    logits_2 = tf.matmul(relu_layer, W_2) + b_2
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_2))
    regularizer = tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2)
    cost = tf.reduce_mean(cost + beta * regularizer)
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Construct model
    pred = tf.nn.softmax(logits_2)  # Softmax
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            print('epoch = ', epoch)
            train_dataset, train_labels = randomize(train_dataset, train_labels)
            avg_cost = 0.
            total_batch = int(train_dataset.shape[0] / batch_size)
            print('total_batch = ', total_batch)
            # Loop over all batches
            for i in range(total_batch):
                if i % 100 == 0:
                    print('i = ', i)
                    print("Test Accuracy:", accuracy.eval({x: test_dataset, y: test_labels}))
                    print("Train Accuracy:", accuracy.eval({x: train_dataset, y: train_labels}))
                batch_xs, batch_ys = train_dataset[i * batch_size:(i + 1) * batch_size], train_labels[i * batch_size:(
                                                                                                                         i + 1) * batch_size]
                # Fit training using batch data
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        print("Accuracy:", accuracy.eval({x: test_dataset, y: test_labels}))


prob1()


def prob4():
    # Parameters
    learning_rate = 0.5
    training_epochs = 3
    batch_size = 128
    display_step = 1
    beta = 0.01

    hidden_nodes_1 = 1024
    hidden_nodes_2 = int(hidden_nodes_1 * 0.5)

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, image_size * image_size])  # mnist data image of shape 28*28=784
    y = tf.placeholder(tf.float32, [None, num_labels])  # 0-9 digits recognition => 10 classes

    # Set model weights
    W_1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_nodes_1],
                                          stddev=math.sqrt(2.0 / (image_size * image_size))))
    b_1 = tf.Variable(tf.zeros([hidden_nodes_1]))
    W_2 = tf.Variable(tf.truncated_normal([hidden_nodes_1, hidden_nodes_2], stddev=math.sqrt(2.0 / (hidden_nodes_1))))
    b_2 = tf.Variable(tf.zeros([hidden_nodes_2]))
    W_3 = tf.Variable(tf.truncated_normal([hidden_nodes_2, num_labels], stddev=math.sqrt(2.0 / (hidden_nodes_2))))
    b_3 = tf.Variable(tf.zeros([num_labels]))

    # Minimize error using cross entropy
    # cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    logits_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)
    # relu_layer_1 = tf.nn.relu(logits_1)
    logits_2 = tf.nn.relu(tf.matmul(logits_1, W_2) + b_2)
    # relu_layer_2 = tf.nn.relu(logits_2)
    logits_3 = tf.matmul(logits_2, W_3) + b_3
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_3))
    regularizer = tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2) + tf.nn.l2_loss(W_3)
    cost = tf.reduce_mean(cost + beta * regularizer)
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Construct model
    pred = tf.nn.softmax(logits_3)  # Softmax
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            print('epoch = ', epoch)
            train_dataset, train_labels = randomize(train_dataset, train_labels)
            avg_cost = 0.
            total_batch = int(train_dataset.shape[0] / batch_size)
            print('total_batch = ', total_batch)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = train_dataset[i * batch_size:(i + 1) * batch_size], train_labels[i * batch_size:(
                                                                                                                     i + 1) * batch_size]

                if i % 500 == 0:
                    print('i = ', i)
                    print("Test Accuracy:", accuracy.eval({x: test_dataset, y: test_labels}))
                    print("Train Accuracy:", accuracy.eval({x: batch_xs, y: batch_ys}))
                # Fit training using batch data
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        print("Accuracy:", accuracy.eval({x: test_dataset, y: test_labels}))


prob4()
