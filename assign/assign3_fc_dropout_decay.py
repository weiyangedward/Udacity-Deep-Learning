# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import math, time

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

"""
prob4
"""

# Parameters
# learning_rate = 0.5
training_epochs = 1000
batch_size = 128
display_step = 1
beta = 0.01
hidden_nodes_1 = 1024
start_learning_rate = 0.01
keep_rate = 0.7

feature_num = image_size * image_size
layer_num = 5


def dropout_decay_fc(train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels):
    graph = tf.Graph()

    with graph.as_default():
        x = tf.placeholder(tf.float32, [None, feature_num])  # mnist data image of shape 28*28=784
        y = tf.placeholder(tf.float32, [None, num_labels])  # 0-9 digits recognition => 10 classes  # Set model weights
        keep_prob = tf.placeholder(tf.float32)
        """
        build deep-NN of any layer
        """

        def model():
            """
            # tf Graph Input
            x = tf.placeholder(tf.float32, [None, image_size*image_size])  # mnist data image of shape 28*28=784
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
            logits_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)
            logits_2 = tf.nn.relu(tf.matmul(logits_1, W_2) + b_2)
            logits_3 = tf.matmul(logits_2, W_3) + b_3
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_3))
            regularizer = tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2) + tf.nn.l2_loss(W_3)
            cost = tf.reduce_mean(cost + beta * regularizer)
            """

            hidden_nodes = [int(hidden_nodes_1 * np.power(0.5, l)) for l in range(layer_num)]
            layer_sizes = [feature_num] + hidden_nodes + [num_labels]

            Ws = [tf.Variable(
                tf.truncated_normal([layer_sizes[l], layer_sizes[l + 1]], stddev=math.sqrt(2.0 / layer_sizes[l]))) for
                  l in range(layer_num + 1)]
            bs = [tf.Variable(tf.zeros([layer_sizes[l]])) for l in range(1, layer_num + 2)]

            """
            Problem 3
            Introduce Dropout on the hidden layer of the neural network. Remember:
            Dropout should only be introduced during training, not evaluation,
            otherwise your evaluation results would be stochastic as well.
            TensorFlow provides nn.dropout() for that, but you have to make sure it's only inserted during training.
            """
            logits = x
            for k in range(layer_num):
                logits = tf.nn.dropout(tf.nn.relu(tf.matmul(logits, Ws[k]) + bs[k]), keep_prob)
            logits = tf.matmul(logits, Ws[layer_num]) + bs[layer_num]

            pred_logits = x
            for k in range(layer_num):
                pred_logits = tf.nn.relu(tf.matmul(pred_logits, Ws[k]) + bs[k])
            pred_logits = tf.matmul(pred_logits, Ws[layer_num]) + bs[layer_num]

            regularizer = tf.reduce_sum([tf.nn.l2_loss(Ws[k]) for k in range(layer_num + 1)])

            print('hidden_nodes = ', len(hidden_nodes), 'layer_sizes = ', len(layer_sizes), 'Ws = ', len(Ws), 'bs = ',
                  len(bs))

            return logits, pred_logits, regularizer

        logits, pred_logits, regularizer = model()
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
        # cost = tf.reduce_mean(cost + beta * regularizer)

        """
        use learning rate decay:
        """
        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 1000, 0.96, staircase=True)

        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)

        # Construct model
        # pred = tf.nn.softmax(logits_3)  # Softmax
        pred = tf.nn.softmax(pred_logits)
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy for 3000 examples
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initializing the variables
        init = tf.global_variables_initializer()

    num_steps = 200001

    # Launch the graph
    with tf.Session(graph=graph) as sess:
        sess.run(init)
        start_time = time.time()

        # Training cycle
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_xs = train_dataset[offset:(offset + batch_size)]
            batch_ys = train_labels[offset:(offset + batch_size)]  # Fit training using batch data
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys, keep_prob: keep_rate})

        # Display logs per epoch step
            if step % 2000 == 0:
                duration = time.time() - start_time
                print('Minibatch at step %d: (%.3f sec)' % (step, duration))
                print("Train Accuracy: %.1f%%" % (100. * accuracy.eval({x: batch_xs, y: batch_ys})))
                print("Valid Accuracy: %.1f%%" % (100. * accuracy.eval({x: valid_dataset, y: valid_labels})))
                print("Test Accuracy: %.1f%%" % (100. * accuracy.eval({x: test_dataset, y: test_labels})))
                start_time = time.time()

    print("Optimization Finished!")
    print("Accuracy: %.1f%%" % (100.*accuracy.eval({x: test_dataset, y: test_labels})))


dropout_decay_fc(train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels)
