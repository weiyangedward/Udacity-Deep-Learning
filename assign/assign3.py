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

"""
prob1:
Introduce and tune L2 regularization for both logistic and neural network models. Remember that L2 amounts to
adding a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor
t using nn.l2_loss(t). The right amount of regularization should improve your validation / test accuracy.
:return:
"""
# Parameters
# learning_rate = 0.01
# training_epochs = 100
# batch_size = 128
# display_step = 1
# beta = 0.01
#
# hidden_nodes_1 = 1024
#
# # tf Graph Input
# x = tf.placeholder(tf.float32, [None, image_size * image_size])  # mnist data image of shape 28*28=784
# y = tf.placeholder(tf.float32, [None, num_labels])  # 0-9 digits recognition => 10 classes
#
# # Set model weights
# W_1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_nodes_1], stddev=math.sqrt(2.0 / (image_size * image_size))))
# b_1 = tf.Variable(tf.zeros([hidden_nodes_1]))
# W_2 = tf.Variable(tf.truncated_normal([hidden_nodes_1, num_labels], stddev=math.sqrt(2.0 / (hidden_nodes_1))))
# b_2 = tf.Variable(tf.zeros([num_labels]))
#
# # Minimize error using cross entropy
# # cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# logits_1 = tf.matmul(x, W_1) + b_1
# relu_layer = tf.nn.relu(logits_1)
# logits_2 = tf.matmul(relu_layer, W_2) + b_2
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_2))
# regularizer = tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2)
# cost = tf.reduce_mean(cost + beta * regularizer)
# # Gradient Descent
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#
# # Construct model
# pred = tf.nn.softmax(logits_2)  # Softmax
# # Test model
# correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# # Calculate accuracy for 3000 examples
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# # Initializing the variables
# init = tf.global_variables_initializer()
#
# # Launch the graph
# with tf.Session() as sess:
#     sess.run(init)
#
#     # Training cycle
#     for epoch in range(training_epochs):
#         print('epoch = ', epoch)
#         train_dataset, train_labels = randomize(train_dataset, train_labels)
#         avg_cost = 0.
#         total_batch = int(train_dataset.shape[0] / batch_size)
#         print('total_batch = ', total_batch)
#         # Loop over all batches
#         for i in range(total_batch):
#             batch_xs, batch_ys = train_dataset[i*batch_size : (i+1)*batch_size], train_labels[i*batch_size : (i+1)*batch_size]
#
#             if i % 500 == 0:
#                 print('i = ', i)
#                 print("Train Accuracy:", accuracy.eval({x: batch_xs, y: batch_ys}))
#                 print("Test Accuracy:", accuracy.eval({x: test_dataset, y: test_labels}))
#             # Fit training using batch data
#             _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
#             # Compute average loss
#             avg_cost += c / total_batch
#         # Display logs per epoch step
#         if (epoch + 1) % display_step == 0:
#             print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
#
#     print("Optimization Finished!")
#
#     print("Accuracy:", accuracy.eval({x: test_dataset, y: test_labels}))


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
keep_rate = 1

"""
build deep-NN of any layer
"""
feature_num = image_size*image_size
layer_num = 5

x = tf.placeholder(tf.float32, [None, feature_num])  # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, num_labels])  # 0-9 digits recognition => 10 classes  # Set model weights

hidden_nodes = [int(hidden_nodes_1 * np.power(0.5, l)) for l in range(layer_num)]
layer_sizes = [feature_num] + hidden_nodes + [num_labels]

Ws = [tf.Variable(tf.truncated_normal([layer_sizes[l], layer_sizes[l + 1]], stddev=math.sqrt(2.0 / layer_sizes[l]))) for
      l in range(layer_num + 1)]
bs = [tf.Variable(tf.zeros([layer_sizes[l]])) for l in range(1, layer_num + 2)]

keep_prob = tf.placeholder(tf.float32)

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

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

regularizer = tf.reduce_sum([tf.nn.l2_loss(Ws[k]) for k in range(layer_num + 1)])

cost = tf.reduce_mean(cost + beta * regularizer)

print('hidden_nodes = ', len(hidden_nodes), 'layer_sizes = ', len(layer_sizes), 'Ws = ', len(Ws), 'bs = ', len(bs))



# tf Graph Input
# x = tf.placeholder(tf.float32, [None, image_size*image_size])  # mnist data image of shape 28*28=784
# y = tf.placeholder(tf.float32, [None, num_labels])  # 0-9 digits recognition => 10 classes  # Set model weights
#
# W_1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_nodes_1],
#                                       stddev=math.sqrt(2.0 / (image_size * image_size))))
# b_1 = tf.Variable(tf.zeros([hidden_nodes_1]))
# W_2 = tf.Variable(tf.truncated_normal([hidden_nodes_1, hidden_nodes_2], stddev=math.sqrt(2.0 / (hidden_nodes_1))))
# b_2 = tf.Variable(tf.zeros([hidden_nodes_2]))
# W_3 = tf.Variable(tf.truncated_normal([hidden_nodes_2, num_labels], stddev=math.sqrt(2.0 / (hidden_nodes_2))))
# b_3 = tf.Variable(tf.zeros([num_labels]))

# Minimize error using cross entropy
# logits_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)
# logits_2 = tf.nn.relu(tf.matmul(logits_1, W_2) + b_2)
# logits_3 = tf.matmul(logits_2, W_3) + b_3
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits_3))
# regularizer = tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2) + tf.nn.l2_loss(W_3)
# cost = tf.reduce_mean(cost + beta * regularizer)

"""
use learning rate decay:
"""
global_step = tf.Variable(0)  # count the number of steps taken.
learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)

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
            batch_xs, batch_ys = train_dataset[i*batch_size : (i + 1)*batch_size], train_labels[i*batch_size : (i + 1)*batch_size]

            # if i % 500 == 0:
            #     print('i = ', i)

            # Fit training using batch data
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys, keep_prob: keep_rate})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Train Accuracy:", accuracy.eval({x: batch_xs, y: batch_ys}))
            # print("Valid Accuracy:", accuracy.eval({x: valid_dataset, y: valid_labels}))
            print("Test Accuracy:", accuracy.eval({x: test_dataset, y: test_labels}))
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")
    print("Accuracy:", accuracy.eval({x: test_dataset, y: test_labels}))
