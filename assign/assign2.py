from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

import argparse
import os.path
import sys
import time
import math
from six.moves import xrange  # pylint: disable=redefined-builtin

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

learning_rate = 0.01
max_steps = 200000
hidden1 = 128
hidden2 = 32
batch_size = 128
fake_data = False
log_dir = '/tmp/tensorflow/mnist/logs/assign2'

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

min_queue_examples = batch_size * max_steps


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels_vector = reformat(train_dataset, train_labels)
valid_dataset, valid_labels_vector = reformat(valid_dataset, valid_labels)
test_dataset, test_labels_vector = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def inference(images, hidden1_units, hidden2_units):
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))), name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))), name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_units))), name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits


def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def placeholder_inputs(batch_size):
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, data_label, images_pl, labels_pl):
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    #     images_feed, labels_feed = data_set.next_batch(batch_size,fake_data)

    data_batch, label_batch = generate_rand_batch(data_set, data_label)

    feed_dict = {
        images_pl: data_batch,
        labels_pl: label_batch,
    }
    return feed_dict

def fill_feed_dict_offset(data_set, data_label, images_pl, labels_pl, offset):
    batch_xs, batch_ys = data_set[offset:(offset + batch_size)], \
                         data_label[offset:(offset + batch_size)]

    feed_dict = {
        images_pl: batch_xs,
        labels_pl: batch_ys,
    }
    return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set, data_label):
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.shape[0] // batch_size
    num_examples = steps_per_epoch * batch_size
    for step in xrange(steps_per_epoch):
        # print('step = ', step)
        offset = (step * batch_size) % (data_set.shape[0] - batch_size)
        feed_dict = fill_feed_dict_offset(data_set, data_label,
                                   images_placeholder,
                                   labels_placeholder, offset)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))

def generate_rand_batch(dataset, labels):
    rand_index = np.random.choice(dataset.shape[0], batch_size, replace=True)

    dataset_batch = dataset[rand_index]
    labels_batch = labels[rand_index]
    return dataset_batch, labels_batch


# Tell TensorFlow that the model will be built into the default Graph.
with tf.Graph().as_default():

    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = inference(images_placeholder, hidden1, hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = training(loss, learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = evaluation(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in xrange(max_steps):
        # print('step = ', step)
        start_time = time.time()

        # train_data_batch, train_label_batch = generate_rand_batch(train_dataset, train_labels)

        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        feed_dict = fill_feed_dict(train_dataset,train_labels,
                                   images_placeholder,labels_placeholder)

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

        duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
        if step % 100 == 0:
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            # Update the events file.
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        # Save a checkpoint and evaluate the model periodically.
        if (step + 1) % 1000 == 0 or (step + 1) == max_steps:

            # train_data_batch, train_label_batch = generate_rand_batch(train_dataset, train_labels)
            # valid_data_batch, valid_label_batch = generate_rand_batch(valid_dataset, valid_labels)
            # test_data_batch, test_label_batch = generate_rand_batch(test_dataset, test_labels)

            checkpoint_file = os.path.join(log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=step)
            # Evaluate against the training set.
            print('Training Data Eval:')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    train_dataset, train_labels)
            # Evaluate against the validation set.
            print('Validation Data Eval:')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    valid_dataset, valid_labels)
            # Evaluate against the test set.
            print('Test Data Eval:')
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    test_dataset, test_labels)
