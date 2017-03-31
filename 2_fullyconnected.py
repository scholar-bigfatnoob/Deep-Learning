"""
Deep Learning
=============

Assignment 2
------------

Previously in `1_notmnist.ipynb`, we created a pickle with formatted datasets for training,
development and testing on the [notMNIST dataset](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html).

The goal of this assignment is to progressively train deeper and more accurate models using TensorFlow.
"""


from __future__ import print_function
import sys
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True


__author__ = "bigfatnoob"

"""
First reload the data we generated in `1_notmnist.ipynb`.
"""

pickle_file = 'data/notMNIST.pkl'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save
  print("Training Set", train_dataset.shape, train_labels.shape)
  print("Validation Set", valid_dataset.shape, valid_labels.shape)
  print("Test Set", test_dataset.shape, test_labels.shape)


"""
Reformat into a shape that's more adapted to the models we're going to train:
- data as a flat matrix,
- labels as float 1-hot encodings.
"""

IMAGE_SIZE = 28
NUM_LABELS = 10


def reformat(dataset, labels):
  dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


"""
We're first going to train a multinomial logistic regression using simple gradient descent.

TensorFlow works like this:
* First you describe the computation that you want to see performed: what the inputs, the variables, and the operations look like. These get created as nodes over a computation graph. This description is all contained within the block below:

      with graph.as_default():
          ...

* Then you can run the operations on this graph as many times as you want by calling `session.run()`, providing it outputs to fetch from the graph that get returned. This runtime operation is all contained in the block below:

      with tf.Session(graph=graph) as session:
          ...

Let's load all the data into TensorFlow and build the computation graph corresponding to our training:
"""

def gradient_descent():
  # With gradient descent training, even this much data is prohibitive.
  # Subset the training data for faster turnaround.
  train_subset = 10000

  graph = tf.Graph()
  with graph.as_default():
    # Input data.
    # Load the training, validation and test data into constants that are
    # attached to the graph.
    tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
    tf_train_labels = tf.constant(train_labels[:train_subset, :])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random values following a (truncated)
    # normal distribution. The biases get initialized to zero.
    weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, NUM_LABELS]))
    biases = tf.Variable(tf.zeros([NUM_LABELS]))
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    NUM_STEPS = 801
    with tf.Session(graph=graph) as session:
      # This is a one-time operation which ensures the parameters get initialized as
      # we described in the graph: random weights for the matrix, zeros for the
      # biases.
      tf.global_variables_initializer().run()
      print("Initialized")
      for step in range(NUM_STEPS):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy
        # arrays.
        _, l, t_pred = session.run([optimizer, loss, train_prediction])
        if step % 100 == 0:
          print("Loss at step %d: %f" % (step, l))
          print("Training Accuracy: %.1f" % accuracy(t_pred, train_labels[:train_subset, :]))
          # Calling .eval() on valid_prediction is basically like calling run(), but
          # just to get that one numpy array. Note that it recomputes all its graph
          # dependencies.
          print("Validation Accuracy: %.1f" % accuracy(valid_prediction.eval(), valid_labels))
      print("Test Accuracy: %.1f" % accuracy(test_prediction.eval(), test_labels))

# gradient_descent()

"""
Let's run this computation and iterate:
"""


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))) / predictions.shape[0]



"""
Let's now switch to stochastic gradient descent training instead, which is much faster.

The graph will be similar, except that instead of holding all the training data into a constant node,
 we create a `Placeholder` node which will be fed actual data at every call of `session.run()`.
"""
BATCH_SIZE = 128


def stochastic_gradient_descent():
  print("No layer - Stochastic Gradient Descent")
  graph = tf.Graph()
  with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE))
    tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    # Variables.
    weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, NUM_LABELS]))
    biases = tf.Variable(tf.zeros([NUM_LABELS]))
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

  num_steps = 3001
  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
      # Pick an offset within the training data, which has been randomized.
      # Note: we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
      batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
      # Prepare a dictionary telling the session where to feed the minibatch.
      # The key of the dictionary is the placeholder node of the graph to be fed,
      # and the value is the numpy array to feed to it.
      feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
      __, l, t_pred = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
      if step % 500 == 0:
        print("Loss at step %d: %f" % (step, l))
        print("Training Accuracy: %.1f" % accuracy(t_pred, batch_labels))
        print("Validation Accuracy: %.1f" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test Accuracy: %.1f" % accuracy(test_prediction.eval(), test_labels))

stochastic_gradient_descent()

"""
---
Problem
-------

Turn the logistic regression example with SGD into a 1-hidden layer neural network with rectified linear units [nn.relu()](https://www.tensorflow.org/versions/r0.7/api_docs/python/nn.html#relu) and 1024 hidden nodes. This model should improve your validation / test accuracy.

---
"""


def log_reg_sgd():
  print("1 layer - Stochastic Gradient Descent")
  graph = tf.Graph()
  with graph.as_default():
    def forward_propagate(inp_layer):
      # Hidden layer
      hidden_input = tf.nn.relu(tf.matmul(inp_layer, input_weights) + input_biases)
      # Output Layer
      return tf.matmul(hidden_input, hidden_weights) + hidden_biases
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE))
    tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    # Variables.
    input_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, NUM_LABELS]))
    input_biases = tf.Variable(tf.zeros([NUM_LABELS]))
    hidden_weights = tf.Variable(tf.truncated_normal([NUM_LABELS, NUM_LABELS]))
    hidden_biases = tf.Variable(tf.zeros([NUM_LABELS]))
    # Forward Propagation
    logits = forward_propagate(tf_train_dataset)
    # Back propagation
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # Predictions
    train_predictions = tf.nn.softmax(logits)
    valid_predictions = tf.nn.softmax(forward_propagate(tf_valid_dataset))
    test_predictions = tf.nn.softmax(forward_propagate(tf_test_dataset))

  num_steps = 3001
  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
      offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
      batch_data = train_dataset[offset:(BATCH_SIZE + offset), :]
      batch_labels = train_labels[offset:(BATCH_SIZE + offset), :]
      feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
      _, l, t_pred = session.run([optimizer, loss, train_predictions], feed_dict=feed_dict)
      if step % 500 == 0:
        print("Loss at step %d: %f" % (step, l))
        print("Training Accuracy: %.1f" % accuracy(t_pred, batch_labels))
        print("Validation Accuracy: %.1f" % accuracy(valid_predictions.eval(), valid_labels))
    print("Test Accuracy: %.1f" % accuracy(test_predictions.eval(), test_labels))

log_reg_sgd()
