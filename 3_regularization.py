"""
Deep Learning
=============

Assignment 3
------------

Previously in `2_fullyconnected.ipynb`, you trained a logistic regression and a neural network model.

The goal of this assignment is to explore regularization techniques.
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import sys
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from six.moves import cPickle as pickle
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
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)


"""
Reformat into a shape that's more adapted to the models we're going to train:
- data as a flat matrix,
- labels as float 1-hot encodings.
"""

IMAGE_SIZE = 28
NUM_LABELS = 10


def reformat(dataset, labels):
  dataset = np.reshape(dataset, (-1, IMAGE_SIZE * IMAGE_SIZE))
  labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

"""
---
Problem 1
---------

Introduce and tune L2 regularization for both logistic and neural network models. Remember that L2 amounts to adding a penalty on the norm of the weights to the loss. In TensorFlow, you can compute the L2 loss for a tensor `t` using `nn.l2_loss(t)`. The right amount of regularization should improve your validation / test accuracy.

---
"""


def log_reg(batch_size, num_steps, beta=0.01, limit_batch=False):
  print("No layer - Stochastic Gradient Descent")
  graph = tf.Graph()
  with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_SIZE * IMAGE_SIZE))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, NUM_LABELS))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    # Variables.
    weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, NUM_LABELS]))
    biases = tf.Variable(tf.zeros([NUM_LABELS]))
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    l2_loss = tf.nn.l2_loss(weights)
    loss = tf.add(loss, beta * l2_loss)
    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

  if limit_batch:
    tr_dataset = train_dataset[:500, :]
    tr_labels = train_labels[:500, :]
  else:
    tr_dataset = train_dataset
    tr_labels = train_labels
  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
      # Pick an offset within the training data, which has been randomized.
      # Note: we could use better randomization across epochs.
      offset = (step * batch_size) % (tr_labels.shape[0] - batch_size)
      batch_data = tr_dataset[offset:(offset + batch_size), :]
      batch_labels = tr_labels[offset:(offset + batch_size), :]
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


def nn_1_hidden_layer(batch_size, num_steps, beta=0.01, use_drop_out=False, use_adaptive_rate=True, limit_batch=False):
  print("1 layer - Neural Network")
  graph = tf.Graph()
  with graph.as_default():
    keep_prob = tf.placeholder(tf.float32)

    def forward_propagate_dropout(inp_layer):
      # Hidden layer
      hidden_input = tf.nn.dropout(tf.nn.relu(tf.matmul(inp_layer, input_weights) + input_biases), keep_prob)
      # Output Layer
      return tf.matmul(hidden_input, hidden_weights) + hidden_biases

    def forward_propagate(inp_layer):
      # Hidden layer
      hidden_input = tf.nn.relu(tf.matmul(inp_layer, input_weights) + input_biases)
      # Output Layer
      return tf.matmul(hidden_input, hidden_weights) + hidden_biases

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_SIZE * IMAGE_SIZE))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, NUM_LABELS))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    # Variables.
    input_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE * IMAGE_SIZE, NUM_LABELS]))
    input_biases = tf.Variable(tf.zeros([NUM_LABELS]))
    hidden_weights = tf.Variable(tf.truncated_normal([NUM_LABELS, NUM_LABELS]))
    hidden_biases = tf.Variable(tf.zeros([NUM_LABELS]))
    # Forward Propagation
    if use_drop_out:
      logits = forward_propagate_dropout(tf_train_dataset)
    else:
      logits = forward_propagate(tf_train_dataset)
    # Back propagation
    l2_loss = tf.nn.l2_loss(input_weights) + tf.nn.l2_loss(input_biases) +\
              tf.nn.l2_loss(hidden_weights) + tf.nn.l2_loss(hidden_biases)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits) + beta*l2_loss)
    if use_adaptive_rate:
      global_step = tf.Variable(0)  # count the number of steps taken.
      learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.90, staircase=True)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    else:
      optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # Predictions
    train_predictions = tf.nn.softmax(logits)
    # Drop outs should only be used on training.
    valid_predictions = tf.nn.softmax(forward_propagate(tf_valid_dataset))
    test_predictions = tf.nn.softmax(forward_propagate(tf_test_dataset))

  if limit_batch:
    tr_dataset = train_dataset[:500, :]
    tr_labels = train_labels[:500, :]
  else:
    tr_dataset = train_dataset
    tr_labels = train_labels
  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
      offset = (step * batch_size) % (tr_labels.shape[0] - batch_size)
      batch_data = tr_dataset[offset:(batch_size + offset), :]
      batch_labels = tr_labels[offset:(batch_size + offset), :]
      feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, keep_prob: 0.5}
      _, l, t_pred = session.run([optimizer, loss, train_predictions], feed_dict=feed_dict)
      if step % 500 == 0:
        print("Loss at step %d: %f" % (step, l))
        print("Training Accuracy: %.1f" % accuracy(t_pred, batch_labels))
        print("Validation Accuracy: %.1f" % accuracy(valid_predictions.eval(), valid_labels))
    print("Test Accuracy: %.1f" % accuracy(test_predictions.eval(), test_labels))

# log_reg(128, 3001)
# nn_1_hidden_layer(128, 3001)

"""
---
Problem 2
---------
Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?

---
"""
# log_reg(128, 3001, limit_batch=True)
# nn_1_hidden_layer(128, 3001, limit_batch=True)

"""
---
Problem 3
---------
Introduce Dropout on the hidden layer of the neural network. Remember: Dropout should only be introduced during training, not evaluation, otherwise your evaluation results would be stochastic as well. TensorFlow provides `nn.dropout()` for that, but you have to make sure it's only inserted during training.

What happens to our extreme overfitting case?

---
"""
nn_1_hidden_layer(128, 3001, use_drop_out=True)


"""
---
Problem 4
---------

Try to get the best performance you can using a multi-layer model! The best reported test accuracy using a deep network is [97.1%](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html?showComment=1391023266211#c8758720086795711595).

One avenue you can explore is to add multiple layers.

Another one is to use learning rate decay:

    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.5, global_step, ...)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

 ---
"""
nn_1_hidden_layer(128, 3001, use_drop_out=False, use_adaptive_rate=True)