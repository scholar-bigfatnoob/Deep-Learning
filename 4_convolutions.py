"""
Deep Learning
=============

Assignment 4
------------

Previously in `2_fullyconnected.ipynb` and `3_regularization.ipynb`, we trained fully connected networks to classify [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) characters.

The goal of this assignment is make the neural network convolutional.
"""

from __future__ import print_function, division
import sys
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from six.moves import cPickle as pickle
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"

# pickle_file = 'data/notMNIST.pkl'
pickle_file = 'data/cleaned_notMNIST.pkl'

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
Reformat into a TensorFlow-friendly shape:
- convolutions need the image data formatted as a cube (width by height by #channels)
- labels as float 1-hot encodings.
"""
IMAGE_SIZE = 28
NUM_LABELS = 10
NUM_CHANNELS = 1 #greyscale


def reformat(dataset, labels):
  dataset = np.reshape(dataset, (-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32)
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


BATCH_SIZE = 16
PATCH_SIZE = 5
DEPTH = 16
NUM_HIDDEN = 64

"""
Let's build a small network with two convolutional layers, followed
by one fully connected layer. Convolutional networks are more expensive
computationally, so we'll limit its depth and number of fully connected nodes.
"""



def cnn_2_layers(tf_train_dataset, tf_train_labels, tf_valid_dataset, tf_test_dataset):
  # Variables
  ## First Convolution Layer
  layer1_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH], stddev=0.1))
  layer1_bias = tf.Variable(tf.zeros([DEPTH]))
  ## Second Convolution Layer
  layer2_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, DEPTH, DEPTH], stddev=0.1))
  layer2_bias = tf.Variable(tf.constant(1.0, shape=[DEPTH]))
  ## Fully Connected Layer
  layer3_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * DEPTH, NUM_HIDDEN], stddev=0.1))
  layer3_bias = tf.Variable(tf.constant(1.0, shape=[NUM_HIDDEN]))
  ## Output Layer
  layer4_weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_LABELS], stddev=0.1))
  layer4_bias = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

  # Model
  def model(data):
    conv1 = tf.nn.conv2d(data, layer1_weights, strides=[1, 2, 2, 1], padding='SAME')
    hidden1 = tf.nn.relu(conv1 + layer1_bias)
    conv2 = tf.nn.conv2d(hidden1, layer2_weights, strides=[1, 2, 2, 1], padding='SAME')
    hidden2 = tf.nn.relu(conv2 + layer2_bias)
    fcc_shape = hidden2.get_shape().as_list()
    hidden2 = tf.reshape(hidden2, [fcc_shape[0], fcc_shape[1] * fcc_shape[2] * fcc_shape[3]])
    hidden3 = tf.nn.relu(tf.matmul(hidden2, layer3_weights) + layer3_bias)
    return tf.matmul(hidden3, layer4_weights) + layer4_bias

  # Training Computation
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

  # Optimizer
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

  # Predictions
  train_predictions = tf.nn.softmax(logits)
  test_predictions = tf.nn.softmax(model(tf_test_dataset))
  valid_predictions = tf.nn.softmax(model(tf_valid_dataset))
  return optimizer, loss, train_predictions, valid_predictions, test_predictions


def conv_net(num_steps, cnn_type):
  graph = tf.Graph()
  with graph.as_default():
    # Input
    tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    optimizer, loss, train_predictions, valid_predictions, \
    test_predictions = cnn_type(tf_train_dataset, tf_train_labels, tf_valid_dataset, tf_test_dataset)

  with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in xrange(num_steps):
      offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
      batch_data = train_dataset[offset:(offset + BATCH_SIZE), :, :, :]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
      feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
      _, l, predictions = session.run([optimizer, loss, train_predictions], feed_dict=feed_dict)
      if step % 50 == 0:
        print('Minibatch loss at step %d: %f' % (step, l))
        print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
        print('Validation accuracy: %.1f%%' % accuracy(valid_predictions.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_predictions.eval(), test_labels))

# conv_net(1001, cnn_2_layers)


def cnn_2_layers_maxpool(tf_train_dataset, tf_train_labels, tf_valid_dataset, tf_test_dataset):
  # Variables
  ## First Convolution Layer
  layer1_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH], stddev=0.1))
  layer1_bias = tf.Variable(tf.zeros([DEPTH]))
  ## Second Convolution Layer
  layer2_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, DEPTH, DEPTH], stddev=0.1))
  layer2_bias = tf.Variable(tf.constant(1.0, shape=[DEPTH]))
  ## Fully Connected Layer
  layer3_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * DEPTH, NUM_HIDDEN], stddev=0.1))
  layer3_bias = tf.Variable(tf.constant(1.0, shape=[NUM_HIDDEN]))
  ## Output Layer
  layer4_weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_LABELS], stddev=0.1))
  layer4_bias = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

  # Model
  def model(data):
    conv1 = tf.nn.conv2d(data, layer1_weights, strides=[1, 1, 1, 1], padding='SAME')
    hidden1 = tf.nn.relu(conv1 + layer1_bias)
    hidden1 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.conv2d(hidden1, layer2_weights, strides=[1, 1, 1, 1], padding='SAME')
    hidden2 = tf.nn.relu(conv2 + layer2_bias)
    hidden2 = tf.nn.max_pool(hidden2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    fcc_shape = hidden2.get_shape().as_list()
    hidden2 = tf.reshape(hidden2, [fcc_shape[0], fcc_shape[1] * fcc_shape[2] * fcc_shape[3]])
    hidden3 = tf.nn.relu(tf.matmul(hidden2, layer3_weights) + layer3_bias)
    return tf.matmul(hidden3, layer4_weights) + layer4_bias

  # Training Computation
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

  # Optimizer
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

  # Predictions
  train_predictions = tf.nn.softmax(logits)
  test_predictions = tf.nn.softmax(model(tf_test_dataset))
  valid_predictions = tf.nn.softmax(model(tf_valid_dataset))
  return optimizer, loss, train_predictions, valid_predictions, test_predictions

# conv_net(1001, cnn_2_layers_maxpool)

def le_net_5(tf_train_dataset, tf_train_labels, tf_valid_dataset, tf_test_dataset):
  keep_prob = 0.9
  beta = 0.01
  # Variables
  ## First Convolution Layer
  layer1_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, NUM_CHANNELS, DEPTH], stddev=0.1))
  layer1_bias = tf.Variable(tf.zeros([DEPTH]))
  ## Second Convolution Layer
  layer2_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, DEPTH, DEPTH], stddev=0.1))
  layer2_bias = tf.Variable(tf.constant(1.0, shape=[DEPTH]))
  ## Fully Connected Layer
  layer3_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * DEPTH, NUM_HIDDEN], stddev=0.1))
  layer3_bias = tf.Variable(tf.constant(1.0, shape=[NUM_HIDDEN]))
  ## Output Layer
  layer4_weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_LABELS], stddev=0.1))
  layer4_bias = tf.Variable(tf.constant(1.0, shape=[NUM_LABELS]))

  # Model
  def model(data, is_train=False):
    conv1 = tf.nn.conv2d(data, layer1_weights, strides=[1, 1, 1, 1], padding='SAME')
    hidden1 = tf.nn.relu(conv1 + layer1_bias)
    hidden1 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    if is_train:
      hidden1 = tf.nn.dropout(hidden1, keep_prob)
    conv2 = tf.nn.conv2d(hidden1, layer2_weights, strides=[1, 1, 1, 1], padding='SAME')
    hidden2 = tf.nn.relu(conv2 + layer2_bias)
    hidden2 = tf.nn.max_pool(hidden2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    if is_train:
      hidden2 = tf.nn.dropout(hidden2, keep_prob)
    fcc_shape = hidden2.get_shape().as_list()
    hidden2 = tf.reshape(hidden2, [fcc_shape[0], fcc_shape[1] * fcc_shape[2] * fcc_shape[3]])
    hidden3 = tf.nn.relu(tf.matmul(hidden2, layer3_weights) + layer3_bias)
    if is_train:
      hidden3 = tf.nn.dropout(hidden3, keep_prob)
    return tf.matmul(hidden3, layer4_weights) + layer4_bias

  # Training Computation
  logits = model(tf_train_dataset, is_train=True)
  l2_loss = tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer1_bias) \
            + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer2_bias) \
            + tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_bias) \
            + tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer4_bias)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits) + beta * l2_loss)

  # Optimizer
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.05, global_step, 1000, 0.90, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  # Predictions
  train_predictions = tf.nn.softmax(logits)
  test_predictions = tf.nn.softmax(model(tf_test_dataset))
  valid_predictions = tf.nn.softmax(model(tf_valid_dataset))
  return optimizer, loss, train_predictions, valid_predictions, test_predictions

conv_net(1001, le_net_5)
