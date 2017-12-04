from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import sys
import itertools
import math
import numpy as np
import tensorflow as tf
import data as dd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix


tf.set_random_seed(0.0)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 100, 100, 3])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 120])
# variable learning rate
lr = tf.placeholder(tf.float32)
# test flag for batch norm
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
# dropout probability
pkeep = tf.placeholder(tf.float32)
pkeep_conv = tf.placeholder(tf.float32)

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()

def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 24  # first convolutional layer output depth
L = 48  # second convolutional layer output depth
M = 64  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, 3, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

W4 = tf.Variable(tf.truncated_normal([40000, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 120], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [120]))

# The model
# batch norm scaling is not useful with relus
# batch norm offsets are used instead of biases
stride = 1  # output is 28x28
Y1l = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1, convolutional=True)
Y1r = tf.nn.relu(Y1bn)
Y1 = tf.nn.dropout(Y1r, pkeep_conv, compatible_convolutional_noise_shape(Y1r))
stride = 2  # output is 14x14
Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2, convolutional=True)
Y2r = tf.nn.relu(Y2bn)
Y2 = tf.nn.dropout(Y2r, pkeep_conv, compatible_convolutional_noise_shape(Y2r))
stride = 2  # output is 7x7
Y3l = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3, convolutional=True)
Y3r = tf.nn.relu(Y3bn)
Y3 = tf.nn.dropout(Y3r, pkeep_conv, compatible_convolutional_noise_shape(Y3r))

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 40000])

Y4l = tf.matmul(YY, W4)
Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)
Y4r = tf.nn.relu(Y4bn)
Y4 = tf.nn.dropout(Y4r, pkeep)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = dd.get_train_data()
    batch_X = batch_X[i*100:(i+1)*100, :, :, :]
    batch_Y = batch_Y[i*100:(i+1)*100, :]

    # learning rate decay
    max_learning_rate = 0.02
    min_learning_rate = 0.0001
    decay_speed = 1600
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False, pkeep: 0.75, pkeep_conv: 1.0})
    sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 1.0, pkeep_conv: 1.0})

for i in range(3): training_step(i, i % 100 == 0, i % 20 == 0)


def train_data(method, data, target):
    """Trains the given data with the specified method.

    :type method: str
    :param method: Name of classifier to use (lr, svm, knn, nn)

    :type data: np.array
    :param data: input data array of (n_samples, n_features)

    :type target: np.array
    :param target: Labels for the data input (n_samples,)

    :rtype: obj
    :retruns: trained sklearn classifier object
    """
    classifier = get_classifier(method)

    if classifier:
        classifier.fit(data, target)
    return classifier


def classify_data(classifier, data):
    """Classifies the given data with the specified method.

    :type classifier: obj
    :param method: Classification object from sklearn

    :type data: np.array
    :param data: input data array of (n_samples, n_features)

    :rtype: np.array
    :returns: array of classification labels (n_samples,)
    """
    return classifier.predict(data)


def get_classifier(method):
    """Creates an sklearn classifier object.

    :type method: str
    :param method: Name of classifier to use (lr, svm, knn, nn)

    :rtype: obj
    :retruns: instantiated sklearn classifier object
    """
    classifier = None
    return classifier


def perform_cross_validation(method, data, target, folds=10):
    """Performs cross validation for the given method and data.

    :type method: str
    :param method: Name of classifier to use (lr, svm, knn, nn)

    :type data: np.array
    :param data: input data array of (n_samples, n_features)

    :type target: np.array
    :param target: Labels for the data input (n_samples,)

    :type folds: int
    :param folds: Number of folds to perform in cross validation

    :rtype: np.array
    :retruns: Accuracy of each fold (folds,)
    """
    classifier = get_classifier(method)
    accuracy = cross_validate(classifier, data, target, cv=folds, n_jobs=-1)
    return accuracy['test_score']


def get_confusion_matrix(method, feature, data, target):
    """Generates a confusion matrix for the given method and data.

    :type method: str
    :param method: Name of classifier to use (lr, svm, knn, nn)

    :type feature: str
    :param feature: name of the feature being used

    :type data: np.array
    :param data: input data array of (n_samples, n_features)

    :type target: np.array
    :param target: Labels for the data input (n_samples,)

    :type plot: bool
    :param plot: Whether or not to plot the confusion matrix graphically

    :rtype: np.array
    :retruns: array representing the confusion matrix (n_labels, n_labels)
    """
    training_data, testing_data, training_labels, testing_labels = train_test_split(
        data,
        target,
        test_size=0.2,
        train_size=0.8,
        random_state=42)

    classifier = train_data(method, training_data, training_labels)
    classification = classify_data(classifier, testing_data)
    return confusion_matrix(testing_labels, classification)


def plot_confusion_matrix(cm, feature, method):
    """This function plots the confusion matrix using matplotlib.

    **NOTE** This code comes from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    :type cm: np.array
    :param cm: confusion matrix to plot

    :type feature: str
    :param feature: name of the feature that generated the matrix

    :type method: str
    :param method: name of the method that generated the matrix
    """
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        cmap = plt.cm.Blues
    except ImportError:
        print(sys.stderr, 'matplotlib is not installed, so plot could not be generated.')
        return

    title = "%s with %s Confusion Matrix" % (method.upper(), feature.upper())
    classes = [genre for genre, _ in sorted(genre_mapping.items(), key=lambda x: x[1])]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('%s_%s_cm_plot.png' % (feature, method))
    return
