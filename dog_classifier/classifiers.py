from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import sys
import itertools
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix


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
