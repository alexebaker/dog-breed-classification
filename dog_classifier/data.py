from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import os
import scipy
import json
import math
import numpy as np

from sklearn.preprocessing import Normalizer

from dog_classifier import BASE_DIR


def read_training_data(folder='', npy_file='', label_file=''):
    """Reads the training data files.

    :type folder: str
    :param: path to the folder of music files

    :type data_file: str
    :param: path to a npy file of pre-processed audio data

    :type label_file: str
    :param: path to a npy file that has the labels of the data_file

    :rtype: (np.array, np.array)
    :returns: return np.array of audio_data and np.array of labels for the audio data
    """
    if os.path.exists(data_file) and os.path.exists(label_file):
        audio_data = np.load(data_file)
        labels = np.load(label_file)
    else:
        pass
    return None


def get_labels(label_file=''):
    """Reads the label files.

    :type label_file: str
    :param: path to a npy file that has the labels of the data_file

    :rtype: np.array
    :returns: np.array of labels for the audio data (n_samples,)
    """
    if os.path.exists(label_file):
        labels = np.load(label_file)
    else:
        _, labels = read_training_data_files()
    return labels


def read_testing_data(folder='', npy_file='', mapping_file=''):
    """Reads the validation music files.

    :type folder: str
    :param: path to the folder of validation music files

    :type data_file: str
    :param: path to a npy file of pre-processed validation data

    :type mapping_file: str
    :param: path to a json file that has a mapping of which filename is which row in the validation data.

    :rtype: (np.array, dict)
    :returns: return np.array of audio_data and dict of the mapping
    """
    if os.path.exists(data_file) and os.path.exists(mapping_file):
        validation_data = np.load(data_file)
        with open(mapping_file, 'r') as f:
            validation_mapping = json.load(f)
    else:
        pass
    return validation_data, validation_mapping


def normalize_data(data):
    """Normalizes the given data.

    :type data: np.array
    :param: data to normalize (n_smaples, n_features)

    :rtype: np.array
    :returns: np.array of normalized data (n_samples, n_features)
    """
    norm = Normalizer()
    return norm.fit_transform(data)


def save_classification(classification, classification_file, validation_mapping):
    """Saves the classification from the classification algorithm.

    :type classification: list
    :param classification: The classification output from the classifier.

    :type classification_file: File Object
    :param classification_file: File to write the classification to.
    """
    with open(classification_file, 'w') as f:
        print("id,class", file=f)
        for idx, label in enumerate(classification):
            print("%s,%s" % (validation_mapping[str(idx)], str(label)),
                  file=f)
    return
