from __future__ import print_function
from __future__ import unicode_literals


import argparse
import dog_classifier as dc


def parse_args():
    """Parse CLI arguments.

    :rtype: dict
    :returns: Dictonairy of parsed cli arguments.
    """

    # argument parser object
    parser = argparse.ArgumentParser(
        description='Classifies the testing data using the training data.')

    # Add arguments to the parser
    parser.add_argument(
        '--train-data',
        type=str,
        default=dc.train_zip,
        help='Path to the training data folder.')

    parser.add_argument(
        '--test-data',
        type=str,
        default=dc.test_zip,
        help='Path to the validation data folder.')

    parser.add_argument(
        '--cross-validate',
        type=int,
        default=10,
        help='Perform n fold cross validation.')

    parser.add_argument(
        '--confusion-matrix',
        action='store_true',
        help='Calculate the confusion matrix for the given methods.')

    return vars(parser.parse_args())
