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
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Number of images to process in a batch.')

    parser.add_argument('--data-dir', type=str, default=dc.DATA_DIR,
                        help='Path to the CIFAR-10 data directory.')

    parser.add_argument('--eval-interval-secs', type=int, default=60*5,
                        help='How often to run the eval.')

    parser.add_argument('--num-examples', type=int, default=10000,
                        help='Number of examples to run.')

    parser.add_argument('--run-once', type=bool, default=False,
                        help='Whether to run eval only once.')

    parser.add_argument('--max-steps', type=int, default=1000000,
                        help='Number of batches to run.')

    parser.add_argument('--log-device-placement', type=bool, default=False,
                        help='Whether to log device placement.')

    parser.add_argument('--log-frequency', type=int, default=10,
                        help='How often to log results to the console.')

    parser.add_argument('--eval', action='store_true',
                        help='Whether to run eval.')

    parser.add_argument('--test', action='store_true',
                        help='Whether to run eval.')

    parser.add_argument('--no-gpu', action='store_true',
                        help='Whether to run eval.')

    return parser.parse_args()
