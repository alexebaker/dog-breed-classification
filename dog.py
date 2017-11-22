from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np

from dog_classifier import cli
from dog_classifier import data
from dog_classifier import classifiers
from dog_classifier import BASE_DIR


data_dir = os.path.join(BASE_DIR, 'data')


def main():
    """Main entry point."""
    # Parse the command line arguments
    cli_args = cli.parse_args()

    # initialize data
    train_data, labels = data.get_train_data(train_zip=cli_args['train_data'])
    test_data, mapping = data.get_test_data(test_zip=cli_args['test_data'])
    return


if __name__ == "__main__":
    main()
