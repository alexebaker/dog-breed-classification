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
    return


if __name__ == "__main__":
    main()
