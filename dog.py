from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from dog_classifier import train, test


def main():
    """Main entry point."""
    tf.app.run(main=train.start_training)
    tf.app.run(main=test.start_eval)
    return


if __name__ == "__main__":
    main()
