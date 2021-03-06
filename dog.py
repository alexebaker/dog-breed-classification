from __future__ import print_function
from __future__ import unicode_literals

import os
import tensorflow as tf

from dog_classifier import ARGS, train, test


def main():
    """Main entry point."""

    if ARGS.no_gpu:
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = ''
        except KeyError:
            pass

    if not ARGS.test:
        if not ARGS.eval:
            tf.app.run(main=train.start_training)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ''
            tf.app.run(main=test.start_eval)
    else:
        ARGS.batch_size = 1
        tf.app.run(main=test.start_test)
    return


if __name__ == "__main__":
    main()
