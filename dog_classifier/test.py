# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time
import numpy as np
import tensorflow as tf

from datetime import datetime

from dog_classifier import ARGS, classifier, data


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.

    Args:
        saver: Saver.
        summary_writer: Summary writer.
        top_k_op: Top K op.
        summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(
            os.path.join(ARGS.data_dir, 'checkpoints'))
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord,
                                                 daemon=True, start=True))

            num_iter = int(math.ceil(ARGS.num_examples / ARGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * ARGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %3.2f%%' % (datetime.now(), precision*100))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
    return


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        images, labels = classifier.get_images(True)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = classifier.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            classifier.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(ARGS.data_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if ARGS.run_once:
                break
            time.sleep(ARGS.eval_interval_secs)


def start_eval(argv=None):
    evaluate()
    return


def start_test(argv=None):
    save_to_csv(True, None, None)
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        images, image_ids = data.get_test_images(ARGS.data_dir, ARGS.batch_size)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = classifier.inference(images)
        predictions = tf.nn.softmax(logits)

        variable_averages = tf.train.ExponentialMovingAverage(
            classifier.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(os.path.join(ARGS.data_dir, 'checkpoints'))
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord,
                                                    daemon=True, start=True))

                num_iter = int(math.ceil(10357 / ARGS.batch_size))
                step = 0
                while step < num_iter and not coord.should_stop():
                    preds = sess.run([predictions])
                    img_ids = sess.run([image_ids])
                    save_to_csv(False, preds[0], img_ids[0])
                    step += 1

            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)
        return


def save_to_csv(first_run, predictions, image_ids):
    csv_file = os.path.join(ARGS.data_dir, 'classification.csv')
    if first_run:
        with open(csv_file, 'w') as f:
            print("id,affenpinscher,afghan_hound,african_hunting_dog,airedale,american_staffordshire_terrier,appenzeller,australian_terrier,basenji,basset,beagle,bedlington_terrier,bernese_mountain_dog,black-and-tan_coonhound,blenheim_spaniel,bloodhound,bluetick,border_collie,border_terrier,borzoi,boston_bull,bouvier_des_flandres,boxer,brabancon_griffon,briard,brittany_spaniel,bull_mastiff,cairn,cardigan,chesapeake_bay_retriever,chihuahua,chow,clumber,cocker_spaniel,collie,curly-coated_retriever,dandie_dinmont,dhole,dingo,doberman,english_foxhound,english_setter,english_springer,entlebucher,eskimo_dog,flat-coated_retriever,french_bulldog,german_shepherd,german_short-haired_pointer,giant_schnauzer,golden_retriever,gordon_setter,great_dane,great_pyrenees,greater_swiss_mountain_dog,groenendael,ibizan_hound,irish_setter,irish_terrier,irish_water_spaniel,irish_wolfhound,italian_greyhound,japanese_spaniel,keeshond,kelpie,kerry_blue_terrier,komondor,kuvasz,labrador_retriever,lakeland_terrier,leonberg,lhasa,malamute,malinois,maltese_dog,mexican_hairless,miniature_pinscher,miniature_poodle,miniature_schnauzer,newfoundland,norfolk_terrier,norwegian_elkhound,norwich_terrier,old_english_sheepdog,otterhound,papillon,pekinese,pembroke,pomeranian,pug,redbone,rhodesian_ridgeback,rottweiler,saint_bernard,saluki,samoyed,schipperke,scotch_terrier,scottish_deerhound,sealyham_terrier,shetland_sheepdog,shih-tzu,siberian_husky,silky_terrier,soft-coated_wheaten_terrier,staffordshire_bullterrier,standard_poodle,standard_schnauzer,sussex_spaniel,tibetan_mastiff,tibetan_terrier,toy_poodle,toy_terrier,vizsla,walker_hound,weimaraner,welsh_springer_spaniel,west_highland_white_terrier,whippet,wire-haired_fox_terrier,yorkshire_terrier", file=f)
    else:
        with open(csv_file, 'a') as f:
            for idx, preds in enumerate(predictions):
                print("%s,%s" % (image_ids[idx], ','.join(map(str, preds.tolist()))), file=f)
    return
