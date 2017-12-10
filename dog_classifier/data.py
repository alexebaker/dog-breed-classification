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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import random
import numpy as np
import tensorflow as tf

from dog_classifier import ARGS


# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 100
CROP_SIZE = 80

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 120
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


breed_ids = {
    'affenpinscher': 1,
    'afghan_hound': 2,
    'african_hunting_dog': 3,
    'airedale': 4,
    'american_staffordshire_terrier': 5,
    'appenzeller': 6,
    'australian_terrier': 7,
    'basenji': 8,
    'basset': 9,
    'beagle': 10,
    'bedlington_terrier': 11,
    'bernese_mountain_dog': 12,
    'black-and-tan_coonhound': 13,
    'blenheim_spaniel': 14,
    'bloodhound': 15,
    'bluetick': 16,
    'border_collie': 17,
    'border_terrier': 18,
    'borzoi': 19,
    'boston_bull': 20,
    'bouvier_des_flandres': 21,
    'boxer': 22,
    'brabancon_griffon': 23,
    'briard': 24,
    'brittany_spaniel': 25,
    'bull_mastiff': 26,
    'cairn': 27,
    'cardigan': 28,
    'chesapeake_bay_retriever': 29,
    'chihuahua': 30,
    'chow': 31,
    'clumber': 32,
    'cocker_spaniel': 33,
    'collie': 34,
    'curly-coated_retriever': 35,
    'dandie_dinmont': 36,
    'dhole': 37,
    'dingo': 38,
    'doberman': 39,
    'english_foxhound': 40,
    'english_setter': 41,
    'english_springer': 42,
    'entlebucher': 43,
    'eskimo_dog': 44,
    'flat-coated_retriever': 45,
    'french_bulldog': 46,
    'german_shepherd': 47,
    'german_short-haired_pointer': 48,
    'giant_schnauzer': 49,
    'golden_retriever': 50,
    'gordon_setter': 51,
    'great_dane': 52,
    'great_pyrenees': 53,
    'greater_swiss_mountain_dog': 54,
    'groenendael': 55,
    'ibizan_hound': 56,
    'irish_setter': 57,
    'irish_terrier': 58,
    'irish_water_spaniel': 59,
    'irish_wolfhound': 60,
    'italian_greyhound': 61,
    'japanese_spaniel': 62,
    'keeshond': 63,
    'kelpie': 64,
    'kerry_blue_terrier': 65,
    'komondor': 66,
    'kuvasz': 67,
    'labrador_retriever': 68,
    'lakeland_terrier': 69,
    'leonberg': 70,
    'lhasa': 71,
    'malamute': 72,
    'malinois': 73,
    'maltese_dog': 74,
    'mexican_hairless': 75,
    'miniature_pinscher': 76,
    'miniature_poodle': 77,
    'miniature_schnauzer': 78,
    'newfoundland': 79,
    'norfolk_terrier': 80,
    'norwegian_elkhound': 81,
    'norwich_terrier': 82,
    'old_english_sheepdog': 83,
    'otterhound': 84,
    'papillon': 85,
    'pekinese': 86,
    'pembroke': 87,
    'pomeranian': 88,
    'pug': 89,
    'redbone': 90,
    'rhodesian_ridgeback': 91,
    'rottweiler': 92,
    'saint_bernard': 93,
    'saluki': 94,
    'samoyed': 95,
    'schipperke': 96,
    'scotch_terrier': 97,
    'scottish_deerhound': 98,
    'sealyham_terrier': 99,
    'shetland_sheepdog': 100,
    'shih-tzu': 101,
    'siberian_husky': 102,
    'silky_terrier': 103,
    'soft-coated_wheaten_terrier': 104,
    'staffordshire_bullterrier': 105,
    'standard_poodle': 106,
    'standard_schnauzer': 107,
    'sussex_spaniel': 108,
    'tibetan_mastiff': 109,
    'tibetan_terrier': 110,
    'toy_poodle': 111,
    'toy_terrier': 112,
    'vizsla': 113,
    'walker_hound': 114,
    'weimaraner': 115,
    'welsh_springer_spaniel': 116,
    'west_highland_white_terrier': 117,
    'whippet': 118,
    'wire-haired_fox_terrier': 119,
    'yorkshire_terrier': 120,
}


def read_image(filename_queue, eval_data=False):
    """Reads and parses examples from CIFAR10 data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
        filename_queue: A queue of strings with the filenames to read from.

    Returns:
        An object representing a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
            for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """
    reader = tf.WholeFileReader()
    filename, f = reader.read(filename_queue)

    image = tf.image.decode_jpeg(f)

    if not eval_data:
        image = tf.image.resize_images(
            image,
            tf.constant([IMAGE_SIZE, IMAGE_SIZE], tf.int32))
    else:
        image = tf.image.resize_images(
            image,
            tf.constant([CROP_SIZE, CROP_SIZE], tf.int32))

    label = tf.py_func(get_label, [filename], tf.int64)

    return image, label


def read_test_image(filename_queue):
    """Reads and parses examples from CIFAR10 data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
        filename_queue: A queue of strings with the filenames to read from.

    Returns:
        An object representing a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
            for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """
    reader = tf.WholeFileReader()
    filename, f = reader.read(filename_queue)

    image = tf.image.decode_jpeg(f)
    image = tf.image.resize_images(
        image,
        tf.constant([CROP_SIZE, CROP_SIZE], tf.int32))

    image_id = tf.py_func(get_file_id, [filename], tf.string)

    return image, image_id


def get_label(filename):
    #file_id = os.path.splitext(os.path.basename(str(filename)))[0]
    #label_mapping = get_label_mapping()
    #return np.array([_breed_2_id(label_mapping[file_id])-1])
    breed_name = os.path.basename(os.path.dirname(filename))
    split_part = re.search('^n[0-9]{8}-', breed_name).group()
    breed_name = breed_name.split(split_part)[-1].lower()
    return np.array([_breed_2_id(breed_name)-1])


def get_file_id(filename):
    return np.array([os.path.splitext(os.path.basename(str(filename)))[0]])


def get_images(eval_data, data_dir, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    if not eval_data:
        filenames = get_filenames(data_dir)
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = get_filenames(data_dir)
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    image, label = read_image(filename_queue, eval_data=eval_data)

    if not eval_data:
        if random.random() < 0.95:
            image = tf.random_crop(image, [CROP_SIZE, CROP_SIZE, 3])
            image = tf.image.random_flip_left_right(image)
            #image = tf.image.random_flip_up_down(image)
            if random.random() < 0.5:
                tf.image.random_brightness(image, max_delta=63)
                tf.image.random_contrast(image, lower=0.2, upper=1.8)
            else:
                tf.image.random_contrast(image, lower=0.2, upper=1.8)
                tf.image.random_brightness(image, max_delta=63)
        else:
            image = tf.image.resize_image_with_crop_or_pad(
                image,
                CROP_SIZE,
                CROP_SIZE)

    image = tf.image.per_image_standardization(image)

    # Set the shapes of tensors.
    image.set_shape([CROP_SIZE, CROP_SIZE, 3])
    label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=16,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def get_test_image(data_dir):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.

    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = get_test_filenames(data_dir)
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames,
                                                    shuffle=False,
                                                    capacity=10357)

    # Read examples from files in the filename queue.
    image, image_id = read_test_image(filename_queue)
    image = tf.image.per_image_standardization(image)

    # Set the shapes of tensors.
    image.set_shape([CROP_SIZE, CROP_SIZE, 3])
    image_id.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    test_image, test_id = tf.train.batch(
        [image, image_id],
        batch_size=1,
        num_threads=16,
        capacity=10357)

    return test_image, tf.reshape(test_id, [1])


def get_label_mapping(data_dir=ARGS.data_dir):
    label_mapping = {}
    label_file = os.path.join(data_dir, 'labels.csv')
    with open(label_file, 'r') as f:
        for line in f.readlines()[1:]:
            file_id, breed = line.strip().split(',')
            label_mapping[file_id] = breed
    return label_mapping


def get_filenames(data_dir):
    filenames = []
    image_dir = os.path.join(data_dir, 'Images')
    #train_dir = os.path.join(data_dir, 'train')
    #for root, dirs, files in os.walk(train_dir):
    #    for f in files:
    #        if f.endswith('.jpg'):
    #            filenames.append(os.path.join(root, f))

    #random.seed(33)
    #random.shuffle(filenames)

    #split_idx = int(0.9 * len(filenames))
    #train_filenames = filenames[:split_idx]
    #test_filenames = filenames[split_idx:]

    for root, dirs, files in os.walk(image_dir):
        for f in files:
            if f.endswith('.jpg'):
                filenames.append(os.path.join(root, f))
    return filenames


def get_test_filenames(data_dir):
    filenames = []
    train_dir = os.path.join(data_dir, 'test')
    for root, dirs, files in os.walk(train_dir):
        for f in files:
            if f.endswith('.jpg'):
                filenames.append(os.path.join(root, f))
    return filenames


def _breed_2_id(breed):
    return breed_ids[breed]


def _id_2_breed(breed_id):
    for breed, idx in breed_ids.iteritems():
        if idx == breed_id:
            return breed
    return "affenpinscher"
