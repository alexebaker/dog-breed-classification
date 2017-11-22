from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import os
import json
import numpy as np
import dog_classifier as dc

from zipfile import ZipFile

from scipy.sparse import csr_matrix

from skimage.io import imread
from skimage.transform import resize

from sklearn.preprocessing import Normalizer



max_img_height = 2562
min_img_height = 102
max_img_width = 3264
min_img_width = 97

img_height = 100
img_width = 100
n_channels = 3

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


def get_train_data(train_zip=dc.train_zip,
                   train_npy=dc.train_npy,
                   labels_npy=dc.labels_npy):
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
    if os.path.exists(train_npy) and os.path.exists(labels_npy):
        train_data = np.load(train_npy)
        labels = np.load(labels_npy)
    else:
        zf = ZipFile(train_zip)
        images = [img for img in zf.namelist() if img.endswith('.jpg')]
        label_mapping = get_label_mapping()
        n_images = len(images)
        train_data = np.zeros(
            (n_images, img_height, img_width, n_channels),
            dtype=np.float64)
        labels = np.zeros((n_images,), dtype=np.int16)
        for idx, image in enumerate(images):
            imf = zf.open(image)
            file_id = os.path.splitext(os.path.basename(image))[0]
            img = resize(imread(imf), (img_height, img_width), mode='reflect')
            train_data[idx, :, :, :] = img
            labels[idx] = _breed_2_id(label_mapping[file_id])
        np.save(train_npy, train_data)
        np.save(labels_npy, labels)
    return train_data, labels


def get_labels(labels_npy=dc.labels_npy):
    """Reads the label files.

    :type label_file: str
    :param: path to a npy file that has the labels of the data_file

    :rtype: np.array
    :returns: np.array of labels for the audio data (n_samples,)
    """
    if os.path.exists(labels_npy):
        labels = np.load(labels_npy)
    else:
        _, labels = get_train_data()
    return labels


def get_label_mapping(labels_zip=dc.label_zip):
    zf = ZipFile(labels_zip, 'r')
    csv = zf.open('labels.csv')
    label_mapping = {}
    for line in csv.readlines()[1:]:
        file_id, breed = line.strip().split(',')
        label_mapping[file_id] = breed
    return label_mapping


def get_test_data(test_zip=dc.test_zip,
                  test_npy=dc.test_npy,
                  mapping_json=dc.mapping_json):
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
    if os.path.exists(test_npy) and os.path.exists(mapping_json):
        test_data = np.load(test_npy)
        with open(mapping_json, 'r') as f:
            test_mapping = json.load(f)
    else:
        zf = ZipFile(test_zip)
        images = [img for img in zf.namelist() if img.endswith('.jpg')]
        n_images = len(images)
        test_mapping = {}
        test_data = np.zeros(
            (n_images, img_height, img_width, n_channels),
            dtype=np.float64)
        for idx, image in enumerate(images):
            imf = zf.open(image)
            file_id = os.path.splitext(os.path.basename(image))[0]
            img = resize(imread(imf), (img_height, img_width), mode='reflect')
            test_data[idx, :, :, :] = img
            test_mapping[str(idx)] = file_id
        np.save(test_npy, test_data)
        with open(mapping_json, 'w') as f:
            json.dump(test_mapping, f)
    return test_data, test_mapping


def get_test_mapping(mapping_json=dc.mapping_json):
    """Reads the mapping json file.

    :type mapping_file: str
    :param: path to a json file that has a mapping of which filename is which row in the validation data.

    :rtype: dict
    :returns: dict of the mapping
    """
    if os.path.exists(mapping_json):
        with open(mapping_json, 'r') as f:
            test_mapping = json.load(f)
    else:
        _, test_mapping = get_test_data()
    return test_mapping


def normalize_data(data):
    """Normalizes the given data.

    :type data: np.array
    :param: data to normalize (n_smaples, n_features)

    :rtype: np.array
    :returns: np.array of normalized data (n_samples, n_features)
    """
    norm = Normalizer()
    return norm.fit_transform(data)


def save_classification(classification, classification_csv, test_mapping):
    """Saves the classification from the classification algorithm.

    :type classification: list
    :param classification: The classification output from the classifier.

    :type classification_file: File Object
    :param classification_file: File to write the classification to.
    """
    with open(classification_csv, 'w') as f:
        print("id,class", file=f)
        for idx, breed in enumerate(classification):
            print("%s,%s" % (test_mapping[str(idx)], str(breed)),
                  file=f)
    return


def _breed_2_id(breed):
    return breed_ids[breed]


def _id_2_breed(breed_id):
    for breed, idx in breed_ids.iteritems():
        if idx == breed_id:
            return breed
    return "affenpinscher"
