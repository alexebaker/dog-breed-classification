import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

train_zip = os.path.join(DATA_DIR, 'train.zip')
test_zip = os.path.join(DATA_DIR, 'test.zip')
label_zip = os.path.join(DATA_DIR, 'labels.csv.zip')
train_npy = os.path.join(DATA_DIR, 'train.npy')
test_npy = os.path.join(DATA_DIR, 'test.npy')
labels_npy = os.path.join(DATA_DIR, 'labels.npy')
mapping_json = os.path.join(DATA_DIR, 'mapping.json')
