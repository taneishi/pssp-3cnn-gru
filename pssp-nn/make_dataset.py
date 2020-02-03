# Original Code : https://github.com/alrojo/CB513/blob/master/data.py
import numpy as np
import subprocess
import gzip
import os

TRAIN_PATH = '../pssp-data/cullpdb+profile_6133_filtered.npy.gz'
TEST_PATH = '../pssp-data/cb513+profile_split1.npy.gz'
TRAIN_URL = 'http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz'
TEST_URL = 'http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz'

def download_dataset():
    print('[Info] Downloading CB513 dataset ...')
    if not (os.path.isfile(TRAIN_PATH) and os.path.isfile(TEST_PATH)):
        os.makedirs('../pssp-data', exist_ok=True)
        os.system(f'wget -O {TRAIN_PATH} {TRAIN_URL}')
        os.system(f'wget -O {TEST_PATH} {TEST_URL}')

def make_dataset(path):
    data = np.load(gzip.open(path, 'rb'))
    data = data.reshape(-1, 700, 57)

    X = data[:, :, np.arange(21)]
    X = X.transpose(0, 2, 1)
    X = X.astype('float32')

    y = data[:, :, 22:30]
    y = np.array([np.dot(yi, np.arange(8)) for yi in y])
    y = y.astype('float32')

    mask = data[:, :, 30] * -1 + 1
    seq_len = mask.sum(axis=1)
    seq_len = seq_len.astype('float32')

    return X, y, seq_len

'''
def make_dataset(path):
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder

    data = pd.read_csv(path, header=None, sep=' ')

    oe = OneHotEncoder(categories='auto', sparse=False)
    X = oe.fit_transform(data)

    X = np.resize(X, (data.shape[0], N_LEN, N_AA))
    X = X.transpose(0, 2, 1)
    X = X.astype('float32')

    data = pd.read_csv(path.replace('aa', 'pss'), header=None, sep=' ')

    le = LabelEncoder()
    le.fit(data[0])
    for col in data.columns:
        data[col] = le.transform(data[col])
    y = data.astype('float32').values

    seq_len = [N_LEN] * data.shape[0]
    seq_len = np.asarray(seq_len).astype('float32')

    return X, y, seq_len
'''
