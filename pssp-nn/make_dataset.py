import numpy as np
import gzip
import os

TRAIN_PATH = '../pssp-data/cullpdb+profile_6133_filtered.npy.gz'
TEST_PATH = '../pssp-data/cb513+profile_split1.npy.gz'
TRAIN_URL = 'http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz'
TEST_URL = 'http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz'

def download_dataset():
    if not (os.path.isfile(TRAIN_PATH) and os.path.isfile(TEST_PATH)):
        print('[Info] Downloading CB513 and CullPDB dataset ...')
        os.makedirs('../pssp-data', exist_ok=True)
        os.system(f'wget -O {TRAIN_PATH} {TRAIN_URL}')
        os.system(f'wget -O {TEST_PATH} {TEST_URL}')

def make_dataset(path):
    with gzip.open(path, 'rb') as f:
        data = np.load(f)
    data = data.reshape(-1, 700, 57) # 57 features

    X = data[:, :, np.arange(21)] # 20-residues + no-seq
    X = X.transpose(0, 2, 1)
    X = X.astype('float32')

    y = data[:, :, 22:30] # 8-state
    y = np.array([np.dot(yi, np.arange(8)) for yi in y])
    y = y.astype('float32')

    mask = data[:, :, 30] * -1 + 1
    seq_len = mask.sum(axis=1)
    seq_len = seq_len.astype('int')
    
    return X, y, seq_len

if __name__ == '__main__':
    download_dataset()
    X, y, seq_len = make_dataset(TEST_PATH)
    index = 0
    length = seq_len[index]
    print(X[index, :, :length], X[index, :, :length].shape)
    print(y[index][:length])
    print(length)
