import numpy as np
import torch
import os

TRAIN_FILE = 'cullpdb+profile_6133_filtered.npy.gz'
TEST_FILE = 'cb513+profile_split1.npy.gz'

TRAIN_URL = 'http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz'
TEST_URL = 'http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz'

def download_datasets(data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)

    # data acquisition
    print('Downloading %s ...' % TRAIN_URL)
    os.system('wget -c --quiet -O %s %s' % (os.path.join(data_dir, TRAIN_FILE), TRAIN_URL))
    print('Downloading %s ...' % TEST_URL)
    os.system('wget -c --quiet -O %s %s' % (os.path.join(data_dir, TEST_FILE), TEST_URL))

def make_dataset(path, data_dir='data'):
    data = np.load(os.path.join(data_dir, path))
    data = data.reshape(-1, 700, 57) # original 57 features

    X = data[:, :, np.arange(21)] # 20-residues + non-seq
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
    download_datasets()

    print('Build dataset files ...')
    X_train, y_train, seq_len_train = make_dataset(TRAIN_FILE)
    X_test, y_test, seq_len_test = make_dataset(TEST_FILE)

    data = {'X_train':X_train,
            'y_train':y_train,
            'seq_len_train':seq_len_train,
            'X_test':X_test,
            'y_test':y_test,
            'seq_len_test':seq_len_test}

    torch.save(data, 'data/dataset.pt')
