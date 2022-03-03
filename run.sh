#!/bin/bash

mkdir -p data
wget -c -P data http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz
wget -c -P data http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_5926_filtered.npy.gz
wget -c -P data http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz

if [ -d torch ]; then
    source torch/bin/activate
else
    python3 -m venv torch
    source torch/bin/activate
    pip install --upgrade pip
    pip install torch numpy pandas matplotlib
fi

python stats.py
python main.py
