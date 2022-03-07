#!/bin/bash

mkdir -p data
wget -c -P data http://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_5926_filtered.npy.gz
wget -c -P data http://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz

pip install -r requirements.txt

python stats.py
python main.py
