# Example datasets of Protein Secondary Structure Prediction

Protein secondary structure prediction datasets, CullPDB6133 and CB513, and simple implementation in PyTorch.

## Dataset

This directory includes datasets used in ICML 2014 Deep Supervised and Convolutional Generative Stochastic Network paper.

Update 2018-10-28:
The original 'cullpdb+profile_6133.npy.gz' and 'cullpdb+profile_6133_filtered.npy.gz' files uploaded contain duplicates.
The fixed files with duplicates removed are 'cullpdb+profile_5926.npy.gz' and 'cullpdb+profile_5926_filtered.npy.gz'.

The corresponding dataset division for the cullpdb+profile_5926.npy.gz dataset is

- `[0,5430)` training
- `[5435,5690)` test
- `[5690,5926)` validation

As described in the paper two datasets are used. Both are based on protein structures from cullpdb servers.
The difference is that the first one is divided to training/validation/test set,
while the second one is filtered to remove redundancy with CB513 dataset (for the purpose of testing performance on CB513 dataset).

cullpdb+profile_6133.npy.gz is the one with training/validation/test set division;
cullpdb+profile_6133_filtered.npy.gz is the one after filtering for redundancy with CB513. this is used for evaluation on CB513.
cb513+profile_split1.npy.gz is the CB513 features I used.
Note that one of the sequences in CB513 is longer than 700 amino acids, and it is splited to two overlapping sequences and these are the last two samples (i.e. there are 514 rows instead of 513).

It is currently in numpy format as a (N protein x k features) matrix. You can reshape it to (N protein x 700 amino acids x 57 features) first.

The 57 features are:
- `[0,22)`: amino acid residues, with the order of 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq'
- `[22,31)`: Secondary structure labels, with the sequence of 'L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq'
- `[31,33)`: N- and C- terminals;
- `[33,35)`: relative and absolute solvent accessibility, used only for training. (absolute accessibility is thresholded at 15; relative accessibility is normalized by the largest accessibility value in a protein and thresholded at 0.15; original solvent accessibility is computed by DSSP)
- `[35,57)`: sequence profile. Note the order of amino acid residues is ACDEFGHIKLMNPQRSTVWXY and it is different from the order for amino acid residues

The last feature of both amino acid residues and secondary structure labels just mark end of the protein sequence.
`[22,31)` and `[33,35)` are hidden during testing.

The dataset division for the first cullpdb+profile_6133.npy.gz dataset is
- `[0,5600)` training
- `[5605,5877)` test
- `[5877,6133)` validation

For the filtered dataset cullpdb+profile_6133_filtered.npy.gz, all proteins can be used for training and test on CB513 dataset.

|sequence length (train)|sequence length (test)|
|:-:|:-:|
|![](figure/seqlen_train.png)|![](figure/seqlen_test.png)|

|amino acid (train)|amino acid (test)|
|:-:|:-:|
|![](figure/amino_acid_train.png)|![](figure/amino_acid_test.png)|

|secondary structure label(train)|secondary structure label (test)|
|:-:|:-:|
|![](figure/secondary_structure_train.png)|![](figure/secondary_structure_test.png)|

## References

- [Datasets for ICML 2014 Deep Supervised and Convolutional Generative Stochastic Network paper](http://www.princeton.edu/~jzthree/datasets/ICML2014/)
- [takatex/protein-secondary-structure-prediction](https://github.com/takatex/protein-secondary-structure-prediction)
- [Protein Secondary Structure Prediction on CB513](https://paperswithcode.com/sota/protein-secondary-structure-prediction-on-1)
