# Protein secondary Structure Prediction in Deep Learning

PyTorch implementations of protein secondary structure prediction on CB513.

## Dataset

I used CB513 dataset of https://github.com/alrojo/CB513.

|sequence length (train)|sequence length (test)|
|:-:|:-:|
|![](figure/seqlen_train.png)|![](figure/seqlen_test.png)|

|amino acid (train)|amino acid (test)|
|:-:|:-:|
|![](figure/amino_acid_train.png)|![](figure/amino_acid_test.png)|

|secondary structure label(train)|secondary structure label (test)|
|:-:|:-:|
|![](figure/secondary_structure_train.png)|![](figure/secondary_structure_test.png)|

## Usage

You can get more infomations by adding `-h` option.

### pssp

```
python main.py
```

## Acknowledgement

- https://github.com/alrojo/CB513 
- [Li, Zhen; Yu, Yizhou, Protein Secondary Structure Prediction Using Cascaded Convolutional and Recurrent Neural Networks, 2016.](https://arxiv.org/pdf/1604.07176.pdf)
