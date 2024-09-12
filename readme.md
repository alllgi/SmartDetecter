# Indetector
This repo is a python implementation of our Indetector for smart contract clone detection. 


## Required Packages
* **python** 3.8
* **pandas** 1.4.3
* **solidity-parser** 0.1.1
* **sklearn** for model evaluation
* ****



Run the following script to install the required packages.
```shell
pip install -r requirements.txt
```


## Dataset
There are two datasets for clone detection.
One dataset is called FC-pair, which consists of function pairs from smart contracts.
These are real-world smart contracts from Etherscan whose source code is publicly available and verified to match the bytecode deployed on the Ethereum blockchain.The other dataset is specially prepared for InDetector, called SR-pair, which consists of the characteristics of SR-tree pairs. In the comparison, metrics accuracy, recall, precision, and F1 score are all involved.

### Dataset structure in this project
All of the smart contract source code, training data, and testing data in these folders in the following structure respectively.
```shell
${Indetector}
└── datasets
    ├── FC-pair
    │   ├── train.csv
    │   ├── test.csv
    │   ├── train_data
    │   └── test_data
    └── SR-pair
        ├── train.csv
        └── test.csv

```

* `datasets/FC-pair/train.csv`:  This is the label of the training set in FC-pair.
* `datasets/FC-pair/test.csv`: This is the label of the testining set in FC-pair.
* `datasets/FC-pair/train_data`: This is the source code of training set in FC-pair.
* `datasets/FC-pair/test_data`: This is the source code of testing set in FC-pair.
* `datasets/SR-pair/train.csv`:  This is the data of the training set in SR-pair.
* `datasets/SR-pair/test.csv`:  This is the data of the test set in SR-pair.

## Code Files
This folder contains the code for the Indetector. There are a few important  files as follows.

* lightgbm_smart.py - contains the codes for clone detection.
* get_feature.py - contains the codes for feature extraction.
* gen_stree.py - contains the codes for extracting SR-tree.


## Getting started
* To run program, we first put  two contracts in testContracts folder and use this command: python Indetector.py --test.
* Also, you can train this model yourself. Put the training dataset into the data folder，use this command: python Indetector.py --train.

Examples:
```shell
python Indetector.py --test
```









