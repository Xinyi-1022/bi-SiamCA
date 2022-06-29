# Bidirectional Siamese Correlation Analysis Method for Enhancing the Detection of SSVEPs
This is the official repository for the Bidirectional Siamese Correlation Analysis (bi-SiamCA) method for SSVEP-based BCIs. This repository allows you to train and test, the proposed bi-SiamCA model. The framework of our method is shown in Figure 1.
![Siamese](https://user-images.githubusercontent.com/108380876/176339858-86280c8a-be05-4a85-8ad3-6e11f36bae00.png)
Figure 1. The framework of the proposed bi-SiamCA. (a) The details of the forward side, and the backward side has the same structure. (b) The original inputs are sent into the forward side, and the temporal reversed inputs are sent into the backward side. Ns denotes the number of sampling points. The outputs of the two sides are element-wise added to obtain the classification result.
## Requisites
The bi-SiamCA model is implemented in Python 3.8.11, and PyTorch 1.9.0. 
## Preparation
The two datasets (the Benchmark dataset[1] and the first 20 subjects in the wearable SSVEP BCI dataset[2]) can be downloaded from http://bci.med.tsinghua.edu.cn/download.html.
## Data Preprocessing
We performed the same preprocessing for the two datasets. In our study, the signals of the Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, and O2 electrodes were used in the benchmark dataset, and the signals of all eight electrodes were involved in the second dataset. The SSVEP recordings were first downsampled to 250 Hz and then passed through a Chebyshev Type I band-pass filter with 6 Hz to 90 Hz. We applied a notch filter at 50 Hz to remove the common power-line noise.
## Experimental setup
A leave-one(block)-out cross-validation was used for both datasets, which means that each block takes turns as the testing set. We randomly chose a block from the remaining blocks as the validation set three times for each fold. The model was trained on the training set (4 blocks for the benchmark dataset and 8 blocks for the wearable SSVEP BCI dataset) until the classification accuracy did not increase on the validation set in 30 consecutive epochs so that we could obtain three models for each testing block. One of the three models that obtained the best classification result on the validation set was evaluated on the testing set. The validation set we used are given in the released code.
The structures of the forward side and the backward side are independent, so we can train the forward network and the backward network separately, and then ensemble the output of the two networks.
It should be noticed that the sliding window method was used in generating the training set and the validation set, and it was not used in generating the testing set. In the testing stage, we used the data from 0.64 s (0.50 s for onset + 0.14 s for the visual delay) to (0.64 + data_length) s for both datasets. The testing data for all compared methods is the same.
## References
[1] Yijun Wang, Xiaogang Chen, Xiaorong Gao, and Shangkai Gao. A benchmark dataset for SSVEP-based brain–computer interfaces. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(10):1746–1752, 2016.

[2] Fangkun Zhu, Jiang Lu, Guoya Dong, Xiaorong Gao, and Yijun Wang. An open dataset for wearable SSVEP-based brain-computer interfaces. Sensors, 21:1256, 02 2021.

