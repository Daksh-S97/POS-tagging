# POS tagging using Viterbi algorithm and BiLSTM-CRF

This project focuses on sequence labeling with Hidden Markov Models and Deep Learning models. The target domain is part-of-speech tagging on English and Norwegian from the Universal Dependencies dataset. It includes the following components:

- Basic preprocessing of the data
- A naive classifier that tags each word with its most common tag
- Viterbi Tagger using `Hidden Markov Model` in PyTorch
- Bi-LSTM deep learning model using PyTorch
- Bi-LSTM_CRF model using the above components (Viterbi and Bi-LSTM) 

