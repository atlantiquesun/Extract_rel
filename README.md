# extract_rel

## Introduction
This repository contains two simple models (implemented in Keras) for the extraction of attributive properties from word embeddings. It is known that word embeddings are able to capture semantic meaning of words, but I was interested to know how exactly it is captured and whether we can extract them. In [1], the author tested the capability of various distributional models (DM) by posing the task of predicting properties of entities in a supervised setting, and concluded that DMs yielded poor results in predicting attributive properties. Specifically, the author trained a linear SVM classifier for each of the property and ran a binary classification task for each of the property. This approach neglects the interaction between the distributional representations for entities and properties. The three simple models in this repository are developed with the aim to address this shortcoming.

## Models
1. model_multiplication.py
- convolutional layers to encode property embeddings 
- take the element-wise product of the encoded embeddings and entity embeddings
- dense neural network for classification
2. model_concatenation.py
- directly input dot products of entity and property embeddings
- dense neural network for classification

## Dataset
To allow fair comparison, the dataset used to test this model is the same dataset as that used by [1] (McRae Feature Norms dataset). The original dataset can be found at [2].


[1] https://www.aclweb.org/anthology/P15-2119 

[2] https://sites.google.com/site/kenmcraelab/norms-data
