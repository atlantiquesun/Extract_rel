# extract_rel

Introduction:
This repository consists of models that I built for the extraction of attributive properties from word embeddings. It is known that word embeddings are able to capture semantic meaning of words, but I was interested to know how exactly it is captured and whether we can extract them. In [1], the author tested the capability of various distributional models (DM) by posing the task of predicting properties of entities in a supervised setting, and concluded that DMs yielded poor results in predicting attributive properties. Specifically, the author trained a linear SVM classifier for each of the property and ran a binary classification task for each of the property. This approach neglects the interaction between the distributional representations for entities and properties. To address this shortcoming, I developed a model that allows for the interaction between entity and property vectors. 

Model:
1) model_multiplication

2) model_concatenation
3) model_distance

Dataset:



[1] https://www.aclweb.org/anthology/P15-2119
