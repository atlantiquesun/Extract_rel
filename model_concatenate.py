#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 21:53:17 2018

@author: yiransun
"""

import pandas as pd
import numpy as np
from random import randint,choice
from copy import deepcopy
from keras.models import Model, Sequential
from keras.layers import Input, dot,Dot,Dense, merge, Reshape,Multiply,Activation, Conv1D, Lambda,  Flatten, MaxPooling1D,Concatenate
from keras.regularizers import l2
from keras.constraints import MinMaxNorm
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
import numpy.random as rng

'''
Helper Functions
'''
def exclude(l1,l2):
    l=deepcopy(l1)
    for i in l2:
        l.remove(i)
    return l

def W_init(shape,name=None):
    '''
    initialise weights
    '''
    values=rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)

def b_init(shape,name=None):
    '''
    initialise bias
    '''
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)
def W2_init(shape,name=None):
    '''
    initialise weights as 1
    '''
    values=np.zeros((shape))+1
    return K.variable(values,name=name)

def multiply_weights(x):
    return np.multiply(x[0],x[1])

def f_score(prediction,val_label):
    '''
    prediction: a np.ndarray of floats, can be of of any value
    val_label: a np.ndarray of floats, only 1 and 0
    '''
    prediction=list(prediction.flatten())
    l=[]
    for i in prediction:
        if i<0.5:l.append(0)
        else:l.append(1)
    r=list(val_label)
    fn=0
    fp=0
    tn=0
    tp=0
    for i in range(len(l)):
        if l[i]==1:
            if r[i]==1:tp+=1
            else:fp+=1
        else:
            if r[i]==1:fn+=1
            else:tn+=1
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    print(precision,recall)
    print(fn,fp,tn,tp)
    f=(2*(precision*recall)/(precision+recall))
    return f

df=pd.read_csv('/Users/yiransun/Desktop/attributive.csv',sep=';')
df=df.drop('BR_Label',axis=1)

attributes=['is_black','is_brown','is_green','is_white','is_yellow','is_large','is_small','is_long','is_round','is_loud','is_dangerous','is_fast']
valid_attributes=[x[3:] for x in attributes]
df=df.loc[df['Feature'].isin(attributes)].reset_index(drop=True)
df['Feature']=df['Feature'].str[3:]
df=df.sample(frac=1).reset_index(drop=True)


'''
Read and Preprocess the Data
'''
pos_entity_word=list(df['Concept'])
pos_attribute_word=list(df['Feature'])
dic={}
for i in range(len(pos_entity_word)):
    if ')' not in pos_entity_word[i] and pos_entity_word[i] in words:
        if pos_entity_word[i] not in dic:
            dic[pos_entity_word[i]]=[pos_attribute_word[i]]
        else:
            dic[pos_entity_word[i]].append(pos_attribute_word[i])
valid_entities=list(dic.keys())

df=df.loc[df['Concept'].isin(valid_entities)].reset_index(drop=True)
pos_entity_word=list(df['Concept'])
pos_attribute_word=list(df['Feature'])

#1.add labels to positive samples
#2.add negative samples with labels
#3.shuffle
#4.retrieve word lists
#5.make them into vector lists
#6.separate them into training and test/validation datasets
pos_labels=np.zeros((len(pos_entity_word),))+1
df['Label']=pos_labels

neg_entity_word=[]
neg_attribute_word=[]
neg_labels=np.zeros((700,))
for i in range(700):
    a=choice(valid_entities)
    neg_entity_word.append(a)
    b=choice(exclude(valid_attributes,dic[a]))
    neg_attribute_word.append(b)
apx=pd.DataFrame([neg_entity_word,neg_attribute_word,list(neg_labels)]).transpose().rename(columns={0:'Concept',1:'Feature',2:'Label'})

df=pd.concat([df,apx]).sample(frac=1).reset_index(drop=True)
val_label=np.asarray(list(df['Label'])[920:])
train_label=np.asarray(list(df['Label'])[:920])


dot_product=[]
for i in range(1380):
    dot_product.append(np.multiply(word_vectors[df['Concept'][i]],word_vectors[df['Feature'][i]]).reshape((100,)))
train_dot=np.asarray(dot_product[:920])
val_dot=np.asarray(dot_product[920:])


    
    
'''
Network
'''
input_shape=(100,)
dot_input=Input(input_shape)


model1=Dense(150)(dot_input)
model1=Dense(100)(model1)
model1=Dense(60)(model1)
#model1=MaxPooling1D()(model1)
model1=Dense(30)(model1)
#model1=Dense(100, kernel_regularizer=l2(1e-3),kernel_initializer=W_init, bias_initializer=b_init)(model1)
model1=Dense(1,activation='relu')(model1)
target=Reshape((1,))(model1)



model=Model(inputs=[dot_input],outputs=target)
model.compile(loss="binary_crossentropy",metrics=['accuracy'],optimizer=Adam())



model.fit(x=[train_dot],y=train_label,batch_size=32,epochs=17)
score=model.evaluate([val_dot],val_label)
prediction=model.predict([val_dot])
print(score)
print(f_score(prediction,val_label))
