# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 12:09:38 2022

@author: bmiller
"""
from keras.models import Sequential  
from keras.layers import * 
#import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras import layers

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
import re
#import nltk
#from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import Conv1D
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import keras.metrics
import math
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

rawdata=pd.read_csv(r'c:/users/bmiller/.spyder-py3/sentiment/careerpaths.csv')
rawdata.isnull().values.any()
rawdata.head()

data=pd.DataFrame()
data['occcode']=['']
data['Prevocccode']=['']

data=rawdata[0:]

# define example
#data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
Xvalues = data['occcode']
Yvalues = data['Prevocccode']
values=array(pd.concat([Xvalues,Yvalues]))
#values=array(valueslist)
print(values)


# integer encode
Master_label_encoder = LabelEncoder()
# Fit label encoder to all values
Master_label_encoder = Master_label_encoder.fit(values)

# Integer encode X and Y
Xinteger_encoded=Master_label_encoder.transform(array(Xvalues))
Yinteger_encoded=Master_label_encoder.transform(array(Yvalues))
All_integer_encoded=Master_label_encoder.transform(values)

#print(integer_encoded)

# binary encode

# Reshape Integer encoded
All_integer_encoded = All_integer_encoded.reshape(len(All_integer_encoded), 1)
Xinteger_encoded= Xinteger_encoded.reshape(len(Xinteger_encoded), 1)
Yinteger_encoded = Yinteger_encoded.reshape(len(Yinteger_encoded), 1)

# Fit encoder to All
Master_onehot_encoder = OneHotEncoder(sparse=False)
Master_onehot_encoder = Master_onehot_encoder.fit(All_integer_encoded)

#transform X and Y sets 
X_onehot_encoded= Master_onehot_encoder.transform(Xinteger_encoded)
Y_onehot_encoded= Master_onehot_encoder.transform(Yinteger_encoded)


#print(onehot_encoded)
# invert first example
Xinverted = Master_label_encoder.inverse_transform([argmax(X_onehot_encoded[0, :])])



Occcode_train, Occcode_test,X_train_onehot, X_test_onehot, prevocccode_train, prevoccode_test,y_train_onehot,y_test_onehot = train_test_split(data['occcode'],X_onehot_encoded, data['Prevocccode'],Y_onehot_encoded, test_size=0.20, random_state=42)
print(Xinverted)


model = Sequential()

METRICS=[keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.BinaryAccuracy(name='acc'),
      #keras.metrics.AUC(name='loss'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
      ]


#model.add(Dense(128, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
#model.add(Dense(128, activation='sigmoid'))
#model.add(Dropout(0.5)) 
model.add(Dense(1003, activation='softmax'))

model.compile(optimizer='adam', loss='cosine_similarity', metrics=METRICS)
print(model.summary())
history = model.fit(X_train_onehot, y_train_onehot, batch_size=5000, epochs=20, verbose=1, validation_split=0.4 )

score = model.evaluate(X_test_onehot, y_test_onehot, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()


ypredict=model.predict(X_test_onehot[0:1])