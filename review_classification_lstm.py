from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizer import SGD

# read data to dataframe
fname = 'yelp_training_set/yelp_training_set_review.json'
data = pd.read_json(fname, lines=True)

useful_features = ['stars', 'text']
data_useful = data.loc[:, useful_features]
data_useful = data_useful.drop(data['stars'] == 3)
data_useful['label'] = data_useful['stars'].apply(lambda x : 1 if x > 3 else 0)

X_data = data_useful['text']
Y_data = data_useful['label']
print('X shape:', X_data.shape)
print('Y shape:', Y_data.shape)

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, random_state=0)
print('Train shape:', X_train.shape)
print(sum(y_train == 1))
print('Test shape:', X_test.shape)

# process text into tokens
max_num_tokens = 20000
tokenizer = Tokenizer(num_words = max_num_tokens)
tokenizer.fit_on_texts(X_train.values)
print('tokenized num:')
print(len(tokenizer.word_index))

X_train_seq = tokenizer.text_to_sequences(X_train.values)
X_train_seq = pad_sequences(X_train_seq)

X_test_seq = tokenizer.text_to_sequences(X_test.values)
X_test_seq = pad_sequences(X_test_seq)

input_length = X_train_seq.shape[1]
print('text length:', input_length)
model = Sequential()
model.add(Embedding(max_num_tokens, 128, input_length=input_length))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
optimizer = SGD(lr=1e-4, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(X_train_seq, y_train.values, validation_split=0.1, epochs=5)
