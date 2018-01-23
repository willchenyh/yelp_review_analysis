from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD

# read data to dataframe
fname = '../yelp_training_set_review.json'
data = pd.read_json(fname, lines=True)

# drop features
# since we are only classifying positive and negative reviews, take only the useful features
useful_features = ['stars', 'text']
data_useful = data.loc[:, useful_features]
data_useful = data_useful.loc[data['stars'] != 3]

# convert stars to binary values 0 and 1
data_useful['label'] = data_useful['stars'].apply(lambda x : 1 if x > 3 else 0)
data_useful.drop(['stars'], axis=1, inplace=True)

# select a subset with equal number of positive and negative reviews
data_pos = data_useful.loc[data_useful['label'] == 1]
data_neg = data_useful.loc[data_useful['label'] == 0]
num_perclass = data_neg.shape[0]
data_pos_spl = data_pos.sample(n=num_perclass, random_state=100)
data_spl = pd.concat([data_pos_spl, data_neg], ignore_index=True)
data_spl = data_spl.reindex(np.random.permutation(data_spl.index))

# separte x and y data
X_data = data_spl['text']
Y_data = data_spl['label']
print('X shape:', X_data.shape)
print('Y shape:', Y_data.shape)
print()

# check number of positive and negative reviews
print(Y_data.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, random_state=0)
print('Train shape:', X_train.shape)
print('num pos reviews:', sum(y_train == 1))
print('Test shape:', X_test.shape)

# process text into tokens
max_num_tokens = 20000
tokenizer = Tokenizer(num_words = max_num_tokens)
tokenizer.fit_on_texts(X_train.values)
print('tokenized num:')
print(len(tokenizer.word_index))

input_length = 300
print('text length:', input_length)

X_train_seq = tokenizer.texts_to_sequences(X_train.values)
X_train_seq = pad_sequences(X_train_seq, maxlen=input_length)

X_test_seq = tokenizer.texts_to_sequences(X_test.values)
X_test_seq = pad_sequences(X_test_seq, maxlen=input_length)

model = Sequential()
model.add(Embedding(max_num_tokens, 128, input_length=input_length))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 3
model.fit(X_train_seq, y_train.values, validation_split=0.1, epochs=num_epochs, batch_size=128)
model.save('review_binary_classification_weights.h5')

results = model.evaluate(X_test_seq, y_test.values)
print('results:')
print(results)