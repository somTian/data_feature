from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input,Dense,Dropout,BatchNormalization,LSTM,Embedding,SpatialDropout1D,Bidirectional
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from data_help import data_input
xtrain, xtest, ytrain, ytest = data_input()

embeddings_index = {}
f = open('data/glove.6B.300d.txt',encoding='utf-8')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors' % len(embeddings_index))

def LSTMmodel(embedding_matrix = None):
    input = Input(shape=(max_len,))
    embed =Embedding(len(word_index) +1, 300, weights=[embedding_matrix], trainable=False)(input)
    drop1 = SpatialDropout1D(rate=0.3)(embed)
    lstm1 = LSTM(units=100, dropout=0.3, recurrent_dropout=0.3,activation='relu')(drop1)
    # blstm = Bidirectional(LSTM(units=100, dropout=0.3, recurrent_dropout=0.3,activation='relu')(drop1))
    dense1 = Dense(units=1024, activation='relu')(lstm1)
    drop2 = Dropout(rate=0.8)(dense1)
    dense2 = Dense(units=1024,activation='relu')(drop2)
    drop3 = Dropout(rate=0.8)(dense2)
    predict = Dense(units=3,activation='softmax')(drop3)
    model = Model(inputs=input,outputs = predict)
    return model

token = Tokenizer(num_words=None)
max_len = 70
token.fit_on_texts(list(xtrain) + list(xtest))
xtrain_seq = token.texts_to_sequences(xtrain)
xtest_seq = token.texts_to_sequences(xtest)

xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xtest_pad = sequence.pad_sequences(xtest_seq, maxlen=max_len)

word_index = token.word_index

y_train_enc = to_categorical(ytrain)
y_test_enc = to_categorical(ytest)

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model  = LSTMmodel(embedding_matrix)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x = xtrain_pad, y=y_train_enc, batch_size=512, epochs=10, verbose=1, validation_data=(xtest_pad, y_test_enc))

# earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
# model.fit(x = xtrain_pad, y=y_train_enc, batch_size=512, epochs=10, verbose=1, validation_data=(xtest_pad, y_test_enc),callbacks=[earlystop])