from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
embeddings_index = {}
f = open('data/glove.6B.300d.txt',encoding='utf-8')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors' % len(embeddings_index))

stop_words = stopwords.words('english')

def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v/np.sqrt((v**2).sum())

from data_help import data_input
xtrain, xtest, ytrain, ytest = data_input()

x_train_glo = np.array([sent2vec(x) for x in tqdm(xtrain)])
x_test_glo = np.array([sent2vec(x) for x in tqdm(xtest)])

scl = StandardScaler()
x_train_glove_scl = scl.fit_transform(x_train_glo)
x_test_glove_scl = scl.transform(x_test_glo)

y_train_enc = to_categorical(ytrain)
y_test_enc = to_categorical(ytest)

from keras.models import Model
from keras.layers import Input,Dense,Dropout,BatchNormalization

drop_keep = 0.2

def DNNmodel(drop_keep = 0.2):

    input = Input(shape=(300,))
    dense1 = Dense(units=300,activation='relu')(input)
    drop1 = Dropout(rate=drop_keep)(dense1)
    bn1 = BatchNormalization()(drop1)

    dense2 = Dense(units=300,activation='relu')(bn1)
    drop2 = Dropout(rate=drop_keep)(dense2)
    bn2 = BatchNormalization()(drop2)

    predic = Dense(units=3,activation='softmax')(bn2)

    model = Model(inputs = input, outputs = predic)
    return model

model = DNNmodel(drop_keep)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x_train_glove_scl,y_train_enc,batch_size=64,epochs=10,verbose=1,validation_data=(x_test_glove_scl,y_test_enc))




