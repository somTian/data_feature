import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    sample = pd.read_csv('data/sample_submission.csv')
    return train, test, sample

def lableEncode(train):
    '''
    We use the LabelEncoder from scikit-learn to convert text labels to integers, 0, 1 2
    :param train:
    :return:
    '''
    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(train.author.values)
    return y

def trainTestSplit(x,y,test_size = 0.1):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, stratify=y, test_size=test_size)
    return xtrain, xtest, ytrain, ytest

def data_input():
    train, test, sample = load_data()
    y = lableEncode(train)
    x = train.text.values
    xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.1)
    return xtrain, xtest, ytrain, ytest