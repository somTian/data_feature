'''
Our very first model is a simple TF-IDF (Term Frequency - Inverse Document Frequency)
followed by a simple Logistic Regression.
'''
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression

from data_help import data_input
from evaluation import multiclass_logloss

xtrain, xtest, ytrain, ytest = data_input()

tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                      token_pattern=r'\w{1,}', ngram_range=(1,3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                      stop_words='english')

tfv.fit(list(xtrain) + list(xtest))
xtrain_tfv = tfv.transform(xtrain)
xtest_tfv = tfv.transform(xtest)

clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, ytrain)

predicitons = clf.predict_proba(xtest_tfv)

print('TFIDF + LogisticRegression logloss:%0.3f'%multiclass_logloss(ytest, predicitons))  #0.617


'''
但是我们想要更好的分数。 让我们看看不同的数据相同的模型。
不使用TF-IDF，我们也可以使用字数作为特征。 这可以通过scikit-learn使用CountVectorizer轻松完成。
'''
from sklearn.feature_extraction.text import CountVectorizer

ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}', ngram_range=(1,3), stop_words='english')

ctv.fit(list(xtrain) + list(xtest))
xtrain_ctv = ctv.transform(xtrain)
xtest_ctv = ctv.transform(xtest)

clf = LogisticRegression(C=1.0)
clf.fit(xtrain_ctv,ytrain)
predicitons = clf.predict_proba(xtest_ctv)

print('CountVector + LogisticRegression logloss:%0.3f'%multiclass_logloss(ytest, predicitons))

'''
在TFIDF结果上使用朴素贝叶斯
'''
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(xtrain_ctv, ytrain)
predicitons = clf.predict_proba(xtest_ctv)

print('CountVector + Naive Bayes logloss:%0.3f'%multiclass_logloss(ytest, predicitons))