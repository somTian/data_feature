from data_help import data_input
from evaluation import multiclass_logloss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from evaluation import multiclass_logloss

'''
这里我们尝试使用SVM，因为SVM花费时间太长，所以我们先用SVD对TF-idf进行降维
'''

xtrain, xtest, ytrain, ytest = data_input()

tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                      token_pattern=r'\w{1,}', ngram_range=(1,3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                      stop_words='english')

tfv.fit(list(xtrain) + list(xtest))
xtrain_tfv = tfv.transform(xtrain)
xtest_tfv = tfv.transform(xtest)

svd = TruncatedSVD(n_components=150)
svd.fit(xtest_tfv)
xtrain_svd = svd.transform(xtrain_tfv)
xtest_svd = svd.transform(xtest_tfv)

scl = StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xtest_svd_scl = scl.fit_transform(xtest_svd)

# 建立一个简单的SVM模型
model = SVC(C=1.0,probability=True)
model.fit(xtrain_svd_scl,ytrain)
predictions = model.predict(xtest_svd_scl)
print(predictions)

# print("logloss: %0.3f " % multiclass_logloss(ytest, predictions))
