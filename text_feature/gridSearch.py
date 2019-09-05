'''
网格搜索是超参数优化的一种技术。 如果您要使用网格，效率可能不佳，但可以给出良好的效果。
我指定通常应该在这篇文章中使用的参数：http://blog.kaggle.com/2016/07/21/approaching-almost-any-machine-learning-problem-abhishek-thakur/
请记住 这些是我通常使用的参数。 超参数优化有许多其他的方法可能会或可能不会有效。

在本节中，我将使用逻辑回归来讨论网格搜索。

在开始网格搜索之前，我们需要创建一个评分函数。 这是通过使用scikit-learn的make_scorer函数来完成的。
'''
import sklearn.metrics
from evaluation import multiclass_logloss
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.linear_model import LogisticRegression
from data_help import data_input
from sklearn.feature_extraction.text import TfidfVectorizer

xtrain, xtest, ytrain, ytest = data_input()

tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                      token_pattern=r'\w{1,}', ngram_range=(1,3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                      stop_words='english')

tfv.fit(list(xtrain) + list(xtest))
xtrain_tfv = tfv.transform(xtrain)
xtest_tfv = tfv.transform(xtest)

#在开始网格搜索之前，我们需要创建一个评分函数。通过使用scikit-learn的make_scorer函数来构建。
mll_score = metrics.make_scorer(multiclass_logloss, greater_is_better=False, needs_proba=True)
#接下来我们需要一个管道。 为了演示这里，我将使用由SVD, scaling and then logistic regression组成的pipeline。 更好的了解更多的模块在管道中，而不仅仅是一个;）

svd = TruncatedSVD()
scl = preprocessing.StandardScaler()
lr_model = LogisticRegression()

# Create the pipeline
clf = pipeline.Pipeline([('svd',svd),('scl',scl),('lr',lr_model)])

param_grid = {'svd__n_components':[120,180],'lr__C':[0.1,1.0,10],'lr__penalty':['l1','l2']}

#因此，对于SVD我们评估120和180分量和逻辑回归我们评估C的三个不同的值与惩罚l1和l2。 现在我们可以开始对这些参数进行网格搜索。

model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_score, verbose=10, n_jobs=-1, iid=True,refit=True,cv=2)

model.fit(xtrain_tfv, ytrain)
print("Best Score: %0.3f"%model.best_score_)
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))