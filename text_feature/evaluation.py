'''
这个问题要求我们预测文本的作者，即EAP，HPL和MWS。 简单来说，文本分类有3个不同的类。

对于这个特殊的问题，Kaggle已经指定了多类别的的log-loss作为评估指标。
这是按照以下方式实现的（摘自https://github.com/dnouri/nolearn/blob/master/nolearn/lasagne/util.py）
'''
import numpy as np
def multiclass_logloss(actual, predicted, eps=1e-15):
    '''

    :param actual:
    :param predicted:
    :param eps:
    :return:
    '''
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))

    return -1.0 / rows * vsota