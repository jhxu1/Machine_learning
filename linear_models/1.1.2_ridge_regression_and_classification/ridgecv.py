"""
岭回归使用交叉验证，选择最合适的alpha
"""

import numpy as np
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=np.logspace(-1, 3, 13))
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print(reg.alpha_)