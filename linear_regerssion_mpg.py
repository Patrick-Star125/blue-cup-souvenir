import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
import pickle
import joblib


class OLSinearRegression:

    def _ols(self, X, y):
        '''最小二乘法估算w'''
        temp = np.linalg.inv(np.matmul(X.T, X))  # 矩阵求逆
        temp = np.matmul(temp, X.T)
        return np.matmul(temp, y)

        # 也可以使用return np.linalg.inv(X.T @ X)@X.T@y

    def _preprocess_data_X(self, X):
        '''数据预处理'''

        # 扩展X，添加X0=1
        m, n = X.shape
        X_ = np.empty((m, n + 1)) # 用空值填充X_，这样创建矩阵最快
        X_[:, 0] = 1
        X_[:, 1:] = X

        return X_

    def train(self, X_train, y_train):
        '''训练模型'''

        # 预处理X_train(添加X0=1)
        X_train = self._preprocess_data_X(X_train)

        # 使用最小二乘法估算w
        self.w = self._ols(X_train, y_train)

    def predict(self, X):
        '''预测'''
        # 预处理X_train(添加x0=1)
        X = self._preprocess_data_X(X)
        return np.matmul(X, self.w)

    def save(self,filename):
        pickle.dump(self,open(filename,'wb'))

    @staticmethod # 用这个修饰器使下面的方法变成静态方法，可以不用self这个参数
    def from_file(filename):
        return pickle.load(open(filename,'rb'))

# ols=OLSinearRegression()
# X = np.loadtxt('auto-mpg.txt', usecols=(1, 2, 3, 4, 5, 6, 7))
# y = np.loadtxt('auto-mpg.txt', usecols=(0))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#
# ols.train(X_train,y_train)
# ols.save('mpgmodel_linear_9')
