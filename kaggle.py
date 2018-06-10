import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import  train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.metrics import log_loss
def drawScatter(heights, weights,name):
    #创建散点图
    #第一个参数为点的横坐标
    #第二个参数为点的纵坐标
    pyplot.scatter(heights, weights,c=labels,marker='o')
    pyplot.xlabel(heights)
    pyplot.ylabel(weights)
    pyplot.title(name)
    pyplot.show()

def drawHist(heights):
    # 创建直方图
    # 第一个参数为待绘制的定量数据，不同于定性数据，这里并没有事先进行频数统计
    # 第二个参数为划分的区间个数
    pyplot.hist(heights, 100)
    pyplot.xlabel('feature')
    pyplot.ylabel('hist')
    pyplot.title('feature-hist')
    pyplot.show()
if __name__=='__main__':
    a1=[]
    a2=[]
    data = pd.read_csv("train.csv")
    data2=pd.read_csv("test.csv")
    data2=data2.drop(
        ['Name','Ticket','Cabin'], axis=1)
    data2_mean=data2["Age"].mean()
    data2 = data2.fillna(value=data2_mean)
    data = data.drop(
        ['Name','Ticket','Cabin'], axis=1)
    data_mean=data["Age"].mean()
    data=data.fillna(value=data_mean)
    labels=data['Survived']
    data1=data.drop(['Survived'], axis=1)
    train_data=pd.get_dummies(data1, columns=["Embarked", "Sex"])
    train_data=train_data.drop(['Embarked_29.69911764705882'], axis=1)
    drawScatter(train_data["Fare"],train_data["Age"],"fare-age")
    drawScatter(train_data["Fare"], train_data["SibSp"], "fare-sibsp")
    drawScatter(train_data["Fare"], train_data["Parch"], "fare-parch")
    drawHist(train_data["Fare"])
    drawHist(train_data["Age"])
    print(train_data)
    x1 = preprocessing.minmax_scale(train_data, feature_range=(0, 1))
    X_train, X_test, y_train, y_test = train_test_split(x1, labels, train_size=0.7, test_size=0.3)
    clf=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    clf.fit(X_train, y_train)
    ft = clf.predict(X_test)
    print("准确率",clf.score(X_test, y_test))
    ft2 = clf.predict_proba(X_test)
    lo = log_loss(y_test, ft2)
    print("logloss损失值",lo)
    test_data = pd.get_dummies(data2, columns=["Embarked", "Sex"])
    x2=preprocessing.minmax_scale(test_data, feature_range=(0, 1))
    ft2 = clf.predict(x2)
    ft4 = pd.DataFrame(ft2, columns=['predicted'])
    lit=test_data["PassengerId"]
    lit=pd.DataFrame(list(lit),columns=['PassengerId'])
    out = pd.merge(lit, ft4, left_index=True, right_index=True, how='outer')
    out.to_csv('prede1.csv', index=False, sep=' ')