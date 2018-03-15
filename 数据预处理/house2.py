#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/9 9:10

# 导入需要的模块
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 用来绘图的，封装了matplot
# 要注意的是一旦导入了seaborn，
# matplotlib的默认作图风格就会被覆盖成seaborn的格式
import seaborn as sns

from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

data_train = pd.read_csv("./input/train.csv")
print(data_train.shape)
print(data_train.columns)
# print(data_train['SalePrice'].describe())
fig,ax = plt.subplots(figsize=(10,9))
sns.distplot(data_train['SalePrice'],fit=norm)
(mu, sigma) = norm.fit(data_train['SalePrice'])
plt.legend(['norm distribution with $\mu$={:.2f},$\sigma$={:.2f}'.format(mu,sigma)])
plt.ylabel('frequency')

print("Skewness: %f" % data_train['SalePrice'].skew())
print("Kurtosis: %f" % data_train['SalePrice'].kurt())


from sklearn import preprocessing
f_names = ['CentralAir', 'Neighborhood']
# data_train = data_train.reindex(columns=sorted(data_train.columns))
for x in f_names:
    label = preprocessing.LabelEncoder()
    data_train[x] = label.fit_transform(data_train[x])
corrmat = data_train.corr()
fig,ax = plt.subplots(figsize=(10,9))
sns.heatmap(corrmat,vmax=0.8,square=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
# plt.show()

k = 10
cols = corrmat.nlargest(k,columns='SalePrice').index
# cm = np.corrcoef(data_train[cols].values.T)
cm = data_train[cols].corr()
# hm = sns.heatmap(cm, cbar=True, annot=True, square= True, fmt='.2f', annot_kws={'size':10}, yticklabels=cols.values, xticklabels=cols.values)
fig,ax = plt.subplots(figsize=(10,9))
hm = sns.heatmap(cm, cbar=True, annot=True,
                 square=True, fmt='.2f', annot_kws={'size': 10})
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
# fig,ax = plt.subplots()
# sns.pairplot(data_train[cols.values],size=2.5)
# plt.savefig('pairplot.pdf')
# plt.show()

from sklearn import preprocessing
from sklearn import linear_model, svm, gaussian_process
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
cols = cols.drop('SalePrice')
x = data_train[cols].values
y = data_train['SalePrice'].values
x_scaled = preprocessing.StandardScaler().fit_transform(x)
y_scaled = preprocessing.StandardScaler().fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(x_scaled,y_scaled,test_size=0.33,random_state=42)
pass





