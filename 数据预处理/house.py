#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/7 19:56

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler
#%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
from pandas import DataFrame as df
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from scipy import stats
from scipy.stats import norm, skew #for some statistics
import math
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

from subprocess import check_output
# print(check_output(["dir",'/b','input'],shell=True).decode('GBK')) #check the files available in the directory, only work on ubuntu

train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
# sns.distplot(train['SalePrice'])
# var = 'YearBuilt'
# data = pd.concat([train['SalePrice'], train[var]], axis=1)
# fig = sns.boxplot(x=var,y='SalePrice',data=data)
# plt.xticks(rotation=90)

# 画缩放的heatmap
# cormat = train.corr()
# k = 10
# cols = cormat.nlargest(10, 'SalePrice')['SalePrice'].index
# cm = train[cols].corr()#np.corrcoef(train[cols].values.T)
# print(train[cols].values)
# f, ax = plt.subplots(figsize = (12,9))
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot= True, fmt='.2f', annot_kws={'size':10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.xticks(rotation=90)
# plt.yticks(rotation=360)
#plt.show()


# 画pairplot
# k = 10
# cols = train.corr().nlargest(10, 'SalePrice').index.values
# sns.pairplot(train[cols],size=2.5)
# plt.show()

# 找到缺失数据
# df.sort_values()
total = train.isnull().sum().sort_values(ascending=False)
miss_percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
miss_data = pd.concat([total,miss_percent],axis=1,keys=['total','percent'])
# print(miss_data.head(20).to_csv('my.csv'))
# print(miss_data)

# 删除缺失数据
print((miss_data[miss_data['total'] > 1]).index)
train = train.drop((miss_data[miss_data['total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index,0)
print(train.isnull().sum().max())
# print(type(train['SalePrice'].values))

# 标准化数据
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'].values)
lowrange = np.sort(saleprice_scaled)[:10]
highrange = np.sort(saleprice_scaled)[-10:]
print(lowrange)
print(highrange)


# 二元分析
# plt.scatter(train['GrLivArea'],train['SalePrice'])
# plt.xlabel('GrLivArea')
# plt.ylabel('SalePrice')
# plt.show()

# 删除异常值
print(train.shape)
train = train.drop(train[(train['SalePrice']<300000) & (train['GrLivArea']>4000)].index)
print(train.shape)

# 售价与总地下室面积的二元分析
# var = 'TotalBsmtSF'
# data = pd.concat([train['SalePrice'],train[var]],1)
# data.plot.scatter(x=var,y='SalePrice')
# plt.show()
train['SalePrice'] = np.log(train['SalePrice'])
train['GrLivArea'] = np.log(train['GrLivArea'])
# print(train['TotalBsmtSF'].nonzero()[0].shape)
# train[(train['TotalBsmtSF'] > 0)] = 1
# train[(train['TotalBsmtSF'] > 0)] = 1


print('小于e平的地下室',train[train['TotalBsmtSF']<math.e]['TotalBsmtSF'].shape)
print(train.loc[train['TotalBsmtSF']>0,'TotalBsmtSF'].shape)
train.loc[train['TotalBsmtSF']>0,'TotalBsmtSF'] = np.log(train.loc[train['TotalBsmtSF']>0,'TotalBsmtSF'])
print(train['TotalBsmtSF'].head())
#正态概率图
# sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

plt.scatter(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], train[train['TotalBsmtSF']>0]['SalePrice']);plt.show()

# train['SalePrice'] = np.log(train['SalePrice'])

# print(type(saleprice_scaled))

# f, ax = plt.subplots(figsize=(12, 9))
#
# sns.heatmap(cormat,vmax=.8,square=True)
# plt.xticks(rotation=90)
# plt.yticks(rotation=360)
#
# plt.show()
# ##display the first five rows of the train dataset.
# print(train.head(5))
# ##display the first five rows of the test dataset.
# print(test.head(5))

# #check the numbers of samples and features
# print("The train data size before dropping Id feature is : {} ".format(train.shape))
# print("The test data size before dropping Id feature is : {} ".format(test.shape))

# #Save the 'Id' column
# train_ID = train['Id']
# test_ID = test['Id']
#
# #Now drop the  'Id' colum since it's unnecessary for  the prediction process.
# train.drop('Id',axis=1,inplace=True)
# test.drop('Id',axis=1,inplace=True)
#
# # #check again the data size after dropping the 'Id' variable
# # print("\nThe train data size after dropping Id feature is : {} ".format(train.shape))
# # print("The test data size after dropping Id feature is : {} ".format(test.shape))
#
# # #plot the data distribution
# # fig, ax = plt.subplots()
# # ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
# # plt.ylabel('SalePrice', fontsize=13)
# # plt.xlabel('GrLivArea', fontsize=13)
# # plt.show()
#
# #Deleting outliers
# train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index,inplace=True)
#
# # #Check the graphic again
# # fig, ax = plt.subplots()
# # ax.scatter(train['GrLivArea'], train['SalePrice'])
# # plt.ylabel('SalePrice', fontsize=13)
# # plt.xlabel('GrLivArea', fontsize=13)
# # plt.show()
#
# sns.distplot(train['SalePrice'] , fit=norm)
# # Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(train['SalePrice'])
# # print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#
# #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')
#
# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.xlabel('Normal theoretical quantiles')
#
# #We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
# train["SalePrice"] = np.log1p(train["SalePrice"])
#
# #Check the new distribution
# fig = plt.figure()
# sns.distplot(train['SalePrice'] , fit=norm)
#
# # Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(train['SalePrice'])
# # print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#
# #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
#             loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')
#
# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# # plt.show()
#
# #let's first concatenate the train and test data in the same dataframe
# ntrain = train.shape[0]
# ntest = test.shape[0]
# y_train = train.SalePrice.values
# all_data = pd.concat((train, test)).reset_index(drop=True)
# all_data.drop(['SalePrice'], axis=1, inplace=True)
# # print("all_data size is : {}".format(all_data.shape))
#
# #Missing Data
# all_data_na = (all_data.isnull().sum() / len(all_data) * 100).sort_values(ascending=False)
# all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index)
# # print(all_data_na)
#
# f, ax = plt.subplots(figsize=(15, 12))
# plt.xticks(rotation='90')
# sns.barplot(x=all_data_na.index, y=all_data_na)
# plt.xlabel('Features', fontsize=15)
# plt.ylabel('Percent of missing values', fontsize=15)
# plt.title('Percent missing data by feature', fontsize=15)
#
# corrmat = train.corr()
# fig, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=0.9, square=True)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# ax.set_yticklabels(ax.get_yticklabels(),rotation=0)
# # plt.show()
#
#
# #Imputing missing values
# all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
# all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
# all_data["Alley"] = all_data["Alley"].fillna("None")
# all_data["Fence"] = all_data["Fence"].fillna("None")
# all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
#
# #Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
# all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].apply(
#     lambda x: x.fillna(x.median()))
#
# #GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None
# for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
#     all_data[col] = all_data[col].fillna('None')
#
# #GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
# for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
#     all_data[col] = all_data[col].fillna(0)
#
# #BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement
# for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
#     all_data[col] = all_data[col].fillna(0)
#
# #BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.
# for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
#     all_data[col] = all_data[col].fillna('None')
#
# #MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
# all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
# all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
#
# #MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'
# all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
#
# #Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
# all_data = all_data.drop(['Utilities'], axis=1)
#
# all_data["Functional"] = all_data["Functional"].fillna("Typ")
# all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
# all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
# all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
# all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
# all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
#
# from sklearn.preprocessing import LabelEncoder
# all_data_na = (all_data.isnull().sum() / len(all_data) * 100).sort_values(ascending=False)
# all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index)
# # print('after fillna the na percentage is {}\n',format(all_data_na))
# # print(all_data.dtypes)
#
# all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
# all_data['OverallCond'] = all_data['OverallCond'].apply(str)
# all_data['MoSold'] = all_data['MoSold'].apply(str)
#
# # category_cols = all_data.dtypes[all_data.dtypes == 'object'].index
# # # print(category_cols)
# # ## 将文字类型的值转换为类别
# # for category_col in category_cols:
# #     labencoder = LabelEncoder()
# #     all_data[category_col] = labencoder.fit_transform(all_data[category_col])
#
#
# # 加入一个房屋总面积特征
# all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
#
# numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
# skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
# # print("\nSkew in numerical features: \n")
# skewness = pd.DataFrame({'Skew' :skewed_feats})
# skewness = skewness[abs(skewness) > 0.75]
# skewed_features = skewness.index
# from scipy.special import boxcox1p
#
# lam = 0.15
# for feat in skewed_features:
#     all_data[feat] = boxcox1p(all_data[feat], lam)
#
# all_data = pd.get_dummies(all_data)
# print(all_data.dtypes.value_counts())
# train = all_data[:ntrain]
# test = all_data[ntrain:]
#
# from sklearn.model_selection import KFold,cross_val_score
# from sklearn.linear_model import Lasso
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import RobustScaler
# n_folds = 5
# def rmsle_cv(model):
#     cv = KFold(n_folds,shuffle=True,random_state=42)
#     score = cross_val_score(model,train.values,y_train,scoring='neg_mean_squared_error',cv=cv)
#     rmse = np.sqrt(-score)
#     return rmse
#
#
# lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
# lasso.fit(train,y_train)
# prediction = np.exp(lasso.predict(test))
# df = pd.DataFrame({'SalePrice':prediction},index=test_ID)
# df.to_csv('result.csv')
# # print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#
#
#
#
#
#
#
#
#
#
#
