#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/13 20:27

#import some necessary librairies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
from scipy import stats
from scipy.stats import norm, skew #for some statistics
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points
from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# print(train.head(5))

print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

print("\nThe train data size after dropping Id feature is : {} ".format(train.shape))
print("The test data size after dropping Id feature is : {} ".format(test.shape))

# fig, ax = plt.subplots()
# ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.show()

# var = 'GrLivArea'
# # data = pd.concat([train['SalePrice'],train[var]],1)
# f, ax = plt.subplots()
# plt.scatter(x=train['GrLivArea'],y=train['SalePrice'])
# plt.show()

print(train.shape)
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
print(train.shape)


# 观察SalePrcie数据分布
# sns.distplot(train['SalePrice'] , fit=norm)
#
# # Get the fitted parameters used by the function
# (mu, sigma) = norm.fit(train['SalePrice'])
# print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
#
# #Now plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
# plt.ylabel('Frequency')
# plt.title('SalePrice distribution')
#
# #Get also the QQ-plot
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()

# 进行log变换
train['SalePrice'] = np.log(train['SalePrice'])
# sns.distplot(train['SalePrice'],fit=norm)
# (mu,sigma) = norm.fit(train['SalePrice'])
# print('the $\mu$ = {:.2f} \n the $\sigma$ = {:.2f}'.format(mu,sigma))
# plt.legend(['Norm distribution( $\mu$ = {:.2f} , $\sigma$ = {:.2f})'.format(mu,sigma)],loc='best')
# plt.ylabel('Frequency')
# plt.xlabel('SalePrice')
# plt.title('SalePrice distribution')
# fig = plt.figure()
# stats.probplot(train['SalePrice'],plot=plt)
# plt.show()

all_data = pd.concat([train,test],ignore_index=True)
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data.drop('SalePrice',1,inplace=True)
print(all_data.shape)

# 处理缺失数据
all_data_na = (all_data.isnull().sum() / len(all_data)) *100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio':all_data_na})
# print(missing_data.head(20))

# 画数据丢失率
# sns.barplot(all_data_na.index,all_data_na.values)
# plt.xticks(rotation=90)
# plt.title('data missing percent')
# plt.show()

# 画数据相关性图
corrmat = train.corr()
# plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat,vmax=0.9,square=True)
# plt.xticks(rotation=90)
# plt.yticks(rotation=360)
# plt.show()

# print(np.abs(corrmat['SalePrice']).sort_values())
var = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MSSubClass', 'MasVnrType']
# all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
for v in var:
    all_data[v] = all_data[v].fillna('None')
# print('为null的值',max(all_data[var].isnull().sum()))

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))
var = ['GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       'BsmtFullBath', 'BsmtHalfBath','MasVnrArea']
for v in var:
    all_data[v] = all_data[v].fillna(0)

var = ['Electrical','MSZoning','KitchenQual','Exterior1st','Exterior2nd','SaleType']
for v in var:
    all_data[v] = all_data[v].fillna(all_data[v].mode()[0])

# print(all_data['Utilities'].value_counts())
all_data.drop('Utilities',1,inplace=True)

all_data["Functional"] = all_data["Functional"].fillna("Typ")
print(max(all_data.isnull().sum()))

#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder

# my1 = pd.get_dummies(all_data).to_csv('my1.csv')
columns = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')
# columns = all_data.columns.values
# print(type(all_data['Street'].values))
for column in columns:
    lb_make = LabelEncoder()
    all_data[column] = lb_make.fit_transform(all_data[column].values)
print('Shape all_data: {}'.format(all_data.shape))


all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

# 算一下数据斜度
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(skew).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})

# 进行box-cox变换
skewness = skewness[abs(skewness)>0.75]
# print('斜度大于0.75的有{}个'.format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)

all_data = pd.get_dummies(all_data)
print(all_data.shape)


# 建立模型

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

train = all_data[:ntrain]
test = all_data[ntrain:]
# 交叉验证
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring='neg_mean_squared_error',cv=kf))
    return rmse

# 基本模型
# LASSO Regression :拉索回归
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

#Elastic Net Regression :弹性网络回归
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

# Kernel Ridge Regression :内核岭回归
KRR = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)

#Gradient Boosting Regression : :
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)

#XGBoost :
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

#LightGBM :
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

# 单个模型的结果
# score = rmsle_cv(lasso)
# print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(ENet)
# print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(KRR)
# print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(GBoost)
# print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(model_xgb)
# print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(model_lgb)
# print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

# class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
#     def __init__(self, models):
#         self.models = models
#     def fit(self,X,y):
#         self.models_ = [clone(x) for x in self.models]
#
#         for model in self.models_:
#             model.fit(X,y)
#         return self
#
#     def predict(self,X):
#         predictions = np.column_stack([model.predict(X) for model in self.models_])
#         return np.mean(predictions, axis=1)

    # def score(self, X, y, sample_weight=None):
    #     return np.mean(self.predict(X))


# averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))
# score = rmsle_cv(averaged_models)
# print(score.mean(),score.std())

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [[]] * len(self.base_models)
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        # d = len(self.base_models_)
        # for base_models in self.base_models_:
        #     a = [model.predict(X) for model in base_models]
        #     c = len(base_models)
        #     for model in base_models:
        #         b = model.predict(X)
        #         pass
        meta_features = np.column_stack([np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
                                         for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR, GBoost, model_xgb, model_lgb),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
stacked_averaged_models.fit(train.values,y_train)
final_result = stacked_averaged_models.predict(train.values)
final_result = pd.DataFrame({'final_result':final_result})
final_result.to_csv('result.csv')
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))






