"""
+---------------------------------------------------------------+
| Main function/script                                          |
+---------------------------------------------------------------+
------------------------------------------------------------------
Copyright: 2019 Wang,Zehui (wzehui@hotmail.com)
@author: Wang,Zehui
"""

import os
import yaml
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from utils.AudioProcessing import process

# load parameter from *.yml
def fParseConfig(sFile):
    # get config file
    with open(sFile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


# load parameter file
cfg = fParseConfig('param.yml')

# load feature
feature_data = pd.read_csv(cfg['FeaturePath'])
labels = []
for i in range(len(feature_data)):
    labels.append(int(feature_data.Label[i]))
feature = []
feature_pre = []
for i in range(feature_data.shape[0]):
    # creat selected feature data for training
    temp = feature_data.iloc[i][cfg['FeatureSelection']]
    feature.append(temp)
    # creat feature from raw audio for pre-classify
    temp_pre = pd.Series()
    path = feature_data.iloc[i][1]

    # check the difference between system sepration symbol
    # windows: \, Linux and MacOS: /
    if path.find(os.sep):
        path = path.split('\\')
        path = os.sep.join(path)
        # path = ''.join(path.split())    # delete space within the path

    y = process(cfg['FilePath'], path)
    # for j in range(len(y.mean_)):
    #     temp_pre = temp_pre.append(pd.Series(y.mean_[j], index=['PCA_' + str(j)]))

    for j in range(y.shape[1]):
        temp_pre = temp_pre.append(pd.Series(y[0, j], index=['PCA_' + str(j)]))

    feature_pre.append(temp_pre)
    print('Data Processing... <-------------- {:.1f}% -------------->'.format(100*(i/feature_data.shape[0])))

# initialization of Random Forest
params_pre = {'max_depth': [5,10,15,20],
          'min_samples_split': [2],
          'min_samples_leaf': [2],
          'n_estimators': [50,100,150,200]
          }
params = {'max_depth': [5,10,15,20,25],
          'min_samples_split': [2],
          'n_estimators': [50,100,150,200],
          'min_samples_leaf': [2],
          }
CrossValidation = StratifiedKFold(n_splits=10, shuffle=True, random_state=cfg['RandomState'])
rf = RandomForestClassifier(oob_score=True, random_state=cfg['RandomState'])
rf_pre = RandomForestClassifier(oob_score=True, random_state=cfg['RandomState'])
rf_grid_pre = GridSearchCV(estimator=rf_pre, param_grid=params_pre, iid=True, cv=CrossValidation, verbose=4)
rf_grid = GridSearchCV(estimator=rf, param_grid=params, iid=True, cv=CrossValidation, verbose=4)
# split training and test data
X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(feature_pre, labels, test_size=0.1, random_state=cfg['RandomState'])
X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size=0.1, random_state=cfg['RandomState'])
# # export name of test dataset, change FeatureSelection to [0]
# with open('/Users/wzehui/Documents/MA/Model/DataDistribution.txt', 'w') as f:
#     f.write('Test Data Name:')
#     f.write('\n')
#     f.write(str(X_test))

# pre-train
PreClassify = [-1] * len(X_train_pre)
index = index_tmp = range(len(X_train_pre))
index_test_cv = []
SplitNum = 10
BER_cv = np.arange(SplitNum, dtype=float)

for i in range(SplitNum):

    index_tmp = list(set(index_tmp) - set(index_test_cv))
    random.seed(cfg['RandomState'])
    index_test_cv = random.sample(index_tmp, round(len(index_tmp)/SplitNum))
    index_train_cv = list(set(index) - set(index_test_cv))
    print("Cross-Validation time:{}".format(i+1))

    X_train_cv = []
    y_train_cv = []
    X_test_cv = []
    y_test_cv = []
    for j in range(len(index_train_cv)):
        X_train_cv.append(X_train_pre[index_train_cv[j]])
        y_train_cv.append(y_train_pre[index_train_cv[j]])

    for j in range(len(index_test_cv)):
        X_test_cv.append(X_train_pre[index_test_cv[j]])
        y_test_cv.append(y_train_pre[index_test_cv[j]])

    rf_grid_pre.fit(X_train_cv, y_train_cv)

    y_pred_cv = rf_grid_pre.predict(X_test_cv)

    CM_cv = confusion_matrix(y_test_cv, y_pred_cv)

    BER_cv[i] = 0.5 * (CM_cv[0][1]/(CM_cv[0][0]+CM_cv[0][1]) + CM_cv[1][0]/(CM_cv[1][0]+CM_cv[1][1]))
    print("BER_{}:{}".format(i+1, BER_cv[i]))

    for j in range(len(index_test_cv)):
        PreClassify[index_test_cv[j]] = pd.Series(y_pred_cv[j], index=['preClassify'])

    SplitNum -= 1

for i in range(len(X_train)):
    X_train[i] = X_train[i].append(PreClassify[i])


# Train
rf_grid.fit(X_train, y_train)
# print(rf_grid.best_estimator_.feature_importances_)

# pre-test
y_pred_pre = rf_grid_pre.predict(X_test_pre)

PreClassify = [-1] * len(X_test)
for i in range(len(X_test)):
    PreClassify[i] = pd.Series(y_pred_pre[i], index=['preClassify'])

for i in range(len(X_test)):
    X_test[i] = X_test[i].append(PreClassify[i])

# Test
y_pred_grid = rf_grid.predict(X_test)

# Criterion
CM_grid = confusion_matrix(y_test, y_pred_grid)
BER_grid = 0.5 * (CM_grid[0][1]/(CM_grid[0][0]+CM_grid[0][1]) + CM_grid[1][0]/(CM_grid[1][0]+CM_grid[1][1]))

# save the best parameter
with open(cfg['BestParamPath'], 'w') as f:
    f.write('best pre-classifiy parameters:' + str(rf_grid_pre.best_params_))
    f.write('\n')
    f.write('best parameters:' + str(rf_grid.best_params_))
    f.write('\n')
    f.write('BER:' + str(BER_grid))
    f.write('\n')
    f.write('Importance:' + str(rf_grid.best_estimator_.feature_importances_))
