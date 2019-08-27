"""
+---------------------------------------------------------------+
| Main function/script                                          |
+---------------------------------------------------------------+
------------------------------------------------------------------
Copyright: 2019 Wang,Zehui (wzehui@hotmail.com)
@author: Wang,Zehui
"""

# import
import yaml
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from utils.AudioProcessing import process


def para_config(sFile):
    # get config file
    with open(sFile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    return cfg


# load parameter file
cfg = para_config('param.yml')
feature_data = pd.read_csv(cfg['FeaturePath'])
# load label
labels = []
for i in range(len(feature_data)):
    labels.append(int(feature_data.Label[i]))
# load feature
feature = []
for i in range(feature_data.shape[0]):
    temp = feature_data.iloc[i][cfg['FeatureSelection']]
    if cfg['FeatureExtraction']:
        path = feature_data.iloc[i][1]
        # check the difference between system sepration symbol
        # windows: \, Linux and MacOS: /
        if path.find(os.sep):
            path = path.split('\\')
            path = os.sep.join(path)
        y = process(cfg['FilePath'], path)
        for j in range(len(y.mean_)):
            temp = temp.append(pd.Series(y.mean_[j], index=['PCA_' + str(j)]))

        # for j in range(y.shape[1]):
        #     temp = temp.append(pd.Series(y[0, j], index=['PCA_' + str(j)]))
    feature.append(temp)
    print('Data Processing... <-------------- {:.1f}% -------------->'.format(100 * (i / feature_data.shape[0])))
# initialization
X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size=0.2, random_state=cfg['RandomState'])
# # change FeatureSelection to [0]
# with open('/Users/wzehui/Documents/MA/Model/DataDistribution_B.txt', 'w') as f:
#     f.write('Test Data Name:')
#     f.write('\n')
#     f.write(str(X_test))
params = {'max_depth': range(5, 105, 5),  # [10,20,30,40,50,60,70,80,100,200],
          'min_samples_split': [2],
          'min_samples_leaf': [2],
          'n_estimators': range(25, 300, 25),  # [50,100,150,200,250,300,350]
          }
CrossValidation = StratifiedKFold(n_splits=10, shuffle=True, random_state=cfg['RandomState'])
rf = RandomForestClassifier(oob_score=True, random_state=cfg['RandomState'])
rf_grid = GridSearchCV(estimator=rf, param_grid=params, iid=False, cv=CrossValidation, verbose=4)
# Train
rf_grid.fit(X_train, y_train)
# print(rf_grid.best_estimator_.feature_importances_)
# Test
y_pred_grid = rf_grid.predict(X_test)
# Criterion
CM_grid = confusion_matrix(y_test, y_pred_grid)
BER_grid = 0.5 * (CM_grid[0][1]/(CM_grid[0][0]+CM_grid[0][1]) + CM_grid[1][0]/(CM_grid[1][0]+CM_grid[1][1]))
# Save Model
with open(cfg['BestParamPath'], 'w') as f:
    f.write(str(rf_grid.best_params_))
    f.write('\n')
    f.write('BER:' + str(BER_grid))
    f.write('\n')
    f.write(str(rf_grid.best_estimator_.feature_importances_))
