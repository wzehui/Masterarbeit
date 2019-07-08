"""
+---------------------------------------------------------------+
| Main function/script                                          |
+---------------------------------------------------------------+
------------------------------------------------------------------
Copyright: 2019 Wang,Zehui (wzehui@hotmail.com)
@author: Wang,Zehui
"""

import yaml
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


def fParseConfig(sFile):
    # get config file
    with open(sFile, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    return cfg


# load parameter file
cfg = fParseConfig('param.yml')

feature_data = pd.read_csv(cfg['FeaturePath'])

labels = []
for i in range(len(feature_data)):
    labels.append(int(feature_data.Label[i]))

feature = []
for i in range(feature_data.shape[0]):
    temp = feature_data.iloc[i]
    feature.append(temp[cfg['FeatureSelection']])

# initialization
X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size=0.2, random_state=cfg['RandomState'])

# # change FeatureSelection to [0]
# with open('/Users/wzehui/Documents/MA/Model/DataDistribution_T.txt', 'w') as f:
#     f.write('Test Data Name:')
#     f.write('\n')
#     f.write(str(X_test))

params = {'max_depth': [12],
          'min_samples_split': [2],
          'n_estimators': [200],
          'min_samples_leaf': [2],
          } # S: 10, 13, 3, 80, 0.174
CrossValidation = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg['RandomState'])
rf = RandomForestClassifier(oob_score=True, random_state=cfg['RandomState'])
rf_grid = GridSearchCV(estimator=rf, param_grid=params, iid=True, cv=CrossValidation, n_jobs=-1, verbose=4)

# Train
rf_grid.fit(X_train, y_train)
# print(rf_grid.best_estimator_.feature_importances_)

# Test
y_pred_grid = rf_grid.predict(X_test)

# Criterion
CM_grid = confusion_matrix(y_test, y_pred_grid)
BER_grid = 0.5 * (CM_grid[0][1]/(CM_grid[0][0]+CM_grid[0][1]) + CM_grid[1][0]/(CM_grid[1][0]+CM_grid[1][1]))

with open(cfg['BestParamPath'], 'w') as f:
    f.write(str(rf_grid.best_params_))
    f.write('\n')
    f.write('BER:' + str(BER_grid))
    f.write('\n')
    f.write(str(rf_grid.best_estimator_.feature_importances_))
