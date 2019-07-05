import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import pandas

from utils.DataPreprocessing import *

feature_path = '/Users/wzehui/Documents/MA/Daten/feature/feature_S.csv'
feature_index = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

feature_data = pd.read_csv(feature_path)

labels = []
for i in range(len(feature_data)):
    labels.append(int(feature_data.Label[i]))

feature = []
for i in range(feature_data.shape[0]):
    temp = feature_data.iloc[i]
    feature.append(temp[feature_index])

X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size=0.3, random_state=50)

# clf = tree.DecisionTreeClassifier()
# clf_rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=0)

grid = RandomForestClassifier(random_state=0)
params = {'max_depth': [8,9,10,11,12],
          'min_samples_split': [11,12,13,15],
          'min_samples_leaf': [2,3,4,5,6],
          'n_estimators': [60,70,80,90]
          } # S: 10, 13, 3, 80, 0.174

grid_clf = GridSearchCV(estimator=grid, param_grid=params, n_jobs=-1, verbose=4)

# clf.fit(X_train, y_train)
# clf_rf.fit(X_train, y_train)
# print(clf_rf.feature_importances_)

grid_clf.fit(X_train, y_train)

# tree.plot_tree(clf.fit(iris.data, iris.target))

#
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("Sopra")

# y_pred = clf.predict(X_test)
# y_pred_rf = clf_rf.predict(X_test)
y_pred_grid = grid_clf.predict(X_test)

# CM = confusion_matrix(y_test, y_pred)
# CM_rf = confusion_matrix(y_test, y_pred_rf)
CM_grid = confusion_matrix(y_test, y_pred_grid)

# BER = 0.5 * (CM[0][1]/(CM[0][0]+CM[0][1]) + CM[1][0]/(CM[1][0]+CM[1][1]))
# BER_rf = 0.5 * (CM_rf[0][1]/(CM_rf[0][0]+CM_rf[0][1]) + CM_rf[1][0]/(CM_rf[1][0]+CM_rf[1][1]))
BER_grid = 0.5 * (CM_grid[0][1]/(CM_grid[0][0]+CM_grid[0][1]) + CM_grid[1][0]/(CM_grid[1][0]+CM_grid[1][1]))

