from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import librosa
from sklearn.decomposition import PCA
import random
from sklearn.model_selection import StratifiedKFold


FeaturePath = '/Users/wzehui/Documents/MA/Daten/quellcode/result_mat/feature_S.csv'
FilePath = '/Users/wzehui/Documents/MA/Daten/quellcode/sounddb/'
feature_index = [4,5,6,7,8,9,10,11,12,13,14,15,16,17]

rf = RandomForestClassifier(oob_score=True, random_state=55)

feature_data = pd.read_csv(FeaturePath)

labels = []
for i in range(len(feature_data)):
    labels.append(int(feature_data.Label[i]))

# dra = lyr = 0
# for i in range(len(labels)):
#     if labels[i] == 0:
#         dra += 1
#     elif labels[i] == 1:
#         lyr += 1

feature = []
feature_pre = []

for i in range(feature_data.shape[0]):

    temp = feature_data.iloc[i][feature_index]
    feature.append(temp)

    temp_pre = pd.Series()
    path = feature_data.iloc[i][1]
    y, fs = librosa.load(FilePath + path + '_m.wav')

    pre_emphasis = 0.97
    y = np.append(y[0], (y[1:] - pre_emphasis * y[:-1]))

    zero_temp = np.zeros(90000)
    if y.size < 90000:
        zero_temp[0:y.size] = y[0:]
    else:
        zero_temp[0:] = y[0:90000]

    y = zero_temp

    frame_size = 0.025
    frame_stride = 0.01
    nmel = 72

    frame_length = frame_size * fs
    frame_length = int(round(frame_length))

    frame_step = frame_stride * fs
    frame_step = int(round(frame_step))

    y = librosa.stft(y, n_fft=frame_length, hop_length=frame_step, win_length=None, window='hann', center=True, pad_mode='reflect')
    y = np.abs(y) ** 2
    y = librosa.feature.melspectrogram(sr=fs, S=y, n_fft=frame_length, hop_length=frame_step, power=1.0, n_mels=nmel)

    y = librosa.feature.mfcc(S=y, sr=fs, n_mfcc=16, dct_type=2, norm='ortho')
    pca = PCA(n_components=1)
    y = pca.fit(y)
    for j in range(len(y.mean_)):
        temp_pre = temp_pre.append(pd.Series(y.mean_[j], index=['PCA_' + str(j)]))

    feature_pre.append(temp_pre)

X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size=0.2, random_state=55)
X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(feature_pre, labels, test_size=0.2, random_state=55)


PreClassify = [-1] * len(X_train_pre)
index = index_tmp = range(len(X_train_pre))
index_test_cv = []

SplitNum = 10

params_pre = {'max_depth': [12],
          'min_samples_split': [2],
          'min_samples_leaf': [2],
          'n_estimators': [50]
          }

rf_pre = GridSearchCV(estimator=rf, param_grid=params_pre, n_jobs=-1, iid=False, verbose=4)

for i in range(SplitNum):

    index_tmp = list(set(index_tmp) - set(index_test_cv))
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

    rf_pre.fit(X_train_cv, y_train_cv)

    y_pred_cv = rf_pre.predict(X_test_cv)

    CM_cv = confusion_matrix(y_test_cv, y_pred_cv)

    BER_cv = 0.5 * (CM_cv[0][1]/(CM_cv[0][0]+CM_cv[0][1]) + CM_cv[1][0]/(CM_cv[1][0]+CM_cv[1][1]))
    print("BER_{}:{}".format(i+1, BER_cv))

    for j in range(len(index_test_cv)):
        PreClassify[index_test_cv[j]] = pd.Series(y_pred_cv[j], index=['preClassify'])

    SplitNum -= 1

for i in range(len(X_train)):
    X_train[i] = X_train[i].append(PreClassify[i])

y_pred_pre = rf_pre.predict(X_test_pre)
PreClassify = [-1] * len(X_test)
for i in range(len(X_test)):
    PreClassify[i] = pd.Series(y_pred_pre[i], index=['preClassify'])

for i in range(len(X_test)):
    X_test[i] = X_test[i].append(PreClassify[i])


params = {'max_depth': [12],
          'min_samples_split': [2],
          'n_estimators': [50],
          'min_samples_leaf': [2],
          }
CrossValidation = StratifiedKFold(n_splits=10, shuffle=True, random_state=55)
rf_grid = GridSearchCV(estimator=rf, param_grid=params, iid=True, cv=CrossValidation, verbose=4)

# Train
rf_grid.fit(X_train, y_train)
print(rf_grid.best_estimator_.feature_importances_)

# Test
y_pred_grid = rf_grid.predict(X_test)

# Criterion
CM_grid = confusion_matrix(y_test, y_pred_grid)
BER_grid = 0.5 * (CM_grid[0][1]/(CM_grid[0][0]+CM_grid[0][1]) + CM_grid[1][0]/(CM_grid[1][0]+CM_grid[1][1]))

