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
import librosa
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn import svm


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
    temp = feature_data.iloc[i][cfg['FeatureSelection']]

    if cfg['FeatureExtraction']:
        path = feature_data.iloc[i][1]

        # check the difference between system sepration symbol
        # windows: \, Linux and MacOS: /
        if path.find(os.sep):
            path = path.split('\\')
            path = os.sep.join(path)
            # path = ''.join(path.split())    # delete space within the path

        y, fs = librosa.load(cfg['FilePath'] + path + '_m.wav')

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

        y = librosa.stft(y, n_fft=frame_length, hop_length=frame_step, win_length=None, window='hann', center=True,
                         pad_mode='reflect')
        y = np.abs(y) ** 2
        y = librosa.feature.melspectrogram(S=y, n_mels=nmel)
        y = librosa.power_to_db(np.abs(y) ** 2)
        y = librosa.feature.mfcc(S=y, sr=fs, n_mfcc=64, dct_type=2, norm='ortho')
        pca = PCA(n_components=1)
        y = pca.fit(y)
        # for j in range(len(y.mean_)):
        #     temp = temp.append(pd.Series(y.mean_[j], index=['PCA_' + str(j)]))
        for j in range(y.components_.shape[1]):
            temp = temp.append(pd.Series(y.components_[0, j], index=['PCA_' + str(j)]))

    feature.append(temp)

# initialization
X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size=0.1, random_state=cfg['RandomState'])

# # change FeatureSelection to [0]
# with open('/Users/wzehui/Documents/MA/Model/DataDistribution_B.txt', 'w') as f:
#     f.write('Test Data Name:')
#     f.write('\n')
#     f.write(str(X_test))
clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=cfg['RandomState'])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
ACC = clf.score(X_test, y_test)

# Criterion
CM_grid = confusion_matrix(y_test, y_pred)
BER_grid = 0.5 * (CM_grid[0][1]/(CM_grid[0][0]+CM_grid[0][1]) + CM_grid[1][0]/(CM_grid[1][0]+CM_grid[1][1]))


