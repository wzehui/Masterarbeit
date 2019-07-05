from utils.DataPreprocessing import *

csv_path = '/Users/wzehui/Documents/MA/Daten/index/Sopran.csv'
file_path = '/Users/wzehui/Documents/MA/Daten/quellcode/sounddb/'
feature_path = '/Users/wzehui/Documents/MA/Daten/quellcode/sounddb/feature_S.csv'
feature_index = [5, 7, 8]


def verify(data):
    PHE = 38
    VF = 6.6
    Extent = 82

    TP = 0; TN = 0; FP = 0; FN = 0
    acc = 0
    dra = 0
    lyr = 0
    error_name = []

    for i in range(0, len(data)):
        a = data[i]
        if a[0][0][0] > PHE and a[0][0][1] > VF:
            label = 1
        elif a[0][0][1] > VF and a[0][0][2] <= Extent:
            label = 1
        elif a[0][0][2] <= Extent and a[0][0][0] > PHE:
            label = 1
        else:
            label = 0

        if label == a[1] == 0:
            TN += 1
            acc += 1
        elif label == 0 and label != a[1]:
            FN += 1
        elif label == a[1] == 1:
            TP += 1
            acc += 1
        elif label == 1 and label != a[1]:
            FP += 1

        if a[1] == 0:
            dra += 1
        elif a[1] == 1:
            lyr += 1

        if label != a[1]:
            error_name += [data.feature_data.Name[i]]
            # print(data.feature_data.Name[i])
            # print(a[0][0][0], a[0][0][1], a[0][0][2])

    # print('\n dramatish/lyrisch : {:.0f}%/{:.0f}%'.format(dra/len(data)*100, lyr/len(data)*100))
    print('\n dramatish/lyrisch : {}/{}'.format(dra, lyr))
    print('\n True/False : {}/{}, accuracy:{:.0f}%'.format(acc, len(data)-acc, acc/len(data)*100))
    # print('\no frequency analyse items: {}'.format(err))

    print('\n TN/FP TP/FN: {}/{} {:.0f}% {}/{} {:.0f}%'.format(TN, FP, TN/(TN+FP)*100, TP, FN, TP/(TP+FN)*100))

    return error_name

training_index, val_index, test_index = process_index(csv_path, 0, 0, 0)


index = training_index

train_set = PreprocessFeature(feature_path, feature_index, index)

# a = train_set[11]

error_python = verify(train_set)

error_matlab = pd.read_csv('/Users/wzehui/Documents/error.csv')
error_matlab = error_matlab.values.tolist()

# n = 0
# for i in range(0, len(error_matlab)):
#     a = error_matlab[i]
#     a = "".join(a)
#     if a not in error_python:
#         n += 1
#         print(a)
# print(n)


