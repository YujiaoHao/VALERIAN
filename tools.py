# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 13:59:28 2022
tools
@author: haoyu
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
import seaborn as sn
from sklearn.manifold import TSNE

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


def balanced_sample_maker(X, y, sample_size, random_seed=42):
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}
    ids = [i for i in range(X.shape[0])]

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels (made all class have the same # of samples)
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=False).tolist()
        balanced_copy_idx += over_sample_idx
    np.random.shuffle(balanced_copy_idx)

    data_train = X[balanced_copy_idx]
    labels_train = y[balanced_copy_idx]
    if ((len(data_train)) == (sample_size * len(uniq_levels))):
        print('number of sampled example ', sample_size * len(uniq_levels), 'number of sample per class ', sample_size,
              ' #classes: ', len(list(set(uniq_levels))))
    else:
        print('number of samples is wrong ')

    labels, values = zip(*Counter(labels_train).items())
    # print('number of classes ', len(list(set(labels_train))))
    check = all(x == values[0] for x in values)
    print(check)
    if check == True:
        print('Good all classes have the same number of examples')
    else:
        print('Repeat again your sampling your classes are not balanced')
    indexes = np.arange(len(labels))
    # width = 0.5
    # plt.bar(indexes, values, width)
    # plt.xticks(indexes + width * 0.5, labels)
    # plt.show()

    other_idx = [] #get data not in the train set
    for gb_level, gb_idx in groupby_levels.items():
        other_idx += list(set(gb_idx) - set(balanced_copy_idx))
    np.random.shuffle(other_idx)

    data_val = X[other_idx]
    labels_val = y[other_idx]

    return data_train, labels_train, data_val, labels_val


# X_train,y_train=balanced_sample_maker(X,y,10)


def align_with_timestamp(X, Y):
    if X[0, 0] < Y[0, 0]:
        ind = np.where(X[:, 0] < Y[0, 0])
        X = np.delete(X, ind, 0)
    elif Y[0, 0] < X[0, 0]:
        ind = np.where(Y[:, 0] < X[0, 0])
        Y = np.delete(Y, ind, 0)
    if X[-1, 0] > Y[-1, 0]:
        ind = np.where(X[:, 0] > Y[-1, 0])
        X = np.delete(X, ind, 0)
    elif Y[-1, 0] > X[-1, 0]:
        ind = np.where(Y[:, 0] > X[-1, 0])
        Y = np.delete(Y, ind, 0)
    return X, Y


def check_redun(data):
    uniques = np.unique(data[:, 0], return_index=True)
    return data[uniques[1], :]


def interpolate_rawdata(data, ts):
    """Make the data evenly sampled by data interpolation
    INPUT: 
        data -- input time series data
        ts -- timestamp
    OUTPUT: 
        interpolated_data -- Nx3 array containing the sensor raw data (no timestamp)
                             1-3 column: x,y,z sensor output data
    """
    # Get interpolation function in terms of timestamp and sensor data
    interpolate_f = interp1d(data[:, 0], data[:, [1, 2, 3]], kind='linear', axis=0)

    # note that this variable only contains the sensor data, no timestamp included
    interpolated_data = interpolate_f(ts)
    data = np.hstack((ts.reshape(-1, 1), interpolated_data))
    return data


# def interpolate_rawdata (data, ts):
#    """Make the data evenly sampled by data interpolation
#    INPUT: 
#        data -- input time series data
#        ts -- timestamp
#    OUTPUT: 
#        interpolated_data -- Nx3 array containing the sensor raw data (no timestamp)
#                             1-3 column: x,y,z sensor output data
#    """
#    # Get interpolation function in terms of timestamp and sensor data
#    interpolate_f = interp1d(data[:,0], data[:, [1,2,3,4,5,6]], kind='linear', axis=0)
#
#    # note that this variable only contains the sensor data, no timestamp included
#    interpolated_data = interpolate_f(ts)
#
#    return interpolated_data

def emwfilter(x, a, axis=0):
    """The exponential moving average filter y(n) = (1-a)*x(n)+a*y(n-1)
    INPUT: 
        x -- input time series
        a -- weight
    OUTPUT: 
        y -- filter data
    """
    y = signal.lfilter([1 - a], [1, -a], x, axis)

    # remove the artifacts at the beginning
    scale = 1 - np.power(a, np.linspace(1, x.shape[0] + 1, x.shape[0])).reshape(x.shape[0], 1)

    return y / scale


def butter_lowpass(cutoff, nyq_freq, order=4):
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, np.ravel(data))
    return y


def remove_gravity(data, fc_hpf, fs):
    """Remove the gravity component from accelerometer data
    INPUT: 
        data -- input accelerometer data Nx3 (x, y, z axis)
        fc_hpf -- high pass filter cutoff frequency
    OUTPUT: 
        hpf_output -- filtered accelerometer data Nx3 (x, y, z axis)
    """
    # compute the coefficient
    a_emw = np.exp(-2 * np.pi * fc_hpf / fs)

    # the number of accelerometer readings
    num_data = data[:, 0].shape[0]
    # hpf_output is used to store the filtered accelerometer data by high pass filter
    hpf_output = np.zeros((num_data, 3))

    # compute linear acceleration for x axis
    acc_X = data[:, 0].reshape(num_data, 1)
    emwfilter_output = emwfilter(acc_X, a_emw)
    hpf_output[:, 0] = emwfilter_output.reshape(1, num_data)

    # compute linear acceleration for y axis
    acc_Y = data[:, 1].reshape(num_data, 1)
    emwfilter_output = emwfilter(acc_Y, a_emw)
    hpf_output[:, 1] = emwfilter_output.reshape(1, num_data)

    # compute linear acceleration for z axis
    acc_Z = data[:, 2].reshape(num_data, 1)
    emwfilter_output = emwfilter(acc_Z, a_emw)
    hpf_output[:, 2] = emwfilter_output.reshape(1, num_data)

    return hpf_output


def remove_noise(data, fc_lpf, fs):
    """Remove noise from accelerometer data via low pass filter
    INPUT: 
        data -- input accelerometer data Nx3 (x, y, z axis)
        fc_lpf -- low pass filter cutoff frequency
    OUTPUT: 
        lpf_output -- filtered accelerometer data Nx3 (x, y, z axis)
    """

    # the number of accelerometer readings
    num_data = data[:, 0].shape[0]
    # lpf_output is used to store the filtered accelerometer data by low pass filter
    lpf_output = np.zeros((num_data, 3))

    # compute linear acceleration for x axis
    acc_X = data[:, 0]
    butterfilter_output = butter_lowpass_filter(acc_X, fc_lpf, fs / 2)
    lpf_output[:, 0] = butterfilter_output.reshape(1, num_data)

    # compute linear acceleration for y axis
    acc_Y = data[:, 1]
    butterfilter_output = butter_lowpass_filter(acc_Y, fc_lpf, fs / 2)
    lpf_output[:, 1] = butterfilter_output.reshape(1, num_data)

    # compute linear acceleration for z axis
    acc_Z = data[:, 2]
    butterfilter_output = butter_lowpass_filter(acc_Z, fc_lpf, fs / 2)
    lpf_output[:, 2] = butterfilter_output.reshape(1, num_data)

    return lpf_output


# def normalize_data(data):
#    #check if there are nans
#    check_nan = np.argwhere(np.isnan(data))
#    if check_nan.size!=0:
#        print('there are nans!')
#        n = check_nan.shape[0]
#        for i in range(n):
#            ind1 = check_nan[i][0]
#            ind2 = check_nan[i][1]
#            data[ind1,ind2] = data[int(ind1-1),ind2]
#    res = normalize(data, axis=0, norm='max')
#    return res

def cal_acc(acc):
    m = acc.shape[0]
    res = np.zeros(m)
    for i in range(m):
        res[i] = np.sqrt(acc[i, 0] ** 2 + acc[i, 1] ** 2 + acc[i, 2] ** 2)
    return res


def normalize_data(data):
    res = normalize(data, axis=0, norm='max')
    return res


# #normalize by max mag
# def normalize_data(data):
#     #check if there are nans
# #    check_nan = np.argwhere(np.isnan(data))
# #    if check_nan.size!=0:
# #        print('there are nans!')
# #        n = check_nan.shape[0]
# #        for i in range(n):
# #            ind1 = check_nan[i][0]
# #            ind2 = check_nan[i][1]
# #            data[ind1,ind2] = data[int(ind1-1),ind2]
#     mag1 = cal_acc(data)
#     #normalize by divide the max of mag
#     obx = np.zeros((data.shape[0],data.shape[1]))
#     obx = data/mag1.max()
#     return obx

def plot_processed_accdata(raw_data, lpf_data, norm_data, ts, title):
    # visualize the accelerometer data after data pre-processing
    fig = plt.figure(figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')

    plt.subplot(3, 3, 1)
    plt.plot(ts, raw_data[:, 0], 'k-')
    plt.ylabel('Acc X')
    plt.xlabel('time (ms)')

    plt.subplot(3, 3, 2)
    plt.plot(ts, lpf_data[:, 0], 'k-')
    plt.ylabel('Acc X LPF')
    plt.xlabel('time (ms)')

    plt.subplot(3, 3, 3)
    plt.plot(ts, norm_data[:, 0], 'k-')
    plt.ylabel('Acc X Norm')
    plt.xlabel('time (ms)')

    plt.subplot(3, 3, 4)
    plt.plot(ts, raw_data[:, 1], 'r-')
    plt.ylabel('Acc Y')
    plt.xlabel('time (ms)')

    plt.subplot(3, 3, 5)
    plt.plot(ts, lpf_data[:, 1], 'r-')
    plt.ylabel('Acc Y LPF')
    plt.xlabel('time (ms)')

    plt.subplot(3, 3, 6)
    plt.plot(ts, norm_data[:, 1], 'r-')
    plt.ylabel('Acc Y Norm')
    plt.xlabel('time (ms)')

    plt.subplot(3, 3, 7)
    plt.plot(ts, raw_data[:, 2], 'g-')
    plt.ylabel('Acc Z')
    plt.xlabel('time (ms)')

    plt.subplot(3, 3, 8)
    plt.plot(ts, lpf_data[:, 2], 'g-')
    plt.ylabel('Acc Z LPF')
    plt.xlabel('time (ms)')

    plt.subplot(3, 3, 9)
    plt.plot(ts, norm_data[:, 2], 'g-')
    plt.ylabel('Acc Z Norm')
    plt.xlabel('time (ms)')

    fig.suptitle(title)
    plt.show()


def plot_raw(rawdata):
    fig, ax = plt.subplots(3, 1)
    ts = np.arange(0, rawdata.shape[0], step=1)
    ax[0].plot(ts, rawdata[:, 0], 'k-')
    ax[1].plot(ts, rawdata[:, 1], 'r-')
    ax[2].plot(ts, rawdata[:, 2], 'g-')
    plt.show()


def plot_acti(rawdata, name):
    fig, ax = plt.subplots(3, 1)
    plt.title(name)
    ts = np.arange(0, rawdata.shape[0], step=1)
    ax[0].plot(ts, rawdata[:, 0], 'k-')
    ax[1].plot(ts, rawdata[:, 1], 'r-')
    ax[2].plot(ts, rawdata[:, 2], 'g-')
    plt.show()


def check_nan(data):
    # check if there are nans
    check_nan = np.argwhere(np.isnan(data))
    if check_nan.size != 0:
        n = check_nan.shape[0]
        print('there are nans!', n)
        data = np.delete(data, check_nan[:, 0], 0)
    return data


def convert_list(my_list):
    length = len(my_list)
    if length == 0:
        return
    res = my_list[0]
    for i in range(1, length):
        res = np.vstack((res, my_list[i]))
    return res


def convert_y(my_list):
    length = len(my_list)
    if length == 0:
        return
    res = my_list[0]
    for i in range(1, length):
        res = np.concatenate((res, my_list[i]))
    return res


def make_context_window(X_raw, L, s):
    m, n = X_raw.shape
    res = []
    i = 0
    while i <= m:
        ind1 = i * (L - s)
        ind2 = ind1 + L
        if ind1 >= m:
            break
        if ind2 > m and ind1 < m:
            mzeros = np.zeros((ind2 - m, n))
            temp = np.vstack((X_raw[ind1:m, :], mzeros))
            res.append(temp)
            break
        res.append(X_raw[ind1:ind2, :])
        i = i + 1
    res = np.stack(res, axis=0)
    return res


def label_per_window(X):
    m, n, o = X.shape
    mones = np.ones((n, 1))
    for i in range(m):
        temp = X[i, :, 12:13]
        unique, counts = np.unique(temp, return_counts=True)
        label = unique[counts.argmax(axis=0)] * mones
        X[i, :, :] = np.hstack((X[i, :, :12], label))
    return X


SLIDING_WINDOW_LENGTH = 100


def load_tensor(name):
    # Read the array from disk
    new_data = np.loadtxt('./SplittedData/bysubject/' + name + '.txt')

    # Note that this returned a 2D array!
    print(new_data.shape)

    # However, going back to 3D is easy if we know the 
    # original shape of the array
    new_data = new_data.reshape((-1, SLIDING_WINDOW_LENGTH, 12))
    return new_data


# Save processed windowed data
def save_tensor(data, name):
    # Write the array to disk
    with open("./ProcessedData/ready/Sub" + str(name) + "_data.txt", 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(data.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in data:
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.  
            np.savetxt(outfile, data_slice, fmt='%-7.2f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')


# =============================================================================
# A:Walking 
# B:Jogging
# C:Stairs 
# D:Sitting
# E:Standing 
# M:Kicking Soccer Ball 
# P:Dribblinlg Basketball 
# =============================================================================
# #evaluate and print predict result
# def eval_perf(ground_truth, predicted_event):
#     print('Accuracy score is: ')
#     acc = accuracy_score(ground_truth, predicted_event)
#     print(acc)
#     print('Confusion Matrix is:')
#     my_matrix = confusion_matrix(ground_truth, predicted_event)
#     my_matrix_n = normalize(my_matrix, axis=1,norm = 'l1')
#     print(pd.DataFrame(my_matrix_n).applymap(lambda x: '{:.2%}'.format(x)).values)

#     target_names = ['walk','jog','stairs','sit','stand','soccer',
#                     'basketball']
#     df_cm = pd.DataFrame(my_matrix_n, index = [i for i in target_names],
#                   columns = [i for i in target_names])
#     plt.figure(figsize = (10,7))
#     sn.heatmap(df_cm, annot=True)
#     print(classification_report(ground_truth, predicted_event, target_names=target_names))  
#     return acc

# evaluate and print predict result
def eval_perf(ground_truth, predicted_event):
    print('Accuracy score is: ')
    acc = accuracy_score(ground_truth, predicted_event)
    print(acc)
    print('Confusion Matrix is:')
    my_matrix = confusion_matrix(ground_truth, predicted_event)
    my_matrix_n = normalize(my_matrix, axis=1, norm='l1')
    print(pd.DataFrame(my_matrix_n).applymap(lambda x: '{:.2%}'.format(x)).values)

    target_names = ['walkF', 'walkL', 'walkR', 'upstairs', 'downstairs', 'run',
                    'jump', 'sit', 'stand', 'lying']
    df_cm = pd.DataFrame(my_matrix_n, index=[i for i in target_names],
                         columns=[i for i in target_names])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    print(classification_report(ground_truth, predicted_event, target_names=target_names))
    return acc


# =============================================================================
# do t-sne of test data
# =============================================================================
def tsne(test, ytest):
    data1 = np.hstack((test, ytest.reshape(-1, 1)))
    Y = TSNE(n_components=2).fit_transform(data1)

    fig, ax = plt.subplots()
    groups = pd.DataFrame(Y, columns=['x', 'y']).assign(category=ytest).groupby('category')
    listact = ['walkF', 'walkL', 'walkR', 'upstairs', 'downstairs', 'run',
               'jump', 'sit', 'stand', 'lying']
    # listact = ['walk','jog','stairs','sit','stand','soccer','basketball']
    # colors = cm.rainbow(np.linspace(0, 1, len(listact)))
    ind = 0
    for name, points in groups:
        f = listact[int(name - 1)]
        print(f)
        # ax.scatter(points.x, points.y, label=f, color=colors[ind])
        ax.scatter(points.x, points.y, label=f)
        ind += 1
    ax.legend()
