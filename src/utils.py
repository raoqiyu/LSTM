# -*- coding: utf-8 -*-
"""
@author: raoqiyu
Created on Thu Oct 15 20:18:36 2015

util functions
"""

import pickle

import numpy
import theano
import matplotlib.pyplot as plt


def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)


def ortho_weight(ndim1, ndim2):
    W = numpy.random.randn(ndim1, ndim2)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)


def generateData(data, duration=0.75, n_skip=150):
    """
    generate short sequence from long sequence train data

    :param data: long sequence train data
    :return: short sequence data
    """
    n_samples = len(data)
    n_steps = len(data[0][0])

    n_duration = duration

    n_newSteps = int(n_steps * n_duration)
    n_end = int(n_steps * (1 - n_duration))
    new_data = []
    for sample in range(n_samples):
        for step in range(0, n_end, n_skip):
            d_x = data[sample][0][step:step + n_newSteps]
            d_y = data[sample][1][step:step + n_newSteps]
            new_data.append((d_x, d_y))

    # Shuffle
    idx_list = numpy.arange(len(new_data), dtype="int32")
    numpy.random.seed(123)
    numpy.random.shuffle(idx_list)
    new_data_shuffled = []
    for idx in idx_list:
        new_data_shuffled.append(new_data[idx])

    return new_data_shuffled


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return list(zip(list(range(len(minibatches))), minibatches))


def parallelize_data(x):
    """Create a matrix from several train data and label to do parallel computing
    """
    n_samples = len(x)
    n_steps = len(x[0])
    n_feature_size = len(x[0][0])

    X = numpy.zeros((n_steps, n_samples, n_feature_size)).astype(theano.config.floatX)
    for idx, sample in enumerate(x):
        X[:, idx] = sample

    return X


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x = prepare_data([data[t][0] for t in valid_index])
        y = prepare_data([data[t][1] for t in valid_index])
        preds = f_pred(x)
        valid_err += numpy.sum((y - preds) ** 2)
    total = len(data) * len(data[0][1])
    valid_err /= total
    return numpy.sqrt(valid_err)


def evaluate(f_pred, prepare_data, data):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    n_samples = len(data)
    n_steps = len(data[0][0])
    # kf = get_minibatches_idx(len(data), 5)
    if len(data[0]) == 2:
        x = prepare_data([data[t][0] for t in range(n_samples)])
        y = prepare_data([data[t][1] for t in range(n_samples)])
        y_preds = numpy.zeros(y.shape, dtype=y.dtype)
        kf = get_minibatches_idx(n_samples, 50)
        for _, valid_index in kf:
            x_preds = x[:, valid_index, :]
            preds = f_pred(x_preds)
            y_preds[:, valid_index, :] = preds
    elif len(data[0]) == 3:
        x1 = prepare_data([data[t][0] for t in range(n_samples)])
        x2 = prepare_data([data[t][1] for t in range(n_samples)])
        y = prepare_data([data[t][2] for t in range(n_samples)])
        y_preds = numpy.zeros(y.shape, dtype=y.dtype)
        kf = get_minibatches_idx(n_samples, 50)
        for _, valid_index in kf:
            x1_preds = x1[:, valid_index, :]
            x2_preds = x2[:, valid_index, :]
            preds = f_pred(x1_preds,x2_preds)
            y_preds[:, valid_index, :] = preds
    # preds = f_pred(x)
    preds_flatten = y_preds.flatten('F')
    y_flatten = y.flatten('F')
    RMSE, CC, CCC = rater_statistics(y_flatten, preds_flatten)

    return RMSE, CCC


def predict_Stage1(f_pred, prepare_data, data, feature, emodim='arousal', partitions='train',generating=True,comparation=True):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    print('Evaluate on data:', emodim, partitions)
    n_samples = len(data)
    n_steps = len(data[0][0])
    # kf = get_minibatches_idx(len(data), 5)
    x = prepare_data([data[t][0] for t in range(n_samples)])
    y = prepare_data([data[t][1] for t in range(n_samples)])

    preds = f_pred(x)

    preds_flatten = preds.flatten('F')
    y_flatten = y.flatten('F')

    # Compute performance
    RMSE, CC, CCC = rater_statistics(y_flatten, preds_flatten)
    print("Before smoothing:", RMSE, CC, CCC)
    if comparation:
        smooth_label(preds_flatten, 0.04)
        RMSE, CC, CCC = rater_statistics(y_flatten, preds_flatten)
        print("0.04   smoothing:", RMSE, CC, CCC)

        smooth_label(preds_flatten, 0.08)
        RMSE, CC, CCC = rater_statistics(y_flatten, preds_flatten)
        print("0.08   smoothing:", RMSE, CC, CCC)

        smooth_label(preds_flatten, 1.)
        RMSE, CC, CCC = rater_statistics(y_flatten, preds_flatten)
        print("1.00   smoothing:", RMSE, CC, CCC)

        filter(preds_flatten, 0.03)
        RMSE, CC, CCC = rater_statistics(y_flatten, preds_flatten)
        print("Filter 0.25:", RMSE, CC, CCC)

        filter(preds_flatten, 0.03)
        RMSE, CC, CCC = rater_statistics(y_flatten, preds_flatten)
        print("Filter 0.30:", RMSE, CC, CCC)

        filter(preds_flatten, 0.03)
        RMSE, CC, CCC = rater_statistics(y_flatten, preds_flatten)
        print("Filter 0.35:", RMSE, CC, CCC)

        filter(preds_flatten, 0.03)
        RMSE, CC, CCC = rater_statistics(y_flatten, preds_flatten)
        print("Filter 0.40:", RMSE, CC, CCC)

    if generating:
            # Concatenate data
        file_name = '/'.join(['data/training',feature,emodim,partitions+emodim.capitalize()])
        print("Generating data for second stage:",file_name)
        save_result(preds, y, file_name)
    fname = 'predictions/Stage1/Basic/4BLSTM/' + emodim + '_' + partitions + '.arff'
    # preds : (n_steps * n_sample)
    with open(fname, 'w') as f:
        for i in range(n_samples):
            single_file = open('predictions/Stage1/Basic/4BLSTM/' + emodim + '/' + partitions + '/' + partitions + '_' + str(i)+'.txt','w')
            for j in range(n_steps):
                s = ','.join([partitions + '_' + str(i), str(j * 0.04), str(preds[j, i, 0])])
                s += '\n'
                single_file.write(str(preds[j, i, 0])+'\n')
                f.write(s)
            single_file.close()



    return RMSE, CC, CCC
    # -------------------------------------

def predict_Stage2(f_pred, prepare_data, data, feature, emodim='arousal', partitions='train',generating=True):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    print('Evaluate on data:', emodim, partitions)
    n_samples = len(data)
    n_steps = len(data[0][0])
    # kf = get_minibatches_idx(len(data), 5)
    x = prepare_data([data[t][0] for t in range(n_samples)])
    y = prepare_data([data[t][1] for t in range(n_samples)])

    preds = f_pred(x)

    # Compute performance
    preds_flatten = preds.flatten('F')
    y_flatten = y.flatten('F')
    RMSE, CC, CCC = rater_statistics(y_flatten, preds_flatten)
    print(RMSE, CC, CCC)

    if generating:
        fname = 'predictions/Stage2/' + emodim + '_' + partitions + '.arff'
        # preds : (n_steps * n_sample)
        with open(fname, 'w') as f:
            for i in range(n_samples):
                single_file = open('predictions/Stage2/' + emodim + '/' + partitions + '/' + partitions + '_' + str(i)+'.txt','w')
                for j in range(n_steps):
                    s = ','.join([partitions + '_' + str(i), str(j * 0.04), str(preds[j, i, 0])])
                    s += '\n'
                    single_file.write(str(preds[j, i, 0])+'\n')
                    f.write(s)
                single_file.close()

    return RMSE, CC, CCC
    # -------------------------------------

def filter(data, k=0.03):
    for i in range(9):
        for j in range(1, 7501):
            data[i * 7501 + j] = (1 - k) * data[i * 7501 + j - 1] + k * data[i * 7501 + j]


def save_result(x, y, file_name):
    n_samples = 9
    n_steps = 7501
    data = []
    with open(file_name+'Second.pkl', 'wb') as f:
        for i in range(n_samples):
            data_x = [];
            data_y = []
            for j in range(n_steps):
                data_x.append(x[j, i, :].tolist())
                data_y.append(y[j, i, :].tolist())
            data.append((data_x, data_y))

        pickle.dump(data, f)


def rater_statistics(r1, r2):
    # MSE
    RMSE = numpy.sqrt(numpy.mean((r1 - r2) ** 2))

    # CC
    r1_mean = numpy.mean(r1)
    r2_mean = numpy.mean(r2)
    r1_std = numpy.std(r1)
    r2_std = numpy.std(r2)
    mean_centprod = numpy.mean((r1 - r1_mean) * (r2 - r2_mean))
    CC = mean_centprod / (r1_std * r2_std)

    # CCC
    r1_var = r1_std ** 2
    r2_var = r2_std ** 2
    CCC = (2.0 * mean_centprod) / (r1_var + r2_var + (r1_mean - r2_mean) ** 2)
    return RMSE, CC, CCC


def visualize(data, labels, fname):
    x = []
    for i in range(len(data[0])):
        x.append([n[i] for n in data])

    plt.title("Model Errors")
    plt.xlabel("Iterations")
    plt.ylabel("Errors")
    plt.axis([0, max(x[0]) + 10, 0, 1])
    for d, n in zip(x[1:], labels):
        plt.plot(x[0], d, label=n, linewidth=4)
        plt.legend()
    plt.savefig(fname)


def temporal_pooling(data, window_size=1., with_label=True):
    """
    Temporal pooling can get the statics of the successive frames.
    Mean pooling can get the context information

    :param data:
    :param window_size: seconds
    :return:
    """
    new_data = []
    n_samples = len(data)
    n_steps = len(data[0][0])
    n_pooling_step = int(window_size / 0.004)

    for d in data:
        d_x = d[0];
        d_y = d[1]
        new_x = [];
        new_y = []
        for i in range(n_steps - n_pooling_step):
            new_x.append(numpy.mean(d_x[i:i + n_pooling_step], axis=0).tolist())
            if with_label:
                new_y.append(numpy.mean(d_y[i:i + n_pooling_step], axis=0).tolist())
            else:
                new_y.append(d_y[i])
        for i in range(n_steps - n_pooling_step, n_steps):
            new_x.append(numpy.mean(d_x[i - n_pooling_step:i], axis=0).tolist())
            if with_label:
                new_y.append(numpy.mean(d_y[i - n_pooling_step:i], axis=0).tolist())
            else:
                new_y.append(d_y[i])
        new_data.append((new_x, new_y))

    return new_data


def smooth_label(data, window_size=1., method='mean'):
    new_data = numpy.zeros(data.shape)
    n_step = int(window_size / 0.004)
    for i in range(9):
        for j in range(7501 - n_step):
            if method == 'mean':
                data[i * 7501 + j] = numpy.mean(data[i * 7501 + j:i * 7501 + j + n_step])
            elif method == 'max':
                data[i * 7501 + j] = numpy.mean(data[i * 7501 + j:i * 7501 + j + n_step])


def add_noise(data):
    noisy_data = numpy.array(data)
    noisy_data = noisy_data + numpy.random.normal(0, 0.1, noisy_data.shape)
    return noisy_data.tolist()
