# -*- coding: utf-8 -*-
"""
@author: raoqiyu
Created on Tue Oct 13 19:00:49 2015


Long-Short Term Memory

In a traditional recurrent neural network, during the gradient signal can end
up being multiplied a large number of times (as the number of time steps) by the
weight matrix associated  with  the connections between the neurons of the
recurrent  hidden layer. This means that,  the  magnitude of weights in the
transition can have a strong impact on the learning process.

Vanishing Gradients ----- the gradient signal gets so small that learning either
becomes very slow or stops working altogether. It can also make more difficult
the task of learning long-term dependencies in the data.

Exploding Gradients ----- the weights in the matrix are large, it can lead to a
situation where the gradient signal is so large that it can cause learning to
diverge.

Memory cell:
    - A neuron with a self-recurrent connection: a connection to itself, has
        a weight of 1.0 (can remain constant from on time step to another.)

    - Input  Gate: allow incoming signal to alter the state of the memory cell
        or block it
    - Output Gate: allow the state of the memory cell to have an effect on other
        cell or prevent it.
    - Forget Gate: modulate the memory cell's self-recurrent connection,
        allowing the cell to remember or forget its previous state, as needed.

Equations:
x(t) is the input to the memory cell at time t
i(t) is the Input Gate status at time t
c(t) is the memory cell status at time t
o(t) is the Output Gate status at time t
f(t) is the Forget Gate statue at time t
h(t) is the memory cell's output at time t

Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo and Vo are weight matrices
bi, bf, bc and bo are bias vectors

    i(t)  =  sigmoid( Wi * x(t) + Ui * h(t-1) + bi)
    f(t)  =  sigmoid( Wf * x(t) + Uf * h(t-1) + bf)
    o(t) = sigmoid( Wo * x(t) + Uo * h(t-1) + bo)
    c'(t) =  sigmoid( Wc * x(t) + Uc * h(t-1) + bc)
    c(t)  =  i(t) * c'(t) + f(t) * c(t-1)
    #o(t)  =  sigmoid( Wo * x(t) + Uo * h(t-1) + Vo * c(t) + bo)
    h(t)  =  o(t) * tanh(C(t))

Implementation Note: (Not implemented yet)
We can concatenate the four matrices W* into a single weight matrix W
( W = [Wi, Wf, Wo, Wc]) and performing the same concatenation on the weight
matrices U* to produce the matrix U ( U = [Ui, Uf, Uo, Uc]) and the bias b*
to produce the vector b ( b = [bi, bf, bo, bc]). Then
    z = sigmoid( W * x(t) + U * h(t-1) + b).
The result is then sliced to obtain the pre-nonlinear activations for i(t),
f(t), c(t), and o(t).

"""

from math import ceil
import time

import theano.tensor as T
import numpy as np

from utils import *


class LSTM(object):
    def __init__(self):
        """
        :type   patience:    int
        :param  patience:    Number of epoch to wait before early stop if no progress

        :type   learning_rate:
        :param  learning_rate:  Learning rate for gradient update

        :type   saveto:  string
        :param  saveto:  The model will be saved there

        :type   saveFreq:   int
        :param  saveFreq:   Save the parameters after every saveFreq updates

        :type   validFreq:  int
        :param  validFreq:  Compute the validation error after this number of updates

        :type   batch_size: int
        :param  batch_size: The batch size during training

        :type   epochs:     int
        :param  epochs:     The maximum number of epoch to run

        :type   dispFreq:    int
        :param  dispFreq:    Display the training progress every N updates

        :type   reload_model:    string
        :param  reload_model:    Path to a saved model we want to start from
        """
        self.n_input = 0;
        self._input = None
        self.n_hiddens = [];
        self.n_output = 0;
        self._output = None
        self.n_layer = 0
        self.layers = []
        self.lrate = T.scalar('learning_rate')

        self.options = {
            "epochs":500,
            "batch_size":16,
            "learning_rate":1e-3,
            "patience":10,
            "L1_penalty":None,
            "L2_penalty":1e-5,
            "shuffle":True,
            "saveto":'model/best/model',
            "saveFreq":1110,
            "dispFreq":50,
            "reload_model":None,
            "validFreq":-1,
            "valid_batch_size":64,
            "optimizer":None
        }

        self.noise = theano.shared(numpy_floatX(0.))

        # input:  x [nsteps, ni], row 0 corresponds to the time step t=0
        self.x = T.tensor3('x', dtype=theano.config.floatX)
        self.y = T.tensor3('y', dtype=theano.config.floatX)
        self.x2 = T.tensor3('x2', dtype=theano.config.floatX)
        # self.x = T.matrix('x', dtype='int64')
        self.mask = T.matrix('mask', dtype=theano.config.floatX)
        # self.y = T.vector('y', dtype='int64')

        self._input = self.x
        self._output = self.y
        self.params = []

        # -----------------------------Building Model----------------------------
        print('\n', '-' * 20, 'Building Model', '-' * 20)

    def add(self, layer):
        """ Build a deep neural network, add one layer each time
                :type  layer:  lstm_layer
                :param layer:  a hidden lstm layer to be added to this model
                """
        # Update model's I/O
        if layer._name == "LSTM_MASK":
            if self.n_layer == 0:
                Wemb = (0.01 * np.random.rand(10000, layer.n_output)).astype(theano.config.floatX)
                self.Wemb = theano.shared(value=Wemb, name='Wemb', borrow=True)
                x_wemb = self.Wemb[self.x.flatten()].reshape((self.x.shape[0],
                                                              self.x.shape[1], layer.n_output))
                self.params.append(self.Wemb)
                layer.perform(x_wemb, self.mask)
                self.n_input = layer.n_input
            else:
                assert (self.n_hiddens[-1] == layer.n_input)
                layer.perform(self._output, self.mask)
            self.n_hiddens.append(layer.n_output)
            self._output = layer.output
            # Add this layer to the network
            self.layers.append(layer)
            self.n_layer += 1
            self.params += layer.params
            print("Add a LSTM layer")
        elif layer._name == 'BLSTM_MASK':
            if self.n_layer == 0:
                Wemb = (0.01 * np.random.rand(10000, layer.n_output)).astype(theano.config.floatX)
                self.Wemb = theano.shared(value=Wemb, name='Wemb', borrow=True)
                x_wemb = self.Wemb[self.x.flatten()].reshape((self.x.shape[0],
                                                              self.x.shape[1], layer.n_output))
                self.params.append(self.Wemb)
                layer.perform([x_wemb, x_wemb], self.mask)
                self.n_input = layer.n_input
            else:
                assert (self.n_hiddens[-1] == layer.n_input)
                layer.perform(self._output, self.mask)
            self.n_hiddens.append(layer.n_output)
            self._output = layer.output
            # Add this layer to the network
            self.layers.append(layer)
            self.n_layer += 1
            self.params += layer.params
            print("Add a BLSTM layer")
        elif layer._name == "LSTM":
            if self.n_layer == 0:
                layer.perform(self.x)
                self.n_input = layer.n_input
            else:
                assert (self.n_hiddens[-1] == layer.n_input)
                layer.perform(self._output)
            self.n_hiddens.append(layer.n_output)
            self._output = layer.output
            # Add this layer to the network
            self.layers.append(layer)
            self.n_layer += 1
            self.params += layer.params
            print("Add a LSTM layer")
        elif layer._name == 'BLSTM':
            if self.n_layer == 0:
                layer.perform([self._input, self._input])
                self.n_input = layer.n_input
            elif self.layers[-1]._name != 'BLSTM':
                layer.perform([self._output, self._output])
            else:
                assert (self.n_hiddens[-1] == layer.n_input)
                layer.perform(self._output)
            self.n_hiddens.append(layer.n_output)
            self._output = layer.output
            # Add this layer to the network
            self.layers.append(layer)
            self.n_layer += 1
            self.params += layer.params
            print("Add a BLSTM layer")
        elif layer._name == 'hidden layer':
            if self.n_layer == 0:
                layer.perform(self._input)
                self.n_input = layer.n_input
            else:
                assert (self.n_hiddens[-1] == layer.n_input)
                layer.perform(self._output)
            self.n_hiddens.append(layer.n_output)
            self._output = layer.output
            # Add this layer to the network
            self.layers.append(layer)
            self.n_layer += 1
            self.params += layer.params
            print("Add a hidden layer")
        elif layer._name == "DropOut":
            if self.layers[-1]._name == "BLSTM":
                _output = []
                layer.perform(self._output[0], self.noise)
                _output.append(layer.output)
                layer.perform(self._output[1], self.noise)
                _output.append(layer.output)
                self._output = _output
            else:
                layer.perform(self._output, self.noise)
                self._output = layer.output
            print("Add a Dropout layer")
        elif layer._name == "imdb_activate":
            assert (self.n_hiddens[-1] == layer.n_input)
            layer.perform(self._output, self.mask, self.noise)
            self._output = layer.output

            # Add this layer to the network
            self.n_output = layer.n_output
            self.layers.append(layer)
            self.n_hiddens.append(layer.n_output)
            self.n_layer += 1
            self.params += layer.params
            print("Add a imdb activation layer")
        elif layer._name == "avec_activate":
            assert (self.n_hiddens[-1] == layer.n_input)
            layer.perform(self._output, self.noise)
            self._output = layer.output

            # Add this layer to the network
            self.n_output = layer.n_output
            self.layers.append(layer)
            self.n_hiddens.append(layer.n_output)
            self.n_layer += 1
            self.params += layer.params
            print("Add a avec activation layer")
        elif layer._name == "linear_fusion":
            assert (self.n_layers == 0)
            self._input = [self.x, self.x2]
            layer.perform(self._input)

            # Add this layer to the network
            self.n_input = layer.n_input
            self.n_hiddens.append(layer.n_output)
            self._output = layer.output
            self.layers.append(layer)
            self.n_layer += 1
            self.params += layer.params
            self.n_output = layer.n_output
            print("Add a avec activation layer. This layer must be the first layer in this code")

    def setup(self, options):
        for k in options:
            assert (k in self.options)
            self.options[k] = options[k]

    def compile(self, options):
        self.setup(options)
        print(self)
        # self.y_pred_prob = theano.function([self._input, self.mask], self._output,
        #                                    name='y_pred_prob')
        # self.y_pred = theano.function([self._input, self.mask],
        #                               self._output.argmax(axis=1), name='y_pred')
        self.predict = theano.function([self._input], self._output, name='predict')
        off = 1e-4
        if self._output.dtype == 'float16':
            off = 1e-2
        # train_cost = - T.log(self._output[T.arange(self.x.shape[1]), self.y] + off).mean()
        # print("loss : square")
        # train_cost =  T.sum((self._output - self.y)**2)
        print("loss : abs")
        # pred_mean = T.mean(self._output)
        train_cost = T.sum(abs(self._output - self.y))
        # train_cost =  T.sum((self._output - self.y)**2)*2/(self._output.shape[0]*self._output.shape[1])
        # y = T.reshape(self.y,[self.y.shape[1], self.y.shape[0]])
        # y_pred = T.reshape(self._output,[self.y.shape[1], self.y.shape[0]])
        #
        # def _cost(y,y_p):
        #     cost  = T.mean( (y - y_p)*(y_p - T.mean(y_p)) )
        #     return cost
        #
        # train_cost, _ = theano.scan(_cost,sequences=[y, y_pred])
        # train_cost = -T.mean(train_cost)

        if self.options["L1_penalty"] is not None:
            L1_reg = 0.
            for p in self.params:
                L1_reg += abs(p).sum()
            L1_reg *= elf.options["L1_penalty"]
            train_cost += L1_reg

        if self.options["L2_penalty"] is not None:
            print("L2 Penalty: last layers")
            L2_reg = 0.
            for p in self.layers[-1].params:
                L2_reg += (p ** 2).sum()
            L2_reg *= self.options["L2_penalty"]
            train_cost += L2_reg
        assert (self.options["optimizer"] is not None)
        self.train, self.update = self.options["optimizer"].compile(params=self.params,
                                                                    x=self.x,
                                                                    y=self.y,
                                                                    cost=train_cost,
                                                                    )

    def save(self, file):
        print("Save model")
        with open(file, 'wb') as f:
            pickle.dump(self.n_layer, f)
            # Save input layer
            for p in self.params:
                pickle.dump(p.get_value(), f)
            # Save hidden layer and output layer
            for l in self.layers:
                for p in l.params:
                    pickle.dump(p.get_value(), f)

    def load(self, file):
        print("Load model, the model loaded must have the same network struct with this")
        with open(file, 'rb') as f:
            n_layer = pickle.load(f)
            assert self.n_layer == n_layer
            # load input layer
            for p in self.params:
                p.set_value(pickle.load(f))
            # Save hidden layer and output layer
            for l in self.layers:
                for p in l.params:
                    p.set_value(pickle.load(f))

    def __str__(self):
        model_config = ''
        # -----------------------------Model Configuration------------------------
        title1 = '\n' + '-' * 20 + 'Model Configuration' + '-' * 20
        nlayers = str(self.n_layer + 1) + " layers"
        layers = "Input  Layer : " + str(self.n_input)
        for idx, l in enumerate(self.layers[:-1]):
            layers += "\nHidden Layer : " + str(self.n_hiddens[idx]) + " " + str(l._name)
        layers += "\nOutput Layer : " + str(self.n_hiddens[-1]) + " " + self.layers[-1]._name
        # -----------------------------Model Options----------------------------
        title2 = '\n' + '-' * 20 + 'Model Paramter' + '-' * 20
        model_config = '\n'.join([title1, nlayers, layers, title2, str(self.options)])

        return model_config

    def fit(self, trainData, validData, testData):
        # -----------------------------Training----------------------------------
        print('\n', '-' * 20, 'Trainig', '-' * 20)
        history_rmse = []
        history_ccc = []
        bad_count = 0

        self.noise.set_value(0.)
        train_rmse, train_ccc = evaluate(self.predict, parallelize_data,
                                         trainData)
        valid_rmse, valid_ccc = evaluate(self.predict, parallelize_data,
                                         validData)
        test_rmse, test_ccc = evaluate(self.predict, parallelize_data,
                                       testData)
        print("Before Training:")
        print("Train Data:", train_rmse, train_ccc)
        print("valid Data:", valid_rmse, valid_ccc)
        print("Test  Data:", test_rmse, test_ccc)

        if self.options["validFreq"] == -1:
            self.options["validFreq"] = ceil(len(trainData) / self.options["batch_size"])
        # if self.saveFreq == -1:
        #            saveFreq = len(self.trainData[0]) / batch_size

        n_iter = 0  # iteration number.
        early_stop = False  # if early stop or not
        start_time = time.clock()
        best_test_rmse = np.inf
        best_test_ccc = 0
        for e in range(self.options["epochs"]):
            n_samples = 0
            print('Epoch', e)
            if np.mod(e, 20) == 0: print(end='', flush=True)
            kf = get_minibatches_idx(len(trainData), self.options["batch_size"],
                                     shuffle=self.options["shuffle"])

            for _, train_index in kf:
                self.noise.set_value(1.)
                n_iter += 1

                # Select the random examples for this minibatch
                x = [trainData[t][0] for t in train_index]
                y = [trainData[t][1] for t in train_index]

                # Get the data in numpy.ndarray format
                # Do parallel computing
                # return training data of shape (n_steps, n_samples, n_feature_size)
                x = parallelize_data(x)
                y = parallelize_data(y)
                n_samples += x.shape[1]

                cost = self.train(x, y)
                self.update()
                # print(self.lstm.Wh.get_value())

                # Check whether there is error(NaN)
                if np.isnan(cost) or np.isinf(cost):
                    print('NaN detected.')
                    return 1., 1., 1.

                # Check whether display training progress or not
                if np.mod(n_iter, self.options["dispFreq"]) == 0:
                    print('     Update', n_iter, 'Cost', cost)

                # Check whether save to path or not (not impletemented yet)

                # Check wether needed to do validation
                if np.mod(n_iter, self.options["validFreq"]) == 0:
                    self.noise.set_value(0.)
                    train_rmse, train_ccc = evaluate(self.predict, parallelize_data,
                                                     trainData)
                    valid_rmse, valid_ccc = evaluate(self.predict, parallelize_data,
                                                     validData)
                    test_rmse, test_ccc = evaluate(self.predict, parallelize_data,
                                                   testData)
                    print("\nTrain Data:", train_rmse, train_ccc)
                    print("valid Data:", valid_rmse, valid_ccc)
                    print("Test  Data:", test_rmse, test_ccc, "\n")
                    history_rmse.append([n_iter, train_rmse, valid_rmse, test_rmse])
                    history_ccc.append([n_iter, train_ccc, valid_ccc, test_ccc])
                    # Check if this param is the best param
                    if test_rmse <= best_test_rmse:
                        self.save(self.options["saveto"] + "_rmse.pkl")
                        print('Saving rmse model')
                        best_test_rmse = test_rmse
                        # bad_count = 0

                    if test_ccc >= best_test_ccc:
                        self.save(self.options["saveto"] + "_ccc.pkl")
                        print('Saving ccc model')
                        best_test_ccc = test_ccc
                        bad_count = 0

                    # Early Stop
                    if (len(history_rmse) > self.options["patience"] and
                                test_rmse >= np.array(history_rmse)[:-self.options["patience"],
                                             -1].min()):
                        bad_count += 1
                        if bad_count > self.options["patience"]:
                            print('Early Stop!')
                            early_stop = True
                            break

            if early_stop:
                break

        end_time = time.clock()
        self.noise.set_value(0.)

        with open(self.options["saveto"] + "_err_rmse.pkl", 'wb') as f:
            pickle.dump(history_rmse, f)
        with open(self.options["saveto"] + "_err_ccc.pkl", 'wb') as f:
            pickle.dump(history_ccc, f)
        # visualize(history_errors, ["train error", "valid error", "test error"], self.options["saveto"]+"_errors.eps")
        test_rmse = np.array(history_rmse)[:, -1].min()
        test_ccc = np.array(history_ccc)[:, -1].max()
        print("\nThe Best test rmse:", test_rmse)
        print("The Best test ccc:", test_ccc)

        print('Test with best Param')
        self.load(self.options["saveto"] + "_rmse.pkl")
        train_rmse, train_ccc = evaluate(self.predict, parallelize_data,
                                         trainData)
        valid_rmse, valid_ccc = evaluate(self.predict, parallelize_data,
                                         validData)
        test_rmse, test_ccc = evaluate(self.predict, parallelize_data,
                                       testData)
        print("Train Data:", train_rmse, train_ccc)
        print("valid Data:", valid_rmse, valid_ccc)
        print("Test  Data:", test_rmse, test_ccc, "\n\n")

        print('The src run for %d epochs, with %f sec/epochs' %
              (e + 1, (end_time - start_time) / (1. * (e + 1))))
        print('Training took %0.fs' % (end_time - start_time))

        return [(train_rmse, train_ccc), (valid_rmse, valid_ccc), (test_rmse, test_ccc)]

    def pretrainning(self, trainData):
        print('\n', '-' * 20, 'Pretrainig', '-' * 20)
        history_rmse = []
        history_ccc = []
        bad_count = 0
        n_samples = len(trainData)
        pratrainingData = []
        for i in range(n_samples):
            pratrainingData.append((trainData[i][1], trainData[i][1]))

        if self.options["validFreq"] == -1:
            self.options["validFreq"] = ceil(len(trainData) / self.options["batch_size"])

        early_stop = False  # if early stop or not
        start_time = time.clock()
        best_train_rmse = np.inf
        best_train_ccc = 0
        for e in range(self.options["epochs"]):
            n_samples = 0
            print('Epoch', e)

            kf = get_minibatches_idx(len(trainData), self.options["batch_size"],
                                     shuffle=self.options["shuffle"])

            for _, train_index in kf:
                self.noise.set_value(1.)

                # Select the random examples for this minibatch
                x = [pratrainingData[t][0] for t in train_index]

                # Get the data in numpy.ndarray format
                # Do parallel computing
                # return training data of shape (n_steps, n_samples, n_feature_size)
                x = parallelize_data(x)
                n_samples += x.shape[1]

                cost = self.train(x, x)
                self.update()
                # print(self.lstm.Wh.get_value())

                # Check whether there is error(NaN)
                if np.isnan(cost) or np.isinf(cost):
                    print('NaN detected.')
                    return 1., 1., 1.

                    # Check whether save to path or not (not impletemented yet)

            # Check wether needed to do validation
            self.noise.set_value(0.)
            train_rmse, train_ccc = evaluate(self.predict, parallelize_data,
                                             pratrainingData)

            print("\nTrain Data:", train_rmse, train_ccc)

            history_rmse.append(train_rmse)
            history_ccc.append(train_ccc)
            # Check if this param is the best param
            if train_rmse <= best_train_rmse:
                self.save(self.options["saveto"] + "pretraining__rmse.pkl")
                print('Saving rmse model')
                best_train_rmse = train_rmse
                bad_count = 0

            if train_ccc >= best_train_ccc:
                self.save(self.options["saveto"] + "pretraining__ccc.pkl")
                print('Saving ccc model')
                best_train_ccc = train_ccc
                # bad_count = 0

            # Early Stop
            if (len(history_rmse) > self.options["patience"] and
                        train_rmse >= np.array(history_rmse)[:-self.options["patience"]].min()):
                bad_count += 1
                if bad_count > self.options["patience"]:
                    print('Early Stop!')
                    early_stop = True
                    break

            if early_stop:
                break

        end_time = time.clock()
        self.noise.set_value(0.)

        with open(self.options["saveto"] + "pretraining__err_rmse.pkl", 'wb') as f:
            pickle.dump(history_rmse, f)
        with open(self.options["saveto"] + "pretraining__err_ccc.pkl", 'wb') as f:
            pickle.dump(history_ccc, f)
        # visualize(history_errors, ["train error", "valid error", "test error"], self.options["saveto"]+"_errors.eps")
        test_rmse = np.array(history_rmse).min()
        test_ccc = np.array(history_ccc).max()
        print("\nThe Best test rmse:", test_rmse)
        print("The Best test ccc:", test_ccc)

        print('Test with best Param')
        self.load(self.options["saveto"] + "pretraining__rmse.pkl")
        train_rmse, train_ccc = evaluate(self.predict, parallelize_data,
                                         pratrainingData)
        print("Train Data:", train_rmse, train_ccc)

        print('The src run for %d epochs, with %f sec/epochs' %
              (e + 1, (end_time - start_time) / (1. * (e + 1))))
        print('Training took %0.fs' % (end_time - start_time))

        return (train_rmse, train_ccc)
