# -*- coding: utf-8 -*-
"""


@author: raqoiyu
Created on Mon Nov 23 09:52:54 2015
"""
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

from utils import numpy_floatX, ortho_weight


class hidden_layer(object):
    """Output layer for AVEC dataset
    """

    def __init__(self, n_input, n_output):
        self._name = "hidden layer"
        self.n_input = n_input
        self.n_output = n_output
        self.input = None
        self.output = None
        w = 0.01 * np.random.randn(n_input, n_output).astype(theano.config.floatX)
        self.w = theano.shared(value=w, name='w', borrow=True)
        # b : model's output weights
        b = np.zeros((n_output,)).astype(theano.config.floatX)
        self.b = theano.shared(value=b, name='b', borrow=True)
        self.params = [self.w, self.b]

    def perform(self, x):
        n_steps = x.shape[0]
        self.input = x

        def _step(x_t):
            o = T.nnet.sigmoid(T.dot(x_t, self.w) + self.b)
            return o

        # h0 and c0 are initialized randomly
        o, _ = theano.scan(_step, sequences=[x],
                           outputs_info=None,
                           name='avec_output', n_steps=n_steps)
        self.output = o


class DropOut(object):
    """ A dropout layer
    """

    def __init__(self):
        self._name = "DropOut"

    def perform(self, x, noise):
        self.input = x
        rng = RandomStreams(123)
        y = T.switch(noise, (x * rng.binomial(x.shape, p=0.5, n=1, dtype=x.dtype))
                     , x * 0.5)
        self.output = y


class imdb_activate(object):
    def __init__(self, n_input, n_output):
        self._name = "imdb_activate"
        self.n_input = n_input
        self.n_output = n_output

        # w : model's output weights
        w = 0.01 * np.random.randn(n_input, n_output).astype(theano.config.floatX)
        self.w = theano.shared(value=w, name='w', borrow=True)
        # b : model's output weights
        b = np.zeros((n_output,)).astype(theano.config.floatX)
        self.b = theano.shared(value=b, name='b', borrow=True)
        self.params = [self.w, self.b]

    def perform(self, x, mask, noise):
        # Mean Pooling
        x = (x * mask[:, :, None]).sum(axis=0)
        x = x / mask.sum(axis=0)[:, None]
        # Activation
        self.output = T.nnet.softmax(T.dot(x, self.w) + self.b)


class avec_activate(object):
    """Output layer for AVEC dataset
    """

    def __init__(self, n_input, n_output):
        self._name = "avec_activate"
        self.n_input = n_input
        self.n_output = n_output
        self.input = None
        self.output = None
        w = 0.01 * np.random.randn(n_input, n_output).astype(theano.config.floatX)
        self.w = theano.shared(value=w, name='w', borrow=True)
        # b : model's output weights
        b = np.zeros((n_output,)).astype(theano.config.floatX)
        self.b = theano.shared(value=b, name='b', borrow=True)
        self.params = [self.w, self.b]

    def perform(self, x, noise):
        n_steps = x.shape[0]
        self.input = x

        def _step(x_t):
            o = T.dot(x_t, self.w) + self.b
            return o

        # h0 and c0 are initialized randomly
        o, _ = theano.scan(_step, sequences=[x],
                           outputs_info=None,
                           name='avec_output', n_steps=n_steps)
        self.output = o


class bi_avec_activate(object):
    """Output layer for AVEC dataset
    """

    def __init__(self, n_input, n_output):
        self._name = "avec_activate"
        self.n_input = n_input
        self.n_output = n_output
        self.input = None
        self.output = None

        # w : model's output weights
        w1 = 0.01 * np.random.randn(n_input, n_output).astype(theano.config.floatX)
        self.w1 = theano.shared(value=w1, name='w1', borrow=True)
        w2 = 0.01 * np.random.randn(n_input, n_output).astype(theano.config.floatX)
        self.w2 = theano.shared(value=w2, name='w2', borrow=True)
        # b : model's output weights
        b = np.zeros((n_output,)).astype(theano.config.floatX)
        self.b = theano.shared(value=b, name='b', borrow=True)
        self.params = [self.w1, self.w2, self.b]

    def perform(self, x, noise):
        x1, x2 = x[0], x[1]
        n_steps = x1.shape[0]
        self.input = x

        def _step(x1_t, x2_t):
            o = T.dot(x1_t, self.w1) + T.dot(x2_t, self.w2) + self.b
            return o

        # h0 and c0 are initialized randomly
        o, _ = theano.scan(_step, sequences=[x1, x2],
                           outputs_info=None,
                           name='avec_output', n_steps=n_steps)
        self.output = o


class bi_imdb_activate(object):
    def __init__(self, n_input, n_output):
        self._name = "imdb_activate"
        self.n_input = n_input
        self.n_output = n_output

        # w : model's output weights
        w1 = 0.01 * np.random.randn(n_input, n_output).astype(theano.config.floatX)
        self.w1 = theano.shared(value=w1, name='w1', borrow=True)
        w2 = 0.01 * np.random.randn(n_input, n_output).astype(theano.config.floatX)
        self.w2 = theano.shared(value=w2, name='w2', borrow=True)
        # b : model's output weights
        b = np.zeros((n_output,)).astype(theano.config.floatX)
        self.b = theano.shared(value=b, name='b', borrow=True)
        self.params = [self.w1, self.w2, self.b]

    def perform(self, x, mask, noise):
        x1, x2 = x[0], x[1]
        # Mean Pooling
        x1 = (x1 * mask[:, :, None]).sum(axis=0)
        x1 = x1 / mask.sum(axis=0)[:, None]
        x2 = (x2 * mask[:, :, None]).sum(axis=0)
        x2 = x2 / mask.sum(axis=0)[:, None]
        # Activation
        self.output = T.nnet.softmax(T.dot(x1, self.w1) + T.dot(x2, self.w2) + self.b)


class lstm_layer(object):
    """A Long-Short Term Memory layer"""

    def __init__(self, n_input, n_output):
        """

        :type  nc: int
        :param nc: dimension of input vector

        :type  nh: int
        :param nh: number of hidden units in this layer

        :type  no: int
        :param no: dimension of output vector
        """

        # Parameter of this lstm layer
        self._name = "LSTM"
        self.n_input = n_input
        self.n_output = n_output
        # Wh = [ Wi, Wc, Wf, Wo]
        Wh = np.concatenate([np.random.randn(n_input, n_output).astype(theano.config.floatX),
                             np.random.randn(n_input, n_output).astype(theano.config.floatX),
                             np.random.randn(n_input, n_output).astype(theano.config.floatX),
                             np.random.randn(n_input, n_output).astype(theano.config.floatX)]
                            , axis=1)
        # Wh = np.concatenate([ortho_weight(n_input, n_output), ortho_weight(n_input, n_output),
        #                      ortho_weight(n_input, n_output), ortho_weight(n_input, n_output)]
        #                     , axis=1)
        self.Wh = theano.shared(value=Wh, name='Wh', borrow=True)
        # U = [Ui, Uc, Uf, Uo]
        Uh = np.concatenate([ortho_weight(n_output, n_output), ortho_weight(n_output, n_output),
                             ortho_weight(n_output, n_output), ortho_weight(n_output, n_output)]
                            , axis=1)
        self.Uh = theano.shared(value=Uh, name='Uh', borrow=True)

        # bh = [bi, bc, bf, bo]
        bh = np.zeros((n_output * 4,)).astype(theano.config.floatX)
        self.bh = theano.shared(value=bh, name='bh', borrow=True)

        self.params = [self.Wh, self.Uh, self.bh]

    def perform(self, x):
        nsteps = x.shape[0]
        # if x.ndim == 3:
        #     n_samples = x.shape[1]
        # else:
        #     n_samples = 1
        #
        n_samples = x.shape[1]

        def _slice(x_t, idx, ndim):
            if x_t.ndim == 3:
                return x_t[:, :, idx * ndim: (idx + 1) * ndim]
            return x_t[:, idx * ndim:(idx + 1) * ndim]

        def _step(x_t, h_tm1, c_tm1):
            # z = sigmoid( W * x(t) + U * h(t-1) + b)
            # zi =  W * x(t) + U * h(t-1) + b
            zi = T.dot(x_t, self.Wh) + T.dot(h_tm1, self.Uh) + self.bh
            # zi = T.dot(h_tm1, self.Uh)
            # zi += x_t
            # W = [Wi, Wf, Wo, Wc], U = [Ui, Uf, Uo, Uc],  b = [bi, bf, bo, bc]
            i = T.nnet.sigmoid(_slice(zi, 0, self.n_output))
            f = T.nnet.sigmoid(_slice(zi, 1, self.n_output))
            o = T.nnet.sigmoid(_slice(zi, 2, self.n_output))
            c = T.tanh(_slice(zi, 3, self.n_output))

            c = f * c_tm1 + i * c;

            h = o * T.tanh(c)
            # output at each time
            # s = softmax(w * h_t + b)
            return [h, c]

        # h0 and c0 are initialized randomly
        h0 = T.alloc(numpy(0.), n_samples, self.n_output);
        c0 = T.alloc(numpy(0.), n_samples, self.n_output)
        h0 = theano.tensor.unbroadcast(h0, 1);
        c0 = theano.tensor.unbroadcast(c0, 1)
        [h, c], _ = theano.scan(fn=_step, sequences=x,
                                outputs_info=[h0, c0],
                                n_steps=nsteps)
        self.input = x
        self.output = h


class blstm_layer(object):
    """A Bidirectional Long-Short Term Memory layer
    """

    def __init__(self, n_input, n_output):
        """

        :type  nc: int
        :param nc: dimension of input vector

        :type  nh: int
        :param nh: number of hidden units in this layer

        :type  no: int
        :param no: dimension of output vector
        """

        # Parameter of this lstm layer
        self._name = "BLSTM"
        self.n_input = n_input
        self.n_output = n_output
        # Wh = [ Wi, Wc, Wf, Wo]
        Wh = np.concatenate([np.random.randn(n_input, n_output).astype(theano.config.floatX),
                             np.random.randn(n_input, n_output).astype(theano.config.floatX),
                             np.random.randn(n_input, n_output).astype(theano.config.floatX),
                             np.random.randn(n_input, n_output).astype(theano.config.floatX)]
                            , axis=1)
        self.Wh = theano.shared(value=Wh, name='Wh', borrow=True)
        # U = [Ui, Uc, Uf, Uo]
        Uh = np.concatenate([ortho_weight(n_output, n_output), ortho_weight(n_output, n_output),
                             ortho_weight(n_output, n_output), ortho_weight(n_output, n_output)]
                            , axis=1)
        self.Uh = theano.shared(value=Uh, name='Uh', borrow=True)

        # bh = [bi, bc, bf, bo]
        bh = np.zeros((n_output * 4,)).astype(theano.config.floatX)
        self.bh = theano.shared(value=bh, name='bh', borrow=True)

        Wh_reverse = np.concatenate([np.random.randn(n_input, n_output).astype(theano.config.floatX),
                                     np.random.randn(n_input, n_output).astype(theano.config.floatX),
                                     np.random.randn(n_input, n_output).astype(theano.config.floatX),
                                     np.random.randn(n_input, n_output).astype(theano.config.floatX)]
                                    , axis=1)
        self.Wh_reverse = theano.shared(value=Wh_reverse, name='Wh_reverse', borrow=True)
        # U = [Ui, Uc, Uf, Uo]
        Uh_reverse = np.concatenate([ortho_weight(n_output, n_output), ortho_weight(n_output, n_output),
                                     ortho_weight(n_output, n_output), ortho_weight(n_output, n_output)]
                                    , axis=1)
        self.Uh_reverse = theano.shared(value=Uh_reverse, name='Uh_reverse', borrow=True)

        # bh = [bi, bc, bf, bo]
        bh_reverse = np.zeros((n_output * 4,)).astype(theano.config.floatX)
        self.bh_reverse = theano.shared(value=bh_reverse, name='bh_reverse', borrow=True)

        self._output = np.zeros(2, )
        self.params = [self.Wh, self.Uh, self.bh, self.Wh_reverse, self.Uh_reverse, self.bh_reverse]

    def perform(self, x):
        x1, x2 = x[0], x[1]
        nsteps = x1.shape[0]
        n_samples = x1.shape[1]
        # if x1.ndim == 3:
        #     n_samples = x1.shape[1]
        # else:
        #     n_samples = 1
        #
        def _slice(x_t, idx, ndim):
            if x_t.ndim == 3:
                return x_t[:, :, idx * ndim: (idx + 1) * ndim]
            return x_t[:, idx * ndim:(idx + 1) * ndim]

        def _step(x_t, h_tm1, c_tm1, W, U, b):
            # z = sigmoid( W * x(t) + U * h(t-1) + b)
            # zi =  W * x(t) + U * h(t-1) + b
            zi = T.dot(x_t, W) + T.dot(h_tm1, U) + b
            # zi = T.dot(h_tm1, self.Uh)
            # zi += x_t
            # W = [Wi, Wf, Wo, Wc], U = [Ui, Uf, Uo, Uc],  b = [bi, bf, bo, bc]
            i = T.nnet.sigmoid(_slice(zi, 0, self.n_output))
            f = T.nnet.sigmoid(_slice(zi, 1, self.n_output))
            o = T.nnet.sigmoid(_slice(zi, 2, self.n_output))
            c = T.tanh(_slice(zi, 3, self.n_output))

            c = f * c_tm1 + i * c;

            h = o * T.tanh(c)
            # output at each time
            # s = softmax(w * h_t + b)
            return h, c

        # h0 and c0 are initialized randomly
        h0 = T.alloc(numpy_floatX(0.), n_samples, self.n_output);
        c0 = T.alloc(numpy_floatX(0.), n_samples, self.n_output)
        h0 = theano.tensor.unbroadcast(h0, 1);
        c0 = theano.tensor.unbroadcast(c0, 1)
        [h, c], _ = theano.scan(_step, sequences=[x1],
                                outputs_info=[h0, c0],
                                non_sequences=[self.Wh, self.Uh, self.bh],
                                name='blstm_layers', n_steps=nsteps)

        [h_reverse, c_reverse], _ = theano.scan(_step, sequences=[x2],
                                                outputs_info=[h0, c0],
                                                non_sequences=[self.Wh_reverse, self.Uh_reverse, self.bh_reverse],
                                                name='blstm_layers_reverse', n_steps=nsteps,
                                                go_backwards=True)
        self.input = x
        self.output = [h, h_reverse]


class lstm_layer_mask(object):
    """A Long-Short Term Memory layer"""

    # noinspection PyUnresolvedReferences
    def __init__(self, n_input, n_output):
        """

        :type  nc: int
        :param nc: dimension of input vector

        :type  nh: int
        :param nh: number of hidden units in this layer

        :type  no: int
        :param no: dimension of output vector
        """

        # Parameter of this lstm layer
        self._name = "LSTM_MASK"
        self.n_input = n_input
        self.n_output = n_output
        # Wh = [ Wi, Wc, Wf, Wo]
        Wh = np.concatenate([ortho_weight(n_input, n_output), ortho_weight(n_input, n_output),
                             ortho_weight(n_input, n_output), ortho_weight(n_input, n_output)]
                            , axis=1)
        self.Wh = theano.shared(value=Wh, name='Wh', borrow=True)
        # U = [Ui, Uc, Uf, Uo]
        Uh = np.concatenate([ortho_weight(n_output, n_output), ortho_weight(n_output, n_output),
                             ortho_weight(n_output, n_output), ortho_weight(n_output, n_output)]
                            , axis=1)
        self.Uh = theano.shared(value=Uh, name='Uh', borrow=True)

        # bh = [bi, bc, bf, bo]
        bh = np.zeros((n_output * 4,)).astype(theano.config.floatX)
        self.bh = theano.shared(value=bh, name='bh', borrow=True)

        self.params = [self.Wh, self.Uh, self.bh]

    def perform(self, x, mask):
        nsteps = x.shape[0]
        if x.ndim == 3:
            n_samples = x.shape[1]
        else:
            n_samples = 1
        #
        def _slice(x_t, idx, ndim):
            if x_t.ndim == 3:
                return x_t[:, :, idx * ndim: (idx + 1) * ndim]
            return x_t[:, idx * ndim:(idx + 1) * ndim]

        def _step(m_t, x_t, h_tm1, c_tm1):
            # z = sigmoid( W * x(t) + U * h(t-1) + b)
            # zi =  W * x(t) + U * h(t-1) + b
            zi = T.dot(x_t, self.Wh) + T.dot(h_tm1, self.Uh) + self.bh
            # zi = T.dot(h_tm1, self.Uh)
            # zi += x_t
            # W = [Wi, Wf, Wo, Wc], U = [Ui, Uf, Uo, Uc],  b = [bi, bf, bo, bc]
            i = T.nnet.sigmoid(_slice(zi, 0, self.n_output))
            f = T.nnet.sigmoid(_slice(zi, 1, self.n_output))
            o = T.nnet.sigmoid(_slice(zi, 2, self.n_output))
            c = T.tanh(_slice(zi, 3, self.n_output))

            c = f * c_tm1 + i * c;
            c = m_t[:, None] * c + (1. - m_t)[:, None] * c_tm1

            h = o * T.tanh(c)
            h = m_t[:, None] * h + (1. - m_t)[:, None] * h_tm1
            # output at each time
            # s = softmax(w * h_t + b)
            return h, c

        # h0 and c0 are initialized randomly
        [h, c], _ = theano.scan(_step, sequences=[mask, x],
                                outputs_info=[T.alloc(numpy(0.),
                                                      n_samples, self.n_output),
                                              T.alloc(numpy(0.),
                                                      n_samples, self.n_output)],
                                name='LSTM_MASK_layers', n_steps=nsteps)
        self.input = x
        self.output = h


class blstm_layer_mask(object):
    """A Bidirectional Long-Short Term Memory layer
    """

    def __init__(self, n_input, n_output):
        """

        :type  nc: int
        :param nc: dimension of input vector

        :type  nh: int
        :param nh: number of hidden units in this layer

        :type  no: int
        :param no: dimension of output vector
        """

        # Parameter of this lstm layer
        self._name = "BLSTM_MASK"
        self.n_input = n_input
        self.n_output = n_output
        # Wh = [ Wi, Wc, Wf, Wo]
        Wh = np.concatenate([ortho_weight(n_input, n_output), ortho_weight(n_input, n_output),
                             ortho_weight(n_input, n_output), ortho_weight(n_input, n_output)]
                            , axis=1)
        self.Wh = theano.shared(value=Wh, name='Wh', borrow=True)
        # U = [Ui, Uc, Uf, Uo]
        Uh = np.concatenate([ortho_weight(n_output, n_output), ortho_weight(n_output, n_output),
                             ortho_weight(n_output, n_output), ortho_weight(n_output, n_output)]
                            , axis=1)
        self.Uh = theano.shared(value=Uh, name='Uh', borrow=True)

        # bh = [bi, bc, bf, bo]
        bh = np.zeros((n_output * 4,)).astype(theano.config.floatX)
        self.bh = theano.shared(value=bh, name='bh', borrow=True)

        Wh_reverse = np.concatenate([ortho_weight(n_input, n_output), ortho_weight(n_input, n_output),
                                     ortho_weight(n_input, n_output), ortho_weight(n_input, n_output)]
                                    , axis=1)
        self.Wh_reverse = theano.shared(value=Wh_reverse, name='Wh_reverse', borrow=True)
        # U = [Ui, Uc, Uf, Uo]
        Uh_reverse = np.concatenate([ortho_weight(n_output, n_output), ortho_weight(n_output, n_output),
                                     ortho_weight(n_output, n_output), ortho_weight(n_output, n_output)]
                                    , axis=1)
        self.Uh_reverse = theano.shared(value=Uh_reverse, name='Uh_reverse', borrow=True)

        # bh = [bi, bc, bf, bo]
        bh_reverse = np.zeros((n_output * 4,)).astype(theano.config.floatX)
        self.bh_reverse = theano.shared(value=bh_reverse, name='bh_reverse', borrow=True)

        self._output = np.zeros(2, )
        self.params = [self.Wh, self.Uh, self.bh, self.Wh_reverse, self.Uh_reverse, self.bh_reverse]

    def perform(self, x, mask):
        x1, x2 = x[0], x[1]
        nsteps = x1.shape[0]
        if x1.ndim == 3:
            n_samples = x1.shape[1]
        else:
            n_samples = 1
        #
        def _slice(x_t, idx, ndim):
            if x_t.ndim == 3:
                return x_t[:, :, idx * ndim: (idx + 1) * ndim]
            return x_t[:, idx * ndim:(idx + 1) * ndim]

        def _step(m_t, x_t, h_tm1, c_tm1, W, U, b):
            # z = sigmoid( W * x(t) + U * h(t-1) + b)
            # zi =  W * x(t) + U * h(t-1) + b
            zi = T.dot(x_t, W) + T.dot(h_tm1, U) + b
            # zi = T.dot(h_tm1, self.Uh)
            # zi += x_t
            # W = [Wi, Wf, Wo, Wc], U = [Ui, Uf, Uo, Uc],  b = [bi, bf, bo, bc]
            i = T.nnet.sigmoid(_slice(zi, 0, self.n_output))
            f = T.nnet.sigmoid(_slice(zi, 1, self.n_output))
            o = T.nnet.sigmoid(_slice(zi, 2, self.n_output))
            c = T.tanh(_slice(zi, 3, self.n_output))

            c = f * c_tm1 + i * c;
            c = m_t[:, None] * c + (1. - m_t)[:, None] * c_tm1

            h = o * T.tanh(c)
            h = m_t[:, None] * h + (1. - m_t)[:, None] * h_tm1
            # output at each time
            # s = softmax(w * h_t + b)
            return h, c

        # h0 and c0 are initialized randomly
        [h, c], _ = theano.scan(_step, sequences=[mask, x1],
                                outputs_info=[T.alloc(numpy(0.),
                                                      n_samples, self.n_output),
                                              T.alloc(numpy(0.),
                                                      n_samples, self.n_output)],
                                non_sequences=[self.Wh, self.Uh, self.bh],
                                name='blstm_layers', n_steps=nsteps)

        [h_reverse, c_reverse], _ = theano.scan(_step, sequences=[mask, x2],
                                                outputs_info=[T.alloc(numpy(0.),
                                                                      n_samples, self.n_output),
                                                              T.alloc(numpy(0.),
                                                                      n_samples, self.n_output)],
                                                non_sequences=[self.Wh_reverse, self.Uh_reverse, self.bh_reverse],
                                                name='blstm_layers_reverse', n_steps=nsteps,
                                                go_backwards=True)
        self.input = x
        self.output = [h, h_reverse]


class temporal_pooling_layer(object):
    """A Long-Short Term Memory layer"""

    def __init__(self, window_size=1.):
        """

        :type  nc: int
        :param nc: dimension of input vector

        :type  nh: int
        :param nh: number of hidden units in this layer

        :type  no: int
        :param no: dimension of output vector
        """

        # Parameter of this lstm layer
        self._name = "temporal_pooling"
        self.window_size = window_size

    def perform(self, x):
        n_steps = x.shape[0]
        # if x.ndim == 3:
        #     n_samples = x.shape[1]
        # else:
        #     n_samples = 1
        #
        n_samples = x.shape[1]
        h = T.tensor3()
        n_pooling_step = int(self.window_size / 0.004)

        def mean_pooling(x_t, *x_rest):
            n_sample = len(n_steps)
            for i in range(n_samples): x_t += x_rest[i]
            return x_t / (1 + n_samples)

        # h, _ = theano.scan(mean_pooling, sequences=[dict)])

        self.output = h


class linear_fusion_layer(object):
    """Output layer for AVEC dataset
    """

    def __init__(self, n_input, n_output):
        self._name = "liner_fusion"
        self.n_input = n_input
        self.n_output = n_output
        self.input = None
        self.output = None

        # w : model's output weights
        w1 = 0.01 * np.random.randn(n_input, n_output).astype(theano.config.floatX)
        self.w1 = theano.shared(value=w1, name='w1', borrow=True)
        w2 = 0.01 * np.random.randn(n_input, n_output).astype(theano.config.floatX)
        self.w2 = theano.shared(value=w2, name='w2', borrow=True)
        # b : model's output weights
        b = np.zeros((n_output,)).astype(theano.config.floatX)
        self.b = theano.shared(value=b, name='b', borrow=True)
        self.params = [self.w1, self.w2, self.b]

    def perform(self, x):
        x1, x2 = x[0], x[1]
        n_steps = x1.shape[0]
        self.input = x

        def _step(x1_t, x2_t):
            o = T.dot(x1_t, self.w1) + T.dot(x2_t, self.w2) + self.b
            return o

        # h0 and c0 are initialized randomly
        o, _ = theano.scan(_step, sequences=[x1, x2],
                           outputs_info=None,
                           name='avec_output', n_steps=n_steps)
        self.output = o


class attention_fusion_lstm_layer(object):
    """A Long-Short Term Memory layer"""

    def __init__(self, n_input, n_output):
        """

        :type  nc: int
        :param nc: dimension of input vector

        :type  nh: int
        :param nh: number of hidden units in this layer

        :type  no: int
        :param no: dimension of output vector
        """

        # Parameter for LSTM
        self._name = "attention_fusion"
        self.n_input = n_input
        self.n_output = n_output
        # Wh = [ Wi, Wc, Wf, Wo]
        Wh = np.concatenate([np.random.randn(n_input, n_output).astype(theano.config.floatX),
                             np.random.randn(n_input, n_output).astype(theano.config.floatX),
                             np.random.randn(n_input, n_output).astype(theano.config.floatX),
                             np.random.randn(n_input, n_output).astype(theano.config.floatX)]
                            , axis=1)
        # Wh = np.concatenate([ortho_weight(n_input, n_output), ortho_weight(n_input, n_output),
        #                      ortho_weight(n_input, n_output), ortho_weight(n_input, n_output)]
        #                     , axis=1)
        self.Wh = theano.shared(value=Wh, name='Wh', borrow=True)
        # U = [Ui, Uc, Uf, Uo]
        Uh = np.concatenate([ortho_weight(n_output, n_output), ortho_weight(n_output, n_output),
                             ortho_weight(n_output, n_output), ortho_weight(n_output, n_output)]
                            , axis=1)
        self.Uh = theano.shared(value=Uh, name='Uh', borrow=True)

        # bh = [bi, bc, bf, bo]
        bh = np.zeros((n_output * 4,)).astype(theano.config.floatX)
        self.bh = theano.shared(value=bh, name='bh', borrow=True)

        self.params = [self.Wh, self.Uh, self.bh]

        # Parameter for Attention singal
        Wa = np.random.rand(n_input,n_input)
        Ua = np.random.rand(n_output,n_input)
        Va = np.random.rand(n_input,1)
        self.Wa = theano.shared(value=Wa, name='Wa', borrow=True)
        self.Ua = theano.shared(value=Ua, name='Ua', borrow=True)
        self.Va = theano.shared(value=Va, name='Va', borrow=True)

    def perform(self, x):
        x1, x2 = x[0], x[1]
        nsteps = x1.shape[0]
        # if x.ndim == 3:
        #     n_samples = x.shape[1]
        # else:
        #     n_samples = 1
        #
        n_samples = x.shape[1]

        def compute_context_vector(x1_t, x2_t, h_tm1):
            e1 = T.dot(T.tanh(T.dot(x1_t,self.Wa) + T.dot(h_tm1, self.Ua)),self.Va)
            e2 = T.dot(T.tanh(T.dot(x2_t, self.Wa) + T.dot(h_tm1, self.Ua)), self.Va)

            a1 = e1/(e1+e2)
            a2 = e2/(e1 + e2)

            context_vector = T.dot(x1_t,a2) + T.dot(x2_t,a2)
            return context_vector


        def _slice(x_t, idx, ndim):
            if x_t.ndim == 3:
                return x_t[:, :, idx * ndim: (idx + 1) * ndim]
            return x_t[:, idx * ndim:(idx + 1) * ndim]

        def _step(x1_t, x2_t, h_tm1, c_tm1):

            context_vector = compute_context_vector(x1_t, x2_t, h_tm1)

            # z = sigmoid( W * x(t) + U * h(t-1) + b)
            # zi =  W * x(t) + U * h(t-1) + b
            zi = T.dot(context_vector, self.Wh) + T.dot(h_tm1, self.Uh) + self.bh
            # zi = T.dot(h_tm1, self.Uh)
            # zi += x_t
            # W = [Wi, Wf, Wo, Wc], U = [Ui, Uf, Uo, Uc],  b = [bi, bf, bo, bc]
            i = T.nnet.sigmoid(_slice(zi, 0, self.n_output))
            f = T.nnet.sigmoid(_slice(zi, 1, self.n_output))
            o = T.nnet.sigmoid(_slice(zi, 2, self.n_output))
            c = T.tanh(_slice(zi, 3, self.n_output))

            c = f * c_tm1 + i * c;

            h = o * T.tanh(c)
            # output at each time
            # s = softmax(w * h_t + b)
            return [h, c]

        # h0 and c0 are initialized randomly
        h0 = T.alloc(numpy(0.), n_samples, self.n_output);
        c0 = T.alloc(numpy(0.), n_samples, self.n_output)
        h0 = theano.tensor.unbroadcast(h0, 1);
        c0 = theano.tensor.unbroadcast(c0, 1)
        [h, c], _ = theano.scan(fn=_step, sequences=x,
                                outputs_info=[h0, c0],
                                n_steps=nsteps)
        self.input = x
        self.output = h