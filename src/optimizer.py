# -*- coding: utf-8 -*-
"""
@author: raoqiyu
Created on Sat Oct 17 09:08:11 2015

Optimizer object
"""

from  theano import tensor as T, theano

from utils import *


class SGD(object):
    # noinspection PyUnresolvedReferences
    def __init__(self, l1, l2=0.0, lr=0.05, decay=1e-6, momentum=0.9):
        self.name = 'SGD'
        self.l1 = theano.shared(numpy_floatX(l1), name='l1')
        self.l2 = theano.shared(numpy_floatX(l2), name='l2')
        self.lr = theano.shared(numpy_floatX(lr), name='lr')
        self.decay = theano.shared(numpy_floatX(decay), name='decay')
        self.momentum = theano.shared(numpy_floatX(momentum), name='momentum')

    # noinspection PyUnresolvedReferences
    def compile(self, parameters, x, mask, y, cost):
        weight_decay = ((self.w ** 2).sum()) * self.decay
        cost += weight_decay
        gradients = [T.grad(cost, param) for param in self.params]

        updates = [(p, p - self.lr * g) for p, g in zip(parameters, gradients)]
        train = theano.function([x, mask, y, self.lr], cost, updates=updates)
        return train


class ADADELTA(object):
    def __init__(self):
        self.name = 'Adadelta'

    # noinspection PyUnresolvedReferences
    def compile(self, params, x, y, cost):
        gradients = [T.grad(cost, param) for param in params]

        zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                      name='%s_grad' % p)
                        for p in params]
        running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                     name='%s_rup2' % p)
                       for p in params]
        running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                        name='%s_rgrad2' % p)
                          for p in params]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, gradients)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
                 for rg2, g in zip(running_grads2, gradients)]

        train = theano.function([x, y], cost, updates=zgup + rg2up,
                                name='adadelta_train')

        updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
                 for zg, ru2, rg2 in zip(zipped_grads,
                                         running_up2,
                                         running_grads2)]

        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
                 for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(params, updir)]

        update = theano.function([], [], updates=ru2up + param_up,
                                 on_unused_input='ignore',
                                 name='adadelta_update')
        return train, update
