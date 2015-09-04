# Author: Kyle Kastner
# License: BSD 3-clause
import numpy as np
import theano
from theano import tensor


class sgd(object):
    """
    Vanilla SGD
    """
    def __init__(self, params):
        pass

    def updates(self, params, grads, learning_rate):
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            updates.append((param, param - learning_rate * grad))
        return updates


class sgd_momentum(object):
    """
    SGD with momentum
    """
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads, learning_rate, momentum):
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            updates.append((param, param - learning_rate * grad))
            updates.append((memory, momentum * memory + (1. - momentum) * grad))
        return updates


class sgd_nesterov(object):
    """
    SGD with nesterov momentum

    Based on example from Yann D.

    See Formula 7 from
    Advances in Optimizing Recurrent Neural Networks
    Y. Benio, N. Boulanger-Lewandowski, R. Pascanu
    """
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads, learning_rate, momentum):
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            update = momentum * memory - learning_rate * grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates


class rmsprop(object):
    """
    RMSProp with nesterov momentum and gradient rescaling
    """
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads, learning_rate, momentum, rescale=5.):
        grad_norm = tensor.sqrt(sum(map(lambda x: tensor.sqr(x).sum(), grads)))
        scaling_num = rescale
        scaling_den = tensor.maximum(rescale, grad_norm)
        scaling = scaling_num / scaling_den
        # Magic constants
        decay = 0.9
        minimum_grad = 1E-6
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            grad *= scaling
            memory = self.memory_[n]
            decayed = decay * memory + (1 - decay) * grad ** 2
            grad_scale = tensor.sqrt(decayed + minimum_grad)
            grad = grad / grad_scale
            update = momentum * memory - learning_rate * grad
            update2 = momentum * momentum * memory - (
                1 + momentum) * learning_rate * grad
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates


class adagrad(object):
    """
    Adagrad optimizer
    """
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]

    def updates(self, params, grads, learning_rate, eps=1E-8):
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            m_t = memory + grad ** 2
            g_t = grad / (eps + tensor.sqrt(m_t))
            p_t = param - learning_rate * g_t
            updates.append((memory, m_t))
            updates.append((param, p_t))
        return updates


class adam(object):
    """
    Adam optimizer

    Based on implementation from @NewMu / Alex Radford
    """
    def __init__(self, params):
        self.memory_ = [theano.shared(np.zeros_like(p.get_value()))
                        for p in params]
        self.velocity_ = [theano.shared(np.zeros_like(p.get_value()))
                          for p in params]
        self.itr_ = theano.shared(np.array(0.).astype(theano.config.floatX))

    def updates(self, params, grads, learning_rate, b1=0.1, b2=0.001, eps=1E-8):
        updates = []
        itr = self.itr_
        i_t = itr + 1.
        fix1 = 1. - (1. - b1) ** i_t
        fix2 = 1. - (1. - b2) ** i_t
        lr_t = learning_rate * (tensor.sqrt(fix2) / fix1)
        for n, (param, grad) in enumerate(zip(params, grads)):
            memory = self.memory_[n]
            velocity = self.velocity_[n]
            m_t = (b1 * grad) + ((1. - b1) * memory)
            v_t = (b2 * tensor.sqr(grad)) + ((1. - b2) * velocity)
            g_t = m_t / (tensor.sqrt(v_t) + eps)
            p_t = param - (lr_t * g_t)
            updates.append((memory, m_t))
            updates.append((velocity, v_t))
            updates.append((param, p_t))
        updates.append((itr, i_t))
        return updates
