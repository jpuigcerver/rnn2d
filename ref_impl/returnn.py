import numpy as np
import theano
import theano.tensor as T
import ctc

from math import sqrt
from numpy.random import uniform

from MultiDirectionalTwoDLSTMOp import MultiDirectionalTwoDLSTMOpInstance

FORGET_GATE_Y_INITIAL_BIAS = 1.0
FORGET_GATE_X_INITIAL_BIAS = 1.0

def create_bias(n_cells, name):
    b = np.zeros((5 * n_cells,), dtype=theano.config.floatX)
    b[1 * n_cells:2 * n_cells] = FORGET_GATE_Y_INITIAL_BIAS
    b[2 * n_cells:3 * n_cells] = FORGET_GATE_X_INITIAL_BIAS
    return theano.shared(b, borrow=True, name=name)


def create_xavier_weights(shape, name):
    p = shape[0] + np.prod(shape[1:]) * 4
    W = np.asarray(uniform(low  =-sqrt(6) / sqrt(p),
                           high = sqrt(6) / sqrt(p),
                           size=shape),
                   dtype=theano.config.floatX)
    return theano.shared(W, borrow=True, name=name)


def create_lstm2d_weights(n_in, n_out, name_suffix):
    W, U, V = create_xavier_weights((n_in,  5 * n_out), 'W' + name_suffix), \
              create_xavier_weights((n_out, 5 * n_out), 'U' + name_suffix), \
              create_xavier_weights((n_out, 5 * n_out), 'V' + name_suffix)
    return W, U, V


def create_lstm2d(x, sizes, n_in, n_out, projection='concat', name='mdlstm', init_params=None):
    assert projection is None or projection == 'concat' or \
        projection == 'mean' or projection == 'sum'
    if not init_params:
        b1 = create_bias(n_out, 'b1_%s' % name)
        b2 = create_bias(n_out, 'b2_%s' % name)
        b3 = create_bias(n_out, 'b3_%s' % name)
        b4 = create_bias(n_out, 'b4_%s' % name)
        W1, U1, V1 = create_lstm2d_weights(n_in, n_out, '1_%s' % name)
        W2, U2, V2 = create_lstm2d_weights(n_in, n_out, '2_%s' % name)
        W3, U3, V3 = create_lstm2d_weights(n_in, n_out, '3_%s' % name)
        W4, U4, V4 = create_lstm2d_weights(n_in, n_out, '4_%s' % name)
    else:
        b1, b2, b3, b4 = theano.shared(init_params['b1'], borrow=True, name='b1_%s' % name), \
                         theano.shared(init_params['b2'], borrow=True, name='b2_%s' % name), \
                         theano.shared(init_params['b3'], borrow=True, name='b3_%s' % name), \
                         theano.shared(init_params['b4'], borrow=True, name='b4_%s' % name)
        W1, W2, W3, W4 = theano.shared(init_params['W1'], borrow=True, name='W1_%s' % name), \
                         theano.shared(init_params['W2'], borrow=True, name='W2_%s' % name), \
                         theano.shared(init_params['W3'], borrow=True, name='W3_%s' % name), \
                         theano.shared(init_params['W4'], borrow=True, name='W4_%s' % name)
        U1, U2, U3, U4 = theano.shared(init_params['U1'], borrow=True, name='U1_%s' % name), \
                         theano.shared(init_params['U2'], borrow=True, name='U2_%s' % name), \
                         theano.shared(init_params['U3'], borrow=True, name='U3_%s' % name), \
                         theano.shared(init_params['U4'], borrow=True, name='U4_%s' % name)
        V1, V2, V3, V4 = theano.shared(init_params['V1'], borrow=True, name='V1_%s' % name), \
                         theano.shared(init_params['V2'], borrow=True, name='V2_%s' % name), \
                         theano.shared(init_params['V3'], borrow=True, name='V3_%s' % name), \
                         theano.shared(init_params['V4'], borrow=True, name='V4_%s' % name)
    # 2D-LSTM operation
    y = MultiDirectionalTwoDLSTMOpInstance(x,
                                           W1, W2, W3, W4,
                                           U1, U2, U3, U4,
                                           V1, V2, V3, V4,
                                           b1, b2, b3, b4,
                                           sizes)
    # Concatenate
    y = T.stack(y[:4], axis=-1)
    if projection == 'mean':
        y = y.mean(axis=-1)
    elif projection == 'sum':
        y = y.sum(axis=-1)
    elif projection == 'concat':
        y = y.reshape((y.shape[0], y.shape[1], y.shape[2], y.shape[3] * y.shape[4]))

    # Return output and parameters
    return y, sizes, [b1, b2, b3, b4, W1, W2, W3, W4, U1, U2, U3, U4, V1, V2, V3, V4]


class Model:
    def __init__(self, n_in, layers, collapse_type='concat', init_params=None):
        self.x = T.tensor4(name='x', dtype='float32')  # Input images, HxWxNxD
        self.z = T.matrix(name='z', dtype='float32')   # Sizes of the images

        self.w = []
        nx, nz = self.x, self.z
        for i, n_out in enumerate(layers):
            init_p = None if not init_params else init_params[i]
            ct = collapse_type if isinstance(collapse_type, str) \
                 else collapse_type[i]
            nx, nz, nw = create_lstm2d(nx, nz, n_in, n_out,
                                       projection=ct, init_params=init_p)
            n_in = n_out * 4
            self.w.extend(nw)

        self._output = nx
        self._output_size = nz
        self._callf = theano.function(
            inputs=self.input, outputs=[self.output, self.output_size])

    @property
    def input(self):
        return [self.x, self.z]

    @property
    def output(self):
        return self._output

    @property
    def output_size(self):
        return self._output_size

    @property
    def parameters(self):
        return self.w

    def __call__(self, x, z):
        return self._callf(x, z)


class Collapse:
    def __init__(self, model, collapse_type='sum'):
        assert(collapse_type == 'sum' or collapse_type == 'mean')
        self._model = model
        if collapse_type == 'sum':
            self._output = model.output.sum(axis=0)
        elif collapse_type == 'mean':
            self._output = model.output.mean(axis=0)

        self._output_size = model._output_size[:,1]
        self._callf = theano.function(
            inputs=self.input, outputs=[self.output, self.output_size])

    @property
    def input(self):
        return self._model.input

    @property
    def output(self):
        return self._output

    @property
    def output_size(self):
        return self._output_size

    @property
    def parameters(self):
        return self._model.parameters

    def __call__(self, x, z):
        return self._callf(x, z)


class Loss:
    def __init__(self, model):
        self._yhat = T.vector(name='yhat', dtype='int32')
        self._ylen = T.vector(name='ylen', dtype='int32')
        self._model = model
        self._loss = ctc.cpu_ctc_th(model.output, model.output_size,
                                    self._yhat, self._ylen).sum()
        self._callf = theano.function(inputs=self.input, outputs=self.output)

    @property
    def input(self):
        return self._model.input + [self._yhat, self._ylen]

    @property
    def output(self):
        return self._loss

    @property
    def parameters(self):
        return self._model.parameters

    def __call__(self, x, z):
        return self._callf(x, z)


class SGD:
    def __init__(self, loss, lr = 0.0001):
        gradParameters = T.grad(loss.output, loss.parameters)
        self._train_function = theano.function(
            inputs=loss.input, outputs=loss.output,
            updates=map(lambda x: (x[0], x[0] - lr * x[1]),
                        zip(loss.parameters, gradParameters)))

    def train(self, *args):
        return self._train_function(*args)
