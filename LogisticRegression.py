import theano
from theano import tensor as T

import numpy as np

class LogisticRegression:
    '''multi-Class Logistic Regression
    '''
    def __init__(self, input, n_in, n_out):
        '''
        n_in: number of features
        #n_out: label dimension
        '''
        #initialize weights and bias with 0 or random
        self.W = theano.shared(np.zeros((n_in, n_out), dtype = theano.config.floatX),
                                name = 'W', borrow = True)
        self.b = theano.shared(np.zeros((n_out,), dtype = theano.config.floatX),
                                name = 'b', borrow = True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)
        self.param = [self.W, self.b]
        self.input = input

    def negative_log_likelihood(self, y):
        '''
            y: true label
        '''
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("Wrong dimension:", ('y:',y.type,"y_pred:", y_pred.type))

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
