import theano
from theano import tensor as T
import timeit
import os
import sys

import numpy as np

class DenoiseAE:
    def __init__(self, numpy_rng, theano_rng, input = None, n_visible = 28 * 28,
                n_hidden = 500, W = None, bhid = None, bvis = None):
        '''
        when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.
            numpy_rng: number random generator used to generate weights
            theano_rng: Theano random generator; if None is given one is
                        generated based on a seed drawn from `rng`
            W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None
            bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
            bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
        '''
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        #initialize W and b
        if not W:
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low = -4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high = 4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size = (n_visible, n_hidden)
                ),
                dtype = theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name = 'input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_value(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_update(self, corruption_level, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_value(tilde_x)
        z = self.get_reconstructed_input(y)

        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis = 1)
        cost = T.mean(L)

        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam)
                    for param, gparam in zip(self.params, gparams)]
        return cost, updates

    def fit(self, datasets = None, learning_rate = 0.13, batch_size = 20, training_epochs = 1):
        if not self.data:
            raise Exception("data can't be empty!")

        index = T.lscalar()
        X = T.matrix('x')
        self.x = X

        train_set_x = datasets
        n_train_batches = train_set_x.get_value(borrow = True).shape[0]// batch_size

        cost, updates = self.get_cost_update(corruption_level = 0.,
                                            learning_rate = learning_rate)
        train_da = theano.function(
                    [index], cost, updates = updates,
                    givens = {
                        X: train_set_x[index * batch_size: (index + 1) * batch_size]
                    } )
        start_time = timeit.default_timer()

        '''training'''
            # go through training epochs
        for epoch in range(training_epochs):
        # go through trainng set
            c = []
            for batch_index in range(n_train_batches):
                c.append(train_da(batch_index))

            print 'Training epoch %d, cost ' % epoch, np.mean(c)

        end_time = timeit.default_timer()

        training_time = (end_time - start_time)

        print ('The 30% corruption code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % (training_time / 60.))
