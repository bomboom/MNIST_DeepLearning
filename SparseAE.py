import theano
from theano import tensor as T
import timeit

import os
import numpy as np

class SparseAE:
    def __init__(self, numpy_rng, theano_rng, input = None, n_visible = 28 * 28,
                n_hidden = 500, W = None, bhid = None, bvis = None, rho = 0.05):
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
                  shared belong the SA and another architecture; if dA should
                  be standalone set this to None
            bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if SA should be standalone set this to None
            bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if SA should be standalone set this to None
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
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
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
        self.rho = rho
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.data = T.dmatrix(name='input')
            self.x = T.dmatrix(name = 'input')
        else:
            self.data = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_value(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def kl_divergence(self, p, p_hat):
        return p * T.log(p) - T.log(p_hat) + (1 - p) * T.log(1 - p) - (1 - p) * T.log(1 - p_hat)

    def sparsity_penalty(self, h, sparse_beta = 0.001):
        sparsity_level = T.extra_ops.repeat(self.rho, self.n_hidden)
        sparsity_penalty = 0
        avg_act = h.mean(axis=0)
        kl_div = self.kl_divergence(sparsity_level, avg_act)
        sparsity_penalty = sparse_beta * kl_div.sum()
        # Implement KL divergence here.
        return sparsity_penalty

    def get_cost_update(self, sparse_beta, learning_rate):
        y = self.get_hidden_value(self.x)
        z = self.get_reconstructed_input(y)

        cost = T.mean(((self.x - z)**2).sum(axis=1))
        sparsity_penal = self.sparsity_penalty(y, sparse_beta)
        cost = cost + sparsity_penal

        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam)
                    for param, gparam in zip(self.params, gparams)]
        return cost, updates

    def fit(self, sparse_beta = 0.1, learning_rate = 0.13, batch_size = 20, training_epochs = 1):
        if not self.data:
            raise Exception("data can't be empty!")

        index = T.lscalar()
        X = T.matrix('x')
        self.x = X

        train_set_x = self.data
        n_train_batches = train_set_x.get_value(borrow = True).shape[0]// batch_size

        cost, updates = self.get_cost_update(sparse_beta = sparse_beta,
                                            learning_rate = learning_rate
                                            )
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

        print (os.path.split(__file__)[1] +
           ' ran for %.2fm' % (training_time / 60.))
