'''Stacked Sparse AutoEncoder
'''

import theano
from theano import tensor as T
import timeit

from DenoiseAE import DenoiseAE
from SparseAE import SparseAE
from LogisticRegression import LogisticRegression

import os
import numpy as np

class HiddenLayer:
    def __init__(self, input, rng, n_in, n_out, W = None, b = None, activ = T.tanh):

        if not W:
            initial_W = np.asarray(
                rng.uniform(
                    low = -np.sqrt(6. / (n_in + n_out)),
                    high = np.sqrt(6. / (n_in + n_out)),
                    size = (n_in, n_out)
                ),
                dtype = theano.config.floatX
            )
            if activ == theano.tensor.nnet.sigmoid:
                initial_W *= 4
            W = theano.shared(value = initial_W, name = 'W', borrow = True)

        if not b:
            initial_b = np.zeros((n_out,), dtype = theano.config.floatX)
            b = theano.shared(value = initial_b, name = 'b', borrow = True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activ is None else activ(lin_output))

        self.params = [self.W, self.b]

class StackedAE:
    def __init__(self, numpy_rng, theano_rng = None, n_ins=784,
                hidden_layers_sizes = [500, 500], n_outs = 10, mode = 'dA'):
        self.sigmoid_layers = []
        self.ae_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        self.x = T.matrix('x')
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data

        for i in range(self.n_layers):
            if i==0:
                input_size = n_ins
                layer_input = self.x
            else:
                input_size = hidden_layers_sizes[i-1]
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng = numpy_rng, input = layer_input,
                                        n_in = input_size,
                                        n_out = hidden_layers_sizes[i],
                                        activ = T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            #initialize dA or sA
            if mode == 'sA':
                ae_layer = SparseAE(numpy_rng = numpy_rng, theano_rng = theano_rng,
                                    input = layer_input, n_visible = input_size,
                                    n_hidden = hidden_layers_sizes[i],
                                    W = sigmoid_layer.W, bhid = sigmoid_layer.b)
            else:
                ae_layer = DenoiseAE(numpy_rng = numpy_rng, theano_rng = theano_rng,
                                    input = layer_input, n_visible = input_size,
                                    n_hidden = hidden_layers_sizes[i],
                                    W = sigmoid_layer.W, bhid = sigmoid_layer.b)
            self.ae_layers.append(ae_layer)

        self.logLayer = LogisticRegression(input = self.sigmoid_layers[-1].output,
                                            n_in = hidden_layers_sizes[-1],
                                            n_out = n_outs)
        self.params.extend(self.logLayer.params)
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the ae corresponding to the layer with same index.
        '''
        index = T.lscalar('index')
        reg_level = T.scalar('reg')
        learning_rate = T.scalar('lr')

        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for ae in self.ae_layers:
            cost, updates = ae.get_cost_update(reg_level, learning_rate)

            fn = theano.function(
                inputs = [
                    index,
                    #theano.In(reg_level, value = 0.2),
                    #theano.In(learning_rate, value = 0.1)
                    reg_level,
                    learning_rate
                ],
                outputs = cost,
                updates = updates,
                givens = {
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            pretrain_fns.append(fn)
        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set
        '''
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        #number of minibatchs
        n_valid_batches = valid_set_x.get_value(borrow = True).shape[0]// batch_size
        n_test_batches = test_set_x.get_value(borrow = True).shape[0] // batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch

         # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score

    def fit(self, datasets, reg_levels = [.1,.2], finetune_lr = 0.1, batch_size = 1,
            pretraining_epochs = 1, pretrain_lr = 0.001, training_epochs = 1):
        if self.n_layers != len(reg_levels):
            raise Exception("regulization length is not equal to number of layers, need to be:", self.n_layers)

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        n_train_batches = train_set_x.get_value(borrow=True).shape[0]//batch_size

        pretraining_fns = self.pretraining_functions(train_set_x = train_set_x,
                                                batch_size = batch_size)
        start_time = timeit.default_timer()

        for i in range(self.n_layers):
            for epoch in range(pretraining_epochs):
                c = []
                for batch_index in range(n_train_batches):
                    c.append(pretraining_fns[i](
                                index = batch_index,
                                reg = reg_levels[i],
                                lr = pretrain_lr))

                print 'Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c))

        end_time = timeit.default_timer()

        print ('The pretraining code for file ' + os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.))

        '''Fine tuning'''
        train_fn, validate_model, test_model = self.build_finetune_functions(
            datasets = datasets,
            batch_size = batch_size,
            learning_rate = finetune_lr
        )

        # early-stopping parameters
        patience = 10 * n_train_batches  # look as this many examples regardless
        patience_increase = 2.  # wait this much longer when a new best is
                                # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = np.inf
        test_score = 0.
        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0

        while (epoch < training_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):
                minibatch_avg_cost = train_fn(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    validation_losses = validate_model()
                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = test_model()
                        test_score = np.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print(
            (
                'Optimization complete with best validation score of %f %%, '
                'on iteration %i, '
                'with test performance %f %%'
            )
            % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
        )
        print ('The training code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.))
