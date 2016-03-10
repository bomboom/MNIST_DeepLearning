'''impelement of LeNet5
'''

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample     #maxPooling

import timeit

import numpy as np
from LogisticRegression import LogisticRegression

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

class Layer:
    def __init__(self, rng, input, filter_shape, image_shape, poolsize = (2,2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        n_in = np.prod(filter_shape[1:])
        #   pooling size
        n_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolsize))

        #initialize
        self.W = theano.shared(
            np.asarray(
                rng.uniform(
                    low = -np.sqrt(6. / (n_in + n_out)),
                    high = np.sqrt(6. / (n_in + n_out)),
                    size = filter_shape
                ),
                dtype = theano.config.floatX
            ),
            borrow = True
        )

        initial_b = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value = initial_b, name = 'b', borrow = True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]

class LeNet:
    def __init__(self, image_shape = [28, 12], filter_shape = [5, 5],
                nkerns = [20, 50], batch_size = 500):
        self.layers = []
        rng = np.random.RandomState(23455)

        # generate symbolic variables for input (x and y represent a
        # minibatch)
        self.x = T.matrix('x')  # data, presented as rasterized images
        self.y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

        layer0_input = self.x.reshape((batch_size, 1, image_shape[0], image_shape[0]))
        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        layer0 = Layer(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 1, image_shape[0], image_shape[0]),
            filter_shape=(nkerns[0], 1, filter_shape[0], filter_shape[0]),
            poolsize=(2, 2)
        )
        self.layers.append(layer0)

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        layer1 = Layer(
            rng,
            input=layer0.output,
            image_shape=(batch_size, nkerns[0], image_shape[1], image_shape[1]),
            filter_shape=(nkerns[1], nkerns[0], filter_shape[1], filter_shape[1]),
            poolsize=(2, 2)
        )
        self.layers.append(layer1)

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        layer2_input = layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(
            input=layer2_input,
            rng = rng,
            n_in=nkerns[1] * 4 * 4,
            n_out=500,
            activ=T.tanh
        )
        self.layers.append(layer2)

        # classify the values of the fully-connected sigmoidal layer
        layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
        self.layers.append(layer3)

        # the cost we minimize during training is the NLL of the model
        self.cost = layer3.negative_log_likelihood(self.y)

    def fit(self, datasets, learning_rate = 0.1, n_epochs = 200, batch_size = 500):
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        #number of minibatchs
        n_train_batches = train_set_x.get_value(borrow = True).shape[0]// batch_size
        n_valid_batches = valid_set_x.get_value(borrow = True).shape[0]// batch_size
        n_test_batches = test_set_x.get_value(borrow = True).shape[0] // batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch

        test_model = theano.function(
            [index],
            self.layers[3].errors(self.y),
            givens={
                self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        validate_model = theano.function(
            [index],
            self.layers[3].errors(self.y),
            givens={
                self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        #updates
        params = self.layers[3].params + self.layers[2].params + self.layers[1].params + self.layers[0].params
        grads = T.grad(self.cost, params)

        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]

        train_model = theano.function(
            [index],
            self.cost,
            updates=updates,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        '''training'''

        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                cost_ij = train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [
                            test_model(i)
                            for i in range(n_test_batches)
                        ]
                        test_score = np.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print 'Optimization complete.'
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print ('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.))
