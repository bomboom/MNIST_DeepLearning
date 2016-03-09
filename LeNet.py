import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample     #maxPooling

import numpy as np

class LeNet:
    def __init__(self, rng, filter_shape, image_shape, poolsize = (2,2)):
        assert image_shape[1] == filter_shape[1]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        n_in = numpy.prod(filter_shape[1:])
        #   pooling size
        n_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))

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

        initial_b = np.zeros((n_out,), dtype = theano.config.floatX)
        self.b = theano.shared(value = initial_b, name = 'b', borrow = True)

        
