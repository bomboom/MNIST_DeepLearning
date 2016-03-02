'''Stacked Sparse AutoEncoder
'''

import theano
from theano import tensor as T
import timeit

import os
import numpy as np

class StackedAE:
    def __init__(self, numpy_rng, theano_rng = None, n_ins=784,
                hidden_layers_sizes = [500, 500], n_out = 10):
        self.sigmoid_layers = []
        self.ae_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        
