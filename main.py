'''MNIST dataset
    train_sample: 60,000    test_sample: 10,000
    image: 28 x 28 pixels

Sparse AutoEncoder + Logistic Regression(stochastic gradient descent)
'''

import cPickle as pickle
import gzip
import numpy as np
from softMaxRegression import softMaxRegression
from DenoiseAE import DenoiseAE
from SparseAE import SparseAE

from PIL import Image

import theano
from theano import tensor as T
import timeit
from theano.tensor.shared_randomstreams import RandomStreams

def shared_dataset(data_xy, borrow = True):
    '''Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory.
    '''
    data_x, data_y = data_xy    #y is label
    shared_x = theano.shared(np.asarray(data_x, dtype = theano.config.floatX),
                                        borrow = borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype = theano.config.floatX),
                                        borrow = borrow)
                        #data on the GPU it has to be stored as floats
                        #during our computations we need them as ints(cast).
    return shared_x, T.cast(shared_y, 'int32')

def load_data(dataset = 'mnist.pkl.gz'):
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding = 'latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

if __name__ == '__main__':
    datasets = load_data()
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    #da = DenoiseAE(numpy_rng = rng, theano_rng = theano_rng, input = datasets[0][0])
    #da.fit()
    sa = SparseAE(numpy_rng = rng, theano_rng = theano_rng, input = datasets[0][0])
    sa.fit()

    processed_data = []
    for i in range(3):
        _in = []
        _in.append(theano.shared(T.nnet.sigmoid(T.dot(datasets[i][0], sa.W) + sa.b).eval()))
        _in.append(datasets[i][1])
        processed_data.append(_in)
    print datasets[0][0].get_value(borrow=True).shape
    print processed_data[0][0].get_value(borrow=True).shape
    softMaxRegression.sgd_optimization_mnist(datasets = processed_data, n_in = 500)


'''plot'''
'''
def scale_to_unit_interval(ndar, eps=1e-8):
  """ Scales all values in the ndarray ndar to be between 0 and 1 """
  ndar = ndar.copy()
  ndar -= ndar.min()
  ndar *= 1.0 / (ndar.max() + eps)
  return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
  """
  Transform an array with one flattened image per row, into an array in
  which images are reshaped and layed out like tiles on a floor.

  This function is useful for visualizing datasets whose rows are images,
  and also columns of matrices for transforming those rows
  (such as the first layer of a neural net).

  :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
  be 2-D ndarrays or None;
  :param X: a 2-D array in which every row is a flattened image.

  :type img_shape: tuple; (height, width)
  :param img_shape: the original shape of each image

  :type tile_shape: tuple; (rows, cols)
  :param tile_shape: the number of images to tile (rows, cols)

  :param output_pixel_vals: if output should be pixel values (i.e. int8
  values) or floats

  :param scale_rows_to_unit_interval: if the values need to be scaled before
  being plotted to [0,1] or not


  :returns: array suitable for viewing as an image.
  (See:`Image.fromarray`.)
  :rtype: a 2-d array with same dtype as X.

  """

  assert len(img_shape) == 2
  assert len(tile_shape) == 2
  assert len(tile_spacing) == 2

  # The expression below can be re-written in a more C style as
  # follows :
  #
  # out_shape = [0,0]
  # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
  #                tile_spacing[0]
  # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
  #                tile_spacing[1]
  out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                      in zip(img_shape, tile_shape, tile_spacing)]

  if isinstance(X, tuple):
      assert len(X) == 4
      # Create an output numpy ndarray to store the image
      if output_pixel_vals:
          out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
      else:
          out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

      #colors default to 0, alpha defaults to 1 (opaque)
      if output_pixel_vals:
          channel_defaults = [0, 0, 0, 255]
      else:
          channel_defaults = [0., 0., 0., 1.]

      for i in range(4):
          if X[i] is None:
              # if channel is None, fill it with zeros of the correct
              # dtype
              out_array[:, :, i] = np.zeros(out_shape,
                      dtype='uint8' if output_pixel_vals else out_array.dtype
                      ) + channel_defaults[i]
          else:
              # use a recurrent call to compute the channel and store it
              # in the output
              out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
      return out_array

  else:
      # if we are dealing with only one channel
      H, W = img_shape
      Hs, Ws = tile_spacing

      # generate a matrix to store the output
      out_array = np.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)


      for tile_row in range(tile_shape[0]):
          for tile_col in range(tile_shape[1]):
              if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                  if scale_rows_to_unit_interval:
                      # if we should scale values to be between 0 and 1
                      # do this by calling the `scale_to_unit_interval`
                      # function
                      this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                  else:
                      this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                  # add the slice to the corresponding position in the
                  # output array
                  out_array[
                      tile_row * (H+Hs): tile_row * (H + Hs) + H,
                      tile_col * (W+Ws): tile_col * (W + Ws) + W
                      ] \
                      = this_img * (255 if output_pixel_vals else 1)
      return out_array

image = Image.fromarray(tile_raster_images(X=sa.W.get_value(borrow=True).T,
                                    img_shape=(28, 28), tile_shape=(10, 10),
                                    tile_spacing=(1, 1)))
image.save('filters_corruption_30.png')
'''
