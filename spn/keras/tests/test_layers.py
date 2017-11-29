import numpy

import theano

from keras.models import Sequential
from spn.keras.layers import SumLayer, ProductLayer


def test_keras_layers():

    #
    # initial weights
    input_vec = numpy.array([[1., 0., 1., 0., 1., 0.],
                             [0., 0., 0., 0., 0., 0.],
                             [1., 0., 0., 1., 0., 1.],
                             [1., 1., 1., 1., 1., 1.]])

    W = numpy.array([[0.6, 0.4, 0., 0., 0., 0.],
                     [0.3, 0.7, 0., 0., 0., 0.],
                     [0., 0., 0.1, 0.9, 0., 0.],
                     [0., 0., 0.7, 0.3, 0., 0.],
                     [0., 0., 0., 0., 0.5, 0.5],
                     [0., 0., 0., 0., 0.2, 0.8]]).T

    W_1 = numpy.array([[1., 0., 1., 0., 1., 0.],
                       [0., 1., 0., 1., 0., 1.],
                       [1., 0., 1., 0., 0., 1.],
                       [0., 1., 0., 1., 1., 0.]]).T
    #
    # creating an architecture
    model = Sequential()

    sum_layer = SumLayer(output_dim=6,
                         input_dim=6,
                         weights=[W])

    # sum_layer.build()
    # sum_input = sum_layer.get_input()
    # sum_output = sum_layer.get_output()
    # f = theano.function([sum_input], sum_output)
    # print(f(input_vec))

    model.add(sum_layer)

    model.add(ProductLayer(output_dim=4,
                           input_dim=6,
                           weights=[W_1]))

    #
    # compiling
    model.compile(loss='mean_squared_error', optimizer='sgd')

    res = model.predict(input_vec, batch_size=4)
    print(res)
    print(res.shape)
