Core layer API
========================

.. currentmodule:: pyvqnet.static.vqnet



Input Definition
------------------------

Input
^^^^^^^^^^^^
.. autofunction:: Input

Examples::

    input_layer = vqnet.Input([3,17,17])
    input_layer.input.shape
    # input tensor shape = [1, 3, 17, 17]

Conv Definition
------------------------

Conv2D
^^^^^^^^^^^^

.. autofunction:: Conv2D

Examples::

    in4d =  vqnet.Input([4,32,32])
    conv2dlayer1 = vqnet.Conv2D(in4d, filters = 16, kernel_size = [3, 3],strides=[2,2],padding="valid")
    conv2dlayer1.info()
    conv2dlayer2 = vqnet.Conv2D(in4d, filters = 16, kernel_size = [3, 3],strides=[1,1],padding="same")
    conv2dlayer2.info()


Conv1D
^^^^^^^^^^^^

.. autofunction:: Conv1D

Examples::

    in4d =  vqnet.Input([1,32])
    conv2dlayer1 = vqnet.Conv1D(in4d, filters = 16, kernel_size = 3,strides=2,padding="valid")
    conv2dlayer1.info()
    conv2dlayer2 = vqnet.Conv1D(in4d, filters = 16, kernel_size = 3,strides=1,padding="same")
    conv2dlayer2.info()


ConvT2D
^^^^^^^^^^^^

.. autofunction:: ConvT2D

Examples::

    in4d =  vqnet.Input([4,5,5])
    conv2dlayer1 = vqnet.ConvT2D(in4d, filters = 16, kernel_size = [3, 3],strides=[2,2],padding="valid")
    conv2dlayer1.info()
    conv2dlayer2 = vqnet.ConvT2D(in4d, filters = 16, kernel_size = [3, 3],strides=[1,1],padding="same")
    conv2dlayer2.info()

Pool Definition
------------------------

MaxPool2D
^^^^^^^^^^^^

.. autofunction:: MaxPool2D

Examples::

    in4d =  vqnet.Input([4,5,5])

    vqnet.MaxPool2D(in4d)
    vqnet.MaxPool2D(in4d, [2, 2])
    l2 = vqnet.MaxPool2D(in4d, [3, 3], [2, 2],"valid")
    l2.info()


MaxPool1D
^^^^^^^^^^^^

.. autofunction:: MaxPool1D

Examples::

    in3d = vqnet.Input([1, 16])
    
    vqnet.MaxPool1D(in3d)
    vqnet.MaxPool1D(in3d, 3, 2)
    l2 = vqnet.MaxPool1D(in3d, 3,2,"valid")
    l2.info()

AvgPool2D
^^^^^^^^^^^^

.. autofunction:: AvgPool2D

Examples::

    in4d =  vqnet.Input([4,5,5])

    vqnet.AvgPool2D(in4d)
    vqnet.AvgPool2D(in4d, [2, 2])
    l2 = vqnet.AvgPool2D(in4d, [3, 3], [2, 2],"valid")
    l2.info()


AvgPool1D
^^^^^^^^^^^^

.. autofunction:: AvgPool1D

Examples::

    in3d = vqnet.Input([1, 16])
    
    vqnet.AvgPool1D(in3d)
    vqnet.AvgPool1D(in3d, 3, 2)
    l2 = vqnet.AvgPool1D(in3d, 3,2,"valid")
    l2.info()

Dense Definition
------------------------

Dense
^^^^^^^^^^^^

.. autofunction:: Dense

Examples::

    input_layer = vqnet.Input([5])
    #output dim 3
    dense_layer = vqnet.Dense(input_layer, 3)
    dense_layer.info()


.. note::
    currently only support one dimension data. e.g. [batchsize,1,ndim]


Dropout Definition
------------------------

Dropout
^^^^^^^^^^^^

.. autofunction:: Dropout

.. note::
    currently  support 1d 2d 3d .

Normalization Definition
------------------------

BatchNormalization2d
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: BatchNormalization2d

BatchNormalization1d
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: BatchNormalization1d
