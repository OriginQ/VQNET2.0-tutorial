Layer API
========================

.. currentmodule:: pyvqnet.static.vqnet


Core layer Function
------------------------------


layer is used to build a net. layer be feed dummpy data(class Tensor) to run single layer's
forward and backward to get output.

forward
^^^^^^^^^^^^

    feed input into this layer and get layer output.

Examples::

    input_layer = vqnet.Input([3,17,17])
    input_layer.forward()
    conv2d_layer = vqnet.Conv2D(input_layer,  16, [3,3], [2,2])
    conv2d_layer.forward()


backward
^^^^^^^^^^^^

    feed input into this layer and get output. 

Examples::

    input_layer = vqnet.Input([3,17,17])
    conv2d_layer = vqnet.Conv2D(input_layer,  16, [3,3], [2,2])
    #input dummpy input
    t1 = PyTensor.range(1,input_layer.input.size).reshape(input_layer.input.shape)
    copyTensor(t1,input_layer.input)
    #input conv2d weights:w b
    w = PyTensor.range(1,conv2d_layer.params[0].size).reshape(conv2d_layer.params[0].shape)
    b = PyTensor.zeros(conv2d_layer.params[1].shape,0)
    conv2d_layer.update_weights(w,b)
    #input deltay ,deltax
    delta_x = PyTensor.zeros(conv2d_layer.input.shape,0)
    delta_y = PyTensor.ones(conv2d_layer.output.shape,0)
    conv2d_layer.update_deltas(delta_y,delta_x)
    
    #layer foward calculation,use w b input
    input_layer.forward()
    conv2d_layer.forward()
    #layer backward calculation use deltay deltax
    conv2d_layer.backward()
    #conv2d_layer.input_delta.print()
    print(conv2d_layer.delta.shape)
    print(conv2d_layer.input_delta.shape)


update_weights
^^^^^^^^^^^^^^^^^^

    update weights and bias value for the layers,support Conv1D Conv2D ConvT2D

Examples::

    input_layer = vqnet.Input([3,17,17])
    conv2d_layer = vqnet.Conv2D(input_layer,  16, [3,3], [2,2])
    #input dummpy input
    t1 = PyTensor.range(1,input_layer.input.size).reshape(input_layer.input.shape)
    copyTensor(t1,input_layer.input)
    #input conv2d weights:w b
    w = PyTensor.range(1,conv2d_layer.params[0].size).reshape(conv2d_layer.params[0].shape)
    b = PyTensor.zeros(conv2d_layer.params[1].shape,0)
    conv2d_layer.update_weights(w,b)

.. note::
    currently only support Conv1D Conv2D ConvT2D.and this function is mainly for **debugging**.


update_deltas
^^^^^^^^^^^^^^^^^^

    update delta_y (delta of output ) and delta_x (delta of input ) for the layers,test on Conv1D Conv2D ConvT2D 

Examples::

    input_layer = vqnet.Input([3,17,17])
    conv2d_layer = vqnet.Conv2D(input_layer,  16, [3,3], [2,2])
    #input dummpy input
    t1 = PyTensor.range(1,input_layer.input.size).reshape(input_layer.input.shape)
    copyTensor(t1,input_layer.input)
    #input conv2d weights:w b
    w = PyTensor.range(1,conv2d_layer.params[0].size).reshape(conv2d_layer.params[0].shape)
    b = PyTensor.zeros(conv2d_layer.params[1].shape,0)
    conv2d_layer.update_weights(w,b)
    #input deltay ,deltax
    delta_x = PyTensor.zeros(conv2d_layer.input.shape,0)
    delta_y = PyTensor.ones(conv2d_layer.output.shape,0)
    conv2d_layer.update_deltas(delta_y,delta_x)

.. note::
    currently only support Conv1D Conv2D ConvT2D.and this function is mainly for **debugging**.




