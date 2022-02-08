Initialize API
========================

.. currentmodule:: pyvqnet.static.vqnet

Initialize layers
------------------------

RandomNormal\_
^^^^^^^^^^^^^^^^^^

.. autofunction:: RandomNormal_

Examples::

    input_layer = vqnet.Input([3,17,17])
    conv2d_layer = vqnet.Conv2D(input_layer, 16, [3,3], [2,2])
    a = conv2d_layer.params[0].getdata()
    vqnet.RandomNormal_(conv2d_layer,m=0.0, s=0.1, seed=1234)
    b = conv2d_layer.params[0].getdata()
    print(a,b)

RandomUniform\_
^^^^^^^^^^^^^^^^^^

.. autofunction:: RandomUniform_

Examples::

    input_layer = vqnet.Input([3,17,17])
    conv2d_layer = vqnet.Conv2D(input_layer, 16, [3,3], [2,2])
    a = conv2d_layer.params[0].getdata()
    vqnet.RandomUniform_(conv2d_layer,min=0.0, max=0.1, seed=1234)
    b = conv2d_layer.params[0].getdata()
    print(a,b)

GlorotNormal\_
^^^^^^^^^^^^^^^^^^

.. autofunction:: GlorotNormal_

Examples::

    input_layer = vqnet.Input([3,17,17])
    conv2d_layer = vqnet.Conv2D(input_layer, 16, [3,3], [2,2])
    a = conv2d_layer.params[0].getdata()
    vqnet.GlorotNormal_(conv2d_layer,seed=1234)
    b = conv2d_layer.params[0].getdata()
    print(a,b)

GlorotUniform\_
^^^^^^^^^^^^^^^^^^

.. autofunction:: GlorotUniform_

Examples::

    input_layer = vqnet.Input([3,17,17])
    conv2d_layer = vqnet.Conv2D(input_layer, 16, [3,3], [2,2])
    a = conv2d_layer.params[0].getdata()
    vqnet.GlorotUniform_(conv2d_layer,seed=1234)
    b = conv2d_layer.params[0].getdata()
    print(a,b)

HeNormal\_
^^^^^^^^^^^^^^^^^^

.. autofunction:: HeNormal_

Examples::

    input_layer = vqnet.Input([3,17,17])
    conv2d_layer = vqnet.Conv2D(input_layer, 16, [3,3], [2,2])
    a = conv2d_layer.params[0].getdata()
    vqnet.HeNormal_(conv2d_layer,seed=1234)
    b = conv2d_layer.params[0].getdata()
    print(a,b)

HeUniform\_
^^^^^^^^^^^^^^^^^^

.. autofunction:: HeUniform_

Examples::

    input_layer = vqnet.Input([3,17,17])
    conv2d_layer = vqnet.Conv2D(input_layer, 16, [3,3], [2,2])
    a = conv2d_layer.params[0].getdata()
    vqnet.HeUniform_(conv2d_layer,seed=1234)
    b = conv2d_layer.params[0].getdata()
    print(a,b)


Constant\_
^^^^^^^^^^^^^^^^^^

.. autofunction:: Constant_

Examples::

    input_layer = vqnet.Input([3,17,17])
    conv2d_layer = vqnet.Conv2D(input_layer, 16, [3,3], [2,2])
    a = conv2d_layer.params[0].getdata()
    vqnet.Constant_(conv2d_layer,v=0.1)
    b = conv2d_layer.params[0].getdata()
    print(a,b)