经典神经网络模块
==================================

以下的经典神经网络模块均支持自动反向传播计算。当您运行前传函数以后，再执行反向函数就可以计算梯度。一个卷积层的简单例子如下：

.. code-block::

    # an image feed into two dimension convolution layer
    b = 2        # batch size 
    ic = 3       # input channels
    oc = 2      # output channels
    hw = 6      # input width and heights

    # two dimension convolution layer
    test_conv = Conv2D(ic,oc,(3,3),(2,2),"same",initializer.ones,initializer.ones)

    # input of shape [b,ic,hw,hw]
    x0 = QTensor(CoreTensor.range(1,b*ic*hw*hw).reshape([b,ic,hw,hw]),requires_grad=True)

    #forward function
    x = test_conv(x0)

    #backward function with autograd
    x.backward()

    print("##W###")
    print(test_conv.weights.grad)
    print("##B###")
    print(test_conv.bias.grad)
    print("##X###")
    print(x0.grad)
    print("##Y###")
    print(x)

.. currentmodule:: pyvqnet.nn


Module类
-------------------------------

abstract calculation module


Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.module.Module

forward
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pyvqnet.nn.module.Module.forward

state_dict 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: pyvqnet.nn.module.Module.state_dict

save_parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.utils.storage.save_parameters

load_parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.utils.storage.load_parameters



经典神经网络层
-------------------------------

Conv1D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.conv.Conv1D

Conv2D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.conv.Conv2D

ConvT2D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.conv.ConvT2D


AvgPool1D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.pooling.AvgPool1D

MaxPool1D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.pooling.MaxPool1D

AvgPool2D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.pooling.AvgPool2D

MaxPool2D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.pooling.MaxPool2D

Embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.embedding.Embedding

BatchNorm2d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.batch_norm.BatchNorm2d

BatchNorm1d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.batch_norm.BatchNorm1d

LayerNorm2d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.layer_norm.LayerNorm2d

LayerNorm1d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.layer_norm.LayerNorm1d

Linear
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.linear.Linear

Dropout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.dropout.Dropout


损失函数层
----------------------------------

MeanSquaredError
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.loss.MeanSquaredError

BinaryCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.loss.BinaryCrossEntropy

CategoricalCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.loss.CategoricalCrossEntropy

SoftmaxCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pyvqnet.nn.loss.SoftmaxCrossEntropy


激活函数
----------------------------------


Activation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.Activation


Sigmoid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.Sigmoid


Softplus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.Softplus


Softsign
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.Softsign


Softmax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.Softmax


HardSigmoid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.HardSigmoid


ReLu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.ReLu


LeakyReLu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.LeakyReLu


ELU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.ELU


Tanh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.nn.activation.Tanh


优化器模块
----------------------------------


Optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.optim.optimizer.Optimizer

adadelta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.optim.adadelta.Adadelta

adagrad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.optim.adagrad.Adagrad

adam
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.optim.adam.Adam

adamax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.optim.adamax.Adamax

rmsprop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.optim.rmsprop.RMSProp

sgd
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyvqnet.optim.sgd.SGD

rotosolve
^^^^^^^^^^^^^^
Rotosolve算法它允许相对于其他参数的固定值直接跳转到单个参数的最佳值，直接找到量子线路最佳参数的优化算法。

.. autoclass:: pyvqnet.optim.rotosolve.Rotosolve

.. figure:: ./images/rotosolve.png


