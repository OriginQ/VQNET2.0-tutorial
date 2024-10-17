
VQNet使用torch后端
#################################################



自2.15.0版本开始,本软件支持使用torch进行底层运算，方便接入主流大模型训练库进行大模型微调。

    .. note::

        :ref:`vqc_api` 中的变分量子计算函数(小写命名，例如 `rx`, `ry`, `rz` 等)，在 ``pyvqnet.backends.set_backend("torch")`` 后，依旧可以输入 ``QTensor``，底层使用 ``torch`` 计算。

        :ref:`qtensor_api` 中的QTensor基本计算函数，在 ``pyvqnet.backends.set_backend("torch")`` 后，依旧可以输入 ``QTensor``，底层使用 ``torch`` 计算。


后端基本设置
*******************************************

set_backend
===============================

.. py:function:: pyvqnet.backends.set_backend(backend_name)

    设置当前计算和储存数据所使用的后端，默认为 "pyvqnet",可设置为 "torch"。
    
    使用 ``pyvqnet.backends.set_backend("torch")`` 后，接口保持不变，但VQNet的 ``QTensor`` 的 ``data`` 成员变量均使用 ``torch.Tensor`` 储存数据，
    并使用torch计算。
    使用 ``pyvqnet.backends.set_backend("pyvqnet")`` 后，VQNet ``QTensor`` 的 ``data`` 成员变量均使用 ``pyvqnet._core.Tensor`` 储存数据，并使用pyvqnet c++库计算。

    .. note::

        该函数修改当前计算后端，在不同backends下得到的 ``QTensor`` 无法在一起运算。

    :param backend_name: backend name

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")

get_backend
===============================

.. py:function:: pyvqnet.backends.get_backend(t=None)

    如果 t 为 None,则获取当前计算后端。
    如果 t 是 QTensor,则根据其 ``data`` 属性返回创建 QTensor 时使用的计算后端。
    如果 "torch" 是使用的后端，则返回 pyvqnet torch api 后端。
    如果 "pyvqnet" 是使用的后端, 则简单地返回“pyvqnet”。
    
    :param t: 当前张量,默认值: None。
    :return: 后端。默认返回 "pyvqnet"。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.backends.get_backend()


兼容torch的神经网络类以及变分量子神经网络模块
******************************************************



TorchModule
===============================

.. py:class:: pyvqnet.nn.torch.TorchModule(*args, **kwargs)

    当用户使用 `torch` 后端时候，定义模型 `Module` 应该继承的基类。该类继承于 ``pyvqnet.nn.Module`` 以及 ``torch.nn.Module``。
    该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    该类的 ``_buffers`` 中的数据为 ``torch.Tensor``类型。
    该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameters``类型。


QModule
===============================

.. py:class:: pyvqnet.nn.torch.QModule(name="")

    当用户使用 `torch` 后端时候，定义量子变分线路模型 `Module` 应该继承的基类。该类继承于 ``pyvqnet.nn.torch.TorchModule`` 以及 ``torch.nn.Module``。
    该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    该类的 ``_buffers`` 中的数据为 ``torch.Tensor``类型。
    该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameters``类型。

Linear
=================================

.. py:class:: pyvqnet.nn.torch.Linear(input_channels, output_channels, weight_initializer=None, bias_initializer=None,use_bias=True, dtype=None, name: str = "")

    线性模块(全连接层),:math:`y = Ax + b` 。
    该类继承于 ``pyvqnet.nn.Module`` 以及 ``torch.nn.Module``，可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    该类的 ``_buffers`` 中的数据为 ``torch.Tensor``类型。
    该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameters``类型。
    

    :param input_channels: `int` - 输入数据通道数。
    :param output_channels: `int` - 输出数据通道数。
    :param weight_initializer: `callable` - 权重初始化函数,默认为空,使用he_uniform。
    :param bias_initializer: `callable` - 偏置初始化参数,默认为空,使用he_uniform。
    :param use_bias: `bool` - 是否使用偏置项, 默认使用。
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 线性层的命名,默认为""。

    :return: 线性层实例。

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Linear
        c1 =2
        c2 = 3
        cin = 7
        cout = 5
        n = Linear(cin,cout)
        input = QTensor(np.arange(1,c1*c2*cin+1).reshape((c1,c2,cin)),requires_grad=True,dtype=pyvqnet.kfloat32)
        y = n.forward(input)
        print(y)
 