
.. _torch_api:

====================================
VQNet使用torch进行底层计算
====================================

    .. important::

        **如需要使用以下功能, 请自行安装2.11.0版本的torch。**
        **如果安装GPU版本的torch，需要使用兼容cuda12.6版本的torch, 否则可能由于 NVIDIA CUDA 运行时库问题导致您的torch无法使用。**
        **本软件安装时候不自动安装 torch 。**

    .. note::

        基于 torch 后端的 ``rx``, ``ry``, ``rz``, ``cnot`` 等变分量子计算接口在 torch 2.11.0 + CUDA 12.6 环境下编译了高性能 CUDA kernel。当检测到 torch 与 CUDA 版本匹配时会自动启用该实现；若不匹配，则自动回退为 native torch 实现。

自2.15.0版本开始,本软件支持使用 `pytorch` 作为计算后端进行底层运算,可接入基于 PyTorch 的模型、代码、第三方库进行二次开发。


    .. important::

        :ref:`vqc_api` 中的变分量子计算函数(小写命名,例如 `rx`, `ry`, `rz` 等), :ref:`qtensor_api` 中的QTensor基本计算函数,以及经典和量子神经网络模块, 在 ``pyvqnet.backends.set_backend("torch")`` 后,可以输入 ``QTensor``,其成员 `data` 从pyvqnet的 ``_core.Tensor`` 变为 ``torch.Tensor`` 计算。

        可使用 ``to_tensor`` 可将 ``torch.Tensor`` 封装为一个 ``QTensor`` 。

        使用 ``pytorch`` 等后端时，所使用的神经网络模块、pyqpanda量子神经网络模块必须继承于 ``pyvqnet.nn.torch.TorchModule``,
        自动微分变分量子模块必须必须继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule``， 否则其中参数无法进行自动微分训练和保存。

        ``pyvqnet.nn.torch.TorchModule`` 和 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型, 
        , ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型,无法使用QTensor接口。


        请注意： ``pyvqnet.backends.set_backend("torch")`` 以及 ``pyvqnet.backends.set_backend("pyvqnet")`` 会修改全局运行后端。
        不同后端配置下创建的 ``QTensor`` ，其底层数据结构不同，无法一起运算。






计算后端基本设置
====================

set_backend
------------------------------------------------

.. py:function:: pyvqnet.backends.set_backend(backend_name)

    该用于切换计算和数据存储后端，可选择使用 pyvqnet 原生计算、C++自动微分、或基于 PyTorch 的后端，从而在不同性能和兼容性需求间灵活切换。默认为 "pyvqnet-ad",可设置为 "torch"(``torch-native`` 与 ``torch`` 已合并,二者效果一致)。
    
    使用 ``pyvqnet.backends.set_backend("pyvqnet")`` 后,VQNet ``QTensor`` 的 ``data`` 成员变量均使用 ``pyvqnet._core.Tensor`` 储存数据,并使用pyvqnet c++库计算,
    自动微分在C++完成。

    使用 ``pyvqnet.backends.set_backend("pyvqnet-ad")`` 与 ``pyvqnet.backends.set_backend("pyvqnet")`` 效果一致。

    使用 ``pyvqnet.backends.set_backend("torch")`` 后,接口保持不变,VQNet的 ``QTensor`` 的 ``data`` 成员变量均使用 ``torch.Tensor`` 储存数据。
    :ref:`qtensor_api`， :ref:`vqc_api` 以及 `pyvqnet.nn.torch` 下的接口输入接受 ``QTensor`` 或 ``torch.Tensor`` 类型，输出为 ``torch.Tensor``。

    使用 ``pyvqnet.backends.set_backend("torch-native")`` 与 ``pyvqnet.backends.set_backend("torch")`` 效果一致。
    

    .. warning::

        该函数修改当前计算后端,在不同backends下得到的 ``QTensor`` 无法在一起运算。

    :param backend_name: backend name

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")

get_backend
-------------------------------

.. py:function:: pyvqnet.backends.get_backend(t=None)

    如果 t 为 None,则获取当前计算后端。
    如果 t 是 QTensor,则根据其 ``data`` 属性返回创建 QTensor 时使用的计算后端。
    如果 "torch" 是使用的后端,则返回 pyvqnet torch API 后端。
    如果 "pyvqnet" 是使用的后端, 则简单地返回“pyvqnet”。
    
    :param t: 当前张量,默认值: None。
    :return: 后端。默认返回 "pyvqnet"。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.backends.get_backend()




QTensor函数
===================

在设置 ``torch`` 计算后端后

.. code-block::

    import pyvqnet
    pyvqnet.backends.set_backend("torch")

在 :ref:`qtensor_api` 下的所有成员函数,创建函数,数学函数,逻辑函数,矩阵变换等均使用torch进行计算。使用 ``QTensor.data`` 可获取torch数据。

使用 ``to_tensor`` 可将 ``torch.Tensor`` 封装为一个 ``QTensor`` 。



经典神经模块
============================================

基类
------------------------------------------------

TorchModule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchModule(*args, **kwargs)

    当用户使用 `torch` 后端时候,定义模型 `Module` 应该继承的基类。
    
    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` 。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。
        使用torch后端情况时候,所有模块应该继承于该类。

    .. warning::

        该类以及其派生类仅适用于 ``pyvqnet.backends.set_backend("torch")`` , 不要与默认 ``pyvqnet.nn`` 下的 ``Module`` 混用。
    
        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。


 
    .. py:method:: pyvqnet.nn.torch.TorchModule.forward(x, *args, **kwargs)

        TorchModule类抽象前向计算函数。

        :param x: 输入QTensor。
        :param \*args: 非关键字可变参数。
        :param \*\*kwargs: 关键字可变参数。

        :return: 输出QTensor,内部的data是 ``torch.Tensor`` 。

        Example::

            import numpy as np
            from pyvqnet.tensor import QTensor
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            from pyvqnet.nn.torch import Conv2D
            b = 2
            ic = 3
            oc = 2
            test_conv = Conv2D(ic, oc, (3, 3), (2, 2), "valid")
            x0 = QTensor(np.arange(1, b * ic * 5 * 5 + 1).reshape([b, ic, 5, 5]),
                        requires_grad=True,
                        dtype=pyvqnet.kfloat32)
            x = test_conv.forward(x0)
            print(x)



    .. py:method:: pyvqnet.nn.torch.TorchModule.state_dict(destination=None, prefix='')

        返回包含模块整个状态的字典:包括参数和缓存值。
        键是对应的参数和缓存值名称。

        :param destination: 返回保存模型内部模块,参数的字典。
        :param prefix: 使用的参数和缓存值的命名前缀。

        :return: 包含模块整个状态的字典。

        Example::

            from pyvqnet.nn.torch import Conv2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = Conv2D(2,3,(3,3),(2,2),"same")
            print(test_conv.state_dict().keys())

    .. py:method:: pyvqnet.nn.torch.TorchModule.load_state_dict(state_dict,strict=True)
        
        将参数和缓冲区从 :attr:`state_dict` 复制到此模块及其子模块。

        :param state_dic: 包含参数和持久缓冲区的字典。
        :param strict: 是否严格执行 state_dict 中的键与模型的 `state_dict()` 匹配,默认: True。

        :return: 如果发生错误,则返回错误消息。
 
        Example::
 
            from pyvqnet.nn.torch import TorchModule,Conv2D
            import pyvqnet

            import pyvqnet.utils
            pyvqnet.backends.set_backend("torch")
            class Net(TorchModule):
                def __init__(self):
                    super(Net, self).__init__()
                    self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5),
                        stride=(1, 1), padding="valid")

                def forward(self, x):
                    return super().forward(x)

            model = Net()
            pyvqnet.utils.storage.save_parameters(model.state_dict(), "tmp.model")
            model_param = pyvqnet.utils.storage.load_parameters("tmp.model")
            model.load_state_dict(model_param)

    .. py:method:: pyvqnet.nn.torch.TorchModule.toGPU(device: int = DEV_GPU_0)

        将模块和其子模块的参数和缓冲数据移动到指定的 GPU 设备中。

        device 指定存储其内部数据的设备。 当device >= DEV_GPU_0时,数据存储在GPU上。如果您的计算机有多个GPU,
        则可以指定不同的设备来存储数据。例如device = DEV_GPU_1 , DEV_GPU_2, DEV_GPU_3, ... 表示存储在不同序列号的GPU上。
        
        .. warning::

            Module在不同GPU上无法进行计算。
            如果您尝试在 ID 超过验证 GPU 最大数量的 GPU 上创建 QTensor,将引发 Cuda 错误。

        :param device: 当前保存QTensor的设备,默认:DEV_GPU_0。device= pyvqnet.DEV_GPU_0,存储在第一个 GPU 中,device = DEV_GPU_1,存储在第二个 GPU 中,依此类推
        :return: Module 移动到 GPU 设备。

        Example::

            from pyvqnet.nn.torch import ConvT2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = ConvT2D(3, 2, [4,4], [2, 2], (0,0))
            test_conv = test_conv.toGPU()
            print(test_conv.backend)
            #1000

    .. py:method:: pyvqnet.torch.TorchModule.toCPU()

        将模块和其子模块的参数和缓冲数据移动到特定的 CPU 设备中。

        :return: Module 移动到 CPU 设备。

        Example::

            from pyvqnet.nn.torch import ConvT2D
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            test_conv = ConvT2D(3, 2, [4,4], [2, 2], (0,0))
            test_conv = test_conv.toCPU()
            print(test_conv.backend)
            #0


TorchModuleList
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.nn.torch.TorchModuleList(modules = None)

    该模块用于将子 ``TorchModule`` 保存在列表中。 TorchModuleList 可以像普通的 Python 列表一样被索引, 它包含的内部参数等可以被保存起来。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` 以及 ``pyvqnet.nn.ModuleList``,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param modules: ``pyvqnet.nn.torch.TorchModule`` 列表

    :return: 一个TorchModuleList 类

    Example::

        from pyvqnet.tensor import *
        from pyvqnet.nn.torch import TorchModule,Linear,TorchModuleList

        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class M(TorchModule):
            def __init__(self):
                super(M, self).__init__()
                self.pqc2 = TorchModuleList([Linear(4,1), Linear(4,1)
                ])

            def forward(self, x):
                y = self.pqc2[0](x)  + self.pqc2[1](x)
                return y

        mm = M()



TorchParameterList
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchParameterList(value=None)

    该模块用于将子 ``pyvqnet.nn.Parameter`` 保存在列表中。 TorchParameterList 可以像普通的 Python 列表一样被索引, 它包含的Parameter的内部参数等可以被保存起来。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` 以及 ``pyvqnet.nn.ParameterList``,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param value: nn.Parameter 列表

    :return: 一个TorchParameterList 类

    Example::

        from pyvqnet.tensor import *
        from pyvqnet.nn.torch import TorchModule,Linear,TorchParameterList
        import pyvqnet.nn as nn
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        class MyModule(TorchModule):
            def __init__(self):
                super().__init__()
                self.params = TorchParameterList([nn.Parameter((10, 10)) for i in range(10)])
            def forward(self, x):

                # ParameterList can act as an iterable, or be indexed using ints
                for i, p in enumerate(self.params):
                    x = self.params[i // 2] * x + p * x
                return x

        model = MyModule()
        print(model.state_dict().keys())


TorchSequential
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.TorchSequential(*args)

    模块将按照传递的顺序添加模块。或者,也可以将模块的 ``OrderedDict`` 传入。 ``Sequential`` 的 ``forward()`` 方法接受任何输入,并将其转发给它的第一个模块。
    然后将输出依次链接到其后每个模块的输入、最后返回最后一个模块的输出。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` 以及 ``pyvqnet.nn.Sequential``,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param args: 添加的Module

    :return: 一个 TorchSequential 类

    Example::
        
        import pyvqnet
        from collections import OrderedDict
        from pyvqnet.tensor import *
        from pyvqnet.nn.torch import TorchModule,Conv2D,ReLu,\
            TorchSequential
        pyvqnet.backends.set_backend("torch")
        model = TorchSequential(
                    Conv2D(1,20,(5, 5)),
                    ReLu(),
                    Conv2D(20,64,(5, 5)),
                    ReLu()
                )
        print(model.state_dict().keys())

        model = TorchSequential(OrderedDict([
                    ('conv1', Conv2D(1,20,(5, 5))),
                    ('relu1', ReLu()),
                    ('conv2', Conv2D(20,64,(5, 5))),
                    ('relu2', ReLu())
                ]))
        print(model.state_dict().keys())


模型参数保存和载入
--------------------------------------------

使用 :ref:`save_parameters` 中的 ``save_parameters`` 以及 ``load_parameters`` 可以进行 ``TorchModule`` 模型参数以字典形式保存到文件中,其中数值以 `numpy.ndarray` 保存。
或从文件中读取参数文件。但请注意,文件中不保存模型结构,需要用户手动构建模型结构。
你也可以直接使用 ``torch.save`` 以及 ``torch.load`` 去直接读取 ``torch`` 模型参数,因为 ``TorchModule`` 的参数是以 ``torch.Tensor`` 储存的。




经典神经网络模块
--------------------------------------------

以下经典神经网络模块均继承于继承于 ``pyvqnet.nn.Module`` 以及 ``torch.nn.Module``,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。
 

Linear
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Linear(input_channels, output_channels, weight_initializer=None, bias_initializer=None,use_bias=True, dtype=None, name: str = "")

    线性模块(全连接层)。
    :math:`y = x@A.T + b`
    
    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
    该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。
    

    :param input_channels: `int` - 输入数据通道数。
    :param output_channels: `int` - 输出数据通道数。
    :param weight_initializer: `callable` - 权重初始化函数,默认为空,使用he_uniform。
    :param bias_initializer: `callable` - 偏置初始化参数,默认为空,使用he_uniform。
    :param use_bias: `bool` - 是否使用偏置项, 默认使用。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 线性层的命名,默认为""。

    :return: 线性层实例。

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Linear
        pyvqnet.backends.set_backend("torch")
        c1 =2
        c2 = 3
        cin = 7
        cout = 5
        n = Linear(cin,cout)
        input = QTensor(np.arange(1,c1*c2*cin+1).reshape((c1,c2,cin)),requires_grad=True,dtype=pyvqnet.kfloat32)
        y = n.forward(input)
        print(y)

Conv1D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Conv1D(input_channels:int,output_channels:int,kernel_size:int ,stride:int= 1,padding = "valid",use_bias:bool = True,kernel_initializer = None,bias_initializer =None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    在输入上进行一维卷积运算。 Conv1D模块的输入具有形状(batch_size、input_channels、in_height)。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param input_channels: `int` - 输入数据的通道数。
    :param output_channels: `int` - 输出数据的通道数。
    :param kernel_size: `int` - 卷积核的尺寸. 卷积核形状 = [output_channels,input_channels/group,kernel_size,1]。
    :param stride: `int` - 步长, 默认为1。
    :param padding: `str|int` - 填充选项, 它可以是一个字符串 {'valid', 'same'} 或一个整数,给出应用在输入上的填充量。 默认 "valid"。
    :param use_bias: `bool` - 是否使用偏置项, 默认使用。
    :param kernel_initializer: `callable` - 卷积核初始化方法。默认为空,使用kaiming_uniform。
    :param bias_initializer: `callable` - 偏置初始化方法。默认为空,使用kaiming_uniform。
    :param dilation_rate: `int` - 空洞大小,默认:  1。
    :param group: `int` -  分组卷积的分组数. Default: 1。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 模块的名字,default:""。

    :return: 一维卷积实例。

    .. warning::

        ``padding='valid'`` 不进行填充。

        ``padding='same'`` 补零填充输入, 输出的out_height 为 = ceil(in_height / stride),不支持 stride>1 的情况。

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Conv1D
        pyvqnet.backends.set_backend("torch")
        b= 2
        ic =3
        oc = 2
        test_conv = Conv1D(ic,oc,3,2)
        x0 = QTensor(np.arange(1,b*ic*5*5 +1).reshape([b,ic,25]),requires_grad=True,dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)

Conv2D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Conv2D(input_channels:int,output_channels:int,kernel_size:tuple,stride:tuple=(1, 1),padding="valid",use_bias = True,kernel_initializer=None,bias_initializer=None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    在输入上进行二维卷积运算。 Conv2D模块的输入具有形状(batch_size, input_channels, height, width)。

    .. warning::
    
        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param input_channels: `int` - 输入数据的通道数。
    :param output_channels: `int` - 输出数据的通道数。
    :param kernel_size: `tuple|list` - 卷积核的尺寸. 卷积核形状 = [output_channels,input_channels/group,kernel_size,kernel_size]。
    :param stride: `tuple|list` - 步长, 默认为 (1, 1)|[1,1]。
    :param padding: `str|tuple` - 填充选项, 它可以是一个字符串 {'valid', 'same'} 或一个整数元组,给出在两边应用的隐式填充量。 默认 "valid"。
    :param use_bias: `bool` - 是否使用偏置项, 默认使用。
    :param kernel_initializer: `callable` - 卷积核初始化方法。默认为空,使用kaiming_uniform。
    :param bias_initializer: `callable` - 偏置初始化方法。默认为空,使用kaiming_uniform。
    :param dilation_rate: `int` - 空洞大小,默认:  1。
    :param group: `int` -  分组卷积的分组数. Default: 1。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 模块的名字,default:""。

    :return: 二维卷积实例。

    .. note::

        ``padding='valid'`` 不进行填充。
        ``padding='same'`` 补零填充输入, 输出的out_height 为 = ceil(in_height / stride),不支持 stride>1 的情况。

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Conv2D
        pyvqnet.backends.set_backend("torch")
        b= 2
        ic =3
        oc = 2
        test_conv = Conv2D(ic,oc,(3,3),(2,2))
        x0 = QTensor(np.arange(1,b*ic*5*5+1).reshape([b,ic,5,5]),requires_grad=True,dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)

ConvT2D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.ConvT2D(input_channels,output_channels,kernel_size,stride=[1, 1],padding=(0,0),use_bias="True", kernel_initializer=None,bias_initializer=None, dilation_rate: int = 1, out_padding = (0,0), group: int = 1, dtype = None, name = "")

    在输入上进行二维转置卷积运算。 Conv2D模块的输入具有形状(batch_size, input_channels, height, width)。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param input_channels: `int` - 输入数据的通道数。
    :param output_channels: `int` - 输出数据的通道数。
    :param kernel_size: `tuple|list` - 卷积核的尺寸,卷积核形状 = [input_channels,output_channels/group,kernel_size,kernel_size]。 
    :param stride: `tuple|list` - 步长, 默认为 (1, 1)|[1,1]。
    :param padding: `tuple` - 填充选项, 一个整数元组,给出在两边应用的隐式填充量。 默认 (0,0)。
    :param use_bias: `bool` - 是否使用偏置项, 默认使用。
    :param kernel_initializer: `callable` - 卷积核初始化方法。默认为空,使用kaiming_uniform。
    :param bias_initializer: `callable` - 偏置项初始化方法。默认为空,使用kaiming_uniform。
    :param dilation_rate: `int` - 空洞大小,默认:  1。
    :param out_padding: 在输出形状中每个维度的一侧添加的额外尺寸。默认值:(0,0)
    :param group: `int` -  分组卷积的分组数. Default: 1。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 模块的名字,default:""。

    :return: 二维转置卷积实例。
    
    .. note::

        ``padding='valid'`` 不进行填充。
        ``padding='same'`` 补零填充输入,输出的height 为 = ceil(height / stride)。

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import ConvT2D
        pyvqnet.backends.set_backend("torch")
        test_conv = ConvT2D(3, 2, (3, 3), (1, 1))
        x = QTensor(np.arange(1, 1 * 3 * 5 * 5+1).reshape([1, 3, 5, 5]), requires_grad=True,dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)

AvgPool1D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.AvgPool1D(kernel, stride, padding=0, name = "")

    对一维输入进行平均池化。输入具有形状(batch_size, input_channels, in_height)。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param kernel: 平均池化的窗口大小。
    :param stride: 窗口移动的步长。
    :param padding: 填充选项, 整数指定填充长度。 默认 0。
    :param name: 模块的名字,default:""。

    :return: 一维平均池化层实例。

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import AvgPool1D
        pyvqnet.backends.set_backend("torch")
        test_mp = AvgPool1D([3],[2],0)
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)



MaxPool1D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.MaxPool1D(kernel, stride, padding=0,name="")

    对一维输入进行最大池化。输入具有形状(batch_size, input_channels, in_height)。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param kernel: 最大池化的窗口大小。
    :param stride: 窗口移动的步长。
    :param padding: 填充选项,整数指定填充长度。 默认 0。
    :param name: 命名,默认为""。

    :return: 一维最大池化层实例。


    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import MaxPool1D
        pyvqnet.backends.set_backend("torch")
        test_mp = MaxPool1D([3],[2],0)
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)


AvgPool2D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.AvgPool2D( kernel, stride, padding=(0,0),name="")

    对二维输入进行平均池化。输入具有形状(batch_size, input_channels, height, width)。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param kernel: 平均池化的窗口大小。
    :param stride: 窗口移动的步长。
    :param padding: 填充选项, 包含2个整数的元组,整数为两个维度上的填充长度。 默认:(0,0)。
    :param name: 命名,默认为""。

    :return: 二维平均池化层实例。


    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import AvgPool2D
        pyvqnet.backends.set_backend("torch")
        test_mp = AvgPool2D([2,2],[2,2],1)
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
 

MaxPool2D
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.MaxPool2D(kernel, stride, padding=(0,0),name="")

    对二维输入进行最大池化。输入具有形状(batch_size, input_channels, height, width)。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param kernel: 最大池化的窗口大小。
    :param stride: 窗口移动的步长。
    :param padding: 填充选项, 包含2个整数的元组,整数为两个维度上的填充长度。 默认: (0,0)。
    :param name: 命名,默认为""。

    :return: 二维最大池化层实例。



    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import MaxPool2D
        pyvqnet.backends.set_backend("torch")
        test_mp = MaxPool2D([2,2],[2,2],(0,0))
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)


Embedding
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Embedding(num_embeddings, embedding_dim, weight_initializer=xavier_normal, dtype=None, name: str = "")

    该模块通常用于存储词嵌入并使用索引检索它们。模块的输入是索引列表,输出是对应的词嵌入。
    该层的输入应该是kint64。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param num_embeddings: `int` - 嵌入字典的大小。
    :param embedding_dim: `int` - 每个嵌入向量的大小
    :param weight_initializer: `callable` - 参数初始化方式,默认正态分布。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 嵌入层的命名,默认为""。

    :return: a Embedding 实例。

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import Embedding
        pyvqnet.backends.set_backend("torch")
        vlayer = Embedding(30,3)
        x = QTensor(np.arange(1,25).reshape([2,3,2,2]),dtype= pyvqnet.kint64)
        y = vlayer(x)
        print(y)



BatchNorm2d
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.BatchNorm2d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5,affine = True, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")
    
    在 4D 输入(B、C、H、W)上应用批归一化。参照论文
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ 。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    其中 :math:`\gamma` 和 :math:`\beta` 为待训练参数。此外,默认情况下,在训练期间,该层会继续运行估计其计算的均值和方差,然后在评估期间用于归一化。平均方差均值保持默认动量 0.1。

    :param channel_num: `int` - 输入通道数。
    :param momentum: `float` - 计算指数加权平均时的动量,默认为 0.1。
    :param epsilon: `float` - 数值稳定参数, 默认 1e-5。
    :param affine: `bool` - 一个布尔值,当设置为 ``True`` 时,此模块具有可学习的每通道仿射参数,初始化为 1(用于权重)和 0(用于偏差)。默认值:``True``。
    :param beta_initializer: `callable` - beta的初始化方式,默认全零初始化。
    :param gamma_initializer: `callable` - gamma的的初始化方式,默认全一初始化。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 批归一化层命名,默认为""。

    :return: 二维批归一化层实例。

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import BatchNorm2d
        pyvqnet.backends.set_backend("torch")
        b = 2
        ic = 2
        test_conv = BatchNorm2d(ic)

        x = QTensor(np.arange(1, 17).reshape([b, ic, 4, 1]),
                    requires_grad=True,
                    dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)


BatchNorm1d
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.BatchNorm1d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5, affine = True, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")

    在 2D 输入 (B,C) 上进行批归一化操作。 参照论文
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ 。

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    其中 :math:`\gamma` 和 :math:`\beta` 为待训练参数。此外,默认情况下,在训练期间,该层会继续运行估计其计算的均值和方差,然后在评估期间用于归一化。平均方差均值保持默认动量 0.1。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param channel_num: `int` - 输入通道数。
    :param momentum: `float` - 计算指数加权平均时的动量,默认为 0.1。
    :param epsilon: `float` - 数值稳定性常数,默认为 1e-5。
    :param affine: `bool` - 一个布尔值,当设置为 ``True`` 时,此模块具有可学习的每通道仿射参数,初始化为 1(用于权重)和 0(用于偏差)。默认值:``True``。
    :param beta_initializer: `callable` - beta的初始化方式,默认全零初始化。
    :param gamma_initializer: `callable` - gamma的的初始化方式,默认全一初始化。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 批归一化层命名,默认为""。

    :return: 一维批归一化层实例。

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import BatchNorm1d
        pyvqnet.backends.set_backend("torch")
        test_conv = BatchNorm1d(4)

        x = QTensor(np.arange(1, 17).reshape([4, 4]),
                    requires_grad=True,
                    dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)


LayerNormNd
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.nn.torch.LayerNormNd(normalized_shape: list, epsilon: float = 1e-5,affine=True, dtype=None, name="")

    在任意输入的后D个维度上进行层归一化。具体方式如论文所述:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    对于像 (B,C,H,W,D) 这样的输入, ``norm_shape`` 可以是 [C,H,W,D],[H,W,D],[W,D] 或 [D] .

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param norm_shape: `float` - 标准化形状。
    :param epsilon: `float` - 数值稳定性常数,默认为 1e-5。
    :param affine: `bool` - 一个布尔值,当设置为 ``True`` 时,此模块具有可学习的每通道仿射参数,初始化为 1(用于权重)和 0(用于偏差)。默认值:``True``。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字, 默认为""。

    :return: 一个 LayerNormNd 类

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32
        from pyvqnet.nn.torch import LayerNormNd
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        ic = 4
        test_conv = LayerNormNd([2,2])
        x = QTensor(np.arange(1,17).reshape([2,2,2,2]),requires_grad=True,dtype=kfloat32)
        y = test_conv.forward(x)
        print(y)
         

LayerNorm2d
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.nn.torch.LayerNorm2d(norm_size:int, epsilon:float = 1e-5, affine=True, dtype=None, name="")

    在 4D 输入上进行层归一化。具体方式如论文所述:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    平均值和标准差是在除去第一个维度以外的剩余维度数据上计算的。对于像 (B,C,H,W) 这样的输入, ``norm_size`` 应该等于 C * H * W。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param norm_size: `float` - 归一化大小,应该等于 C * H * W。
    :param epsilon: `float` - 数值稳定性常数,默认为 1e-5。
    :param affine: `bool` - 一个布尔值,当设置为 ``True`` 时,此模块具有可学习的每通道仿射参数,初始化为 1(用于权重)和 0(用于偏差)。默认值:``True``。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字, 默认为""。

    :return: 二维层归一化实例。

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import LayerNorm2d
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        ic = 4
        test_conv = LayerNorm2d(8)
        x = QTensor(np.arange(1,17).reshape([2,2,4,1]),requires_grad=True,dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)



LayerNorm1d
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.nn.torch.LayerNorm1d(norm_size:int, epsilon:float = 1e-5, affine=True, dtype=None, name="")
    
    在 2D 输入上进行层归一化。具体方式如论文所述:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    均值和标准差是在最后一个维度大小上计算的,其中“norm_size” 是 ``norm_size`` 的值。


    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param norm_size: `float` - 归一化大小,应该等于最后一维大小。
    :param epsilon: `float` - 数值稳定性常数,默认为 1e-5。
    :param affine: `bool` - 一个布尔值,当设置为 ``True`` 时,此模块具有可学习的每通道仿射参数,初始化为 1(用于权重)和 0(用于偏差)。默认值:``True``。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字, 默认为""。

    :return: 一维层归一化实例。

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import LayerNorm1d
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        test_conv = LayerNorm1d(4)
        x = QTensor(np.arange(1,17).reshape([4,4]),requires_grad=True,dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)



GroupNorm
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.nn.torch.GroupNorm(num_groups: int, num_channels: int, epsilon = 1e-5, affine = True, dtype = None, name = "")

    对小批量输入应用组归一化。输入: :math:`(N, C, *)` 其中 :math:`C=\mathrm{num\_channels}` , 输出: :math:`(N, C, *)` 。

    此层实现论文 `组归一化 <https://arxiv.org/abs/1803.08494>`__ 中描述的操作。

    .. math::
        
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    输入通道被分成 :attr:`num_groups` 组,每组包含 ``num_channels / num_groups`` 个通道。:attr:`num_channels` 必须能被 :attr:`num_groups` 整除。平均值和标准差是在每个组中分别计算的。如果 :attr:`affine` 为 ``True``,则 :math:`\gamma` 和 :math:`\beta` 是可学习的。每个通道仿射变换参数向量,大小为 :attr:`num_channels`。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。

    :param num_groups (int): 将通道分成的组数
    :param num_channels (int): 输入中预期的通道数
    :param eps: 添加到分母的值,以实现数值稳定性。默认值:1e-5
    :param affine: 一个布尔值,当设置为 ``True`` 时,此模块具有可学习的每通道仿射参数,初始化为 1(用于权重)和 0(用于偏差)。默认值: ``True``。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字, 默认为""。

    :return: GroupNorm 类对象

    Example::

        import numpy as np
        from pyvqnet.tensor import QTensor,kfloat32
        from pyvqnet.nn.torch import GroupNorm
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        test_conv = GroupNorm(2,10)
        x = QTensor(np.arange(0,60*2*5).reshape([2,10,3,2,5]),requires_grad=True,dtype=kfloat32)
        y = test_conv.forward(x)
        print(y)

Dropout
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.nn.torch.Dropout(dropout_rate = 0.5)

    Dropout 模块。dropout 模块将一些单元的输出随机设置为零,同时根据给定的 dropout_rate 概率升级其他单元。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param dropout_rate: `float` - 神经元被设置为零的概率。
    :param name: 这个模块的名字, 默认为""。

    :return: Dropout实例。

    Example::

        import numpy as np
        from pyvqnet.nn.torch import Dropout
        from pyvqnet.tensor import arange
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        b = 2
        ic = 2
        x = arange(-1 * ic * 2 * 2.0,
                            (b - 1) * ic * 2 * 2).reshape([b, ic, 2, 2])
        droplayer = Dropout(0.5)
        droplayer.train()
        y = droplayer(x)
        print(y)



DropPath
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.DropPath(dropout_rate = 0.5,name="")

    DropPath 模块将逐样本丢弃路径(随机深度)。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param dropout_rate: `float` - 神经元被设置为零的概率。
    :param name: 这个模块的名字, 默认为""。

    :return: DropPath实例。

    Example::

        import pyvqnet.nn.torch as nn
        import pyvqnet.tensor as tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = tensor.randu([4])
        y = nn.DropPath()(x)
        print(y)
        #[0.2008128,0.3908308,0.7102265,0.3784221]

Pixel_Shuffle 
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Pixel_Shuffle(upscale_factors, name="")

    重新排列形状为:(*, C * r^2, H, W)  的张量
    到形状为 (*, C, H * r, W * r) 的张量,其中 r 是尺度变换因子。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param upscale_factors: 增加尺度变换的因子
    :param name: 这个模块的名字, 默认为""。

    :return:
            Pixel_Shuffle 模块

    Example::

        from pyvqnet.nn.torch import Pixel_Shuffle
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        ps = Pixel_Shuffle(3)
        inx = tensor.ones([5,2,3,18,4,4])
        inx.requires_grad = True
        y = ps(inx)


Pixel_Unshuffle 
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Pixel_Unshuffle(downscale_factors, name="")

    通过重新排列元素来反转 Pixel_Shuffle 操作. 将 (*, C, H * r, W * r) 形状的张量变化为 (*, C * r^2, H, W) ,其中 r 是缩小因子。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param downscale_factors: 增加尺度变换的因子
    :param name: 这个模块的名字, 默认为""。

    :return:
            Pixel_Unshuffle 模块

    Example::

        from pyvqnet.nn.torch import Pixel_Unshuffle
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        ps = Pixel_Unshuffle(3)
        inx = tensor.ones([5, 2, 3, 2, 12, 12])
        inx.requires_grad = True
        y = ps(inx)



GRU
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.GRU(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    门控循环单元 (GRU) 模块。支持多层堆叠,双向配置。单层单向GRU的计算公式如下:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。


    :param input_size: 输入特征维度。
    :param hidden_size:  隐藏特征维度。
    :param num_layers: 堆叠GRU层数, 默认: 1。
    :param batch_first: 如果为 True, 则输入形状为 [batch_size,seq_len,feature_dim],
     如果为 False, 则输入形状为 [seq_len,batch_size,feature_dim],默认为 True。
    :param use_bias: 如果为 False,该模块不适用偏置项,默认: True。
    :param bidirectional: 如果为 True, 变为双向GRU, 默认: False。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字, 默认为""。

    :return: GRU 实例

    Example::
        
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import GRU
        from pyvqnet.tensor import tensor

        rnn2 = GRU(4, 6, 2, batch_first=False, bidirectional=True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])

        output, hn = rnn2(input, h0)


RNN 
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    循环神经网络(RNN)模块,使用 :math:`\tanh` 或 :math:`\text{ReLU}` 作为激活函数。支持双向,多层配置。
    单层单向RNN计算公式如下:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    如果 :attr:`nonlinearity` 是 ``'relu'``, 则 :math:`\text{ReLU}` 将替代 :math:`\tanh`。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。


    :param input_size: 输入特征维度。
    :param hidden_size:  隐藏特征维度。
    :param num_layers: 堆叠RNN层数, 默认: 1。
    :param nonlinearity: 非线性激活函数,默认为 ``'tanh'``。
    :param batch_first: 如果为 True, 则输入形状为 [batch_size,seq_len,feature_dim],
     如果为 False, 则输入形状为 [seq_len,batch_size,feature_dim],默认为 True。
    :param use_bias: 如果为 False, 该模块不适用偏置项,默认: True。
    :param bidirectional: 如果为 True,变为双向RNN,默认: False。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字, 默认为""。

    :return: RNN 实例

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import RNN
        from pyvqnet.tensor import tensor

        rnn2 = RNN(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        output, hn = rnn2(input, h0)




LSTM
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    长短期记忆(LSTM)模块。支持双向LSTM, 堆叠多层LSTM等配置。单层单向LSTM计算公式如下:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。


    :param input_size: 输入特征维度。
    :param hidden_size:  隐藏特征维度。
    :param num_layers: 堆叠LSTM层数,默认: 1。
    :param batch_first: 如果为 True,则输入形状为 [batch_size,seq_len,feature_dim],
     如果为 False, 则输入形状为 [seq_len,batch_size,feature_dim],默认为 True。
    :param use_bias: 如果为 False,该模块不适用偏置项, 默认: True。
    :param bidirectional: 如果为 True,变为双向LSTM, 默认: False。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字, 默认为""。

    :return: LSTM 实例

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import LSTM
        from pyvqnet.tensor import tensor

        rnn2 = LSTM(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        c0 = tensor.ones([4, 3, 6])
        output, (hn, cn) = rnn2(input, (h0, c0))


Dynamic_GRU
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Dynamic_GRU(input_size,hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    将多层门控循环单元 (GRU) RNN 应用于动态长度输入序列。

    第一个输入应该是定义了可变长度的批处理序列输入
    通过 ``tensor.PackedSequence`` 类。
    ``tensor.PackedSequence`` 类可以构造为
    连续调用下一个函数: ``pad_sequence`` 、 ``pack_pad_sequence``。

    Dynamic_GRU 的第一个输出也是一个 ``tensor.PackedSequence`` 类,
    可以使用 ``tensor.pad_pack_sequence`` 将其解压缩为普通 QTensor。

    对于输入序列中的每个元素,每一层计算以下公式:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。


    :param input_size: 输入特征维度。
    :param hidden_size: 隐藏的特征维度。
    :param num_layers: 循环层数。 默认值:1
    :param batch_first: 如果为 True,输入形状提供为 [批大小,序列长度,特征维度]。如果为 False,输入形状提供为 [序列长度,批大小,特征维度],默认为 True。
    :param use_bias: 如果为False,则该层不使用偏置权重b_ih和b_hh。 默认值:True。
    :param bidirectional: 如果为真,则成为双向 GRU。 默认值:False。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字, 默认为""。

    :return: 一个 Dynamic_GRU 类

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import Dynamic_GRU
        from pyvqnet.tensor import tensor
        seq_len = [4,1,2]
        input_size = 4
        batch_size =3
        hidden_size = 2
        ml = 2
        rnn2 = Dynamic_GRU(input_size,
                        hidden_size=2,
                        num_layers=2,
                        batch_first=False,
                        bidirectional=True)

        a = tensor.arange(1, seq_len[0] * input_size + 1).reshape(
            [seq_len[0], input_size])
        b = tensor.arange(1, seq_len[1] * input_size + 1).reshape(
            [seq_len[1], input_size])
        c = tensor.arange(1, seq_len[2] * input_size + 1).reshape(
            [seq_len[2], input_size])

        y = tensor.pad_sequence([a, b, c], False)

        input = tensor.pack_pad_sequence(y,
                                        seq_len,
                                        batch_first=False,
                                        enforce_sorted=False)

        h0 = tensor.ones([ml * 2, batch_size, hidden_size])

        output, hn = rnn2(input, h0)

        seq_unpacked, lens_unpacked = \
        tensor.pad_packed_sequence(output, batch_first=False)

Dynamic_RNN 
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Dynamic_RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    将循环神经网络 RNN 应用于动态长度输入序列。

    第一个输入应该是定义了可变长度的批处理序列输入
    通过 ``tensor.PackedSequence`` 类。
    ``tensor.PackedSequence`` 类可以构造为
    连续调用下一个函数: ``pad_sequence`` 、 ``pack_pad_sequence``。

    Dynamic_RNN 的第一个输出也是一个 ``tensor.PackedSequence`` 类,
    可以使用 ``tensor.pad_pack_sequence`` 将其解压缩为普通 QTensor。

    循环神经网络(RNN)模块,使用 :math:`\tanh` 或 :math:`\text{ReLU}` 作为激活函数。支持双向,多层配置。
    单层单向RNN计算公式如下:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    如果 :attr:`nonlinearity` 是 ``'relu'``, 则 :math:`\text{ReLU}` 将替代 :math:`\tanh`。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。


    :param input_size: 输入特征维度。
    :param hidden_size:  隐藏特征维度。
    :param num_layers: 堆叠RNN层数, 默认: 1。
    :param nonlinearity: 非线性激活函数,默认为 ``'tanh'``。
    :param batch_first: 如果为 True, 则输入形状为 [批大小,序列长度,特征维度],
     如果为 False, 则输入形状为 [序列长度,批大小,特征维度],默认为 True。
    :param use_bias: 如果为 False, 该模块不适用偏置项,默认: True。
    :param bidirectional: 如果为 True,变为双向RNN,默认: False。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字, 默认为""。

    :return: Dynamic_RNN 实例

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import Dynamic_RNN
        from pyvqnet.tensor import tensor
        seq_len = [4,1,2]
        input_size = 4
        batch_size =3
        hidden_size = 2
        ml = 2
        rnn2 = Dynamic_RNN(input_size,
                        hidden_size=2,
                        num_layers=2,
                        batch_first=False,
                        bidirectional=True,
                        nonlinearity='relu')

        a = tensor.arange(1, seq_len[0] * input_size + 1).reshape(
            [seq_len[0], input_size])
        b = tensor.arange(1, seq_len[1] * input_size + 1).reshape(
            [seq_len[1], input_size])
        c = tensor.arange(1, seq_len[2] * input_size + 1).reshape(
            [seq_len[2], input_size])

        y = tensor.pad_sequence([a, b, c], False)

        input = tensor.pack_pad_sequence(y,
                                        seq_len,
                                        batch_first=False,
                                        enforce_sorted=False)

        h0 = tensor.ones([ml * 2, batch_size, hidden_size])

        output, hn = rnn2(input, h0)

        seq_unpacked, lens_unpacked = \
        tensor.pad_packed_sequence(output, batch_first=False)




Dynamic_LSTM
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Dynamic_LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    将长短期记忆(LSTM) RNN 应用于动态长度输入序列。

    第一个输入应该是定义了可变长度的批处理序列输入
    通过 ``tensor.PackedSequence`` 类。
    ``tensor.PackedSequence`` 类可以构造为
    连续调用下一个函数: ``pad_sequence`` 、 ``pack_pad_sequence``。

    Dynamic_LSTM 的第一个输出也是一个 ``tensor.PackedSequence`` 类,
    可以使用 ``tensor.pad_pack_sequence`` 将其解压缩为普通 QTensor。

    循环神经网络(RNN)模块,使用 :math:`\tanh` 或 :math:`\text{ReLU}` 作为激活函数。支持双向,多层配置。
    单层单向RNN计算公式如下:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。


    :param input_size: 输入特征维度。
    :param hidden_size:  隐藏特征维度。
    :param num_layers: 堆叠LSTM层数,默认: 1。
    :param batch_first: 如果为 True,则输入形状为 [批大小,序列长度,特征维度],
     如果为 False, 则输入形状为 [序列长度,批大小,特征维度],默认为 True。
    :param use_bias: 如果为 False,该模块不适用偏置项, 默认: True。
    :param bidirectional: 如果为 True,变为双向LSTM, 默认: False。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字, 默认为""。

    :return: Dynamic_LSTM 实例

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.nn.torch import Dynamic_LSTM
        from pyvqnet.tensor import tensor

        input_size = 2
        hidden_size = 2
        ml = 2
        seq_len = [3, 4, 1]
        batch_size = 3
        rnn2 = Dynamic_LSTM(input_size,
                            hidden_size=hidden_size,
                            num_layers=ml,
                            batch_first=False,
                            bidirectional=True)

        a = tensor.arange(1, seq_len[0] * input_size + 1).reshape(
            [seq_len[0], input_size])
        b = tensor.arange(1, seq_len[1] * input_size + 1).reshape(
            [seq_len[1], input_size])
        c = tensor.arange(1, seq_len[2] * input_size + 1).reshape(
            [seq_len[2], input_size])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], False)

        input = tensor.pack_pad_sequence(y,
                                        seq_len,
                                        batch_first=False,
                                        enforce_sorted=False)

        h0 = tensor.ones([ml * 2, batch_size, hidden_size])
        c0 = tensor.ones([ml * 2, batch_size, hidden_size])

        output, (hn, cn) = rnn2(input, (h0, c0))

        seq_unpacked, lens_unpacked = \
        tensor.pad_packed_sequence(output, batch_first=False)

 


Interpolate
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Interpolate(size = None, scale_factor = None, mode = "nearest", align_corners = None,  recompute_scale_factor = None, name = "")

    向下/向上对输入进行采样。

    目前只支持四维输入数据。

    输入尺寸的解释形式为 `B x C x H x W`。

    可用于选择的 `mode` 有 ``nearest`` 、``bilinear`` 、``bicubic``.

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param size: 输出大小,默认为None。
    :param scale_factor: 缩放因子,默认为None。
    :param mode: 用于上采样的算法  ``nearest`` | ``bilinear`` | ``bicubic``.
    :param align_corners:  从几何学角度看,我们将输入和输出的像素点视为方形而不是点。输入和输出的像素点视为正方形,而不是点。
            如果设置为 `true`,输入和输出张量将根据其角像素的中心点对齐。角像素的中心点对齐,保留角像素的值。
            如果设置为 `false`,输入和输出张量将按其角像素的角点对齐,而角像素的值将保留。角像素的角点对齐,插值会使用边缘值填充
            对超出边界的值进行填充,从而使此操作与输入大小无关。
            当 ``scale_factor`` 保持不变时。这只有在 ``mode`` 为 ``bilinear`` 时才有效。
    :param recompute_scale_factor: 重新计算缩放因子,以便在插值计算中使用。 当 ``scale_factor`` 作为参数传递时,它将用于来计算输出尺寸。
    :param name: 模块名字.

    Example::

        from pyvqnet.nn.torch import Interpolate
        from pyvqnet.tensor import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(1)

        mode_ = "bilinear"
        size_ = 3

        model = Interpolate(size=size_, mode=mode_)
        input_vqnet = tensor.randu((1, 1, 6, 6),
                                dtype=pyvqnet.kfloat32,
                                requires_grad=True)
        output_vqnet = model(input_vqnet)

SDPA
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.SDPA(attn_mask=None,dropout_p=0.,scale=None,is_causal=False)

    构造计算查询、键和值张量的缩放点积注意力的类。
    
    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param attn_mask: 注意掩码；默认值: 无。shape 必须可广播到注意权重的形状。
    :param dropout_p: Dropout 概率；默认值: 0,如果大于 0.0,则应用 dropout。
    :param scale: 在 softmax 之前应用的缩放因子,默认值: 无。
    :param is_causal: 默认值: False,如果设置为 true,则当掩码为方阵时,注意力掩码为下三角矩阵。如果同时设置了 attn_mask 和 is_causal,则会引发错误。
    :return: 一个SDPA类

    Example::
    
        from pyvqnet.nn.torch import SDPA
        from pyvqnet import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        model = SDPA(tensor.QTensor([1.]))

    .. py:method:: forward(query,key,value)

        进行前向计算

        :param query: query输入QTensor。
        :param key: key输入QTensor。
        :param value: key输入QTensor。
        :return: SDPA计算返回的QTensor。

        Example::
        
            from pyvqnet.nn.torch import SDPA
            from pyvqnet import tensor
            import pyvqnet
            pyvqnet.backends.set_backend("torch")

            import numpy as np

            model = SDPA(tensor.QTensor([1.]))

            query_np = np.random.randn(3, 3, 3, 5).astype(np.float32) 
            key_np = np.random.randn(3, 3, 3, 5).astype(np.float32)   
            value_np = np.random.randn(3, 3, 3, 5).astype(np.float32) 

            query_p = tensor.QTensor(query_np, dtype=pyvqnet.kfloat32, requires_grad=True)
            key_p = tensor.QTensor(key_np, dtype=pyvqnet.kfloat32, requires_grad=True)
            value_p = tensor.QTensor(value_np, dtype=pyvqnet.kfloat32, requires_grad=True)

            out_sdpa = model(query_p, key_p, value_p)

            out_sdpa.backward(pyvqnet.tensor.ones_like(out_sdpa))

损失函数接口
------------------------

MeanSquaredError
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.MeanSquaredError(name="")

    计算输入 :math:`x` 和目标值 :math:`y` 之间的均方根误差。

    若平方根误差可由如下函数描述:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    :math:`x` 和 :math:`y` 是任意形状的 QTensor , 总 :math:`n` 个元素的均方根误差由下式计算。

    .. math::
        \ell(x, y) =
            \operatorname{mean}(L)

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param name: 这个模块的名字, 默认为""。
    :return: 一个均方根误差实例。

    均方根误差前向计算函数的所需参数:

        x: :math:`(N, *)` 预测值,其中 :math:`*` 表示任意维度。

        y: :math:`(N, *)`, 目标值, 和输入一样维度的 QTensor 。


    .. note::

            请注意,跟 PyTorch 等框架不同的是,以下MeanSquaredError函数的前向函数中,第一个参数为目标值,第二个参数为预测值。


    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        from pyvqnet.nn.torch import MeanSquaredError
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        y = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
                    requires_grad=False,
                    dtype=kfloat64)
        x = QTensor([[0.1, 0.05, 0.7, 0, 0.05, 0.1, 0, 0, 0, 0]],
                    requires_grad=True,
                    dtype=kfloat64)

        loss_result = MeanSquaredError()
        result = loss_result(y, x)
        print(result)



BinaryCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.BinaryCrossEntropy(name="")

    测量目标和输入之间的平均二元交叉熵损失。

    未做平均运算的二元交叉熵如下式:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    若 :math:`N` 为批的大小,则平均二元交叉熵.

    .. math::
        \ell(x, y) = \operatorname{mean}(L)
    
    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param name: 这个模块的名字, 默认为""。
    :return: 一个平均二元交叉熵实例。

    平均二元交叉熵误差前向计算函数的所需参数:

        x: :math:`(N, *)` 预测值,其中 :math:`*` 表示任意维度。

        y: :math:`(N, *)`, 目标值,和输入一样维度的 QTensor 。

    .. note::

            请注意,跟 PyTorch 等框架不同的是,BinaryCrossEntropy函数的前向函数中,第一个参数为目标值,第二个参数为预测值。



    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import BinaryCrossEntropy
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[0.3, 0.7, 0.2], [0.2, 0.3, 0.1]], requires_grad=True)
        y = QTensor([[0.0, 1.0, 0], [0.0, 0, 1]], requires_grad=False)

        loss_result = BinaryCrossEntropy()
        result = loss_result(y, x)
        result.backward(pyvqnet.tensor.ones_like(result))
        print(result)


CategoricalCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.CategoricalCrossEntropy(name="")

    该损失函数将 LogSoftmax 和 NLLLoss 同时计算的平均分类交叉熵。

    损失函数计算方式如下,其中 class 为目标值的对应分类标签:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: 这个模块的名字, 默认为""。
    :return: 平均分类交叉熵实例。

    误差前向计算函数的所需参数:

        x: :math:`(N, *)` 预测值,其中 :math:`*` 表示任意维度。

        y: :math:`(N, *)`, 目标值,和输入一样维度的 QTensor 。必须为64位整数,kint64。

    .. note::

            请注意,跟 PyTorch 等框架不同的是,CategoricalCrossEntropy函数的前向函数中,第一个参数为目标值,第二个参数为预测值。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32,kint64
        from pyvqnet.nn.torch import CategoricalCrossEntropy
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]], requires_grad=True,dtype=kfloat32)
        y = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], requires_grad=False,dtype=kint64)
        loss_result = CategoricalCrossEntropy()
        result = loss_result(y, x)
        print(result)



SoftmaxCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.SoftmaxCrossEntropy(name="")

    该损失函数将 LogSoftmax 和 NLLLoss 同时计算的平均分类交叉熵,并具有更高的数值稳定性。

    损失函数计算方式如下,其中 class 为目标值的对应分类标签:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: 这个模块的名字, 默认为""。
    :return: 一个Softmax交叉熵损失函数实例

    误差前向计算函数的所需参数:

        x: :math:`(N, *)` 预测值,其中 :math:`*` 表示任意维度。

        y: :math:`(N, *)`, 目标值,和输入一样维度的 QTensor 。必须为64位整数,kint64。

    .. note::

            请注意,跟 PyTorch 等框架不同的是,SoftmaxCrossEntropy函数的前向函数中,第一个参数为目标值,第二个参数为预测值。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat32, kint64
        from pyvqnet.nn.torch import SoftmaxCrossEntropy
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                    requires_grad=True,
                    dtype=kfloat32)
        y = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
                    requires_grad=False,
                    dtype=kint64)
        loss_result = SoftmaxCrossEntropy()
        result = loss_result(y, x)
        result.backward(pyvqnet.tensor.ones_like(result))
        print(result)



NLL_Loss
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.NLL_Loss(name="")

    平均负对数似然损失。 对C个类别的分类问题很有用。

    `x` 是模型给出的概率形式的似然量。其尺寸可以是 :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` 。 `y` 是损失函数期望的真值,包含 :math:`[0, C-1]` 的类别索引。

    .. math::

        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = -  
            \sum_{n=1}^N \frac{1}{N}x_{n,y_n} \quad

    :param name: 这个模块的名字, 默认为""。
    :return: 一个NLL_Loss损失函数实例

    误差前向计算函数的所需参数:

        x: :math:`(N, *)`,损失函数的输出预测值,可以为多维变量。

        y: :math:`(N, *)`,损失函数目标值。必须为64位整数,kint64。

    .. note::

        请注意,跟 PyTorch 等框架不同的是,NLL_Loss函数的前向函数中,第一个参数为目标值,第二个参数为预测值。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kint64
        from pyvqnet.nn.torch import NLL_Loss
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ])
        x=x.reshape([1, 3, 1, 5])
        x.requires_grad = True
        y = QTensor([[[2, 1, 0, 0, 2]]], dtype=kint64)

        loss_result = NLL_Loss()
        result = loss_result(y, x)
        print(result)


CrossEntropyLoss
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.CrossEntropyLoss(name="")

    该函数计算LogSoftmax以及NLL_Loss在一起的损失。

    `x` 是包含未做归一化的输出.它的尺寸可以为 :math:`(C)` , :math:`(N, C)` 二维或 :math:`(N, C, d_1, d_2, ..., d_K)` 多维。

    损失函数的公式如下,其中 class 为目标值的对应分类标签:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: 这个模块的名字, 默认为""。
    :return: 一个CrossEntropyLoss损失函数实例

    误差前向计算函数的所需参数:

        x: :math:`(N, *)`,损失函数的输出,可以为多维变量。

        y: :math:`(N, *)`,损失函数期望的真值。必须为64位整数,kint64。

    .. note::

            请注意,跟 PyTorch 等框架不同的是,CrossEntropyLoss函数的前向函数中,第一个参数为目标值,第二个参数为预测值。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet import kint64
        from pyvqnet.nn.torch import CrossEntropyLoss
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ])
        x=x.reshape([1, 3, 1, 5])
        x.requires_grad = True
        y = QTensor([[[2, 1, 0, 0, 2]]], dtype=kint64)

        loss_result = CrossEntropyLoss()
        result = loss_result(y, x)
        print(result)


激活函数
---------------------

Sigmoid
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Sigmoid(name:str="")

    Sigmoid激活函数层。

    .. math::
        \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}
    
    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param name: 激活函数层的命名,默认为""。

    :return: 一个Sigmoid激活函数层实例。

    Example::

        from pyvqnet.nn.torch import Sigmoid
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Sigmoid()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)


Softplus
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Softplus(name:str="")

    Softplus激活函数层。

    .. math::
        \text{Softplus}(x) = \log(1 + \exp(x))

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param name: 激活函数层的命名,默认为""。

    :return: 一个Softplus激活函数层实例。

    Example::

        from pyvqnet.nn.torch import Softplus
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Softplus()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))

Softsign
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Softsign(name:str="")

    Softsign 激活函数层。

    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}


    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param name: 激活函数层的命名,默认为""。

    :return: 一个Softsign 激活函数层实例。

    Example::

        from pyvqnet.nn.torch import Softsign
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Softsign()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))



Softmax
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Softmax(dim:int = -1,name:str="")

    Softmax 激活函数层。

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}


    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param dim: 计算的维度(最后一个轴为-1),默认值 = -1。
    :param name: 激活函数层的命名,默认为""。

    :return: 一个Softmax 激活函数层实例。

    Example::

        from pyvqnet.nn.torch import Softmax
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Softmax()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))


HardSigmoid
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.HardSigmoid(name:str="")

    HardSigmoid 激活函数层。

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{ if } x \le -3, \\
            1 & \text{ if } x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}


    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param name: 激活函数层的命名,默认为""。

    :return: 一个HardSigmoid 激活函数层实例。

    Example::

        from pyvqnet.nn.torch import HardSigmoid
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = HardSigmoid()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))


ReLu
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.ReLu(name:str="")

    ReLu 整流线性单元激活函数层。

    .. math::
        \text{ReLu}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        0, & \text{ if } x \leq 0
        \end{cases}


    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param name: 激活函数层的命名,默认为""。

    :return: 一个ReLu 激活函数层实例。

    Example::

        from pyvqnet.nn.torch import ReLu
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = ReLu()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))

        


LeakyReLu
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.LeakyReLu(alpha:float=0.01,name:str="")

    LeakyReLu 带泄露的修正线性单元激活函数层。

    .. math::
        \text{LeakyRelu}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \alpha * x, & \text{ otherwise }
        \end{cases}


    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param alpha: LeakyRelu 系数,默认:0.01。
    :param name: 激活函数层的命名,默认为""。

    :return: 一个LeakyReLu 激活函数层实例。

    Example::

        from pyvqnet.nn.torch import LeakyReLu
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = LeakyReLu()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))



Gelu
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Gelu(approximate="tanh", name="")
    
    应用高斯误差线性单元函数:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    当近似参数为 'tanh' 时, GELU 通过以下方式估计:

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))


    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param approximate: 近似计算方式,默认为"tanh"。
    :param name: 激活函数层的命名,默认为""。

    :return: Gelu 激活函数层实例。

    Example::

        from pyvqnet.tensor import randu, ones_like
        from pyvqnet.nn.torch import Gelu
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        qa = randu([5,4])
        qb = Gelu()(qa)



ELU
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.ELU(alpha:float=1,name:str="")

    ELU 指数线性单位激活函数层。

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}


    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param alpha: ELU 系数,默认:1。
    :param name: 激活函数层的命名,默认为""。

    :return: ELU 激活函数层实例。

    Example::

        from pyvqnet.nn.torch import ELU
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = ELU()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))


Tanh
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Tanh(name:str="")

    Tanh双曲正切激活函数.

    .. math::
        \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}


    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param name: 激活函数层的命名,默认为""。

    :return: Tanh 激活函数层实例。

    Example::

        from pyvqnet.nn.torch import Tanh
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Tanh()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))


以下损失函数用于基于强化学习/偏好的大模型微调(RLHF/DPO/PPO/GRPO/SFT)。

sft_loss
-----------------------------

.. py:function:: pyvqnet.torch.trl.sft_loss(model, input_ids, labels, ignore_index=-100)

    监督式微调(SFT)交叉熵损失。

    :param model: 神经网络模块,前向传播返回 logits (B, L, V)。
    :param input_ids: 输入 token ID 序列 (B, L)。
    :param labels: 目标 token ID 标签 (B, L)。
    :param ignore_index: 忽略的标签索引,默认为 -100。
    :return: SFT 损失值。

    Example::

        import torch
        torch.manual_seed(42)
        import torch.nn as nn
        from pyvqnet.torch.trl import sft_loss

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(8, 8)
                self.head = nn.Linear(8, 8)
            def forward(self, x):
                return self.head(self.embed(x))

        loss = sft_loss(TinyLM(), torch.tensor([[1,2,3,4]]), torch.tensor([[2,3,4,5]]))
        print(loss.item())
        # 2.1718

dpo_loss
-----------------------------

.. py:function:: pyvqnet.torch.trl.dpo_loss(policy_model, ref_model, chosen_ids, rejected_ids, chosen_mask, rejected_mask, beta=0.1)

    DPO (Direct Preference Optimization) 标准 sigmoid 偏好损失。通过最大化偏好与非偏好序列之间的隐式奖励差异进行优化。

    :param policy_model: 策略网络,前向传播返回 logits (B, L, V)。
    :param ref_model: 参考网络,前向传播返回 logits (B, L, V)。
    :param chosen_ids: 偏好序列的 token ID (B, L_chosen)。
    :param rejected_ids: 非偏好序列的 token ID (B, L_rejected)。
    :param chosen_mask: 偏好序列的注意力掩码。
    :param rejected_mask: 非偏好序列的注意力掩码。
    :param beta: KL 正则化系数,默认为 0.1。
    :return: DPO 损失值。

    Example::

        import torch
        torch.manual_seed(42)
        import torch.nn as nn
        from pyvqnet.torch.trl import dpo_loss

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(8, 8)
                self.head = nn.Linear(8, 8)
            def forward(self, x):
                return self.head(self.embed(x))

        policy, ref = TinyLM(), TinyLM()
        loss = dpo_loss(policy, ref,
            torch.tensor([[0,1,2,3,4,5]]), torch.tensor([[5,4,3,2,1,0]]),
            torch.tensor([[1,1,1,1,1,1]]), torch.tensor([[1,1,1,1,1,1]]),
            beta=0.1)
        print(loss.item())
        # 0.7008

ppo_loss
-----------------------------

.. py:function:: pyvqnet.torch.trl.ppo_loss(policy_model, value_model, ref_model, query_responses, context_length, response_mask, old_logprobs, old_values, advantages, returns, cliprange=0.2, cliprange_value=0.2, vf_coef=1.0, temperature=1.0)

    PPO (Proximal Policy Optimization) 策略与价值函数联合损失。包含裁剪的替代策略损失和价值函数损失。

    :param policy_model: 策略网络,前向传播返回 logits。
    :param value_model: 价值网络,前向传播返回标量值。
    :param ref_model: 参考网络,用于 KL 惩罚。
    :param query_responses: 查询与响应拼接的 token ID 序列 (B, L)。
    :param context_length: 查询部分的长度,用于区分查询与响应。
    :param response_mask: 响应部分掩码 (B, L),1=响应 token。
    :param old_logprobs: 旧策略下的对数概率。
    :param old_values: 旧价值网络的估计值。
    :param advantages: 优势函数估计。
    :param returns: 折扣回报。
    :param cliprange: 策略裁剪范围,默认为 0.2。
    :param cliprange_value: 价值函数裁剪范围,默认为 0.2。
    :param vf_coef: 价值函数损失系数,默认为 1.0。
    :param temperature: 采样温度,默认为 1.0。
    :return: PPO 损失值。

    Example::

        import torch
        torch.manual_seed(42)
        import torch.nn as nn
        from pyvqnet.torch.trl import ppo_loss

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(8, 8)
                self.head = nn.Linear(8, 8)
            def forward(self, x):
                return self.head(self.embed(x))

        class TinyValue(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(8, 8)
                self.head = nn.Linear(8, 1)
            def forward(self, x):
                return self.head(self.embed(x)).squeeze(-1)

        loss = ppo_loss(TinyLM(), TinyValue(), TinyLM(),
            torch.tensor([[0,1,2,3,4,5,6,7]]), 2,
            torch.ones(1,6), torch.zeros(1,6), torch.zeros(1,6),
            torch.ones(1,6), torch.ones(1,6))
        print(loss.item())
        # 1.3638

grpo_loss
-----------------------------

.. py:function:: pyvqnet.torch.trl.grpo_loss(policy_model, ref_model, input_ids, completion_mask, old_per_token_logps, advantages, beta=0.0, epsilon=0.2, epsilon_low=None, epsilon_high=None)

    GRPO (Group Relative Policy Optimization) 裁剪替代损失。将多个补全结果分组计算优势。

    :param policy_model: 策略网络,前向传播返回 logits。
    :param ref_model: 参考网络或 None。
    :param input_ids: 提示与补全拼接的 token ID 序列 (B*G, L),其中 G 为组大小。
    :param completion_mask: 补全部分掩码 (B*G, L),1=补全 token,0=提示/填充。
    :param old_per_token_logps: 旧策略下的逐 token 对数概率 (B*G, T)。
    :param advantages: 组内的优势函数估计。
    :param beta: KL 惩罚系数,默认为 0.0。
    :param epsilon: PPO 裁剪范围,默认为 0.2。
    :param epsilon_low: 裁剪下限,默认为 None(使用 epsilon)。
    :param epsilon_high: 裁剪上限,默认为 None(使用 epsilon)。
    :return: GRPO 损失值。

    Example::

        import torch
        torch.manual_seed(42)
        import torch.nn as nn
        from pyvqnet.torch.trl import grpo_loss

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(8, 8)
                self.head = nn.Linear(8, 8)
            def forward(self, x):
                return self.head(self.embed(x))

        loss = grpo_loss(TinyLM(), TinyLM(),
            torch.tensor([[0,1,2,3,4],[5,6,7,0,1]]),
            torch.tensor([[0,0,1,1,1],[0,1,1,1,1]]),
            torch.zeros(2,3), torch.tensor([1.0, -0.5]),
            beta=0.0, epsilon=0.2)
        print(loss.item())
        # 0.1405

reward_loss
-----------------------------

.. py:function:: pyvqnet.torch.trl.reward_loss(model, chosen_ids, rejected_ids, margin=None, center_coef=None)

    奖励模型对比损失。通过最大化偏好与非偏好序列之间的奖励差异训练奖励模型。

    :param model: 奖励模型,前向传播返回标量奖励值。
    :param chosen_ids: 偏好序列的 token ID。
    :param rejected_ids: 非偏好序列的 token ID。
    :param margin: 对比间隔,默认为 None。
    :param center_coef: 奖励中心化系数,默认为 None。
    :return: 奖励模型损失值。

    Example::

        import torch
        torch.manual_seed(42)
        import torch.nn as nn
        from pyvqnet.torch.trl import reward_loss

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(8, 8)
                self.head = nn.Linear(8, 8)
            def forward(self, x):
                return self.head(self.embed(x))

        loss = reward_loss(TinyLM(), torch.tensor([[1,2,3]]), torch.tensor([[3,2,1]]))
        print(loss.item())
        # 0.7172

优化器模块
---------------------------------------------

对于继承于 `TorchModule` 的VQNet的经典和量子线路模块,对其中的参数 `model.paramters()` 可继续使用 :ref:`Optimizer` 下的除 `Rotosolve` 以外的VQNet优化器进行参数优化。

对于继承于 `TorchModule` 的VQNet的经典和量子线路模块,其中参数同样可以被 `torch.nn.Module.parameters()` 获取,可同样使用 torch 的优化器进行优化。


使用pyqpanda进行计算的量子变分线路训练函数
------------------------------------------

以下是使用pyqpanda3进行线路计算的训练变分量子线路接口。



.. warning::

    以下TorchQcloud3QuantumLayer,TorchQpanda3QuantumLayer接口的量子计算部分使用pyqpanda3 https://qcloud.originqc.com.cn/document/qpanda-3/index.html。


TorchQcloud3QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当您安装最新版本pyqpanda3,可以使用本接口定义一个变分线路,并提交到originqc的真实芯片上运行。

.. py:class:: pyvqnet.qnn.pq3.torch.qpanda3_layer.TorchQcloud3QuantumLayer(origin_qprog_func, qcloud_token, para_num, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift", submit_kwargs={}, query_kwargs={})

    使用 pyqpanda3的本源量子 https://qcloud.originqc.com.cn/  真实芯片的抽象计算模块。 它提交参数化量子电路到真实芯片并获得测量结果。
    如果 diff_method == "random_coordinate_descent" ,该层将随机选择单个参数来计算梯度,其他参数将保持为零。参考:https://arxiv.org/abs/2311.00088

    .. note::

        qcloud_token 为您到 https://qcloud.originqc.com.cn/ 中申请的 API token。
        origin_qprog_func 需要返回pypqanda3.core.QProg类型的数据,如果没有设置测量观测量pauli_str_dict,需要保证该QProg中已经插入了measure。
        origin_qprog_func 的形式必须按照如下:

        origin_qprog_func(input,param )
        
            `input`: 输入1~2维经典数据,二维的情况下,第一个维度为批处理大小。
            
            `param`: 输入一维的变分量子线路的待训练参数。

    .. warning::

        该类继承于 ``pyvqnet.nn.Module`` 以及 ``torch.nn.Module``,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。



    :param origin_qprog_func: QPanda 构建的变分量子电路函数,必须返回QProg。
    :param qcloud_token: `str` - 量子机的类型或用于执行的云令牌。
    :param para_num: `int` - 参数数量,参数是大小为[para_num]的QTensor。
    :param pauli_str_dict: `dict|list` - 表示量子电路中泡利运算符的字典或字典列表。 默认为“无”,则进行测量操作,如果输入泡利算符的字典,则会计算单个期望或者多个期望。
    :param shot: `int` - 测量次数。 默认值为 1000。
    :param initializer: 参数值的初始化器。 默认为“无”,使用0~2*pi正态分布。
    :param dtype: 参数的数据类型。 默认值为 None,即使用默认数据类型pyvqnet.kfloat32。
    :param name: 模块的名称。 默认为空字符串。
    :param diff_method: 梯度计算的微分方法。 默认为“parameter_shift”,"random_coordinate_descent"。
    :param submit_kwargs: 用于提交量子电路的附加关键字参数,默认:{"if_print_qcloud_log":False,"chip_id":"WK_C180","is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False,"server_ip_address":"","use_qwc":True},当设置test_qcloud_fake为True则本地CPUQVM模拟。
    :param query_kwargs: 用于查询量子结果的附加关键字参数,默认:{"timeout":2,"print_query_info":True,"sub_circuits_split_size":1}。
    :return: 一个可以计算量子电路的模块。

    Example::

        import pyqpanda3.core as pq
        import pyvqnet
        from pyvqnet.qnn.vqc.sv.torch import TorchQcloud3QuantumLayer

        pyvqnet.backends.set_backend("torch")
        def qfun(input,param):

            m_qlist = range(6)
            cbits = range(6)
            measure_qubits = [0,2]
            m_prog = pq.QProg()
            cir = pq.QCircuit()
            cir<<pq.RZ(m_qlist[0],input[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[1])
            cir<<pq.RY(m_qlist[1],param[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[2])
            cir<<pq.RZ(m_qlist[1],input[1])
            cir<<pq.RY(m_qlist[2],param[1])
            cir<<pq.H(m_qlist[2])
            m_prog<<cir

            for idx, ele in enumerate(measure_qubits):
                m_prog << pq.measure(m_qlist[ele], cbits[idx])  # pylint: disable=expression-not-assigned
            return m_prog

        l = TorchQcloud3QuantumLayer(qfun,
                        "your_api_token",
                        2,
                        pauli_str_dict=None,
                        shots = 1000,
                        initializer=None,
                        dtype=None,
                        name="",
                        diff_method="parameter_shift",
                        submit_kwargs={"test_qcloud_fake":True},
                        query_kwargs={})
        x = pyvqnet.tensor.QTensor([[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2]],requires_grad= True)
        y = l(x)
        print(y)
        y.backward(pyvqnet.tensor.ones_like(y))
        print(l.m_para.grad)
        print(x.grad)

        def qfun2(input,param ):

            m_qlist = range(6)
            cbits = range(6)
            measure_qubits = [0,2]
            m_prog = pq.QProg()
            cir = pq.QCircuit()
            cir<<pq.RZ(m_qlist[0],input[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[1])
            cir<<pq.RY(m_qlist[1],param[0])
            cir<<pq.CNOT(m_qlist[0],m_qlist[2])
            cir<<pq.RZ(m_qlist[1],input[1])
            cir<<pq.RY(m_qlist[2],param[1])
            cir<<pq.H(m_qlist[2])
            m_prog<<cir

            return m_prog
        l = TorchQcloud3QuantumLayer(qfun2,
                "your_api_token",
                2,

                pauli_str_dict={'Z0 X1':10,'':-0.5,'Y2':-0.543},
                shots = 1000,
                initializer=None,
                dtype=None,
                name="",
                diff_method="parameter_shift",
                submit_kwargs={"test_qcloud_fake":True},
                query_kwargs={})
        x = pyvqnet.tensor.QTensor([[0.56,1.2],[0.56,1.2],[0.56,1.2],[0.56,1.2]],requires_grad= True)
        y = l(x)
        print(y)
        y.backward(pyvqnet.tensor.ones_like(y))
        print(l.m_para.grad)
        print(x.grad)



TorchQpanda3QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如您更加熟悉pyqpanda3语法,可以使用该接口TorchQpanda3QuantumLayer。

.. py:class:: pyvqnet.qnn.pq3.torch.qpanda3_layer.TorchQpanda3QuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

	变分量子层的抽象计算模块。对一个参数化的量子线路使用pyqpanda3进行仿真,得到测量结果。该变分量子层继承了VQNet框架的梯度计算模块,可以使用参数移位法等计算线路参数的梯度,训练变分量子线路模型或将变分量子线路嵌入混合量子和经典模型。
    
    :param qprog_with_measure: 用pyqpanda3构建的量子线路 运行和测量函数。
    :param para_num: `int` - 参数个数。
    :param diff_method: 求解量子线路参数梯度的方法,"parameter_shift"或"finite_diff"，默认为 "parameter_shift"。 。
    :param delta: 有限差分计算梯度时的 \delta。
    :param dtype: 参数的数据类型,默认: None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字, 默认为""。

    :return: 一个可以计算量子线路的模块。

    .. note::
        qprog_with_measure是pyQPanda中定义的量子线路函数 :https://qcloud.originqc.com.cn/document/qpanda-3/db/d6c/tutorial_circuit_and_program.html.。
        
        此函数必须包含以下参数作为函数入参（即使某个参数未实际使用）,否则无法在本函数中正常运行。

        量子线路函数 qprog_with_measure (input,param)的使用可参考下面的例子。
        
        `input`: 输入一维经典数据。如果没有,输入 None。
        
        `param`: 输入一维的变分量子线路的待训练参数。


    Example::

        import pyqpanda3.core as pq
        from pyvqnet.qnn.pq3 import ProbsMeasure
        import numpy as np
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import TorchQpanda3QuantumLayer
        def pqctest (input,param):
            num_of_qubits = 4

            m_machine = pq.CPUQVM()# outside
        
            qubits =range(num_of_qubits)

            circuit = pq.QCircuit()
            circuit<<pq.H(qubits[0])
            circuit<<pq.H(qubits[1])
            circuit<<pq.H(qubits[2])
            circuit<<pq.H(qubits[3])

            circuit<<pq.RZ(qubits[0],input[0])
            circuit<<pq.RZ(qubits[1],input[1])
            circuit<<pq.RZ(qubits[2],input[2])
            circuit<<pq.RZ(qubits[3],input[3])

            circuit<<pq.CNOT(qubits[0],qubits[1])
            circuit<<pq.RZ(qubits[1],param[0])
            circuit<<pq.CNOT(qubits[0],qubits[1])

            circuit<<pq.CNOT(qubits[1],qubits[2])
            circuit<<pq.RZ(qubits[2],param[1])
            circuit<<pq.CNOT(qubits[1],qubits[2])

            circuit<<pq.CNOT(qubits[2],qubits[3])
            circuit<<pq.RZ(qubits[3],param[2])
            circuit<<pq.CNOT(qubits[2],qubits[3])

            prog = pq.QProg()
            prog<<circuit

            rlt_prob = ProbsMeasure(m_machine,prog,[0,2])
            return rlt_prob

        pqc = TorchQpanda3QuantumLayer(pqctest,3)

        #classic data as input
        input = QTensor([[1.0,2,3,4],[4,2,2,3],[3,3,2,2]],requires_grad=True)

        #forward circuits
        rlt = pqc(input)

        print(rlt)

        grad = pyvqnet.tensor.ones(rlt.data.shape)*1000
        #backward circuits
        rlt.backward(grad)

        print(pqc.m_para.grad)
        print(input.grad)



基于态矢的变分量子线路模块
===============================================


基类
--------------------------------------------------

编写变分量子线路模型需要继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule``。

QModule
^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.QModule(name="")

    当用户使用 `torch` 后端时候,定义量子变分线路模型 `Module` 应该继承的基类。
    该类继承于 ``pyvqnet.nn.torch.TorchModule`` 以及 ``torch.nn.Module``。
    该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    .. note::

        该类以及其派生类仅适用于 ``pyvqnet.backends.set_backend("torch")`` , 不要与默认 ``pyvqnet.nn`` 下的 ``Module`` 混用。
    
        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。



QMachine
^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.QMachine(num_wires, dtype=pyvqnet.kcomplex64,grad_mode="",save_ir=False)

    变分量子计算的模拟器类,包含states属性为量子线路的statevectors。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` 以及 ``pyvqnet.qnn.QMachine`` 。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    .. warning::
        
        在每次运行一个完整的量子线路之前,必须使用 `pyvqnet.qnn.vqc.sv.torch.QMachine.reset_states(batchsize)` 将模拟器里面初态重新初始化,并且广播为
        (batchsize,*) 维度从而适应批量数据训练。

    :param num_wires: 量子比特数。
    :param dtype: 计算数据的数据类型。默认值是pyvqnet.kcomplex64,对应的参数精度为pyvqnet.kfloat32。
    :param grad_mode: 梯度计算模式,可为 "adjoint",默认值:"",使用自动微分模拟。
    :param save_ir: 设置为True时,将操作保存到originIR,默认值:False。

    :return: 输出一个QMachine对象。

    Example::
        
        from pyvqnet.qnn.vqc.sv.torch import QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        qm = QMachine(4)
        print(qm.states)


    .. py:method:: reset_states(batchsize)

        将模拟器里面初态重新初始化,并且广播为
        (batchsize,*) 维度从而适应批量数据训练。

        :param batchsize: 批处理维度。


变分量子逻辑门模块
^^^^^^^^^^^^^^^^^^^^^^^^


以下 ``pyvqnet.qnn.vqc.sv.torch`` 中的函数接口直接支持 ``torch`` 后端的 ``QTensor`` 进行计算。

.. csv-table:: 已支持pyvqnet.qnn.vqc.sv.torch接口列表
   :file: ./images/same_apis_from_vqc.csv


以下量子线路模块继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule``,其中计算使用 ``torch.Tensor`` 进行计算。


.. warning::

    该类以及其派生类仅适用于 ``pyvqnet.backends.set_backend("torch")`` , 不要与默认 ``pyvqnet.nn`` 下的 ``Module`` 混用。

    这些类如果有非参数成员变量 ``_buffers`` ,则其中的数据为 ``torch.Tensor`` 类型。
    这些类如果有参数成员变量 ``_parmeters`` ,则其中的数据为 ``torch.nn.Parameter`` 类型。

I
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个I逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params: 是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 I 逻辑门实例

    Example::
        
        from pyvqnet.qnn.vqc.sv.torch import I,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = I(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


Hadamard
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.Hadamard(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个Hadamard逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 Hadamard 逻辑门实例

    Example::
        
        from pyvqnet.qnn.vqc.sv.torch import Hadamard,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = Hadamard(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


T
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.T(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个T逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 T 逻辑门实例

    Example::
        
        from pyvqnet.qnn.vqc.sv.torch import T,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = T(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



S
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.S(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个S逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 S 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.sv.torch import S,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = S(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


PauliX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.PauliX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliX 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.sv.torch import PauliX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PauliX(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


PauliY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.PauliY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliY 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.sv.torch import PauliY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PauliY(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



PauliZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.PauliZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliZ 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.sv.torch import PauliZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PauliZ(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



X1
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.X1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个X1逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 X1 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.sv.torch import X1,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = X1(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


RX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.RX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RX逻辑门类 。


    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import RX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RX(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



RY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.RY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import RY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RY(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


RZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.RZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import RZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RZ(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CRX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.CRX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import CRX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CRX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CRY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.CRY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import CRY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CRY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CRZ
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.sv.torch.CRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import CRZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



U1
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.U1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U1逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U1 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import U1,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = U1(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

U2
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.sv.torch.U2(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U2逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U2 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import U2,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = U2(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


U3
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.sv.torch.U3(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U3逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U3 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import U3,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = U3(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



CNOT
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.CNOT(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CNOT逻辑门类,也可称为CX。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CNOT 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import CNOT,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CNOT(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

CY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.CY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import CY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CY(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.CZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import CZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CZ(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)




CR
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.CR(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CR逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CR 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import CR,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        device = QMachine(4)
        layer = CR(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



SWAP
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.sv.torch.SWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SWAP逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 SWAP 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import SWAP,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = SWAP(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CSWAP
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.CSWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SWAP逻辑门类 。

    .. math:: CSWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}.

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CSWAP 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import CSWAP,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CSWAP(wires=[0,1,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

RXX
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.sv.torch.RXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RXX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RXX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import RXX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

RYY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.RYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RYY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RYY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import RYY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


RZZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.RZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个RZZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import RZZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



RZX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.RZX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RZX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import RZX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RZX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

Toffoli
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.sv.torch.Toffoli(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个Toffoli逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 Toffoli 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import Toffoli,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = Toffoli(wires=[0,2,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

IsingXX
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.sv.torch.IsingXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingXX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingXX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import IsingXX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


IsingYY
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.sv.torch.IsingYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingYY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingYY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import IsingYY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


IsingZZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.IsingZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingZZ逻辑门类 。


    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingZZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import IsingZZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


IsingXY
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.sv.torch.IsingXY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingXY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingXY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import IsingXY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingXY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


PhaseShift
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.sv.torch.PhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PhaseShift逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PhaseShift 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import PhaseShift,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PhaseShift(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


MultiRZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.MultiRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个MultiRZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 MultiRZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import MultiRZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = MultiRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



SDG
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.sv.torch.SDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SDG逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 SDG 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.sv.torch import SDG,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = SDG(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)




TDG
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.TDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SDG逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 TDG 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.sv.torch import TDG,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = TDG(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



ControlledPhaseShift
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.sv.torch.ControlledPhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个ControlledPhaseShift逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 ControlledPhaseShift 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.sv.torch import ControlledPhaseShift,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = ControlledPhaseShift(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



MultiControlledX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.MultiControlledX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False,control_values=None)
    
    定义一个MultiControlledX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。
    
    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :param control_values: 控制值,默认为None,当比特位为1时控制。

    :return: 一个 MultiControlledX 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import QMachine,MultiControlledX
        from pyvqnet.tensor import QTensor,kcomplex64
        qm = QMachine(4,dtype=kcomplex64)
        qm.reset_states(2)
        mcx = MultiControlledX( 
                        init_params=None,
                        wires=[2,3,0,1],
                        dtype=kcomplex64,
                        use_dagger=False,control_values=[1,0,0])
        y = mcx(q_machine = qm)
        print(qm.states)


测量接口
^^^^^^^^^^^^^^^^^^^^^^

Probability
^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.sv.torch.Probability(wires=None, name="")

    计算量子线路在特定比特上概率测量结果。

    .. warning::
        
        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param wires: 测量比特的索引,列表、元组或者整数。
    :param name: 模块的名字,默认:""。
    :return: 测量结果,QTensor。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import Probability,rx,ry,cnot,QMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
        qm = QMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rz(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,2])
        rz(q_machine=qm,wires=3,params=x[:,[1]])
        ma = Probability(wires=1)
        y =ma(q_machine=qm)


MeasureAll
^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.MeasureAll(obs=None, name="")

    计算量子线路的测量结果,支持输入观测量 ``obs``。其格式可以为字典格式，用于表示一个由多个Pauli算符组合而成的可观测量；列表形式,表示多个期望值的可观测量列表。
 
    例如:

    {\'X0\': 0.23} 表示在量子比特0上作用PauliX,系数为0.23

    {\'X1 Z2\':2.4,\'Y2\':-0.5} 对应于观测量 2.4 * X1 @ Z2 - 0.5 * Y2

    [{\'X1 Z2\':4,\'Z1 Z0\':3},{\'X1 Y2 Z0\':3.5}] 对应于两个观测量 4 * X1 @ Z2 + 3 * Z1 @ Z0 以及 3.5 * X1 @ Y2 @ Z0 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param obs: observable。
    :param name: 模块的名字,默认:""。
    :return: 一个 MeasureAll 测量方法实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import MeasureAll,rx,ry,cnot,QMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
        qm = QMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rz(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,2])
        rz(q_machine=qm,wires=3,params=x[:,[1]])
        obs_list = [{
            "Z0 Z1" :2
        }, {
            "X1 X0" :2
        }]
        ma = MeasureAll(obs = obs_list)
        y = ma(q_machine=qm)
        print(y)



Samples
^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.Samples(wires=None, obs=None, shots = 1,name="")

    获取特定线路上的带有 shot 的样本结果

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param wires: 样本量子比特索引。默认值: None,根据运行时使用模拟器的所有比特。
    :param obs: 该值只能设为None。
    :param shots: 样本重复次数,默认值: 1。
    :param name: 此模块的名称,默认值: “”。
    :return: 一个 Samples 测量方法实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import Samples,rx,ry,cnot,QMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)

        qm = QMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rx(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])

        cnot(q_machine=qm,wires=[0,2])
        ry(q_machine=qm,wires=3,params=x[:,[1]])


        ma = Samples(wires=[0,1,2],shots=3)
        y = ma(q_machine=qm)
        print(y)


HermitianExpval
^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.HermitianExpval(obs=None, name="")

    计算量子线路某个厄密特量的期望。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param obs: 厄密特量。
    :param name: 模块的名字,默认:""。
    :return: 一个 HermitianExpval 测量方法实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import QMachine, rx,ry,\
            RX, RY, CNOT, PauliX, PauliZ, VQC_RotCircuit,HermitianExpval
        from pyvqnet.tensor import QTensor, tensor
        from pyvqnet.nn import Parameter
        import numpy as np
        bsz = 3
        H = np.array([[8, 4, 0, -6], [4, 0, 4, 0], [0, 4, 8, 0], [-6, 0, 0, 0]])
        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()
                self.rot_param = Parameter((3, ))
                self.rot_param.copy_value_from(tensor.QTensor([-0.5, 1, 2.3]))
                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)
                self.rx_layer1 = VQC_RotCircuit
                self.ry_layer2 = RY(has_params=True,
                                    trainable=True,
                                    wires=0,
                                    init_params=tensor.QTensor([-0.5]))
                self.xlayer = PauliX(wires=0)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = HermitianExpval(obs = {'wires':(1,0),'observables':tensor.to_tensor(H)})

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                rx(q_machine=self.qm, wires=0, params=x[:, [1]])
                ry(q_machine=self.qm, wires=1, params=x[:, [0]])
                self.xlayer(q_machine=self.qm)
                self.rx_layer1(params=self.rot_param, wire=1, q_machine=self.qm)
                self.ry_layer2(q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                rlt = self.measure(q_machine = self.qm)

                return rlt


        input_x = tensor.arange(1, bsz * 2 + 1,
                                dtype=pyvqnet.kfloat32).reshape([bsz, 2])
        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=2, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward(pyvqnet.tensor.ones_like(batch_y))


量子线路常见模板
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VQC_HardwareEfficientAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial = None,dtype=None)

    论文介绍的Hardware Efficient Ansatz的实现: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param n_qubits: 量子比特数。
    :param single_rot_gate_list: 单个量子比特旋转门列表由一个或多个作用于每个量子比特的旋转门构成。目前支持 Rx、Ry、Rz。
    :param entangle_gate: 非参数化纠缠门。支持 CNOT、CZ。默认值: CNOT。
    :param entangle_rules: 纠缠门在电路中的使用方式。'linear' 表示纠缠门将作用于每个相邻的量子比特。'all' 表示纠缠门将作用于任意两个量子比特。默认值: linear。
    :param depth: 假设的深度,默认值: 1。
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数,默认值: None,此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用float32。
    :return: 一个 VQC_HardwareEfficientAnsatz 实例。

    Example::

        from pyvqnet.nn.torch import TorchModule,Linear,TorchModuleList
        from pyvqnet.qnn.vqc.sv.torch.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ
        from pyvqnet.qnn.vqc.sv.torch import Probability,QMachine
        from pyvqnet import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)

        class QM(TorchModule):
            def __init__(self, name=""):
                super().__init__(name)
                self.linearx = Linear(4,2)
                self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                            entangle_gate="cnot",
                                            entangle_rules="linear",
                                            depth=2)
                self.encode1 = RZ(wires=0)
                self.encode2 = RZ(wires=1)
                self.measure = Probability(wires=[0,2])
                self.device = QMachine(4)
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(x.shape[0])
                y = self.linearx(x)
                self.encode1(params = y[:, [0]],q_machine = self.device,)
                self.encode2(params = y[:, [1]],q_machine = self.device,)
                self.ansatz(q_machine =self.device)
                return self.measure(q_machine =self.device)

        bz =3
        inputx = tensor.arange(1.0,bz*4+1).reshape([bz,4])
        inputx.requires_grad= True
        qlayer = QM()
        y = qlayer(inputx)
        y.backward(pyvqnet.tensor.ones_like(y))
        print(y)



VQC_BasicEntanglerTemplate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.VQC_BasicEntanglerTemplate(num_layer=1, num_qubits=1, rotation="RX", initial=None, dtype=None)

    由每个量子位上的单参数单量子位旋转组成的层,后跟一个闭合链或环组合的多个CNOT门。

    CNOT 门环将每个量子位与其邻居连接起来,最后一个量子位被认为是第一个量子位的邻居。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param num_layer: 量子比特线路层数。
    :param num_qubits: 量子比特数,默认为1。
    :param rotation: 使用单参数单量子比特门,``RX`` 被用作默认值。
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数,默认值: None,此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用float32。
    :return: 返回一个含可训练参数的VQC_BasicEntanglerTemplate实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import QModule,\
            VQC_BasicEntanglerTemplate, Probability, QMachine
        from pyvqnet import tensor


        class QM(QModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_BasicEntanglerTemplate(2,
                                                    4,
                                                    "rz",
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 2])
                self.device = QMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward(pyvqnet.tensor.ones_like(y))
        print(y)



VQC_StronglyEntanglingTemplate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.VQC_StronglyEntanglingTemplate(num_layers=1, num_qubits=1, rotation = "RX", initial = None, dtype: = None)

    由单个量子比特旋转和纠缠器组成的层,参考 `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param num_layers: 重复层数,默认值: 1。
    :param num_qubits: 量子比特数,默认值: 1。
    :param rotation: 要使用的单参数单量子比特门,默认值: `RX`
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数,默认值: None,此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用 float32。
    :return: VQC_BasicEntanglerTemplate 实例


    Example::

        from pyvqnet.nn.torch import TorchModule,Linear,TorchModuleList
        from pyvqnet.qnn.vqc.sv.torch.qcircuit import VQC_StronglyEntanglingTemplate
        from pyvqnet.qnn.vqc.sv.torch import Probability, QMachine
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)
        class QM(TorchModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_StronglyEntanglingTemplate(2,
                                                    4,
                                                    None,
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 1])
                self.device = QMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward(pyvqnet.tensor.ones_like(y))
        print(y)



VQC_QuantumEmbedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.sv.torch.VQC_QuantumEmbedding(  num_repetitions_input, depth_input, num_unitary_layers, num_repetitions,initial = None,dtype = None,name= "")

    使用 RZ,RY,RZ 创建变分量子电路,将经典数据编码为量子态。
    参考 `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param num_repetitions_input: 子模块中输入编码的重复次数。
    :paramdepth_input: 输入维数。
    :param num_unitary_layers: 变分量子门的重复次数。
    :param num_repetitions: 子模块的重复次数。
    :param initial: 参数初始化值,默认为None
    :param dtype: 参数的类型,默认 None,使用float32.
    :param name: 类的名字

    Example::

        from pyvqnet.nn.torch import TorchModule
        from pyvqnet.qnn.vqc.sv.torch.qcircuit import VQC_QuantumEmbedding
        from pyvqnet.qnn.vqc.sv.torch import Probability, QMachine, MeasureAll
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)
        depth_input = 2
        num_repetitions = 2
        num_repetitions_input = 2
        num_unitary_layers = 2
        nq = depth_input * num_repetitions_input
        bz = 12

        class QM(TorchModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_QuantumEmbedding(num_repetitions_input, depth_input,
                                                num_unitary_layers,
                                                num_repetitions, initial=tensor.full([1],12.0),dtype=pyvqnet.kfloat32)

                self.measure = MeasureAll(obs={f"Z{nq-1}":1})
                self.device = QMachine(nq)

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(x.shape[0])
                self.ansatz(x,q_machine=self.device)
                return self.measure(q_machine=self.device)

        inputx = tensor.arange(1.0, bz * depth_input + 1).reshape([bz, depth_input])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward(pyvqnet.tensor.ones_like(y))
        print(y)


ExpressiveEntanglingAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.ExpressiveEntanglingAnsatz(type: int, num_wires: int, depth: int, dtype=None, name: str = "")

    论文 `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_ 中的 19 种不同的ansatz。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param type: 电路类型从 1 到 19,共19种线路。
    :param num_wires: 量子比特数。
    :param depth: 电路深度。
    :param dtype: 参数的数据类型, 默认值: None, 使用 float32。
    :param name: 名字,默认"".

    :return:
        一个 ExpressiveEntanglingAnsatz 实例

    Example::

        from pyvqnet.nn.torch import TorchModule
        from pyvqnet.qnn.vqc.sv.torch.qcircuit import ExpressiveEntanglingAnsatz
        from pyvqnet.qnn.vqc.sv.torch import Probability, QMachine, MeasureAll
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)

        class QModel(TorchModule):
            def __init__(self, num_wires, dtype,grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype,grad_mode=grad_mode)
                self.c1 = ExpressiveEntanglingAnsatz(1,3,2)
                self.measure = MeasureAll(obs={"Z1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.c1(q_machine = self.qm)
                rlt = self.measure(q_machine=self.qm)
                return rlt
            

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward(pyvqnet.tensor.ones_like(batch_y))
        print(batch_y)



vqc_basis_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.sv.torch.vqc_basis_embedding(basis_state,q_machine)

    将n个二进制特征编码到 ``q_machine`` 的n个量子比特的基态。该函数别名 `VQC_BasisEmbedding` 。

    例如, 对于 ``basis_state=([0, 1, 1])``, 在量子系统下其基态为 :math:`|011 \rangle`。

    :param basis_state:  ``(n)`` 大小的二进制输入。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import vqc_basis_embedding,QMachine
        qm  = QMachine(3)
        vqc_basis_embedding(basis_state=[1,1,0],q_machine=qm)
        print(qm.states)




vqc_angle_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.sv.torch.vqc_angle_embedding(input_feat, wires, q_machine: pyvqnet.qnn.vqc.sv.torch.QMachine, rotation: str = "X")

    将 :math:`N` 特征编码到 :math:`n` 量子比特的旋转角度中, 其中 :math:`N \leq n`。
    该函数别名 `VQC_AngleEmbedding` 。

    旋转可以选择为 : 'X' , 'Y' , 'Z', 如 ``rotation`` 的参数定义为:

    * ``rotation='X'`` 将特征用作RX旋转的角度。

    * ``rotation='Y'`` 将特征用作RY旋转的角度。

    * ``rotation='Z'`` 将特征用作RZ旋转的角度。

     ``wires`` 代表旋转门在量子比特上的idx。

    :param input_feat: 表示参数的数组。
    :param wires: 量子比特idx。
    :param q_machine: 量子虚拟机设备。
    :param rotation: 旋转门,默认为“X”。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import vqc_angle_embedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(2)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='X')
        print(qm.states)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Y')
        print(qm.states)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Z')
        print(qm.states)



vqc_amplitude_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.sv.torch.vqc_amplitude_embedding(input_feature, q_machine)

    将 :math:`2^n` 特征编码为 :math:`n` 量子比特的振幅向量。该函数别名 `VQC_AmplitudeEmbedding` 。

    ``VQC_AmplitudeEmbedding`` 是 ``vqc_amplitude_embedding`` 的别名,作用完全相同。

    :param input_feature: 表示参数的numpy数组。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import vqc_amplitude_embedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        vqc_amplitude_embedding(QTensor([3.2,-2,-2,0.3,12,0.1,2,-1]), q_machine=qm)
        print(qm.states)



vqc_iqp_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:function:: pyvqnet.qnn.vqc.vqc_iqp_embedding(input_feat, q_machine: pyvqnet.qnn.vqc.sv.torch.QMachine, rep: int = 1)

    使用IQP线路的对角门将 :math:`n` 特征编码为 :math:`n` 量子比特。该函数别名:  ``VQC_IQPEmbedding`` 。

    编码是由 `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_ 提出。

    通过指定 ``rep`` ,可以重复基本IQP线路。

    :param input_feat: 表示参数的数组。
    :param q_machine: 量子虚拟机设备。
    :param rep: 重复量子线路块次数,默认次数为1。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import vqc_iqp_embedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        vqc_iqp_embedding(QTensor([3.2,-2,-2]), q_machine=qm)
        print(qm.states)        



vqc_rotcircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.sv.torch.vqc_rotcircuit(q_machine, wire, params)

    任意单量子比特旋转的量子逻辑门组合。该函数别名:  ``VQC_RotCircuit`` 。

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.


    :param q_machine: 量子虚拟机设备。
    :param wire: 量子比特索引。
    :param params: 表示参数  :math:`[\phi, \theta, \omega]`。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import vqc_rotcircuit, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        vqc_rotcircuit(q_machine=qm, wire=[1],params=QTensor([2.0,1.5,2.1]))
        print(qm.states)


vqc_crot_circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.sv.torch.vqc_crot_circuit(para,control_qubits,rot_wire,q_machine)

	受控Rot单量子比特旋转的量子逻辑门组合。该函数别名:  ``VQC_CRotCircuit`` 。

    .. math:: CR(\phi, \theta, \omega) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
            0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.
    
    :param para: 表示参数的数组。
    :param control_qubits: 控制量子比特索引。
    :param rot_wire: Rot量子比特索引。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.sv.torch import vqc_crot_circuit,QMachine, MeasureAll
        p = QTensor([2, 3, 4.0])
        qm = QMachine(2)
        vqc_crot_circuit(p, 0, 1, qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)




vqc_controlled_hadamard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.sv.torch.vqc_controlled_hadamard(wires, q_machine)

    受控Hadamard逻辑门量子线路。该函数别名:  ``VQC_Controlled_Hadamard`` 。

    .. math:: CH = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
            0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: 量子比特索引列表, 第一位是控制比特, 列表长度为2。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.sv.torch import vqc_controlled_hadamard,\
            QMachine, MeasureAll

        p = QTensor([0.2, 3, 4.0])
        qm = QMachine(3)
        vqc_controlled_hadamard([1, 0], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_ccz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.sv.torch.vqc_ccz(wires, q_machine)

    受控-受控-Z (controlled-controlled-Z) 逻辑门。该函数别名:  ``VQC_CCZ`` 。

    .. math::

        CCZ =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
        \end{pmatrix}
    
    :param wires: 量子比特下标列表,第一位是控制比特。列表长度为3。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.sv.torch import vqc_ccz,QMachine, MeasureAll
        p = QTensor([0.2, 3, 4.0])

        qm = QMachine(3)

        vqc_ccz([1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_fermionic_single_excitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.sv.torch.vqc_fermionic_single_excitation(weight, wires, q_machine)

    对泡利矩阵的张量积求幂的耦合簇单激励算子。矩阵形式下式给出:

    .. math::

        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    该函数别名:  ``VQC_FermionicSingleExcitation`` 。

    :param weight:  量子比特p上的参数, 只有一个元素.
    :param wires: 表示区间[r, p]中的量子比特索引子集。最小长度必须为2。第一索引值被解释为r,最后一个索引值被解释为p。
                中间的索引被CNOT门作用,以计算量子位集的奇偶校验。
    :param q_machine: 量子虚拟机设备。

    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.sv.torch import vqc_fermionic_single_excitation,\
            QMachine, MeasureAll
        qm = QMachine(3)
        p0 = QTensor([0.5])

        vqc_fermionic_single_excitation(p0, [1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

 


vqc_fermionic_double_excitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.sv.torch.vqc_fermionic_double_excitation(weight, wires1, wires2, q_machine)

    对泡利矩阵的张量积求幂的耦合聚类双激励算子,矩阵形式由下式给出:

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    其中 :math:`\hat{c}` 和 :math:`\hat{c}^\dagger` 是费米子湮灭和
    创建运算符和索引 :math:`r, s` 和 :math:`p, q` 在占用的和
    分别为空分子轨道。 使用 `Jordan-Wigner 变换
    <https://arxiv.org/abs/1208.5986>`_ 上面定义的费米子算子可以写成
    根据 Pauli 矩阵(有关更多详细信息,请参见
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_)

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.}  ) \Big\}

    该函数别名:  ``VQC_FermionicDoubleExcitation`` 。

    :param weight: 可变参数
    :param wires1: 代表的量子比特的索引列表区间 [s, r] 中占据量子比特的子集。第一个索引被解释为 s,最后一索引被解释为 r。 CNOT 门对中间的索引进行操作,以计算一组量子位的奇偶性。
    :param wires2: 代表的量子比特的索引列表区间 [q, p] 中占据量子比特的子集。第一根索引被解释为 q,最后一索引被解释为 p。 CNOT 门对中间的索引进行操作,以计算一组量子位的奇偶性。
    :param q_machine: 量子虚拟机设备。

    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.sv.torch import vqc_fermionic_double_excitation,\
            QMachine, MeasureAll
        qm = QMachine(5)
        p0 = QTensor([0.5])

        vqc_fermionic_double_excitation(p0, [0, 1], [2, 3], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)
 

vqc_uccsd
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.sv.torch.vqc_uccsd(weights, wires, s_wires, d_wires, init_state, q_machine)

    实现酉耦合簇单激发和双激发拟设(UCCSD)。UCCSD 是 VQE 拟设,通常用于运行量子化学模拟。

    在一阶 Trotter 近似内,UCCSD 酉函数由下式给出:

    .. math::

        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}

    其中 :math:`\hat{c}` 和 :math:`\hat{c}^\dagger` 是费米子湮灭和
    创建运算符和索引 :math:`r, s` 和 :math:`p, q` 在占用的和
    分别为空分子轨道。(更多细节见
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    该函数别名:  ``VQC_UCCSD`` 。

    :param weights: 包含参数的大小 ``(len(s_wires)+ len(d_wires))`` 张量
        :math:`\theta_{pr}` 和 :math:`\theta_{pqrs}` 输入 Z 旋转
        ``FermionicSingleExcitation`` 和 ``FermionicDoubleExcitation`` 。
    :param wires: 模板作用的量子比特索引
    :param s_wires: 包含量子比特索引的列表序列 ``[r,...,p]``
        由单一激发产生
        :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`,
        其中 :math:`\vert \mathrm{HF} \rangle` 表示 Hartee-Fock 参考态。
    :param d_wires: 列表序列,每个列表包含两个列表
        指定索引 ``[s, ...,r]`` 和 ``[q,..., p]`` 
        定义双激励 :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle` 。
    :param init_state: 长度 ``len(wires)`` occupation-number vector 表示
        高频状态。 ``init_state`` 在量子比特初始化状态。
    :param q_machine: 量子虚拟机设备。
    
    
    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import vqc_uccsd, QMachine, MeasureAll
        from pyvqnet.tensor import QTensor
        p0 = QTensor([2, 0.5, -0.2, 0.3, -2, 1, 3, 0])
        s_wires = [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]]
        d_wires = [[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]],
                [[0, 1], [4, 5]]]
        qm = QMachine(6)

        vqc_uccsd(p0, range(6), s_wires, d_wires, QTensor([1.0, 1, 0, 0, 0, 0]), qm)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.963802]]


vqc_zfeaturemap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.sv.torch.vqc_zfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.sv.torch.QMachine, data_map_func=None, rep: int = 2)

    一阶泡利 Z 演化电路。

    对于 3 个量子位和 2 次重复,电路表示为:

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘
    
    泡利弦固定为 ``Z``。 因此,一阶展开将是一个没有纠缠门的电路。

    :param input_feat: 表示输入参数的数组。
    :param q_machine: 量子虚拟机。
    :param data_map_func: 参数映射矩阵, 为可调用函数, 设计方式为: ``data_map_func = lambda x: x``。
    :param rep: 模块重复次数。
    
    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import vqc_zfeaturemap, QMachine, hadamard
        from pyvqnet.tensor import QTensor
        qm = QMachine(3)
        for i in range(3):
            hadamard(q_machine=qm, wires=[i])
        vqc_zfeaturemap(input_feat=QTensor([[0.1,0.2,0.3]]),q_machine = qm)
        print(qm.states)
 

vqc_zzfeaturemap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.sv.torch.vqc_zzfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.sv.torch.QMachine, data_map_func=None, entanglement: Union[str, List[List[int]],Callable[[int], List[int]]] = "full",rep: int = 2)

    二阶 Pauli-Z 演化电路。

    对于 3 个量子位、1 个重复和线性纠缠,电路表示为:

    .. parsed-literal::

        ┌───┐┌─────────────────┐
        ┤ H ├┤ U1(2.0*φ(x[0])) ├──■────────────────────────────■────────────────────────────────────
        ├───┤├─────────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[1])) ├┤ X ├┤ U1(2.0*φ(x[0],x[1])) ├┤ X ├──■────────────────────────────■──
        ├───┤├─────────────────┤└───┘└──────────────────────┘└───┘┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[2])) ├──────────────────────────────────┤ X ├┤ U1(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└─────────────────┘                                  └───┘└──────────────────────┘└───┘
    
    其中 ``φ`` 是经典的非线性函数,如果输入两个值则 ``φ(x,y) = (pi - x)(pi - y)``, 输入一个则为 ``φ(x) = x``, 用 ``data_map_func`` 表示如下:
    
    .. code-block::
        
        def data_map_func(x):
            coeff = x if x.shape[-1] == 1 else ft.reduce(lambda x, y: (np.pi - x) * (np.pi - y), x)
            return coeff

    :param input_feat: 表示输入参数的数组。
    :param q_machine: 量子虚拟机。
    :param data_map_func: 参数映射矩阵, 为可调用函数。 
    :param entanglement: 指定的纠缠结构。
    :param rep: 模块重复次数。
    
    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import vqc_zzfeaturemap, QMachine
        from pyvqnet.tensor import QTensor

        qm = QMachine(3)
        vqc_zzfeaturemap(q_machine=qm, input_feat=QTensor([[0.1,0.2,0.3]]))
        print(qm.states)


vqc_allsinglesdoubles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.sv.torch.vqc_allsinglesdoubles(weights, q_machine: pyvqnet.qnn.vqc.sv.torch.QMachine, hf_state, wires, singles=None, doubles=None)

    在这种情况下,我们有四个单激发和双激发来保留 Hartree-Fock 态的总自旋投影。

    由此产生的酉矩阵保留了粒子数量,并在初始 Hartree-Fock 状态和编码多激发配置的其他状态的叠加中准备了n量子位系统。
      
    :param weights: 大小为 ``(len(singles) + len(doubles),)`` 的QTensor,包含按顺序进入 vqc.qCircuit.single_excitation 和 vqc.qCircuit.double_excitation 操作的角度
    :param q_machine: 量子虚拟机。
    :param hf_state: 代表 Hartree-Fock 状态的长度 ``len(wires)`` 占用数向量, ``hf_state`` 用于初始化线路。
    :param wires: 作用的量子位。
    :param singles: 具有single_exitation操作所作用的两个量子位索引的列表序列。
    :param doubles: 具有double_exitation操作所作用的两个量子位索引的列表序列。

    例如,两个电子和六个量子位情况下的量子电路如下图所示:
    
.. image:: ./images/all_singles_doubles.png
    :width: 600 px
    :align: center

|

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import vqc_allsinglesdoubles, QMachine

        from pyvqnet.tensor import QTensor
        qubits = 4
        qm = QMachine(qubits)

        vqc_allsinglesdoubles(q_machine=qm, weights=QTensor([0.55, 0.11, 0.53]), 
                              hf_state = QTensor([1,1,0,0]), singles=[[0, 2], [1, 3]], doubles=[[0, 1, 2, 3]], wires=[0,1,2,3])
        print(qm.states)

vqc_basisrotation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.sv.torch.vqc_basisrotation(q_machine: pyvqnet.qnn.vqc.sv.torch.QMachine, wires, unitary_matrix: QTensor, check=False)

    实现一个电路,提供可用于执行精确的单体基础旋转的整体。线路来自于 `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ 中给出的单粒子费米子确定的酉变换 :math:`U(u)`
    
    .. math::

        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.
    
    :math:`U(u)` 通过使用论文 `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ 中给出的方案。
    

    :param q_machine: 量子虚拟机。
    :param wires: 作用的量子位。
    :param unitary_matrix: 指定基础变换的矩阵。
    :param check: 检测 `unitary_matrix` 是否为酉矩阵。

    Example::

        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.sv.torch import vqc_basisrotation, QMachine
        from pyvqnet.tensor import QTensor
        import numpy as np

        V = np.array([[0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [-0.21271 + 0.34938j, -0.38853 + 0.36497j, 0.61467 - 0.41317j]])

        eigen_vals, eigen_vecs = np.linalg.eigh(V)
        umat = eigen_vecs.T
        wires = range(len(umat))

        qm = QMachine(len(umat))

        vqc_basisrotation(q_machine=qm,
                        wires=wires,
                        unitary_matrix=QTensor(umat, dtype=qm.states.dtype))

        print(qm.states)



QuantumLayerAdjoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.sv.torch.QuantumLayerAdjoint(general_module, use_qpanda=False,name="")


    使用伴随矩阵方式进行梯度计算的可自动微分的变分量子线路层,参考  `Efficient calculation of gradients in classical simulations of variational quantum algorithms <https://arxiv.org/abs/2009.02823>`_ 。

    :param general_module: 一个仅使用 ``pyvqnet.qnn.vqc.sv.torch`` 下量子线路接口搭建的 ``pyvqnet.qnn.vqc.sv.torch.QModule`` 实例。
    :param use_qpanda: 是否使用qpanda线路进行前传,默认:False。
    :param name: 该层名字,默认为""。
    :return: 返回一个 QuantumLayerAdjoint 类实例。


    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` 以及 ``pyvqnet.qnn.vqc.QuantumLayerAdjoint`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    .. warning::
        Module 默认处于 `eval` 模式,如果需要训练参数，需要运行 `train()` 接口进入训练模式。

    .. note::

        general_module 的 QMachine 应设置 grad_method = "adjoint".

        当前支持由如下含参逻辑门 `RX`, `RY`, `RZ`, `PhaseShift`, `RXX`, `RYY`, `RZZ`, `RZX`, `U1`, `U2`, `U3` 以及其他不含参逻辑门构成的变分线路。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet import tensor
        from pyvqnet.qnn.vqc.sv.torch import QuantumLayerAdjoint, \
            QMachine, RX, RY, CNOT, T, \
                MeasureAll, RZ, VQC_HardwareEfficientAnsatz,\
                    QModule

        class QModel(QModule):
            def __init__(self, num_wires, dtype, grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype, grad_mode=grad_mode)
                self.rx_layer = RX(has_params=True, trainable=False, wires=0)
                self.ry_layer = RY(has_params=True, trainable=False, wires=1)
                self.rz_layer = RZ(has_params=True, trainable=False, wires=1)
                self.rz_layer2 = RZ(has_params=True, trainable=True, wires=1)

                self.rot = VQC_HardwareEfficientAnsatz(6, ["rx", "RY", "rz"],
                                                    entangle_gate="cnot",
                                                    entangle_rules="linear",
                                                    depth=5)
                self.tlayer = T(wires=1)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs={
                    "X1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                self.rx_layer(params=x[:, [0]], q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                self.ry_layer(params=x[:, [1]], q_machine=self.qm)
                self.tlayer(q_machine=self.qm)
                self.rz_layer(params=x[:, [2]], q_machine=self.qm)
                self.rz_layer2(q_machine=self.qm)
                self.rot(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)

                return rlt


        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])
        input_x = tensor.broadcast_to(input_x, [40, 3])
        input_x.requires_grad = True
        qunatum_model = QModel(num_wires=6,
                            dtype=pyvqnet.kcomplex64,
                            grad_mode="adjoint")
        adjoint_model = QuantumLayerAdjoint(qunatum_model)
        adjoint_model.train()
        batch_y = adjoint_model(input_x)
        batch_y.backward(pyvqnet.tensor.ones_like(batch_y))





张量网络后端变分量子线路模块
============================================

.. note::

    本模块基于 ``jax`` 实现自动微分与 GPU 加速。默认安装 ``pyvqnet`` 不包含该依赖，请使用 ``pip install jax``（CPU）或 ``pip install jax[cuda12]``（GPU，需 CUDA 12.6）安装。此外还需额外安装 ``tensornetwork``： ``pip install tensornetwork`` 。

张量网络（Tensor Network）通过将复杂的张量分解为多个低维张量的网络，显著降低了计算复杂度。

矩阵乘积态（Matrix Product State, MPS）是张量网络的一种特殊形式，MPS 将量子态表示为一系列矩阵的乘积，从而有效减少参数数量，降低了计算复杂度。

下面接口则是基于 ``torch`` 后端，对张量网络构建量子线路的功能支持，包括对构建量子线路基类、量子逻辑门、量子线路以及测量方法，并通过自动微分模拟代替参数移位法计算参数梯度。

以MPS方式构建量子线路弥补对大比特量子线路构建支持。

.. warning::

        通过 ``TNQMachine`` 中 ``use_mps`` 参数开启MPS构建量子线路功能， 支持大比特(100以及以上)量子线路实现。

.. warning::
        
        批量化与经典模块下使用方式不同，基于vmap的方式，数据以及参数构建线路需降一维输入，即对应态矢模拟时代码 ``x[:,i]`` 需要改为 ``x[i]``, 具体可查看下方接口中样例, 批次化执行必须同时基于 ``TNQMachine`` 和 ``TNQModule`` 并使用 ``TNQMachine`` 的 ``reset_states`` 显式指定批次大小。

基类
--------------------------------------------------

TNQModule
^^^^^^^^^^^^^^^^^^^^^^^^

基于张量网络编写变分量子线路模型需要继承于 ``TNQModule``

.. py:class:: pyvqnet.qnn.vqc.tn.torch.TNQModule(use_jit=False,vectorized_argnums=0,name="")

    在 `torch` 后端下,定义张量网络下量子变分线路模型 `Module` 应该继承的基类。
    该类用于使用张量网络来模块来用语执行量子线路。

    :param use_jit: 开启即时编译功能, 默认为False。
    :param vectorized_argnums: 要被向量化的参数,这些参数应该在同一维共享相同的批次形状,默认值为0
    :param name: 模块名。

    .. note::

        开启 ``use_jit`` 后，模型会使用 ``jax`` 的 ``jit`` 进行即时编译，首次运行将会进行编译，会耗时较长。

    .. note::

        该类以及其派生类仅适用于 ``pyvqnet.backends.set_backend("torch")`` , 不要与默认 ``pyvqnet.nn`` 下的 ``Module`` 混用。

    Example::

        import pyvqnet
        from pyvqnet.nn import Parameter
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import TNQModule
        from pyvqnet.qnn.vqc.tn.torch import TNQMachine, RX, RY, CNOT, PauliX, PauliZ,qmeasure,qcircuit,VQC_RotCircuit
        class QModel(TNQModule):
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)

                self.w = Parameter((2,4,3),initializer=pyvqnet.utils.initializer.quantum_uniform)
                self.cnot = CNOT(wires=[0, 1])
                self.batch_size = batch_size
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(batchsize=self.batch_size)

                def get_cnot(nqubits,qm):
                    for i in range(len(nqubits) - 1):
                        CNOT(wires = [nqubits[i], nqubits[i + 1]])(q_machine = qm)
                    CNOT(wires = [nqubits[len(nqubits) - 1], nqubits[0]])(q_machine = qm)


                def build_circuit(weights, xx, nqubits,qm):
                    def Rot(weights_j, nqubits,qm):#pylint:disable=invalid-name
                        VQC_RotCircuit(qm,nqubits,weights_j)

                    def basisstate(qm,xx, nqubits):
                        for i in nqubits:
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.ry(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])

                    basisstate(qm,xx,nqubits)

                    for i in range(weights.shape[0]):

                        weights_i = weights[i, :, :]
                        for j in range(len(nqubits)):
                            weights_j = weights_i[j]
                            Rot(weights_j, nqubits[j],qm)
                        get_cnot(nqubits,qm)

                build_circuit(self.w, x,range(4),self.qm)

                y= qmeasure.MeasureAll(obs={'Z0': 1})(self.qm)
                return y


        x= pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        y.backward(pyvqnet.tensor.ones_like(y))

TNQMachine
^^^^^^^^^^^^^^^^^^^^^^^^

基于张量网络编写变分量子线路设备需要 ``TNQMachine`` 进行初始化。 

.. py:class:: pyvqnet.qnn.vqc.tn.torch.TNQMachine(num_wires, dtype=pyvqnet.kcomplex64,use_mps=False)

    变分量子计算的模拟器类,包含states属性为量子线路的statevectors。

    .. warning::

        该类继承于 ``pyvqnet.nn.tn.TorchModule`` 以及 ``pyvqnet.qnn.QMachine`` 。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    .. warning::
        
        在每次运行一个完整的量子线路之前,必须使用 `pyvqnet.qnn.vqc.tn.torch.TNQMachine.reset_states(batchsize)` 将模拟器里面初态重新初始化,并且广播为
        (batchsize,*) 维度从而适应批量数据训练。

    .. warning::
        
        在张量网络的量子线路中，默认会开启 ``vmap`` 功能，在线路上的逻辑门参数上均为舍弃了批次维度，
        使用时，调用参数若维度为 [batch_size, \*], 在使用时舍弃第一个batch_size维度, 直接使用后面维度, 如对输入数据x[:,1] -> x[1], 对可训练参数也一致，可参考下列样例中xx, weights用法.


    :param num_wires: 量子比特数。
    :param dtype: 计算数据的数据类型。默认值是pyvqnet.kcomplex64,对应的参数精度为pyvqnet.kfloat32。
    :param use_mps: 是否基于mpscircuit进行模拟, 用于模拟大比特量子线路执行。

    :return: 输出一个TNQMachine对象。

    Example::
        
        import pyvqnet
        from pyvqnet.nn import Parameter
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import TNQModule
        from pyvqnet.qnn.vqc.tn.torch import TNQMachine, RX, RY, CNOT, PauliX, PauliZ,qmeasure,qcircuit,VQC_RotCircuit
        class QModel(TNQModule):
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)

                self.w = Parameter((2,4,3),initializer=pyvqnet.utils.initializer.quantum_uniform)
                self.cnot = CNOT(wires=[0, 1])
                self.batch_size = batch_size
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(batchsize=self.batch_size)

                def get_cnot(nqubits,qm):
                    for i in range(len(nqubits) - 1):
                        CNOT(wires = [nqubits[i], nqubits[i + 1]])(q_machine = qm)
                    CNOT(wires = [nqubits[len(nqubits) - 1], nqubits[0]])(q_machine = qm)


                def build_circuit(weights, xx, nqubits,qm):
                    def Rot(weights_j, nqubits,qm):#pylint:disable=invalid-name
                        VQC_RotCircuit(qm,nqubits,weights_j)

                    def basisstate(qm,xx, nqubits):
                        for i in nqubits:
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.ry(q_machine=qm, wires=i, params=xx[i])
                            qcircuit.rz(q_machine=qm, wires=i, params=xx[i])

                    basisstate(qm,xx,nqubits)

                    for i in range(weights.shape[0]):

                        weights_i = weights[i, :, :]
                        for j in range(len(nqubits)):
                            weights_j = weights_i[j]
                            Rot(weights_j, nqubits[j],qm)
                        get_cnot(nqubits,qm)

                build_circuit(self.w, x,range(4),self.qm)

                y= qmeasure.MeasureAll(obs={'Z0': 1})(self.qm)
                return y


        x= pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        y.backward(pyvqnet.tensor.ones_like(y))

    .. py:method:: get_states()

        获得张量网络中的states。

变分量子逻辑门模块
--------------------------------------------------


以下 ``pyvqnet.qnn.vqc`` 中的函数接口直接支持 ``torch`` 后端的 ``QTensor`` 进行计算，通过 ``pyvqnet.qnn.vqc.tn`` 下调用使用。

.. csv-table:: 已支持pyvqnet.qnn.vqc.sv.torch接口列表
   :file: ./images/same_apis_from_tn.csv

以下量子线路模块继承于 ``pyvqnet.qnn.vqc.tn.torch.TNQModule``,其中计算使用 ``torch.Tensor`` 进行计算。


.. warning::

    该类以及其派生类仅适用于 ``pyvqnet.backends.set_backend("torch")`` , 不要与默认 ``pyvqnet.nn`` 下的 ``Module`` 混用。

    这些类如果有非参数成员变量 ``_buffers`` ,则其中的数据为 ``torch.Tensor`` 类型。
    这些类如果有参数成员变量 ``_parmeters`` ,则其中的数据为 ``torch.nn.Parameter`` 类型。

I
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个I逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params: 是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 I 逻辑门实例

    Example::


        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import I,TNQMachine
        device = TNQMachine(4)
        layer = I(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.get_states())

Hadamard
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.Hadamard(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个Hadamard逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 Hadamard 逻辑门实例

    Example::


        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import Hadamard,TNQMachine
        device = TNQMachine(4)
        layer = Hadamard(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.get_states())

T
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.T(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个T逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 T 逻辑门实例

    Example::


        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import T,TNQMachine
        device = TNQMachine(4)
        layer = T(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.get_states())

S
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.S(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个S逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 S 逻辑门实例。

    Example::


        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import S,TNQMachine
        device = TNQMachine(4)
        layer = S(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.get_states())

PauliX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.PauliX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliX 逻辑门实例。

    Example::


        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import PauliX,TNQMachine
        device = TNQMachine(4)
        layer = PauliX(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.get_states())

PauliY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.PauliY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliY 逻辑门实例。

    Example::


        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import PauliY,TNQMachine
        device = TNQMachine(4)
        layer = PauliY(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.get_states())

PauliZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.PauliZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliZ 逻辑门实例。

    Example::


        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import PauliZ,TNQMachine
        device = TNQMachine(4)
        layer = PauliZ(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.get_states())

RX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.RX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RX逻辑门类 。


    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RX 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import RX,TNQMachine
        device = TNQMachine(4)
        layer = RX(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

RY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.RY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RY 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import RY,TNQMachine
        device = TNQMachine(4)
        layer = RY(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

RZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.RZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RZ 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import RZ,TNQMachine
        device = TNQMachine(4)
        layer = RZ(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

CRX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.CRX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRX 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import CRX,TNQMachine
        device = TNQMachine(4)
        layer = CRX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

CRY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.CRY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRY 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import CRY,TNQMachine
        device = TNQMachine(4)
        layer = CRY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

CRZ
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.torch.CRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRZ 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import CRZ,TNQMachine
        device = TNQMachine(4)
        layer = CRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

U1
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.U1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U1逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U1 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import U1,TNQMachine
        device = TNQMachine(4)
        layer = U1(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

U2
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.torch.U2(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U2逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U2 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import U2,TNQMachine
        device = TNQMachine(4)
        layer = U2(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

U3
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.torch.U3(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U3逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U3 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import U3,TNQMachine
        device = TNQMachine(4)
        layer = U3(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

CNOT
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.CNOT(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CNOT逻辑门类,也可称为CX。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CNOT 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import CNOT,TNQMachine
        device = TNQMachine(4)
        layer = CNOT(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

CY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.CY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CY 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import CY,TNQMachine
        device = TNQMachine(4)
        layer = CY(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

CZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.CZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CZ 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import CZ,TNQMachine
        device = TNQMachine(4)
        layer = CZ(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

CR
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.CR(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CR逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CR 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import CR,TNQMachine
        device = TNQMachine(4)
        layer = CR(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

SWAP
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.torch.SWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SWAP逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 SWAP 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import SWAP,TNQMachine
        device = TNQMachine(4)
        layer = SWAP(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

CSWAP
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.CSWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SWAP逻辑门类 。

    .. math:: CSWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}.

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CSWAP 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import CSWAP,TNQMachine
        device = TNQMachine(4)
        layer = CSWAP(wires=[0,1,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

RXX
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.torch.RXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RXX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RXX 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import RXX,TNQMachine
        device = TNQMachine(4)
        layer = RXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

RYY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.RYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RYY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RYY 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import RYY,TNQMachine
        device = TNQMachine(4)
        layer = RYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

RZZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.RZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个RZZ 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import RZZ,TNQMachine
        device = TNQMachine(4)
        layer = RZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

RZX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.RZX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RZX 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import RZX,TNQMachine
        device = TNQMachine(4)
        layer = RZX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

Toffoli
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.torch.Toffoli(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个Toffoli逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 Toffoli 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import Toffoli,TNQMachine
        device = TNQMachine(4)
        layer = Toffoli(  wires=[0,2,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

IsingXX
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.torch.IsingXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingXX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingXX 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import IsingXX,TNQMachine
        device = TNQMachine(4)
        layer = IsingXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

IsingYY
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.torch.IsingYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingYY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingYY 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import IsingYY,TNQMachine
        device = TNQMachine(4)
        layer = IsingYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

IsingZZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.IsingZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingZZ逻辑门类 。


    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingZZ 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import IsingZZ,TNQMachine
        device = TNQMachine(4)
        layer = IsingZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

IsingXY
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.torch.IsingXY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingXY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingXY 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import IsingXY,TNQMachine
        device = TNQMachine(4)
        layer = IsingXY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

PhaseShift
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.torch.PhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PhaseShift逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PhaseShift 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import PhaseShift,TNQMachine
        device = TNQMachine(4)
        layer = PhaseShift(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

MultiRZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.MultiRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个MultiRZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 MultiRZ 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import MultiRZ,TNQMachine
        device = TNQMachine(4)
        layer = MultiRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())


SDG
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.torch.SDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SDG逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 SDG 逻辑门实例。

    Example::


        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import SDG,TNQMachine
        device = TNQMachine(4)
        layer = SDG(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.get_states())

TDG
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.TDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SDG逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 TDG 逻辑门实例。

    Example::


        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import TDG,TNQMachine
        device = TNQMachine(4)
        layer = TDG(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.get_states())

ControlledPhaseShift
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.torch.ControlledPhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个ControlledPhaseShift逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 ControlledPhaseShift 逻辑门实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import ControlledPhaseShift,TNQMachine
        device = TNQMachine(4)
        layer = ControlledPhaseShift(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.get_states())

常见测量接口
--------------------------------------

VQC_Purity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.torch.VQC_Purity(state, qubits_idx, num_wires, use_tn=False)

    从态矢中计算特定量子比特 ``qubits_idx`` 上的纯度。

    .. math::
        \gamma = \text{Tr}(\rho^2)

    式中 :math:`\rho` 为密度矩阵。标准化量子态的纯度满足 :math:`\frac{1}{d} \leq \gamma \leq 1` ,
    其中 :math:`d` 是希尔伯特空间的维数。
    纯态的纯度是1。

    :param state: TNQMachine.get_states() 获取的量子态
    :param qubits_idx: 要计算纯度的量子比特位索引
    :param num_wires: 量子比特数
    :param use_tn: 张量网络后端时改成True, 默认False

    :return: 对应比特位置上的纯度。

    .. note::
        
        批量化必须搭配TNQModule使用。

    Example::

        import pyvqnet
        from pyvqnet.qnn.vqc.tn.torch import TNQMachine, qcircuit, TNQModule,VQC_Purity
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor

        x = QTensor([[0.7, 0.4], [1.7, 2.4]], requires_grad=True).toGPU()

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)
                self.device = TNQMachine(3)
                
            def forward(self, x):
                self.device.reset_states(2)
                qcircuit.rx(q_machine=self.device, wires=0, params=x[0])
                qcircuit.ry(q_machine=self.device, wires=1, params=x[1])
                qcircuit.ry(q_machine=self.device, wires=2, params=x[1])
                qcircuit.cnot(q_machine=self.device, wires=[0, 1])
                qcircuit.cnot(q_machine=self.device, wires=[2, 1])
                return VQC_Purity([0, 1], 3, self.device)

        model = QM().toGPU()
        y_tn = model(x)
        x.data.retain_grad()
        y_tn.backward(pyvqnet.tensor.ones_like(y_tn))
        print(y_tn)

VQC_VarMeasure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.torch.VQC_VarMeasure(q_machine, obs)

    提供的可观察量 ``obs`` 的方差。

    :param q_machine: 从pyqpanda get_qstate()获取的量子态
    :param obs: 测量观测量,当前支持Hadamard,I,PauliX,PauliY,PauliZ 几种Observable.

    :return: 计算可观测量方差。

    .. note::

        测量结果一般为[b,1],其中b为 q_machine.reset_states(b)的批处理数量b。

    Example::

        import pyvqnet
        from pyvqnet.qnn.vqc.tn.torch import TNQMachine, qcircuit, VQC_VarMeasure, TNQModule,PauliY
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[0.7, 0.4], [0.6, 0.4]], requires_grad=True).toGPU()

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)
                self.device = TNQMachine(3)
                
            def forward(self, x):
                self.device.reset_states(2)
                qcircuit.rx(q_machine=self.device, wires=0, params=x[0])
                qcircuit.ry(q_machine=self.device, wires=1, params=x[1])
                qcircuit.ry(q_machine=self.device, wires=2, params=x[1])
                qcircuit.cnot(q_machine=self.device, wires=[0, 1])
                qcircuit.cnot(q_machine=self.device, wires=[2, 1])
                return VQC_VarMeasure(q_machine= self.device, obs=PauliY(wires=0))
            
        model = QM().toGPU()
        y = model(x)
        x.data.retain_grad()
        y.backward(pyvqnet.tensor.ones_like(y))
        print(y)

        # [[0.9370641],
        # [0.9516521]]


VQC_DensityMatrixFromQstate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.torch.VQC_DensityMatrixFromQstate(state, indices, use_tn=False)

    计算量子态在一组特定量子比特上的密度矩阵。

    :param state: 一维列表状态向量。 这个列表的大小应该是 ``(2**N,)`` 对于量子比特个数 ``N`` ,qstate 应该从 000 ->111 开始。
    :param indices: 所考虑子系统中的量子比特索引列表。
    :param use_tn: 张量网络后端时改成True, 默认False.
    :return: 大小为“(b, 2**len(indices), 2**len(indices))”的密度矩阵,其中b为 q_machine.reset_states(b)的批处理数量b。

    Example::

        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn.torch import TNQMachine, qcircuit, VQC_DensityMatrixFromQstate,TNQModule
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[0.7,0.4],[1.7,2.4]], requires_grad=True).toGPU()
        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name=name, use_jit=True)
                self.device = TNQMachine(3)
                
            def forward(self, x):
                self.device.reset_states(2)
                qcircuit.rx(q_machine=self.device, wires=0, params=x[0])
                qcircuit.ry(q_machine=self.device, wires=1, params=x[1])
                qcircuit.ry(q_machine=self.device, wires=2, params=x[1])
                qcircuit.cnot(q_machine=self.device, wires=[0, 1])
                qcircuit.cnot(q_machine=self.device, wires=[2, 1])
                return VQC_DensityMatrixFromQstate([0,1], 3, self.device)
            
        model = QM().toGPU()
        y = model(x)
        x.data.retain_grad()
        y.backward(pyvqnet.tensor.ones_like(y))
        print(y)

        # [[[0.8155131+0.j        0.1718155+0.j        0.       +0.0627175j
        #   0.       +0.2976855j]
        #  [0.1718155+0.j        0.0669081+0.j        0.       +0.0244234j
        #   0.       +0.0627175j]
        #  [0.       -0.0627175j 0.       -0.0244234j 0.0089152+0.j
        #   0.0228937+0.j       ]
        #  [0.       -0.2976855j 0.       -0.0627175j 0.0228937+0.j
        #   0.1086637+0.j       ]]
        # 
        # [[0.3362115+0.j        0.1471083+0.j        0.       +0.1674582j
        #   0.       +0.3827205j]
        #  [0.1471083+0.j        0.0993662+0.j        0.       +0.1131119j
        #   0.       +0.1674582j]
        #  [0.       -0.1674582j 0.       -0.1131119j 0.1287589+0.j
        #   0.1906232+0.j       ]
        #  [0.       -0.3827205j 0.       -0.1674582j 0.1906232+0.j
        #   0.4356633+0.j       ]]]   



Probability
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.torch.Probability(wires=None, name="")

    计算量子线路在特定比特上概率测量结果。

    .. warning::
        
        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param wires: 测量比特的索引,列表、元组或者整数。
    :param name: 模块的名字,默认:""。
    :return: 测量结果,QTensor。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import Probability, qcircuit, TNQMachine, Hadamard, CNOT
        qm = TNQMachine(2)
        qm.reset_states(2)
        Hadamard(wires=0)(q_machine=qm)
        CNOT(wires=[0, 1])(q_machine=qm)
        ma = Probability(wires=1)
        y = ma(q_machine=qm)
        print(y)


MeasureAll
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.MeasureAll(obs=None, name="")

    计算量子线路的测量结果,支持输入obs为多个或单个泡利算子或哈密顿量。
    例如:

    {\'wires\': [0,  1], \'observables\': [\'x\', \'i\'],\'coefficient\':[0.23,-3.5]}
    或:
    {\'X0\': 0.23}
    或:
    [{\'wires\': [0, 2, 3],\'observables\': [\'X\', \'Y\', \'Z\'],\'coefficient\': [1, 0.5, 0.4]}, {\'wires\': [0, 1, 2],\'observables\': [\'X\', \'Y\', \'Z\'],\'coefficient\': [1, 0.5, 0.4]}]

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param obs: observable。
    :param name: 模块的名字,默认:""。
    :return: 一个 MeasureAll 测量方法实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import MeasureAll, qcircuit, TNQMachine, Hadamard, CNOT
        qm = TNQMachine(2)
        qm.reset_states(2)
        Hadamard(wires=0)(q_machine=qm)
        CNOT(wires=[0, 1])(q_machine=qm)
        obs_list = [{"Z0 Z1": 2}, {"X1 X0": 2}]
        ma = MeasureAll(obs=obs_list)
        y = ma(q_machine=qm)
        print(y)



Samples
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.Samples(wires=None, obs=None, shots = 1,name="")

    获取特定线路上的带有 shot 的样本结果

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param wires: 样本量子比特索引。默认值: None,根据运行时使用模拟器的所有比特。
    :param obs: 该值只能设为None。
    :param shots: 样本重复次数,默认值: 1。
    :param name: 此模块的名称,默认值: “”。
    :return: 一个 Samples 测量方法实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import Samples, TNQMachine, Hadamard, CNOT
        qm = TNQMachine(3)
        qm.reset_states(2)
        Hadamard(wires=0)(q_machine=qm)
        CNOT(wires=[0, 1])(q_machine=qm)
        ma = Samples(wires=[0, 1, 2], shots=3)
        y = ma(q_machine=qm)
        print(y)



HermitianExpval
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.HermitianExpval(obs=None, name="")

    计算量子线路某个厄密特量的期望。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param obs: 厄密特量。
    :param name: 模块的名字,默认:""。
    :return: 一个 HermitianExpval 测量方法实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import TNQModule, TNQMachine, HermitianExpval, RX, RY, CNOT, PauliX, PauliZ, VQC_RotCircuit, rx, ry
        from pyvqnet.tensor import QTensor, tensor
        from pyvqnet.nn import Parameter
        import numpy as np
        bsz = 3
        H = np.array([[8, 4, 0, -6], [4, 0, 4, 0], [0, 4, 8, 0], [-6, 0, 0, 0]])
        class QModel(TNQModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()
                self.rot_param = Parameter((3, ))
                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)
                self.rx_layer1 = VQC_RotCircuit
                self.ry_layer2 = RY(has_params=True,
                                    trainable=True,
                                    wires=0,
                                    init_params=tensor.QTensor([-0.5]))
                self.xlayer = PauliX(wires=0)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = HermitianExpval(obs = {'wires':(1,0),'observables':tensor.to_tensor(H)})

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(bsz)

                rx(q_machine=self.qm, wires=0, params=x[1])
                ry(q_machine=self.qm, wires=1, params=x[0])
                self.xlayer(q_machine=self.qm)
                self.rx_layer1(params=self.rot_param, wire=1, q_machine=self.qm)
                self.ry_layer2(q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                rlt = self.measure(q_machine = self.qm)

                return rlt


        input_x = tensor.arange(1, bsz * 2 + 1,
                                dtype=pyvqnet.kfloat32).reshape([bsz, 2])
        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=2, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward(pyvqnet.tensor.ones_like(batch_y))
 


常见量子线路模版
--------------------------------------------------

VQC_HardwareEfficientAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial = None,dtype=None)

    论文介绍的Hardware Efficient Ansatz的实现: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param n_qubits: 量子比特数。
    :param single_rot_gate_list: 单个量子比特旋转门列表由一个或多个作用于每个量子比特的旋转门构成。目前支持 Rx、Ry、Rz。
    :param entangle_gate: 非参数化纠缠门。支持 CNOT、CZ。默认值: CNOT。
    :param entangle_rules: 纠缠门在电路中的使用方式。'linear' 表示纠缠门将作用于每个相邻的量子比特。'all' 表示纠缠门将作用于任意两个量子比特。默认值: linear。
    :param depth: 假设的深度,默认值: 1。
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数,默认值: None,此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用float32。
    :return: 一个 VQC_HardwareEfficientAnsatz 实例。

    Example::

        from pyvqnet.nn.torch import Linear
        from pyvqnet.qnn.vqc.tn.torch.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ
        from pyvqnet.qnn.vqc.tn.torch import Probability,TNQMachine, TNQModule
        from pyvqnet import tensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)
                self.linearx = Linear(4,2)
                self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                            entangle_gate="cnot",
                                            entangle_rules="linear",
                                            depth=2)
                self.encode1 = RZ(wires=0)
                self.encode2 = RZ(wires=1)
                self.measure = Probability(wires=[0, 2])
                self.device = TNQMachine(4)
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(bz)
                y = self.linearx(x)
                self.encode1(params = y[0],q_machine = self.device,)
                self.encode2(params = y[1],q_machine = self.device,)
                self.ansatz(q_machine =self.device)
                return self.measure(q_machine =self.device)

        bz =3
        inputx = tensor.arange(1.0,bz*4+1).reshape([bz,4])
        inputx.requires_grad= True
        qlayer = QM()
        y = qlayer(inputx)
        y.backward(pyvqnet.tensor.ones_like(y))
        print(y)


VQC_BasicEntanglerTemplate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.VQC_BasicEntanglerTemplate(num_layer=1, num_qubits=1, rotation="RX", initial=None, dtype=None)

    由每个量子位上的单参数单量子位旋转组成的层,后跟一个闭合链或环组合的多个CNOT门。

    CNOT 门环将每个量子位与其邻居连接起来,最后一个量子位被认为是第一个量子位的邻居。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param num_layer: 量子比特线路层数。
    :param num_qubits: 量子比特数,默认为1。
    :param rotation: 使用单参数单量子比特门,``RX`` 被用作默认值。
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数,默认值: None,此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用float32。
    :return: 返回一个含可训练参数的VQC_BasicEntanglerTemplate实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import TNQModule, VQC_BasicEntanglerTemplate, Probability, TNQMachine
        from pyvqnet import tensor


        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_BasicEntanglerTemplate(2,
                                                    4,
                                                    "rz",
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 2])
                self.device = TNQMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward(pyvqnet.tensor.ones_like(y))
        print(y)



VQC_StronglyEntanglingTemplate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.VQC_StronglyEntanglingTemplate(num_layers=1, num_qubits=1, rotation = "RX", initial = None, dtype: = None)

    由单个量子比特旋转和纠缠器组成的层,参考 `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param num_layers: 重复层数,默认值: 1。
    :param num_qubits: 量子比特数,默认值: 1。
    :param rotation: 要使用的单参数单量子比特门,默认值: `RX`
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数,默认值: None,此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用 float32。
    :return: VQC_BasicEntanglerTemplate 实例


    Example::

        from pyvqnet.nn.torch import TorchModule,Linear,TorchModuleList
        from pyvqnet.qnn.vqc.tn.torch.qcircuit import VQC_StronglyEntanglingTemplate
        from pyvqnet.qnn.vqc.tn.torch import Probability, TNQMachine, TNQModule
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)
        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_StronglyEntanglingTemplate(2,
                                                    4,
                                                    None,
                                                    initial=tensor.ones([1, 1]))

                self.measure = Probability(wires=[0, 1])
                self.device = TNQMachine(4)

            def forward(self,x, *args, **kwargs):

                self.ansatz(q_machine=self.device)
                return self.measure(q_machine=self.device)

        bz = 1
        inputx = tensor.arange(1.0, bz * 4 + 1).reshape([bz, 4])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward(pyvqnet.tensor.ones_like(y))
        print(y)

VQC_QuantumEmbedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.torch.VQC_QuantumEmbedding(  num_repetitions_input, depth_input, num_unitary_layers, num_repetitions,initial = None,dtype = None,name= "")

    使用 RZ,RY,RZ 创建变分量子电路,将经典数据编码为量子态。
    参考 `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param num_repetitions_input: 子模块中输入编码的重复次数。
    :paramdepth_input: 输入维数。
    :param num_unitary_layers: 变分量子门的重复次数。
    :param num_repetitions: 子模块的重复次数。
    :param initial: 参数初始化值,默认为None
    :param dtype: 参数的类型,默认 None,使用float32.
    :param name: 类的名字

    Example::

        from pyvqnet.qnn.vqc.tn.torch.qcircuit import VQC_QuantumEmbedding
        from pyvqnet.qnn.vqc.tn.torch import TNQMachine, MeasureAll, TNQModule
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)
        depth_input = 2
        num_repetitions = 2
        num_repetitions_input = 2
        num_unitary_layers = 2
        nq = depth_input * num_repetitions_input
        bz = 12

        class QM(TNQModule):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_QuantumEmbedding(num_repetitions_input, depth_input,
                                                num_unitary_layers,
                                                num_repetitions, initial=tensor.full([1],12.0),dtype=pyvqnet.kfloat32)

                self.measure = MeasureAll(obs={f"Z{nq-1}":1})
                self.device = TNQMachine(nq)

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(bz)
                self.ansatz(x,q_machine=self.device)
                return self.measure(q_machine=self.device)

        inputx = tensor.arange(1.0, bz * depth_input + 1).reshape([bz, depth_input])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward(pyvqnet.tensor.ones_like(y))
        print(y)


ExpressiveEntanglingAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.torch.ExpressiveEntanglingAnsatz(type: int, num_wires: int, depth: int, dtype=None, name: str = "")

    论文 `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_ 中的 19 种不同的ansatz。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param type: 电路类型从 1 到 19,共19种线路。
    :param num_wires: 量子比特数。
    :param depth: 电路深度。
    :param dtype: 参数的数据类型, 默认值: None, 使用 float32。
    :param name: 名字,默认"".

    :return:
        一个 ExpressiveEntanglingAnsatz 实例

    Example::

        from pyvqnet.qnn.vqc.tn.torch.qcircuit import ExpressiveEntanglingAnsatz
        from pyvqnet.qnn.vqc.tn.torch import Probability, TNQMachine, MeasureAll, TNQModule
        from pyvqnet import tensor
        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(25)

        class QModel(TNQModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = TNQMachine(num_wires, dtype=dtype)
                self.c1 = ExpressiveEntanglingAnsatz(1,3,2)
                self.measure = MeasureAll(obs={
                    "Z1":1
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(1)
                self.c1(q_machine = self.qm)
                rlt = self.measure(q_machine=self.qm)
                return rlt
            

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward(pyvqnet.tensor.ones_like(batch_y))
        print(batch_y)


vqc_basis_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_basis_embedding(basis_state,q_machine)

    将n个二进制特征编码到 ``q_machine`` 的n个量子比特的基态。该函数别名 `VQC_BasisEmbedding` 。

    例如, 对于 ``basis_state=([0, 1, 1])``, 在量子系统下其基态为 :math:`|011 \rangle`。

    :param basis_state:  ``(n)`` 大小的二进制输入。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import vqc_basis_embedding,TNQMachine
        qm  = TNQMachine(3)
        vqc_basis_embedding(basis_state=[1,1,0],q_machine=qm)
        print(qm.get_states())




vqc_angle_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_angle_embedding(input_feat, wires, q_machine: pyvqnet.qnn.vqc.tn.torch.TNQMachine, rotation: str = "X")

    将 :math:`N` 特征编码到 :math:`n` 量子比特的旋转角度中, 其中 :math:`N \leq n`。
    该函数别名 `VQC_AngleEmbedding` 。

    旋转可以选择为 : 'X' , 'Y' , 'Z', 如 ``rotation`` 的参数定义为:

    * ``rotation='X'`` 将特征用作RX旋转的角度。

    * ``rotation='Y'`` 将特征用作RY旋转的角度。

    * ``rotation='Z'`` 将特征用作RZ旋转的角度。

     ``wires`` 代表旋转门在量子比特上的idx。

    :param input_feat: 表示参数的数组。
    :param wires: 量子比特idx。
    :param q_machine: 量子虚拟机设备。
    :param rotation: 旋转门,默认为“X”。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import vqc_angle_embedding, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(2)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='X')
        print(qm.get_states())
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Y')
        print(qm.get_states())
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Z')
        print(qm.get_states())


vqc_amplitude_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_amplitude_embedding(input_feature, q_machine)

    将 :math:`2^n` 特征编码为 :math:`n` 量子比特的振幅向量。该函数别名 `VQC_AmplitudeEmbedding` 。

    :param input_feature: 表示参数的numpy数组。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import vqc_amplitude_embedding, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(3)
        vqc_amplitude_embedding(QTensor([3.2,-2,-2,0.3,12,0.1,2,-1]), q_machine=qm)
        print(qm.get_states())


vqc_iqp_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:function:: pyvqnet.qnn.vqc.tn.vqc_iqp_embedding(input_feat, q_machine: pyvqnet.qnn.vqc.tn.torch.TNQMachine, rep: int = 1)

    使用IQP线路的对角门将 :math:`n` 特征编码为 :math:`n` 量子比特。该函数别名:  ``VQC_IQPEmbedding`` 。

    编码是由 `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_ 提出。

    通过指定 ``rep`` ,可以重复基本IQP线路。

    :param input_feat: 表示参数的数组。
    :param q_machine: 量子虚拟机设备。
    :param rep: 重复量子线路块次数,默认次数为1。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import vqc_iqp_embedding, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(3)
        vqc_iqp_embedding(QTensor([3.2,-2,-2]), q_machine=qm)
        print(qm.get_states())        



vqc_rotcircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_rotcircuit(q_machine, wire, params)

    任意单量子比特旋转的量子逻辑门组合。该函数别名:  ``VQC_RotCircuit`` 。

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.


    :param q_machine: 量子虚拟机设备。
    :param wire: 量子比特索引。
    :param params: 表示参数  :math:`[\phi, \theta, \omega]`。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import vqc_rotcircuit, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(3)
        vqc_rotcircuit(q_machine=qm, wire=[1],params=QTensor([2.0,1.5,2.1]))
        print(qm.get_states())


vqc_crot_circuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_crot_circuit(para,control_qubits,rot_wire,q_machine)

	受控Rot单量子比特旋转的量子逻辑门组合。该函数别名:  ``VQC_CRotCircuit`` 。

    .. math:: CR(\phi, \theta, \omega) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
            0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.
    
    :param para: 表示参数的数组。
    :param control_qubits: 控制量子比特索引。
    :param rot_wire: Rot量子比特索引。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn.torch import vqc_crot_circuit,TNQMachine, MeasureAll
        p = QTensor([2, 3, 4.0])
        qm = TNQMachine(2)
        vqc_crot_circuit(p, 0, 1, qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)




vqc_controlled_hadamard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_controlled_hadamard(wires, q_machine)

    受控Hadamard逻辑门量子线路。该函数别名:  ``VQC_Controlled_Hadamard`` 。

    .. math:: CH = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
            0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: 量子比特索引列表, 第一位是控制比特, 列表长度为2。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn.torch import vqc_controlled_hadamard,\
            TNQMachine, MeasureAll

        p = QTensor([0.2, 3, 4.0])
        qm = TNQMachine(3)
        vqc_controlled_hadamard([1, 0], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_ccz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_ccz(wires, q_machine)

    受控-受控-Z (controlled-controlled-Z) 逻辑门。该函数别名:  ``VQC_CCZ`` 。

    .. math::

        CCZ =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
        \end{pmatrix}
    
    :param wires: 量子比特下标列表,第一位是控制比特。列表长度为3。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn.torch import vqc_ccz,TNQMachine, MeasureAll
        p = QTensor([0.2, 3, 4.0])

        qm = TNQMachine(3)

        vqc_ccz([1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_fermionic_single_excitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_fermionic_single_excitation(weight, wires, q_machine)

    对泡利矩阵的张量积求幂的耦合簇单激励算子。矩阵形式下式给出:

    .. math::

        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    该函数别名:  ``VQC_FermionicSingleExcitation`` 。

    :param weight:  量子比特p上的参数, 只有一个元素.
    :param wires: 表示区间[r, p]中的量子比特索引子集。最小长度必须为2。第一索引值被解释为r,最后一个索引值被解释为p。
                中间的索引被CNOT门作用,以计算量子位集的奇偶校验。
    :param q_machine: 量子虚拟机设备。

    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn.torch import vqc_fermionic_single_excitation,\
            TNQMachine, MeasureAll
        qm = TNQMachine(3)
        p0 = QTensor([0.5])

        vqc_fermionic_single_excitation(p0, [1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

 


vqc_fermionic_double_excitation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_fermionic_double_excitation(weight, wires1, wires2, q_machine)

    对泡利矩阵的张量积求幂的耦合聚类双激励算子,矩阵形式由下式给出:

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \{ \theta (\hat{c}_p^\dagger \hat{c}_q^\dagger
        \hat{c}_r \hat{c}_s - \mathrm{H.c.}) \},

    其中 :math:`\hat{c}` 和 :math:`\hat{c}^\dagger` 是费米子湮灭和
    创建运算符和索引 :math:`r, s` 和 :math:`p, q` 在占用的和
    分别为空分子轨道。 使用 `Jordan-Wigner 变换
    <https://arxiv.org/abs/1208.5986>`_ 上面定义的费米子算子可以写成
    根据 Pauli 矩阵(有关更多详细信息,请参见
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_)

    .. math::

        \hat{U}_{pqrs}(\theta) = \mathrm{exp} \Big\{
        \frac{i\theta}{8} \bigotimes_{b=s+1}^{r-1} \hat{Z}_b \bigotimes_{a=q+1}^{p-1}
        \hat{Z}_a (\hat{X}_s \hat{X}_r \hat{Y}_q \hat{X}_p +
        \hat{Y}_s \hat{X}_r \hat{Y}_q \hat{Y}_p +\\ \hat{X}_s \hat{Y}_r \hat{Y}_q \hat{Y}_p +
        \hat{X}_s \hat{X}_r \hat{X}_q \hat{Y}_p - \mathrm{H.c.}  ) \Big\}

    该函数别名:  ``VQC_FermionicDoubleExcitation`` 。

    :param weight: 可变参数
    :param wires1: 代表的量子比特的索引列表区间 [s, r] 中占据量子比特的子集。第一个索引被解释为 s,最后一索引被解释为 r。 CNOT 门对中间的索引进行操作,以计算一组量子位的奇偶性。
    :param wires2: 代表的量子比特的索引列表区间 [q, p] 中占据量子比特的子集。第一根索引被解释为 q,最后一索引被解释为 p。 CNOT 门对中间的索引进行操作,以计算一组量子位的奇偶性。
    :param q_machine: 量子虚拟机设备。

    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn.torch import vqc_fermionic_double_excitation,\
            TNQMachine, MeasureAll
        qm = TNQMachine(5)
        p0 = QTensor([0.5])

        vqc_fermionic_double_excitation(p0, [0, 1], [2, 3], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)
 

vqc_uccsd
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_uccsd(weights, wires, s_wires, d_wires, init_state, q_machine)

    实现酉耦合簇单激发和双激发拟设(UCCSD)。UCCSD 是 VQE 拟设,通常用于运行量子化学模拟。

    在一阶 Trotter 近似内,UCCSD 酉函数由下式给出:

    .. math::

        \hat{U}(\vec{\theta}) =
        \prod_{p > r} \mathrm{exp} \Big\{\theta_{pr}
        (\hat{c}_p^\dagger \hat{c}_r-\mathrm{H.c.}) \Big\}
        \prod_{p > q > r > s} \mathrm{exp} \Big\{\theta_{pqrs}
        (\hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r \hat{c}_s-\mathrm{H.c.}) \Big\}

    其中 :math:`\hat{c}` 和 :math:`\hat{c}^\dagger` 是费米子湮灭和
    创建运算符和索引 :math:`r, s` 和 :math:`p, q` 在占用的和
    分别为空分子轨道。(更多细节见
    `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`_):

    该函数别名:  ``VQC_UCCSD`` 。

    :param weights: 包含参数的大小 ``(len(s_wires)+ len(d_wires))`` 张量
        :math:`\theta_{pr}` 和 :math:`\theta_{pqrs}` 输入 Z 旋转
        ``FermionicSingleExcitation`` 和 ``FermionicDoubleExcitation`` 。
    :param wires: 模板作用的量子比特索引
    :param s_wires: 包含量子比特索引的列表序列 ``[r,...,p]``
        由单一激发产生
        :math:`\vert r, p \rangle = \hat{c}_p^\dagger \hat{c}_r \vert \mathrm{HF} \rangle`,
        其中 :math:`\vert \mathrm{HF} \rangle` 表示 Hartee-Fock 参考态。
    :param d_wires: 列表序列,每个列表包含两个列表
        指定索引 ``[s, ...,r]`` 和 ``[q,..., p]`` 
        定义双激励 :math:`\vert s, r, q, p \rangle = \hat{c}_p^\dagger \hat{c}_q^\dagger \hat{c}_r\hat{c}_s \vert \mathrm{HF} \rangle` 。
    :param init_state: 长度 ``len(wires)`` occupation-number vector 表示
        高频状态。 ``init_state`` 在量子比特初始化状态。
    :param q_machine: 量子虚拟机设备。
    
    
    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import vqc_uccsd, TNQMachine, MeasureAll
        from pyvqnet.tensor import QTensor
        p0 = QTensor([2, 0.5, -0.2, 0.3, -2, 1, 3, 0])
        s_wires = [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]]
        d_wires = [[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]],
                [[0, 1], [4, 5]]]
        qm = TNQMachine(6)

        vqc_uccsd(p0, range(6), s_wires, d_wires, QTensor([1.0, 1, 0, 0, 0, 0]), qm)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.963802]]


vqc_zfeaturemap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_zfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.tn.torch.TNQMachine, data_map_func=None, rep: int = 2)

    一阶泡利 Z 演化电路。

    对于 3 个量子位和 2 次重复,电路表示为:

    .. parsed-literal::

        ┌───┐┌──────────────┐┌───┐┌──────────────┐
        ┤ H ├┤ U1(2.0*x[0]) ├┤ H ├┤ U1(2.0*x[0]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[1]) ├┤ H ├┤ U1(2.0*x[1]) ├
        ├───┤├──────────────┤├───┤├──────────────┤
        ┤ H ├┤ U1(2.0*x[2]) ├┤ H ├┤ U1(2.0*x[2]) ├
        └───┘└──────────────┘└───┘└──────────────┘
    
    泡利弦固定为 ``Z``。 因此,一阶展开将是一个没有纠缠门的电路。

    :param input_feat: 表示输入参数的数组。
    :param q_machine: 量子虚拟机。
    :param data_map_func: 参数映射矩阵, 为可调用函数, 设计方式为: ``data_map_func = lambda x: x``。
    :param rep: 模块重复次数。
    
    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import vqc_zfeaturemap, TNQMachine, hadamard
        from pyvqnet.tensor import QTensor
        qm = TNQMachine(3)
        for i in range(3):
            hadamard(q_machine=qm, wires=[i])
        vqc_zfeaturemap(input_feat=QTensor([[0.1,0.2,0.3]]),q_machine = qm)
        print(qm.get_states())
 

vqc_zzfeaturemap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_zzfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.tn.torch.TNQMachine, data_map_func=None, entanglement: Union[str, List[List[int]],Callable[[int], List[int]]] = "full",rep: int = 2)

    二阶 Pauli-Z 演化电路。

    对于 3 个量子位、1 个重复和线性纠缠,电路表示为:

    .. parsed-literal::

        ┌───┐┌─────────────────┐
        ┤ H ├┤ U1(2.0*φ(x[0])) ├──■────────────────────────────■────────────────────────────────────
        ├───┤├─────────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[1])) ├┤ X ├┤ U1(2.0*φ(x[0],x[1])) ├┤ X ├──■────────────────────────────■──
        ├───┤├─────────────────┤└───┘└──────────────────────┘└───┘┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[2])) ├──────────────────────────────────┤ X ├┤ U1(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└─────────────────┘                                  └───┘└──────────────────────┘└───┘
    
    其中 ``φ`` 是经典的非线性函数,如果输入两个值则 ``φ(x,y) = (pi - x)(pi - y)``, 输入一个则为 ``φ(x) = x``, 用 ``data_map_func`` 表示如下:
    
    .. code-block::
        
        def data_map_func(x):
            coeff = x if x.shape[-1] == 1 else ft.reduce(lambda x, y: (np.pi - x) * (np.pi - y), x)
            return coeff

    :param input_feat: 表示输入参数的数组。
    :param q_machine: 量子虚拟机。
    :param data_map_func: 参数映射矩阵, 为可调用函数。 
    :param entanglement: 指定的纠缠结构。
    :param rep: 模块重复次数。
    
    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import vqc_zzfeaturemap, TNQMachine
        from pyvqnet.tensor import QTensor

        qm = TNQMachine(3)
        vqc_zzfeaturemap(q_machine=qm, input_feat=QTensor([[0.1,0.2,0.3]]))
        print(qm.get_states())


vqc_allsinglesdoubles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_allsinglesdoubles(weights, q_machine: pyvqnet.qnn.vqc.tn.torch.TNQMachine, hf_state, wires, singles=None, doubles=None)

    在这种情况下,我们有四个单激发和双激发来保留 Hartree-Fock 态的总自旋投影。

    由此产生的酉矩阵保留了粒子数量,并在初始 Hartree-Fock 状态和编码多激发配置的其他状态的叠加中准备了n量子位系统。
      
    :param weights: 大小为 ``(len(singles) + len(doubles),)`` 的QTensor,包含按顺序进入 vqc.qCircuit.single_excitation 和 vqc.qCircuit.double_excitation 操作的角度
    :param q_machine: 量子虚拟机。
    :param hf_state: 代表 Hartree-Fock 状态的长度 ``len(wires)`` 占用数向量, ``hf_state`` 用于初始化线路。
    :param wires: 作用的量子位。
    :param singles: 具有single_exitation操作所作用的两个量子位索引的列表序列。
    :param doubles: 具有double_exitation操作所作用的两个量子位索引的列表序列。

    例如,两个电子和六个量子位情况下的量子电路如下图所示:
    
.. image:: ./images/all_singles_doubles.png
    :width: 600 px
    :align: center

|

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import vqc_allsinglesdoubles, TNQMachine

        from pyvqnet.tensor import QTensor
        qubits = 4
        qm = TNQMachine(qubits)

        vqc_allsinglesdoubles(q_machine=qm, weights=QTensor([0.55, 0.11, 0.53]), 
                              hf_state = QTensor([1,1,0,0]), singles=[[0, 2], [1, 3]], doubles=[[0, 1, 2, 3]], wires=[0,1,2,3])
        print(qm.get_states())

vqc_basisrotation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_basisrotation(q_machine: pyvqnet.qnn.vqc.tn.torch.TNQMachine, wires, unitary_matrix: QTensor, check=False)

    实现一个电路,提供可用于执行精确的单体基础旋转的整体。线路来自于 `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ 中给出的单粒子费米子确定的酉变换 :math:`U(u)`
    
    .. math::

        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.
    
    :math:`U(u)` 通过使用论文 `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ 中给出的方案。
    

    :param q_machine: 量子虚拟机。
    :param wires: 作用的量子位。
    :param unitary_matrix: 指定基础变换的矩阵。
    :param check: 检测 `unitary_matrix` 是否为酉矩阵。

    Example::

        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import vqc_basisrotation, TNQMachine
        from pyvqnet.tensor import QTensor
        import numpy as np

        V = np.array([[0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [0.73678 + 0.27511j, -0.5095 + 0.10704j, -0.06847 + 0.32515j],
                    [-0.21271 + 0.34938j, -0.38853 + 0.36497j, 0.61467 - 0.41317j]])

        eigen_vals, eigen_vecs = np.linalg.eigh(V)
        umat = eigen_vecs.T
        wires = range(len(umat))

        qm = TNQMachine(len(umat))

        vqc_basisrotation(q_machine=qm,
                        wires=wires,
                        unitary_matrix=QTensor(umat, dtype=qm.dtype))

        print(qm.get_states())

量子机器学习批量数据张量网络训练测试
--------------------------------------

.. note::
    本测试中 VQNet 的张量网络后端依赖 ``jax`` 与 ``tensornetwork`` ，安装方式参见上文 "张量网络后端变分量子线路模块" 章节中的说明。

本测试沿用上节测试2的线路结构（RY(data) → [RY(param) → CRZ(param) → RY(param) → CRZ(param)] × L），评估张量网络（Tensor Network）模拟后端在变分量子线路批量数据梯度计算中的性能。测试在 Linux 系统下使用 GPU，对基于张量网络收缩模拟的 VQNet JAX TN 后端（ ``pyvqnet.qnn.vqc.tn.torch`` ，torch 前端 + JAX 收缩引擎）与 TensorCircuit（jax 后端），以及基于状态向量模拟的 VQNet（含 ``fused_multi_crz`` 融合算子版本）、TorchQuantum、PennyLane、DeepQuantum、MindQuantum 进行了对比。测试在量子比特数目 n = 4 与 n = 10 、线路深度 layer 10 、批量数据规模 batch size 512/128/32 的配置下分别进行，快速框架统计 20 次运行的平均耗时，较慢框架（PennyLane、MindQuantum）仅统计 2 次以节约时间。

.. image:: ./images/grad_bench_full_n4.png
   :width: 600 px
   :align: center

|

.. image:: ./images/grad_bench_full_n10.png
   :width: 600 px
   :align: center

|

测试环境与软件版本如下：

.. code-block:: text

    +-------------------+--------------------------------------+
    | 项目              | 规格                                 |
    +===================+======================================+
    | CPU               | 2x NVIDIA GeForce RTX 3090           |
    +-------------------+--------------------------------------+
    | CUDA              | 12.6                                 |
    +-------------------+--------------------------------------+
    | python            | 3.10.0                               |
    +-------------------+--------------------------------------+
    | pyvqnet           | 2.18.1                               |
    +-------------------+--------------------------------------+
    | torch             | 2.11.0+cu126                         |
    +-------------------+--------------------------------------+
    | jax               | 0.6.2 (+ jaxlib 0.6.2)               |
    +-------------------+--------------------------------------+
    | tensorcircuit-ng  | 1.9.1                                |
    +-------------------+--------------------------------------+
    | pennylane         | 0.42.3                               |
    +-------------------+--------------------------------------+
    | deepquantum       | 4.5.0                                |
    +-------------------+--------------------------------------+
    | mindquantum       | 0.12.0 (mqvector_gpu)                |
    +-------------------+--------------------------------------+
    | torchquantum      | 0.2.0                                |
    +-------------------+--------------------------------------+

测试代码如下：

.. code-block:: python

    """
    VQC gradient benchmark across quantum ML frameworks.

    Tested software versions (Linux, 2026-09-02):
        python            3.10.0
        pyvqnet           2.18.1 (PyPI)
        torch             2.11.0+cu126
        CUDA              12.6, 2x NVIDIA GeForce RTX 3090
        numpy             2.2.6
        jax               0.6.2 (+ jaxlib 0.6.2, jax-cuda12-plugin 0.6.2)
        tensorcircuit-ng  1.9.1
        pennylane         0.42.3
        deepquantum       4.5.0
        mindquantum       0.12.0 (mqvector_gpu)
        torchquantum      0.2.0
        matplotlib        3.10.7
    """
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from pyvqnet.tensor import tensor
    from pyvqnet.qnn.vqc import RX, RY, RZ, crz, PauliX, PauliY, PauliZ, paulix, pauliy, pauliz, rx, ry, rz, MeasureAll, fused_multi_crz
    from pyvqnet.nn import ParameterDict, Parameter
    from pyvqnet.qnn.vqc import QModule, QMachine
    import numpy as np
    import pyvqnet
    import time

    QuantumDevice = QMachine
    class Encoder(QModule):

        def __init__(self):
            super().__init__()
            pass

        def forward(self, x, qdev):
            raise NotImplementedError

    op_name_dict = {
        "x": PauliX,
        "y": PauliY,
        "z": PauliZ,
        "rx": RX,
        "ry": RY,
        "rz": RZ
    }

    func_name_dict = {
        "x": paulix,
        "y": pauliy,
        "z": pauliz,
        "rx": rx,
        "ry": ry,
        "rz": rz
    }

    class GeneralEncoder(Encoder):
        """Encoder applying a list of gate configs: {'input_idx', 'func', 'wires'}."""

        def __init__(self, func_list):
            super().__init__()
            self.func_list = func_list

        def forward(self, x, qdev):
            for info in self.func_list:
                if op_name_dict[info["func"]].num_params > 0:
                    params = x[:, info["input_idx"]]
                else:
                    params = None

                func_name_dict[info["func"]](qdev,
                                             wires=info["wires"],
                                             params=params)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)


    class VQC_new(QModule):
        """VQC using fused_multi_crz - one parameter vector per layer"""

        def __init__(self, n_wires: int = 4, n_qlayers: int = 1):
            super().__init__()
            self.n_wires = n_wires
            self.n_qlayers = n_qlayers
            self.dev = QuantumDevice(self.n_wires)
            enc_cnt = list()
            for i in range(self.n_wires):
                cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
                enc_cnt.append(cnt)

            self.encoder = GeneralEncoder(enc_cnt)
            self._use_vqnet = True

            # Initialize parameters - each layer has one vector for fused_multi_crz
            self.params_ry1_dct = ParameterDict()
            self.params_ry2_dct = ParameterDict()
            self.params_crx1_dct = ParameterDict()
            self.params_crx2_dct = ParameterDict()

            for k in range(self.n_qlayers):
                # fused_multi_crz: one parameter vector of size n_wires per layer
                self.params_crx1_dct[str(k)] = Parameter([self.n_wires])
                self.params_crx2_dct[str(k)] = Parameter([self.n_wires])
                for i in range(self.n_wires):
                    self.params_ry1_dct[str(i + k * self.n_wires)] = Parameter([1])
                    self.params_ry2_dct[str(i + k * self.n_wires)] = Parameter([1])

            obs_list = []
            for i in range(self.n_wires):
                obs_list.append({f"Z{i}": 1})

            self.measure = MeasureAll(obs=obs_list)

        def forward(self, x):
            q_device = self.dev
            q_device.reset_states(x.shape[0])

            self.encoder(x, q_device)

            for k in range(self.n_qlayers):
                # RY gates
                for i in range(self.n_wires):
                    ry(q_machine=q_device, wires=i, params=self.params_ry1_dct[str(i + k * self.n_wires)])

                # fused_multi_crz - forward direction
                obj_qubits = [(i + 1) % self.n_wires for i in range(self.n_wires - 1, -1, -1)]
                ctrls = list(range(self.n_wires - 1, -1, -1))
                fused_multi_crz(
                    q_machine=q_device,
                    params=self.params_crx1_dct[str(k)],
                    obj_qubits=obj_qubits,
                    ctrls=ctrls)

                # RY gates
                for i in range(self.n_wires):
                    ry(q_machine=q_device, params=self.params_ry2_dct[str(i + k * self.n_wires)], wires=i)

                # fused_multi_crz - reverse direction
                obj_qubits = [(i - 1) % self.n_wires for i in [self.n_wires - 1] + list(range(self.n_wires - 1))]
                ctrls = [self.n_wires - 1] + list(range(self.n_wires - 1))
                fused_multi_crz(
                    q_machine=q_device,
                    params=self.params_crx2_dct[str(k)],
                    obj_qubits=obj_qubits,
                    ctrls=ctrls)

            return self.measure(q_device)


    class VQC(QModule):
        """VQC using individual crz gates - one parameter per gate"""

        def __init__(self, n_wires: int = 4, n_qlayers: int = 1):
            super().__init__()
            self.n_wires = n_wires
            self.n_qlayers = n_qlayers
            self.dev = QuantumDevice(self.n_wires)
            enc_cnt = list()
            for i in range(self.n_wires):
                cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
                enc_cnt.append(cnt)

            self.encoder = GeneralEncoder(enc_cnt)
            self._use_vqnet = True

            # Initialize parameters - one parameter per gate
            self.params_ry1_dct = ParameterDict()
            self.params_ry2_dct = ParameterDict()
            self.params_crx1_dct = ParameterDict()
            self.params_crx2_dct = ParameterDict()

            for k in range(self.n_qlayers):
                for i in range(self.n_wires):
                    self.params_crx1_dct[str(i + k * self.n_wires)] = Parameter([1])
                    self.params_crx2_dct[str(i + k * self.n_wires)] = Parameter([1])
                    self.params_ry1_dct[str(i + k * self.n_wires)] = Parameter([1])
                    self.params_ry2_dct[str(i + k * self.n_wires)] = Parameter([1])

            obs_list = []
            for i in range(self.n_wires):
                obs_list.append({f"Z{i}": 1})

            self.measure = MeasureAll(obs=obs_list)

        def forward(self, x):
            q_device = self.dev
            q_device.reset_states(x.shape[0])

            self.encoder(x, q_device)

            for k in range(self.n_qlayers):
                # RY gates
                for i in range(self.n_wires):
                    ry(q_machine=q_device, wires=i, params=self.params_ry1_dct[str(i + k * self.n_wires)])

                # crz gates - forward direction (control -> target)
                # wires=[control, target], params on control-target pair
                for i in range(self.n_wires - 1, -1, -1):
                    crz(
                        q_machine=q_device,
                        params=self.params_crx1_dct[str(i + k * self.n_wires)],
                        wires=[i, (i + 1) % self.n_wires])

                # RY gates
                for i in range(self.n_wires):
                    ry(q_machine=q_device, params=self.params_ry2_dct[str(i + k * self.n_wires)], wires=i)

                # crz gates - reverse direction
                for i in [self.n_wires - 1] + list(range(self.n_wires - 1)):
                    crz(
                        q_machine=q_device,
                        params=self.params_crx2_dct[str(i + k * self.n_wires)],
                        wires=[i, (i - 1) % self.n_wires])

            return self.measure(q_device)


    def benchmark(f, *args, trials=10, sync_fn=None):
        time0 = time.time()
        r = f(*args)
        if sync_fn:
            sync_fn(r)
        time1 = time.time()
        for _ in range(trials):
            r = f(*args)
        if sync_fn:
            sync_fn(r)
        time2 = time.time()
        if trials > 0:
            time21 = (time2 - time1) / trials
        else:
            time21 = 0
        ts = (time1 - time0, time21)
        print('staging time: %.6f s' % ts[0])
        if trials > 0:
            print('running time: %.6f s' % ts[1])
        return r, ts


    def grad_pyvqnet_vqc_new(b, n, l, trials=10):
        """Test VQC_new (fused_multi_crz)"""
        pyvqnet.backends.set_backend("pyvqnet")
        pyvqnet.utils.set_random_seed(42)  # Set seed before creating layer
        layer = VQC_new(n, l)
        layer.toGPU(1000)

        def get_grad(values):
            r = layer(values)
            r.backward()
            return values.grad

        input = tensor.ones([b, n], device=1000)
        input.requires_grad = True
        return benchmark(get_grad, input, trials=trials,
                         sync_fn=lambda r: r.numpy())


    def grad_pyvqnet_vqc(b, n, l, trials=10):
        pyvqnet.backends.set_backend("pyvqnet")
        pyvqnet.utils.set_random_seed(42)  # Set seed before creating layer
        layer = VQC(n, l)
        layer.toGPU(1001)

        def get_grad(values):
            r = layer(values)
            r.backward()
            return values.grad

        input = tensor.ones([b, n], device=1001)
        input.requires_grad = True
        return benchmark(get_grad, input, trials=trials,
                         sync_fn=lambda r: r.numpy())


    def grad_tq_vqc(b, n, l, trials=10):
        """Test TorchQuantum VQC matching VQNet VQC structure"""
        import torchquantum as tq
        import torch
        import torch.cuda

        class VQC_TQ(tq.QuantumModule):
            """TorchQuantum VQC matching VQNet's VQC structure"""

            def __init__(self, n_wires: int = 4, n_qlayers: int = 1):
                super().__init__()
                self.n_wires = n_wires
                self.n_qlayers = n_qlayers

                enc_cnt = list()
                for i in range(self.n_wires):
                    cnt = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
                    enc_cnt.append(cnt)
                self.encoder = tq.GeneralEncoder(enc_cnt)

                self.params_ry1_dct = tq.QuantumModuleDict()
                self.params_ry2_dct = tq.QuantumModuleDict()
                self.params_crx1_dct = tq.QuantumModuleDict()
                self.params_crx2_dct = tq.QuantumModuleDict()

                for k in range(self.n_qlayers):
                    for i in range(self.n_wires):
                        self.params_ry1_dct[str(i + k * self.n_wires)] = tq.RY(has_params=True, trainable=True)
                        self.params_crx1_dct[str(i + k * self.n_wires)] = tq.CRZ(has_params=True, trainable=True)
                        self.params_ry2_dct[str(i + k * self.n_wires)] = tq.RY(has_params=True, trainable=True)
                        self.params_crx2_dct[str(i + k * self.n_wires)] = tq.CRZ(has_params=True, trainable=True)

                self.measure = tq.MeasureMultipleTimes([{'wires': range(self.n_wires), 'observables': ['z'] * self.n_wires}])

                from torchquantum import QuantumDevice as TQQuantumDevice
                self.dev = TQQuantumDevice(self.n_wires)

            def forward(self, x: torch.Tensor):
                q_device = self.dev
                q_device.reset_states(x.shape[0])
                self.encoder(q_device, x)

                for k in range(self.n_qlayers):
                    for i in range(self.n_wires):
                        self.params_ry1_dct[str(i + k * self.n_wires)](q_device, wires=i)

                    for i in range(self.n_wires - 1, -1, -1):
                        self.params_crx1_dct[str(i + k * self.n_wires)](q_device, wires=[i, (i + 1) % self.n_wires])

                    for i in range(self.n_wires):
                        self.params_ry2_dct[str(i + k * self.n_wires)](q_device, wires=i)

                    for i in [self.n_wires - 1] + list(range(self.n_wires - 1)):
                        self.params_crx2_dct[str(i + k * self.n_wires)](q_device, wires=[i, (i - 1) % self.n_wires])

                return self.measure(q_device)

        torch.manual_seed(42)
        layer = VQC_TQ(n, l)
        layer.to("cuda:1")

        def get_grad(values):
            r = layer(values)
            r.backward(torch.ones_like(r))
            return values.grad

        input = torch.ones([b, n], device="cuda:1")
        input.requires_grad = True
        return benchmark(get_grad, input, trials=trials,
                         sync_fn=lambda _: torch.cuda.synchronize())

    # ──────────────────────────────────────────────
    # PennyLane benchmark  (pennylane == 0.44.1 on py>=3.11, or 0.42.3 on py310;
    # 0.45.x has a regression where default.qubit initialises state on CPU)
    # ──────────────────────────────────────────────

    def grad_pl_vqc(b, n, l, trials=1):
        """PennyLane VQC (default.qubit) matching VQNet VQC structure."""
        import pennylane as qml
        from functools import reduce
        import torch
        assert qml.version() in ("0.44.1", "0.42.3"), (
            f"PennyLane {qml.version()} is not supported; "
            "0.45.x has a GPU device-mismatch regression. Use 0.44.1 (py>=3.11) "
            "or 0.42.3 (py310)."
        )
        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights_ry1, weights_crz1, weights_ry2, weights_crz2):
            for j in range(l):
                for i in range(n):
                    qml.RY(inputs[:, i], wires=i)
                for i in range(n):
                    qml.RY(weights_ry1[j, i], wires=i)
                for i in range(n - 1, -1, -1):
                    qml.CRZ(weights_crz1[j, i], wires=[i, (i + 1) % n])
                for i in range(n):
                    qml.RY(weights_ry2[j, i], wires=i)
                for i in [n - 1] + list(range(n - 1)):
                    qml.CRZ(weights_crz2[j, i], wires=[i, (i - 1) % n])

            obs = reduce(lambda x, y: x @ y, [qml.PauliZ(i) for i in range(n)])
            return qml.expval(obs)

        weight_shapes = {
            "weights_ry1": (l, n),
            "weights_crz1": (l, n),
            "weights_ry2": (l, n),
            "weights_crz2": (l, n),
        }

        def get_grad_pl(inputs):
            torch.manual_seed(42)
            qlayer = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)
            qlayer.to("cuda:0")
            y = qlayer(inputs)
            y.backward(torch.ones_like(y))
            return inputs.grad

        params = torch.ones([b, n], device="cuda:0", requires_grad=True)
        result, ts = benchmark(get_grad_pl, params, trials=trials)

        # ── correctness assertions ──
        assert result is not None, "PennyLane gradient is None"
        assert result.shape == (b, n), f"grad shape {result.shape} != expected ({b}, {n})"
        assert str(result.device) == "cuda:0", f"grad on {result.device}, expected cuda:0"
        assert not torch.isnan(result).any(), "grad contains NaN"
        assert not torch.isinf(result).any(), "grad contains Inf"
        assert not torch.allclose(result, torch.zeros_like(result)), "grad is all zeros"
        return result, ts


    # ──────────────────────────────────────────────
    # DeepQuantum benchmark
    # ──────────────────────────────────────────────

    def grad_dq_vqc(b, n, l, trials=10):
        """DeepQuantum VQC matching VQNet VQC structure."""
        import deepquantum as dq
        import torch
        import torch.cuda
        def get_grad_dq(input_data):
            cir = dq.QubitCircuit(n, reupload=True)
            for j in range(l):
                for i in range(n):
                    cir.ry(wires=i, encode=True)
                for i in range(n):
                    cir.ry(wires=i)
                for i in range(n - 1, -1, -1):
                    cir.crz(control=i, target=(i + 1) % n)
                for i in range(n):
                    cir.ry(wires=i)
                for i in [n - 1] + list(range(n - 1)):
                    cir.crz(control=i, target=(i - 1) % n)
            for w in range(n):
                cir.observable(basis='z', wires=w)
            cir.to("cuda:0")
            cir(data=input_data)
            exp = cir.expectation()
            exp.backward(torch.ones_like(exp))
            return input_data.grad

        params = torch.ones([b, n], device="cuda:0", requires_grad=True)
        return benchmark(get_grad_dq, params, trials=trials,
                         sync_fn=lambda _: torch.cuda.synchronize())


    # ──────────────────────────────────────────────
    # MindQuantum benchmark
    # ──────────────────────────────────────────────

    def grad_mq_vqc(b, n, l, trials=1):
        """MindQuantum VQC with mqvector_gpu backend."""
        from mindquantum.core.circuit import Circuit
        from mindquantum.core.gates import RY, RZ, X
        from mindquantum.core.operators import Hamiltonian, QubitOperator
        from mindquantum.simulator import Simulator

        total_circuit = Circuit()
        for j in range(l):
            layer_enc = Circuit()
            for i in range(n):
                layer_enc += RY(f'enc_{j}_{i}').on(i)
            layer_enc.as_encoder()

            layer_ans = Circuit()
            for i in range(n):
                layer_ans += RY(f'ry1_{j}_{i}').on(i)
            for i in range(n - 1, -1, -1):
                tgt = (i + 1) % n
                ctrl = i
                p = f'crz1_{j}_{i}'
                layer_ans += RZ({p: 0.5}).on(tgt)
                layer_ans += X.on(tgt, ctrl)
                layer_ans += RZ({p: -0.5}).on(tgt)
                layer_ans += X.on(tgt, ctrl)
            for i in range(n):
                layer_ans += RY(f'ry2_{j}_{i}').on(i)
            for i in [n - 1] + list(range(n - 1)):
                tgt = (i - 1) % n
                ctrl = i
                p = f'crz2_{j}_{i}'
                layer_ans += RZ({p: 0.5}).on(tgt)
                layer_ans += X.on(tgt, ctrl)
                layer_ans += RZ({p: -0.5}).on(tgt)
                layer_ans += X.on(tgt, ctrl)
            layer_ans.as_ansatz()

            total_circuit += layer_enc + layer_ans

        obs = ' '.join(f'Z{i}' for i in range(n))
        ham = Hamiltonian(QubitOperator(obs))
        sim = Simulator('mqvector_gpu', n)
        grad_ops = sim.get_expectation_with_grad(ham, total_circuit)
        n_ansatz_params = 4 * n * l
        ansatz_data = np.ones(n_ansatz_params, dtype=np.float32)

        def get_grad_mq(input_data):
            encoder_data = np.tile(input_data, (1, l)).astype(np.float32)
            _, g_enc, _ = grad_ops(encoder_data, ansatz_data)
            return g_enc

        def sync_mq(g_enc):
            g_enc_real = np.asarray(g_enc.real, dtype=np.float32)
            g = np.zeros((b, n), dtype=np.float32)
            for j in range(l):
                g += g_enc_real[:, 0, j * n : (j + 1) * n]
            return g

        input_data = np.ones((b, n), dtype=np.float32)
        return benchmark(get_grad_mq, input_data, trials=trials,
                         sync_fn=sync_mq)

    def grad_tc_torch_interface(b, n, l, trials=1):
        """Test TensorCircuit-NG VQC via torch_interface (PyTorch frontend, JAX backend).

        Circuit runs on TC's JAX backend, but the autograd graph is PyTorch.
        """
        import tensorcircuit as tc
        tc.set_backend("jax")
        import torch
        import jax.numpy as jnp

        device = torch.device("cuda")

        def circuit_single(x, w_ry1, w_crz1, w_ry2, w_crz2):
            c = tc.Circuit(n)
            for k_idx in range(l):
                for i in range(n):
                    c.ry(i, theta=x[i])
                for i in range(n):
                    c.ry(i, theta=w_ry1[k_idx, i])
                for i in range(n - 1, -1, -1):
                    c.crz(i, (i + 1) % n, theta=w_crz1[k_idx, i])
                for i in range(n):
                    c.ry(i, theta=w_ry2[k_idx, i])
                for i in [n - 1] + list(range(n - 1)):
                    c.crz(i, (i - 1) % n, theta=w_crz2[k_idx, i])
            result = sum(c.expectation((tc.gates.z(), [i])) for i in range(n))
            return tc.backend.real(result)

        batched = tc.backend.vmap(circuit_single, vectorized_argnums=(0,))
        f_torch = tc.interfaces.torch_interface(batched, jit=True)

        torch.manual_seed(42)
        w_ry1 = torch.rand(l, n, device=device) * (2 * jnp.pi)
        w_crz1 = torch.rand(l, n, device=device) * (2 * jnp.pi)
        w_ry2 = torch.rand(l, n, device=device) * (2 * jnp.pi)
        w_crz2 = torch.rand(l, n, device=device) * (2 * jnp.pi)

        def get_grad(x):
            x.grad = None
            y = f_torch(x, w_ry1, w_crz1, w_ry2, w_crz2)
            y.sum().backward()
            return x.grad

        inputs = torch.ones([b, n], device=device, requires_grad=True)
        # Warmup JIT
        get_grad(inputs)
        torch.cuda.synchronize()

        return benchmark(get_grad, inputs, trials=trials,
                         sync_fn=lambda _: torch.cuda.synchronize())


    def grad_vqnet_jax_tn_torch_interface(b, n, l, trials=1):
        """Test VQNet JAX TN VQC via TNQModule (PyTorch frontend, JAX backend)."""
        import torch
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn.torch import TNQMachine, TNQModule, qcircuit, qmeasure
        from pyvqnet.nn import Parameter

        class QuantumNet(TNQModule):
            def __init__(self, n_wires, n_layers):
                super().__init__()
                self.n_wires = n_wires
                self.n_layers = n_layers
                self.w_ry1 = Parameter([n_layers, n_wires])
                self.w_crz1 = Parameter([n_layers, n_wires])
                self.w_ry2 = Parameter([n_layers, n_wires])
                self.w_crz2 = Parameter([n_layers, n_wires])
                self.qm = TNQMachine(n_wires, dtype=pyvqnet.kcomplex64)
                self.measure = qmeasure.MeasureAll(
                    obs=[{f"Z{i}": 1.0} for i in range(n_wires)]
                )

            def forward(self, x):
                self.qm.reset_states(x.shape[0])
                for k_idx in range(self.n_layers):
                    for i in range(self.n_wires):
                        qcircuit.ry(q_machine=self.qm, wires=i, params=x[i])
                    for i in range(self.n_wires):
                        qcircuit.ry(q_machine=self.qm, wires=i,
                                    params=self.w_ry1[k_idx, i])
                    for i in range(self.n_wires - 1, -1, -1):
                        qcircuit.crz(q_machine=self.qm,
                                     wires=[i, (i + 1) % self.n_wires],
                                     params=self.w_crz1[k_idx, i])
                    for i in range(self.n_wires):
                        qcircuit.ry(q_machine=self.qm, wires=i,
                                    params=self.w_ry2[k_idx, i])
                    for i in [self.n_wires - 1] + list(range(self.n_wires - 1)):
                        qcircuit.crz(q_machine=self.qm,
                                     wires=[i, (i - 1) % self.n_wires],
                                     params=self.w_crz2[k_idx, i])
                return self.measure(self.qm)

        model = QuantumNet(n, l)
        model.to_gpu()

        def get_grad(x):
            x.grad = None
            y = model(x)
            y.backward(torch.ones_like(y))
            return x.grad

        device = torch.device("cuda")
        inputs = torch.ones([b, n], device=device, requires_grad=True)
        get_grad(inputs)

        return benchmark(get_grad, inputs, trials=trials,
                         sync_fn=lambda _: torch.cuda.synchronize())


    # ──────────────────────────────────────────────
    # Plotting
    # ──────────────────────────────────────────────

    def _parse_trials(results, fw):
        """Extract trials count from results keys like '*-grad-{fw}-t{N}' (exact framework match)."""
        import re
        pat = re.compile(rf'-grad-{re.escape(fw)}-t(\d+)$')
        for k in results:
            m = pat.search(k)
            if m:
                return m.group(1)
        return '?'


    FRAMEWORK_COLORS = {
        'pyvqnet': '#1f77b4',
        'new': '#ff7f0e',
        'tc-torch': '#17becf',
        'vqnet-jax-tn-torch': '#2ca02c',
        'torchquantum': '#9467bd',
        'pl': '#8c564b',
        'dq': '#7f7f7f',
        'mq': '#bcbd22',
    }


    def _fmt_time(t):
        """Human-readable time label for bar annotations."""
        if t >= 1:
            return f'{t:.1f}s'
        if t >= 1e-3:
            return f'{t * 1e3:.1f}ms'
        return f'{t * 1e6:.0f}us'


    FRAMEWORK_METHOD = {
        # 'tn' = tensor-network simulation, 'sv' = statevector simulation
        'pyvqnet': 'sv',
        'new': 'sv',
        'tc-torch': 'tn',
        'vqnet-jax-tn-torch': 'tn',
        'torchquantum': 'sv',
        'pl': 'sv',
        'dq': 'sv',
        'mq': 'sv',
    }
    METHOD_TITLES = {
        'tn': '── Tensor-network based ──',
        'sv': '── Statevector based ──',
    }


    def plot_results(results, output_path="grad_benchmarks_10q_ry_crz.png", frameworks=None):
        """
        Grouped bar chart: one group per batch size, one bar per framework,
        y = running time (log scale), value label on top of each bar.
        Keys format: "{batch}-{n}-{l}-grad-{framework}-t{trials}".
        frameworks: optional subset of result keys to plot (default: all supported).
        """
        import matplotlib.pyplot as plt
        import re
        import numpy as np
        from matplotlib.patches import Patch

        if frameworks is None:
            frameworks = ['pyvqnet', 'new', 'tc-torch', 'vqnet-jax-tn-torch', 'torchquantum', 'pl', 'dq', 'mq']
        labels_base = {
            'pyvqnet': 'pyVQNet',
            'new': 'pyVQNet (fused CRZ)',
            'tc-torch': 'TensorCircuit-NG (torch interface)',
            'vqnet-jax-tn-torch': 'VQNet JAX TN (torch interface)',
            'torchquantum': 'TorchQuantum',
            'pl': 'PennyLane',
            'dq': 'DeepQuantum',
            'mq': 'MindQuantum',
        }

        batch_sizes = sorted({int(k.split('-')[0]) for k in results})
        n_qubits = sorted({k.split('-')[1] for k in results})
        n_layers = sorted({k.split('-')[2] for k in results})
        n_q = int(n_qubits[0]) if n_qubits else 10
        n_l = int(n_layers[0]) if n_layers else 10

        # data[fw][bs] = running time
        data = {}
        for fw in frameworks:
            trials_str = _parse_trials(results, fw)
            label = f"{labels_base[fw]} (t={trials_str})"
            data[fw] = {'label': label, 'times': {}}
            for bs in batch_sizes:
                pat = re.compile(rf'^{bs}-{n_q}-{n_l}-grad-{re.escape(fw)}-t\d+$')
                candidates = [k for k in results if pat.match(k)]
                if candidates:
                    data[fw]['times'][bs] = results[candidates[0]][1]

        fig, ax = plt.subplots(figsize=(13, 6.5))

        n_fw = len(frameworks)
        group_width = 0.82
        bar_w = group_width / n_fw
        x = np.arange(len(batch_sizes))

        all_vals = []
        for i, fw in enumerate(frameworks):
            xs, ys = [], []
            for j, bs in enumerate(batch_sizes):
                t = data[fw]['times'].get(bs)
                if t is not None:
                    xs.append(x[j] - group_width / 2 + bar_w * (i + 0.5))
                    ys.append(t)
            if not xs:
                continue
            all_vals.extend(ys)
            is_tn = FRAMEWORK_METHOD.get(fw) == 'tn'
            ax.bar(xs, ys, width=bar_w * 0.92, color=FRAMEWORK_COLORS[fw],
                   hatch='//' if is_tn else None, edgecolor='#444444', linewidth=0.4,
                   align='center')
            for px, py in zip(xs, ys):
                ax.text(px, py * 1.12, _fmt_time(py), ha='center', va='bottom',
                        fontsize=7, rotation=90)

        # legend grouped by simulation method (tensor network / statevector)
        handles = []
        for method in ('tn', 'sv'):
            handles.append(Patch(facecolor='none', edgecolor='none',
                                 label=METHOD_TITLES[method]))
            for fw in frameworks:
                if FRAMEWORK_METHOD.get(fw) != method or not data[fw]['times']:
                    continue
                handles.append(Patch(
                    facecolor=FRAMEWORK_COLORS[fw], edgecolor='#444444', linewidth=0.4,
                    hatch='//' if method == 'tn' else None,
                    label=data[fw]['label']))
        ax.set_xlabel('Batch Size', fontsize=13)
        ax.set_ylabel('Running Time (s) — log scale', fontsize=13)
        ax.set_title(f'VQC Gradient Benchmark (n_qubits={n_q}, n_layers={n_l})', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(batch_sizes)
        ax.set_yscale('log')
        if all_vals:
            ax.set_ylim(min(all_vals) / 1.6, max(all_vals) * 8)
        ax.legend(handles=handles, fontsize=10, loc='upper left', bbox_to_anchor=(1.01, 1.0))
        ax.grid(True, axis='y', alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')


    def plot_results_main(results, output_path="grad_benchmarks_main.png"):
        """
        只画主力框架对比（7 个框架的分组柱状图）：
        pyVQNet (fused CRZ) / VQNet JAX TN（TNQModule, torch 接口）/ TensorCircuit-NG（torch 接口）/
        TorchQuantum / PennyLane / DeepQuantum / MindQuantum
        """
        plot_results(results, output_path,
                     frameworks=['new', 'vqnet-jax-tn-torch', 'tc-torch', 'torchquantum', 'pl', 'dq', 'mq'])


    def test_3():
        """
        Run all benchmarks across multiple configs.
        Fast frameworks: trials=20; slow frameworks (PennyLane, MindQuantum): trials=2.
        Results are cached to JSON; rerunning with the JSON present only re-plots.
        """
        import os
        import json
        json_path = "compare_grad_calc_results.json"

        # Import tensorcircuit up-front and pin it to the jax backend, so that
        # tensorcircuit and pyvqnet's TN modules coexist correctly in one process.
        try:
            import tensorcircuit
            tensorcircuit.set_backend("jax")
        except Exception as _e:
            print(f"[WARN] tensorcircuit unavailable: {_e}")

        def plot_from(results):
            """Generate full + main bar charts, one pair per n_qubits."""
            ns = sorted({k.split('-')[1] for k in results})
            for n in ns:
                sub = {k: v for k, v in results.items() if k.split('-')[1] == n}
                plot_results(sub, f'grad_bench_full_n{n}.png')
                plot_results_main(sub, f'grad_bench_main_n{n}.png')
                print(f'charts written for n_qubits={n}')

        if os.path.exists(json_path):
            print(f"{json_path} already exists, loading and plotting directly...")
            with open(json_path) as f:
                results = json.load(f)
            print()
            print("=== Loaded Results ===")
            for key, ts in results.items():
                print(f"{key}: staging={ts[0]:.4f}s, running={ts[1]:.4f}s")
            plot_from(results)
            return

        results = {}
        n_list = [4,10,]
        l_list = [10, ]
        b_list = [512, 128,32]
        t_fast, t_slow = 20, 2

        import traceback

        def run_case(name, key, call, seed=False):
            print(str(b) + '-' + str(n) + '-' + str(l) + '-' + 'grad')
            print(name)
            try:
                if seed:
                    pyvqnet.utils.set_random_seed(42)
                _, ts = call()
            except Exception:
                traceback.print_exc()
                print(f"[SKIP] {name} failed, skipping")
                return
            results[key] = ts

        for n in n_list:
            for l in l_list:
                for b in b_list:
                    run_case("grad_pyvqnet_vqc",
                             f'{b}-{n}-{l}-grad-pyvqnet-t{t_fast}',
                             lambda: grad_pyvqnet_vqc(b, n, l, trials=t_fast), seed=True)

                    run_case("grad_pyvqnet_vqc_new",
                             f'{b}-{n}-{l}-grad-new-t{t_fast}',
                             lambda: grad_pyvqnet_vqc_new(b, n, l, trials=t_fast), seed=True)

                    run_case("grad_vqnet_jax_tn_torch_interface",
                             f'{b}-{n}-{l}-grad-vqnet-jax-tn-torch-t{t_fast}',
                             lambda: grad_vqnet_jax_tn_torch_interface(b, n, l, trials=t_fast))

                    run_case("grad_tc_torch_interface",
                             f'{b}-{n}-{l}-grad-tc-torch-t{t_fast}',
                             lambda: grad_tc_torch_interface(b, n, l, trials=t_fast))

                    run_case("grad_torchquantum_vqc",
                             f'{b}-{n}-{l}-grad-torchquantum-t{t_fast}',
                             lambda: grad_tq_vqc(b, n, l, trials=t_fast))

                    run_case("grad_pennylane_vqc",
                             f'{b}-{n}-{l}-grad-pl-t{t_slow}',
                             lambda: grad_pl_vqc(b, n, l, trials=t_slow))

                    run_case("grad_deepquantum_vqc",
                             f'{b}-{n}-{l}-grad-dq-t{t_fast}',
                             lambda: grad_dq_vqc(b, n, l, trials=t_fast))

                    run_case("grad_mindquantum_vqc",
                             f'{b}-{n}-{l}-grad-mq-t{t_slow}',
                             lambda: grad_mq_vqc(b, n, l, trials=t_slow))

        print("\n=== All Results ===")
        for key, ts in results.items():
            print(f"{key}: staging={ts[0]:.4f}s, running={ts[1]:.4f}s")

        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {json_path}")

        plot_from(results)


    if __name__ == "__main__":
        test_3()


分布式接口
=================================================

分布式相关功能,当使用 ``torch`` 计算后端时候,封装使用了torch的 ``torch.distributed`` 的功能, 同样使用 CommController。





CommController
-------------------------

.. py:class:: pyvqnet.distributed.ControlComm.CommController(backend,rank=None,world_size=None)
   :noindex:

    CommController用于控制在cpu、gpu下数据通信的控制器, 通过设置参数 `backend` 来生成cpu(gloo)、gpu(nccl)的控制器。
    这个类会调用 backend,rank,world_size 初始化 ``torch.distributed.init_process_group(backend,rank,world_size)``


    .. note::

        请参考 `torch 分布式接口 <https://pytorch.org/docs/stable/distributed.html>`_  中启动分布式的方法启动。
        当使用 CPU 上进行分布式,请使用 ``gloo`` 而不是 ``mpi`` 配置backend。
        当使用 GPU 上进行分布式,请使用 ``nccl`` 启动  配置backend。

    :ref:`vqnet_dist` 下VQNet自己实现的分布式接口不适用 ``torch`` 计算后端。

    :param backend: 用于生成cpu或者gpu的数据通信控制器,'gloo' 或 'nccl'。
    :param rank: 当前程序所在的进程号。
    :param world_size: 全局所有的进程数量。

    :return:
        CommController 实例。

    Example::

        from pyvqnet.distributed import CommController
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        import os
        import multiprocessing as mp


        def init_process(rank, size):
            """ Initialize the distributed environment. """
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'
            os.environ['LOCAL_RANK'] = f"{rank}"
            pp = CommController("gloo", rank=rank, world_size=size)
            
            local_rank = pp.get_rank()
            print(local_rank)


        if __name__ == "__main__":
            world_size = 2
            processes = []
            mp.set_start_method("spawn")
            for rank in range(world_size):
                p = mp.Process(target=init_process, args=(rank, world_size))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        #python test.py

    .. py:method:: getRank()
        :noindex:

        用于获得当前进程的进程号。

        :return: 返回当前进程的进程号。

        Example::

            from pyvqnet.distributed import CommController
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                pp = CommController("gloo", rank=rank, world_size=size)
                
                local_rank = pp.getRank()
                print(local_rank)


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            #python test.py


    .. py:method:: getSize()
        :noindex:

        用于获得总共启动的进程数。


        :return: 返回总共进程的数量。

        Example::

            from pyvqnet.distributed import CommController
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                pp = CommController("gloo", rank=rank, world_size=size)
                
                local_rank = pp.getSize()
                print(local_rank)


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            #python test.py


    .. py:method:: getLocalRank()
        :noindex:

        在每个进程中通过 ``os.environ['LOCAL_RANK'] = rank`` 获取每个机器的局部进程号。
        需要事先对环境变量 `LOCAL_RANK` 进行设置。

        :return: 当前机器上的当前进程号。

        Example::

            from pyvqnet.distributed import CommController
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                pp = CommController("gloo", rank=rank, world_size=size)
                
                local_rank = pp.getLocalRank()
                print(local_rank )


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            #python test.py
 
    .. py:method:: split_groups(rankL)
        :noindex:

        根据入参设置的进程号列表用于划分多个通信组。

        :param rankL: 进程组列表。
        :return: 包含 ``torch.distributed.ProcessGroup`` 的列表

        Example::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                group = Comm_OP.split_groups([[1,3]])

                num = tensor.to_tensor(np.random.rand(1, 5)+get_local_rank()*10)
                print(f"before rank {Comm_OP.getRank()}  {num}\n")
                
                Comm_OP.reduce_group(num, 1,"sum",group[0])
                print(f"after rank {Comm_OP.getRank()}  {num}\n")
                

            if __name__ == "__main__":
                world_size = 4
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            #python test.py
    .. py:method:: barrier()
        :noindex:

        不同进程的同步。

        :return: 同步操作。

        Example::

            from pyvqnet.distributed import CommController
            import pyvqnet
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                pp = CommController("gloo", rank=rank, world_size=size)
                
                pp.barrier()


            if __name__ == "__main__":
                world_size = 4
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            #python test.py

    .. py:method:: allreduce(tensor, c_op = "avg")
        :noindex:

        支持对数据作allreduce通信。

        :param tensor: 输入数据.
        :param c_op: 计算方式.

        Example::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))
                print(f"rank {Comm_OP.getRank()}  {num}")

                Comm_OP.all_reduce(num, "sum")
                print(f"rank {Comm_OP.getRank()}  {num}")
                

            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            #python test.py

 
    .. py:method:: reduce(tensor, root = 0, c_op = "avg")
        :noindex:

        支持对数据作reduce通信。

        :param tensor: 输入数据。
        :param root: 指定数据返回的节点。
        :param c_op: 计算方式。

        Example::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))
                print(f"before rank {Comm_OP.getRank()}  {num}")
                
                Comm_OP.reduce(num, 1,"sum")
                print(f"after rank {Comm_OP.getRank()}  {num}")
                

            if __name__ == "__main__":
                world_size = 3
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            #python test.py
 
    .. py:method:: broadcast(tensor, root = 0)
        :noindex:

        将指定进程root上的数据广播到所有进程上。

        :param tensor: 输入数据。
        :param root: 指定的节点。

        Example::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))+ rank
                print(f"before rank {Comm_OP.getRank()}  {num}")
                
                Comm_OP.broadcast(num, 1)
                print(f"after rank {Comm_OP.getRank()}  {num}")
                

            if __name__ == "__main__":
                world_size = 3
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            #python test.py

 
    .. py:method:: allgather(tensor)
        :noindex:

        将所有进程上数据allgather到一起。本接口只支持nccl后端。

        :param tensor: 输入数据。

        Example::

            from pyvqnet.distributed import get_local_rank,CommController,get_world_size
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import torch.multiprocessing as mp

            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("nccl", rank=rank, world_size=size)

                num = tensor.QTensor(np.random.rand(5,4),device=pyvqnet.DEV_GPU_0+rank)
                print(f"before rank {Comm_OP.getRank()}  {num}\n")

                num = Comm_OP.all_gather(num)
                print(f"after rank {Comm_OP.getRank()}  {num}\n")


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            #python test.py

    .. py:method:: send(tensor, dest)
        :noindex:

        p2p通信接口。

        :param tensor: 输入数据.
        :param dest: 目的进程.

        Example::

            from pyvqnet.distributed import get_rank,CommController,get_world_size
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))
                recv = tensor.zeros_like(num)
                if get_rank() == 0:
                    Comm_OP.send(num, 1)
                elif get_rank() == 1:
                    Comm_OP.recv(recv, 0)
                print(f"before rank {Comm_OP.getRank()}  {num}")
                print(f"after rank {Comm_OP.getRank()}  {recv}")

            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            #python test.py
 
    .. py:method:: recv(tensor, source)
        :noindex:

        p2p通信接口。

        :param tensor: 输入数据.
        :param source: 接受进程.

        Example::

            from pyvqnet.distributed import get_rank,CommController,get_world_size
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                num = tensor.to_tensor(np.random.rand(1, 5))
                recv = tensor.zeros_like(num)
                if get_rank() == 0:
                    Comm_OP.send(num, 1)
                elif get_rank() == 1:
                    Comm_OP.recv(recv, 0)
                print(f"before rank {Comm_OP.getRank()}  {num}")
                print(f"after rank {Comm_OP.getRank()}  {recv}")

            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            #python test.py

    .. py:method:: allreduce_group(tensor, c_op = "avg", group = None)
        :noindex:

        组内allreduce通信接口。

        :param tensor: 输入数据.
        :param c_op: 计算方法.
        :param group: 通信组.

        Example::

            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                rankL = [[0,1],[2,3]]
                groups = Comm_OP.split_groups(rankL)
                num = tensor.to_tensor(np.ones(5)+get_local_rank()*1000)

                print(f"before rank {Comm_OP.getRank()}  {num}")
                if Comm_OP.getRank() in rankL[0]:
                    Comm_OP.all_reduce_group(num, "sum", groups[0])

                    print(f"after rank {Comm_OP.getRank()}  {num}")

                if Comm_OP.getRank() in rankL[1]:
                    Comm_OP.all_reduce_group(num, "sum", groups[1])

                    print(f"after rank {Comm_OP.getRank()}  {num}")

            if __name__ == "__main__":
                world_size = 4
                mp.set_start_method("spawn")
                processes = []
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            #python test.py

    .. py:method:: reduce_group(tensor, root = 0, c_op = "avg", group = None)
        :noindex:

        组内reduce通信接口。

        :param tensor: 输入数据.
        :param root: 指定全局进程号.
        :param c_op: 计算方法.
        :param group: 通信组.

        Example::
            
            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)
                rankL = [[1,3],[0,2]]
                group = Comm_OP.split_groups([[1,3],[0,2]])

                num = tensor.to_tensor(np.random.rand(1, 5)+get_local_rank()*10)
                print(f"before rank {Comm_OP.getRank()}  {num}\n")
                if Comm_OP.getRank() in rankL[0]:
                    Comm_OP.reduce_group(num, rankL[0][1],"sum",group[0])
                    print(f"after rank {Comm_OP.getRank()}  {num}\n")
                if Comm_OP.getRank() in rankL[1]:
                    Comm_OP.reduce_group(num, rankL[1][1],"sum",group[1])
                    print(f"after rank {Comm_OP.getRank()}  {num}\n")

            if __name__ == "__main__":
                world_size = 4
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            #python test.py

 
    .. py:method:: broadcast_group(tensor, root = 0, group = None)
        :noindex:

        组内broadcast通信接口。

        :param tensor: 输入数据.
        :param root: 指定全局进程号.
        :param group: 通信组.

        Example::
            
            from pyvqnet.distributed import get_local_rank,CommController
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp


            def init_process(rank, size):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                rankL = [[2,3],[0,1,4]]
                group = Comm_OP.split_groups(rankL)

                num = tensor.to_tensor(np.random.rand(1, 5))+ rank*1000
                print(f"before rank {Comm_OP.getRank()}  {num}")
                
                if Comm_OP.getRank() in rankL[0]:
                    Comm_OP.broadcast_group(num, rankL[0][0],group[0])
                    print(f"after rank {Comm_OP.getRank()}  {num}")

                if Comm_OP.getRank() in rankL[1]:
                    Comm_OP.broadcast_group(num, rankL[1][1],group[1])
                    print(f"after rank {Comm_OP.getRank()}  {num}")

            if __name__ == "__main__":
                world_size = 5
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            #python test.py

    .. py:method:: allgather_group(tensor, group = None)
        :noindex:

        组内allgather通信接口,仅支持 `nccl` 后端。

        :param tensor: 输入数据.
        :param group: 通信组.

        Example::
            
            from pyvqnet.distributed import get_local_rank,CommController,get_world_size
            import pyvqnet
            import numpy as np
            from pyvqnet.tensor import tensor
            pyvqnet.backends.set_backend("torch")
            import os
            import multiprocessing as mp
            

            def init_process(rank, size ):
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'
                os.environ['LOCAL_RANK'] = f"{rank}"
                Comm_OP = CommController("nccl", rank=rank, world_size=size)

                group = Comm_OP.split_groups([[0,1]])
                print(f"get_world_size {get_world_size()}")

                num = tensor.QTensor(np.random.rand(5,4)+get_local_rank()*100,device=pyvqnet.DEV_GPU_0+get_local_rank())
                print(f"before rank {Comm_OP.getRank()}  {num}\n")

                num = Comm_OP.allgather_group(num,group[0])
                print(f"after rank {Comm_OP.getRank()}  {num}\n")


            if __name__ == "__main__":
                world_size = 2
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size ))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

            #python test.py