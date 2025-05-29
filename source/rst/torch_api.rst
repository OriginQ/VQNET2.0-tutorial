
.. _torch_api:

====================================
VQNet使用torch进行底层计算
====================================

.. danger::

    **如需要使用以下功能, 请自行安装 torch>=2.4.0 , 本软件安装时候不自动安装 torch 。**

自2.15.0版本开始,本软件支持使用 ``torch`` 作为计算后端进行底层运算,可接入第三方主流大模型训练库进行大模型微调。


    .. warning::

        :ref:`vqc_api` 中的变分量子计算函数(小写命名,例如 `rx`, `ry`, `rz` 等), :ref:`qtensor_api` 中的QTensor基本计算函数,
        在 ``pyvqnet.backends.set_backend("torch")`` 后,可以输入 ``QTensor``,其成员 `data` 从pyvqnet的Tensor变为 ``torch.Tensor`` 计算。

        ``pyvqnet.backends.set_backend("torch")`` 以及 ``pyvqnet.backends.set_backend("pyvqnet")`` 会修改全局运行后端。
        不同后端配置下申请的 ``QTensor`` 无法一起运算。

        使用 ``to_tensor`` 可将 ``torch.Tensor`` 封装为一个 ``QTensor`` 。

计算后端基本设置
====================

set_backend
------------------------------------------------

.. py:function:: pyvqnet.backends.set_backend(backend_name)

    设置当前计算和储存数据所使用的后端,默认为 "pyvqnet",可设置为 "torch"。
    
    使用 ``pyvqnet.backends.set_backend("torch")`` 后,接口保持不变,但VQNet的 ``QTensor`` 的 ``data`` 成员变量均使用 ``torch.Tensor`` 储存数据,
    并使用torch计算。
    
    使用 ``pyvqnet.backends.set_backend("pyvqnet")`` 后,VQNet ``QTensor`` 的 ``data`` 成员变量均使用 ``pyvqnet._core.Tensor`` 储存数据,并使用pyvqnet c++库计算。

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
    如果 "torch" 是使用的后端,则返回 pyvqnet torch api 后端。
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
    pyvqnet.backends.set_backend("pyvqnet")

在 :ref:`qtensor_api` 下的所有成员函数,创建函数,数学函数,逻辑函数,矩阵变换等均使用torch进行计算。使用 ``QTensor.data`` 可获取torch数据。

使用 ``to_tensor`` 可将 ``torch.Tensor`` 封装为一个 ``QTensor`` 。



经典神经网络类以及变分量子神经网络模块
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
 
        Examples::
 
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

        :param device: 当前保存QTensor的设备,默认:DEV_GPU_0。device= pyvqnet.DEV_GPU_0,存储在第一个 GPU 中,devcie = DEV_GPU_1,存储在第二个 GPU 中,依此类推
        :return: Module 移动到 GPU 设备。

        Examples::

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

        Examples::

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
    :math:`y = Ax + b`
    
    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
    该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。
    

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
    :param dilation_rate: `int` - 空洞大小,defaults: 1。
    :param group: `int` -  分组卷积的分组数. Default: 1。
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
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
    :param dilation_rate: `int` - 空洞大小,defaults: 1。
    :param group: `int` -  分组卷积的分组数. Default: 1。
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
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
    :param dilation_rate: `int` - 空洞大小,defaults: 1。
    :param out_padding: 在输出形状中每个维度的一侧添加的额外尺寸。默认值:(0,0)
    :param group: `int` -  分组卷积的分组数. Default: 1。
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
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
    :param strides: 窗口移动的步长。
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
    :param strides: 窗口移动的步长。
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
    :param strides: 窗口移动的步长。
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
    :param strides: 窗口移动的步长。
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
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
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
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
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
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
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
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
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
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
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
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
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

    对小批量输入应用组归一化。输入: :math:`(N, C, *)` 其中 :math:`C=\text{num_channels}` , 输出: :math:`(N, C, *)` 。

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
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
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
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
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
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
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
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
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
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
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
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
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
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
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

    构造计算查询、键和值张量的缩放点积注意力的类。如果输入为cpu下的QTensor,则使用数学公式计算, 如果输入在gpu下QTensor,则使用flash-attention方法计算。
    
    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param attn_mask: 注意掩码；默认值: 无。shape 必须可广播到注意权重的形状。
    :param dropout_p: Dropout 概率；默认值: 0,如果大于 0.0,则应用 dropout。
    :param scale: 在 softmax 之前应用的缩放因子,默认值: 无。
    :param is_causal: 默认值: False,如果设置为 true,则当掩码为方阵时,注意掩码为下三角矩阵。如果同时设置了 attn_mask 和 is_causal,则会引发错误。
    :return: 一个SDPA类

    Examples::
    
        from pyvqnet.nn.torch import SDPA
        from pyvqnet import tensor
        model = SDPA(tensor.QTensor([1.]))

    .. py:method:: forward(query,key,value)

        进行前向计算,如果输入为cpu下的QTensor,则使用数学公式计算, 如果输入在gpu下QTensor,则使用flash-attention方法计算。

        :param query: query输入QTensor。
        :param key: key输入QTensor。
        :param value: key输入QTensor。
        :return: SDPA计算返回的QTensor。

        Examples::
        
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

            out_sdpa.backward()

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

            请注意,跟pytorch等框架不同的是,以下MeanSquaredError函数的前向函数中,第一个参数为目标值,第二个参数为预测值。


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

            请注意,跟pytorch等框架不同的是,BinaryCrossEntropy函数的前向函数中,第一个参数为目标值,第二个参数为预测值。



    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.nn.torch import BinaryCrossEntropy
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        x = QTensor([[0.3, 0.7, 0.2], [0.2, 0.3, 0.1]], requires_grad=True)
        y = QTensor([[0.0, 1.0, 0], [0.0, 0, 1]], requires_grad=False)

        loss_result = BinaryCrossEntropy()
        result = loss_result(y, x)
        result.backward()
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

            请注意,跟pytorch等框架不同的是,CategoricalCrossEntropy函数的前向函数中,第一个参数为目标值,第二个参数为预测值。

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

            请注意,跟pytorch等框架不同的是,SoftmaxCrossEntropy函数的前向函数中,第一个参数为目标值,第二个参数为预测值。

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
        result.backward()
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

        请注意,跟pytorch等框架不同的是,NLL_Loss函数的前向函数中,第一个参数为目标值,第二个参数为预测值。

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

            请注意,跟pytorch等框架不同的是,CrossEntropyLoss函数的前向函数中,第一个参数为目标值,第二个参数为预测值。

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

    Examples::

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

    Examples::

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

    Examples::

        from pyvqnet.nn.torch import Softsign
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Softsign()
        y = layer(QTensor([1.0, 2.0, 3.0, 4.0]))



Softmax
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.torch.Softmax(axis:int = -1,name:str="")

    Softmax 激活函数层。

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}


    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param axis: 计算的维度(最后一个轴为-1),默认值 = -1。
    :param name: 激活函数层的命名,默认为""。

    :return: 一个Softmax 激活函数层实例。

    Examples::

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

    Examples::

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

    Examples::

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

    Examples::

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

    Examples::

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

    Examples::

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

    Examples::

        from pyvqnet.nn.torch import Tanh
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        layer = Tanh()
        y = layer(QTensor([-1, 2.0, -3, 4.0]))

优化器模块
---------------------------------------------

对于继承于 `TorchModule` 的VQNet的经典和量子线路模块,对其中的参数 `model.paramters()` 可继续使用 :ref:`Optimizer` 下的除 `Rotosolve` 以外的VQNet优化器进行参数优化。

对于继承于 `TorchModule` 的VQNet的经典和量子线路模块,其中参数同样可以被 `torch.nn.Module.parameters()` 获取,可同样使用 torch 的优化器进行优化。


使用pyqpanda进行计算的量子变分线路训练函数
------------------------------------------

以下是使用pyqpanda以及pyqpanda3进行线路计算的训练变分量子线路接口。

.. warning::

    以下 TorchQpandaQuantumLayer, TorchQcloudQuantumLayer 的量子计算部分使用pyqpanda2 https://pyqpanda-toturial.readthedocs.io/zh/latest/。

    您需要自行安装pyqpanda2, `pip install pyqpanda` 

TorchQpandaQuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如您更加熟悉pyqpanda2语法,可以使用该接口TorchQpandaQuantumLayer,自定义量子比特 ``qubits`` ,经典比特 ``cbits`` ,后端模拟器 ``machine`` 加入TorchQpandaQuantumLayer的参数 ``qprog_with_measure`` 函数中。

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQpandaQuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

	变分量子层的抽象计算模块。对一个参数化的量子线路使用pyqpanda2进行仿真,得到测量结果。该变分量子层继承了VQNet框架的梯度计算模块,可以使用参数漂移法等计算线路参数的梯度,训练变分量子线路模型或将变分量子线路嵌入混合量子和经典模型。
    
    :param qprog_with_measure: 用pyQPand构建的量子线路运行和测量函数。
    :param para_num: `int` - 参数个数。
    :param diff_method: 求解量子线路参数梯度的方法,“参数位移”或“有限差分”,默认参数偏移。
    :param delta: 有限差分计算梯度时的 \delta。
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字, 默认为""。

    :return: 一个可以计算量子线路的模块。

    .. note::
        qprog_with_measure是pyqpanda2中定义的量子线路函数 :https://pyqpanda-toturial.readthedocs.io/zh/latest/QCircuit.html。
        
        此函数必须包含以下参数作为函数入参（即使某个参数未实际使用）,否则无法在本函数中正常运行。

        与QuantumLayer相比。该接口传入的变分线路运行函数中,用户应该手动创建量子比特和模拟器: https://pyqpanda-toturial.readthedocs.io/zh/latest/QuantumMachine.html,

        如果qprog_with_measure需要quantum measure,用户还需要手动创建需要分配cbits: https://pyqpanda-toturial.readthedocs.io/zh/latest/Measure.html
        
        量子线路函数 qprog_with_measure (input,param,nqubits,ncubits)的使用可参考下面的例子。
        
        `input`: 输入一维经典数据。如果没有,输入 None。
        
        `param`: 输入一维的变分量子线路的待训练参数。


    Example::

        import pyqpanda as pq
        from pyvqnet.qnn import ProbsMeasure
        import numpy as np
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import TorchQpandaQuantumLayer
        def pqctest (input,param):
            num_of_qubits = 4

            m_machine = pq.CPUQVM()# outside
            m_machine.init_qvm()# outside
            qubits = m_machine.qAlloc_many(num_of_qubits)

            circuit = pq.QCircuit()
            circuit.insert(pq.H(qubits[0]))
            circuit.insert(pq.H(qubits[1]))
            circuit.insert(pq.H(qubits[2]))
            circuit.insert(pq.H(qubits[3]))

            circuit.insert(pq.RZ(qubits[0],input[0]))
            circuit.insert(pq.RZ(qubits[1],input[1]))
            circuit.insert(pq.RZ(qubits[2],input[2]))
            circuit.insert(pq.RZ(qubits[3],input[3]))

            circuit.insert(pq.CNOT(qubits[0],qubits[1]))
            circuit.insert(pq.RZ(qubits[1],param[0]))
            circuit.insert(pq.CNOT(qubits[0],qubits[1]))

            circuit.insert(pq.CNOT(qubits[1],qubits[2]))
            circuit.insert(pq.RZ(qubits[2],param[1]))
            circuit.insert(pq.CNOT(qubits[1],qubits[2]))

            circuit.insert(pq.CNOT(qubits[2],qubits[3]))
            circuit.insert(pq.RZ(qubits[3],param[2]))
            circuit.insert(pq.CNOT(qubits[2],qubits[3]))

            prog = pq.QProg()
            prog.insert(circuit)

            rlt_prob = ProbsMeasure([0,2],prog,m_machine,qubits)
            return rlt_prob

        pqc = TorchQpandaQuantumLayer(pqctest,3)

        #classic data as input
        input = QTensor([[1.0,2,3,4],[4,2,2,3],[3,3,2,2]],requires_grad=True)

        #forward circuits
        rlt = pqc(input)

        print(rlt)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #backward circuits
        rlt.backward(grad)

        print(pqc.m_para.grad)
        print(input.grad)


TorchQcloudQuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当您安装最新版本pyqpanda2,可以使用本接口定义一个变分线路,并提交到originqc的真实芯片上运行。

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQcloudQuantumLayer(origin_qprog_func, qcloud_token, para_num, num_qubits, num_cubits, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift", submit_kwargs={}, query_kwargs={})

    使用 pyqpanda QCloud 从版本 3.8.2.2 开始的本源量子真实芯片的抽象计算模块。 它提交参数化量子电路到真实芯片并获得测量结果。
    如果 diff_method == "random_coordinate_descent" ,该层将随机选择单个参数来计算梯度,其他参数将保持为零。参考:https://arxiv.org/abs/2311.00088

    .. note::

        qcloud_token 为您到 https://qcloud.originqc.com.cn/ 中申请的api token。
        origin_qprog_func 需要返回pypqanda.QProg类型的数据,如果没有设置pauli_str_dict,需要保证该QProg中已经插入了measure。
        origin_qprog_func 的形式必须按照如下:

        origin_qprog_func(input,param,qubits,cbits,machine)
        
            `input`: 输入1~2维经典数据,二维的情况下,第一个维度为批处理大小。
            
            `param`: 输入一维的变分量子线路的待训练参数。

            `machine`: 由QuantumBatchAsyncQcloudLayer创建的模拟器QCloud,无需用户额外在函数中定义。
            
            `qubits`: 由QuantumBatchAsyncQcloudLayer创建的模拟器QCloud创建的量子比特,数量为  `num_qubits`, 类型为pyQpanda.Qubits,无需用户额外在函数中定义。
            
            `cbits`: 由QuantumBatchAsyncQcloudLayer分配的经典比特, 数量为  `num_cubits`, 类型为 pyQpanda.ClassicalCondition,无需用户额外在函数中定义。。
            


    :param origin_qprog_func: QPanda 构建的变分量子电路函数,必须返回QProg。
    :param qcloud_token: `str` - 量子机的类型或用于执行的云令牌。
    :param para_num: `int` - 参数数量,参数是大小为[para_num]的QTensor。
    :param num_qubits: `int` - 量子电路中的量子比特数量。
    :param num_cubits: `int` - 量子电路中用于测量的经典比特数量。
    :param pauli_str_dict: `dict|list` - 表示量子电路中泡利运算符的字典或字典列表。 默认为“无”,则进行测量操作,如果输入泡利算符的字典,则会计算单个期望或者多个期望。
    :param shot: `int` - 测量次数。 默认值为 1000。
    :param initializer: 参数值的初始化器。 默认为“无”,使用0~2*pi正态分布。
    :param dtype: 参数的数据类型。 默认值为 None,即使用默认数据类型pyvqnet.kfloat32。
    :param name: 模块的名称。 默认为空字符串。
    :param diff_method: 梯度计算的微分方法。 默认为“parameter_shift”,"random_coordinate_descent"。
    :param submit_kwargs: 用于提交量子电路的附加关键字参数,默认:{"chip_id":pyqpanda.real_chip_type.origin_72,"is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False},当设置test_qcloud_fake为True则本地CPUQVM模拟。
    :param query_kwargs: 用于查询量子结果的附加关键字参数,默认:{"timeout":2,"print_query_info":True,"sub_circuits_split_size":1}。
    :return: 一个可以计算量子电路的模块。
    
    Example::

        import pyqpanda as pq
        import pyvqnet
        from pyvqnet.qnn.vqc.torch import TorchQcloudQuantumLayer

        pyvqnet.backends.set_backend("torch")
        def qfun(input,param, m_machine, m_qlist,cubits):
            measure_qubits = [0,2]
            m_prog = pq.QProg()
            cir = pq.QCircuit()
            cir.insert(pq.RZ(m_qlist[0],input[0]))
            cir.insert(pq.CNOT(m_qlist[0],m_qlist[1]))
            cir.insert(pq.RY(m_qlist[1],param[0]))
            cir.insert(pq.CNOT(m_qlist[0],m_qlist[2]))
            cir.insert(pq.RZ(m_qlist[1],input[1]))
            cir.insert(pq.RY(m_qlist[2],param[1]))
            cir.insert(pq.H(m_qlist[2]))
            m_prog.insert(cir)

            for idx, ele in enumerate(measure_qubits):
                m_prog << pq.Measure(m_qlist[ele], cubits[idx])  # pylint: disable=expression-not-assigned
            return m_prog

        l = TorchQcloudQuantumLayer(qfun,
                        "3047DE8A59764BEDAC9C3282093B16AF1",
                        2,
                        6,
                        6,
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
        y.backward()
        print(l.m_para.grad)
        print(x.grad)

        def qfun2(input,param, m_machine, m_qlist,cubits):
            measure_qubits = [0,2]
            m_prog = pq.QProg()
            cir = pq.QCircuit()
            cir.insert(pq.RZ(m_qlist[0],input[0]))
            cir.insert(pq.CNOT(m_qlist[0],m_qlist[1]))
            cir.insert(pq.RY(m_qlist[1],param[0]))
            cir.insert(pq.CNOT(m_qlist[0],m_qlist[2]))
            cir.insert(pq.RZ(m_qlist[1],input[1]))
            cir.insert(pq.RY(m_qlist[2],param[1]))
            cir.insert(pq.H(m_qlist[2]))
            m_prog.insert(cir)

            return m_prog
        l = TorchQcloudQuantumLayer(qfun2,
                "3047DE8A59764BEDAC9C3282093B16AF",
                2,
                6,
                6,
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
        y.backward()
        print(l.m_para.grad)
        print(x.grad)



.. warning::

    以下TorchQcloud3QuantumLayer,TorchQpanda3QuantumLayer接口的量子计算部分使用pyqpanda3 https://qcloud.originqc.com.cn/document/qpanda-3/index.html。

    如果您使用了本模块下的QCloud功能,在代码中导入pyqpanda2 或 使用pyvqnet的pyqpanda2相关封装接口会有错误。

TorchQcloud3QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

当您安装最新版本pyqpanda3,可以使用本接口定义一个变分线路,并提交到originqc的真实芯片上运行。

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQcloud3QuantumLayer(origin_qprog_func, qcloud_token, para_num, pauli_str_dict=None, shots = 1000, initializer=None, dtype=None, name="", diff_method="parameter_shift", submit_kwargs={}, query_kwargs={})

    使用 pyqpanda3的本源量子真实芯片的抽象计算模块。 它提交参数化量子电路到真实芯片并获得测量结果。
    如果 diff_method == "random_coordinate_descent" ,该层将随机选择单个参数来计算梯度,其他参数将保持为零。参考:https://arxiv.org/abs/2311.00088

    .. note::

        qcloud_token 为您到 https://qcloud.originqc.com.cn/ 中申请的api token。
        origin_qprog_func 需要返回pypqanda3.core.QProg类型的数据,如果没有设置pauli_str_dict,需要保证该QProg中已经插入了measure。
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
    :param submit_kwargs: 用于提交量子电路的附加关键字参数,默认:{"chip_id":pyqpanda.real_chip_type.origin_72,"is_amend":True,"is_mapping":True,"is_optimization":True,"compile_level":3,"default_task_group_size":200,"test_qcloud_fake":False},当设置test_qcloud_fake为True则本地CPUQVM模拟。
    :param query_kwargs: 用于查询量子结果的附加关键字参数,默认:{"timeout":2,"print_query_info":True,"sub_circuits_split_size":1}。
    :return: 一个可以计算量子电路的模块。

    Example::

        import pyqpanda3.core as pq
        import pyvqnet
        from pyvqnet.qnn.vqc.torch import TorchQcloud3QuantumLayer

        pyvqnet.backends.set_backend("torch")
        def qfun(input,param):

            m_qlist = range(6)
            cubits = range(6)
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
                m_prog << pq.measure(m_qlist[ele], cubits[idx])  # pylint: disable=expression-not-assigned
            return m_prog

        l = TorchQcloud3QuantumLayer(qfun,
                        "3047DE8A59764BEDAC9C3282093B16AF1",
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
        y.backward()
        print(l.m_para.grad)
        print(x.grad)

        def qfun2(input,param ):

            m_qlist = range(6)
            cubits = range(6)
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
                "3047DE8A59764BEDAC9C3282093B16AF",
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
        y.backward()
        print(l.m_para.grad)
        print(x.grad)



TorchQpanda3QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如您更加熟悉pyqpanda3语法,可以使用该接口TorchQpanda3QuantumLayer。

.. py:class:: pyvqnet.qnn.vqc.torch.TorchQpanda3QuantumLayer(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01,dtype=None,name="")

	变分量子层的抽象计算模块。对一个参数化的量子线路使用pyqpanda3进行仿真,得到测量结果。该变分量子层继承了VQNet框架的梯度计算模块,可以使用参数漂移法等计算线路参数的梯度,训练变分量子线路模型或将变分量子线路嵌入混合量子和经典模型。
    
    :param qprog_with_measure: 用pyQPand构建的量子线路运行和测量函数。
    :param para_num: `int` - 参数个数。
    :param diff_method: 求解量子线路参数梯度的方法,“参数位移”或“有限差分”,默认参数偏移。
    :param delta: 有限差分计算梯度时的 \delta。
    :param dtype: 参数的数据类型,defaults:None,使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字, 默认为""。

    :return: 一个可以计算量子线路的模块。

    .. note::
        qprog_with_measure是pyQPanda中定义的量子线路函数 :https://qcloud.originqc.com.cn/document/qpanda-3/db/d6c/tutorial_circuit_and_program.html.。
        
        此函数必须包含以下参数作为函数入参（即使某个参数未实际使用）,否则无法在本函数中正常运行。

        量子线路函数 qprog_with_measure (input,param,nqubits,ncubits)的使用可参考下面的例子。
        
        `input`: 输入一维经典数据。如果没有,输入 None。
        
        `param`: 输入一维的变分量子线路的待训练参数。


    Example::

        import pyqpanda3.core as pq
        from pyvqnet.qnn.pq3 import ProbsMeasure
        import numpy as np
        from pyvqnet.tensor import QTensor
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import TorchQpanda3QuantumLayer
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

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #backward circuits
        rlt.backward(grad)

        print(pqc.m_para.grad)
        print(input.grad)


基于自动微分的变分量子线路模块和接口
--------------------------------------------------


基类
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

编写变分量子线路模型需要继承于 ``QModule``。

QModule
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QModule(name="")

    当用户使用 `torch` 后端时候,定义量子变分线路模型 `Module` 应该继承的基类。
    该类继承于 ``pyvqnet.nn.torch.TorchModule`` 以及 ``torch.nn.Module``。
    该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    .. note::

        该类以及其派生类仅适用于 ``pyvqnet.backends.set_backend("torch")`` , 不要与默认 ``pyvqnet.nn`` 下的 ``Module`` 混用。
    
        该类的 ``_buffers`` 中的数据为 ``torch.Tensor`` 类型。
        该类的 ``_parmeters`` 中的数据为 ``torch.nn.Parameter`` 类型。



QMachine
""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QMachine(num_wires, dtype=pyvqnet.kcomplex64,grad_mode="",save_ir=False)

    变分量子计算的模拟器类,包含states属性为量子线路的statevectors。

    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` 以及 ``pyvqnet.qnn.QMachine`` 。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    .. warning::
        
        在每次运行一个完整的量子线路之前,必须使用 `pyvqnet.qnn.vqc.QMachine.reset_states(batchsize)` 将模拟器里面初态重新初始化,并且广播为
        (batchsize,*) 维度从而适应批量数据训练。

    :param num_wires: 量子比特数。
    :param dtype: 计算数据的数据类型。默认值是pyvqnet。kcomplex64,对应的参数精度为pyvqnet.kfloat32。
    :param grad_mode: 梯度计算模式,可为 "adjoint",默认值:"",使用自动微分模拟。
    :param save_ir: 设置为True时,将操作保存到originIR,默认值:False。

    :return: 输出一个QMachine对象。

    Example::
        
        from pyvqnet.qnn.vqc.torch import QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        qm = QMachine(4)
        print(qm.states)


    .. py:method:: reset_states(batchsize)

        将模拟器里面初态重新初始化,并且广播为
        (batchsize,*) 维度从而适应批量数据训练。

        :param batchsize: 批处理维度。


变分量子逻辑门模块
""""""""""""""""""""""""


以下 ``pyvqnet.qnn.vqc`` 中的函数接口直接支持 ``torch`` 后端的 ``QTensor`` 进行计算。

.. csv-table:: 已支持pyvqnet.qnn.vqc接口列表
   :file: ./images/same_apis_from_vqc.csv


以下量子线路模块继承于 ``pyvqnet.qnn.vqc.torch.QModule``,其中计算使用 ``torch.Tensor`` 进行计算。


.. warning::

    该类以及其派生类仅适用于 ``pyvqnet.backends.set_backend("torch")`` , 不要与默认 ``pyvqnet.nn`` 下的 ``Module`` 混用。

    这些类如果有非参数成员变量 ``_buffers`` ,则其中的数据为 ``torch.Tensor`` 类型。
    这些类如果有参数成员变量 ``_parmeters`` ,则其中的数据为 ``torch.nn.Parameter`` 类型。

I
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个I逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params: 是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 I 逻辑门实例

    Example::
        
        from pyvqnet.qnn.vqc.torch import I,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = I(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


Hadamard
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.Hadamard(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个Hadamard逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 Hadamard 逻辑门实例

    Example::
        
        from pyvqnet.qnn.vqc.torch import Hadamard,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = Hadamard(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


T
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.T(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个T逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 T 逻辑门实例

    Example::
        
        from pyvqnet.qnn.vqc.torch import T,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = T(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



S
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.S(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个S逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 S 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.torch import S,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = S(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


PauliX
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.PauliX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliX 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.torch import PauliX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PauliX(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


PauliY
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.PauliY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliY 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.torch import PauliY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PauliY(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



PauliZ
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.PauliZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliZ 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.torch import PauliZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PauliZ(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



X1
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.X1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个X1逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 X1 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.torch import X1,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = X1(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


RX
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RX逻辑门类 。


    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import RX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RX(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



RY
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import RY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RY(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


RZ
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import RZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RZ(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CRX
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CRX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import CRX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CRX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CRY
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CRY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import CRY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CRY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CRZ
""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.CRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import CRZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



U1
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.U1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U1逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U1 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import U1,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = U1(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

U2
""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.U2(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U2逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U2 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import U2,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = U2(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


U3
""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.U3(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U3逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U3 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import U3,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = U3(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



CNOT
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CNOT(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CNOT逻辑门类,也可称为CX。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CNOT 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import CNOT,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CNOT(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

CY
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import CY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CY(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CZ
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import CZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CZ(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)




CR
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CR(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CR逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CR 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import CR,QMachine
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
""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.SWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SWAP逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 SWAP 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import SWAP,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = SWAP(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


CSWAP
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.CSWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
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

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CSWAP 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import CSWAP,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = CSWAP(wires=[0,1,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

RXX
""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.RXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RXX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RXX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import RXX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

RYY
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RYY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RYY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import RYY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


RZZ
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个RZZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import RZZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



RZX
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.RZX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RZX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import RZX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = RZX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

Toffoli
""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.Toffoli(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个Toffoli逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 Toffoli 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import Toffoli,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = Toffoli(wires=[0,2,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

IsingXX
""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.IsingXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingXX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingXX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import IsingXX,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


IsingYY
""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.IsingYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingYY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingYY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import IsingYY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


IsingZZ
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.IsingZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingZZ逻辑门类 。


    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingZZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import IsingZZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


IsingXY
""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.IsingXY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingXY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingXY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import IsingXY,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = IsingXY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


PhaseShift
""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.PhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PhaseShift逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PhaseShift 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import PhaseShift,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = PhaseShift(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


MultiRZ
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.MultiRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个MultiRZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 MultiRZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import MultiRZ,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = MultiRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



SDG
""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.SDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SDG逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 SDG 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.torch import SDG,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = SDG(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)




TDG
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.TDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SDG逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 TDG 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.torch import TDG,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = TDG(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



ControlledPhaseShift
""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.ControlledPhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个ControlledPhaseShift逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 ControlledPhaseShift 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.torch import ControlledPhaseShift,QMachine
        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        device = QMachine(4)
        layer = ControlledPhaseShift(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



MultiControlledX
""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.MultiControlledX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False,control_values=None)
    
    定义一个MultiControlledX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
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
        from pyvqnet.qnn.vqc.torch import QMachine,MultiControlledX
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
"""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.Probability(wires=None, name="")

    计算量子线路在特定比特上概率测量结果。

    .. warning::
        
        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param wires: 测量比特的索引,列表、元组或者整数。
    :param name: 模块的名字,默认:""。
    :return: 测量结果,QTensor。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import Probability,rx,ry,cnot,QMachine,rz
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
"""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.MeasureAll(obs=None, name="")

    计算量子线路的测量结果,支持输入obs为多个或单个泡利算子或哈密顿量。
    例如:

    {\'wires\': [0,  1], \'observables\': [\'x\', \'i\'],\'coefficient\':[0.23,-3.5]}
    或:
    {\'X0\': 0.23}
    或:
    [{\'wires\': [0, 2, 3],\'observables\': [\'X\', \'Y\', \'Z\'],\'coefficient\': [1, 0.5, 0.4]}, {\'wires\': [0, 1, 2],\'observables\': [\'X\', \'Y\', \'Z\'],\'coefficient\': [1, 0.5, 0.4]}]

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param obs: observable。
    :param name: 模块的名字,默认:""。
    :return: 一个 MeasureAll 测量方法实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import MeasureAll,rx,ry,cnot,QMachine,rz
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
            'wires': [0, 2, 3],
            'observables': ['X', 'Y', 'Z'],
            'coefficient': [1, 0.5, 0.4]
        }, {
            'wires': [0, 1, 2],
            'observables': ['X', 'Y', 'Z'],
            'coefficient': [1, 0.5, 0.4]
        }]
        ma = MeasureAll(obs = obs_list)
        y = ma(q_machine=qm)
        print(y)



Samples
"""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.Samples(wires=None, obs=None, shots = 1,name="")

    获取特定线路上的带有 shot 的样本结果

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param wires: 样本量子比特索引。默认值: None,根据运行时使用模拟器的所有比特。
    :param obs: 该值只能设为None。
    :param shots: 样本重复次数,默认值: 1。
    :param name: 此模块的名称,默认值: “”。
    :return: 一个 Samples 测量方法实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import Samples,rx,ry,cnot,QMachine,rz
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
"""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.HermitianExpval(obs=None, name="")

    计算量子线路某个厄密特量的期望。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param obs: 厄密特量。
    :param name: 模块的名字,默认:""。
    :return: 一个 HermitianExpval 测量方法实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import QMachine, rx,ry,\
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
        batch_y.backward()

        print(batch_y)

量子线路常见模板
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VQC_HardwareEfficientAnsatz
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial = None,dtype=None)

    论文介绍的Hardware Efficient Ansatz的实现: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
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
        from pyvqnet.qnn.vqc.torch.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ
        from pyvqnet.qnn.vqc.torch import Probability,QMachine
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
        y.backward()
        print(y)



VQC_BasicEntanglerTemplate
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.VQC_BasicEntanglerTemplate(num_layer=1, num_qubits=1, rotation="RX", initial=None, dtype=None)

    由每个量子位上的单参数单量子位旋转组成的层,后跟一个闭合链或环组合的多个CNOT门。

    CNOT 门环将每个量子位与其邻居连接起来,最后一个量子位被认为是第一个量子位的邻居。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
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
        from pyvqnet.qnn.vqc.torch import QModule,\
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
        y.backward()
        print(y)



VQC_StronglyEntanglingTemplate
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.VQC_StronglyEntanglingTemplate(num_layers=1, num_qubits=1, rotation = "RX", initial = None, dtype: = None)

    由单个量子比特旋转和纠缠器组成的层,参考 `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param num_layers: 重复层数,默认值: 1。
    :param num_qubits: 量子比特数,默认值: 1。
    :param rotation: 要使用的单参数单量子比特门,默认值: `RX`
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数,默认值: None,此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用 float32。
    :return: VQC_BasicEntanglerTemplate 实例


    Example::

        from pyvqnet.nn.torch import TorchModule,Linear,TorchModuleList
        from pyvqnet.qnn.vqc.torch.qcircuit import VQC_StronglyEntanglingTemplate
        from pyvqnet.qnn.vqc.torch import Probability, QMachine
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
        y.backward()
        print(y)



VQC_QuantumEmbedding
""""""""""""""""""""""""""""""""""""""""


.. py:class:: pyvqnet.qnn.vqc.torch.VQC_QuantumEmbedding(  num_repetitions_input, depth_input, num_unitary_layers, num_repetitions,initial = None,dtype = None,name= "")

    使用 RZ,RY,RZ 创建变分量子电路,将经典数据编码为量子态。
    参考 `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
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
        from pyvqnet.qnn.vqc.torch.qcircuit import VQC_QuantumEmbedding
        from pyvqnet.qnn.vqc.torch import Probability, QMachine, MeasureAll
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
                                                num_repetitions, initial=tensor.full([1],12.0),dtype=pyvqnet.kfloat64)

                self.measure = MeasureAll(obs={f"Z{nq-1}":1})
                self.device = QMachine(nq)

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(x.shape[0])
                self.ansatz(x,q_machine=self.device)
                return self.measure(q_machine=self.device)

        inputx = tensor.arange(1.0, bz * depth_input + 1).reshape([bz, depth_input])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)


ExpressiveEntanglingAnsatz
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.ExpressiveEntanglingAnsatz(type: int, num_wires: int, depth: int, dtype=None, name: str = "")

    论文 `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_ 中的 19 种不同的ansatz。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.torch.QModule`` 以及 ``torch.nn.Module``。
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
        from pyvqnet.qnn.vqc.torch.qcircuit import ExpressiveEntanglingAnsatz
        from pyvqnet.qnn.vqc.torch import Probability, QMachine, MeasureAll
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
                self.measure = MeasureAll(obs={
                    'wires': [1],
                    'observables': ['z'],
                    'coefficient': [1]
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.c1(q_machine = self.qm)
                rlt = self.measure(q_machine=self.qm)
                return rlt
            

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()
        print(batch_y)



vqc_basis_embedding
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_basis_embedding(basis_state,q_machine)

    将n个二进制特征编码到 ``q_machine`` 的n个量子比特的基态。该函数别名 `VQC_BasisEmbedding` 。

    例如, 对于 ``basis_state=([0, 1, 1])``, 在量子系统下其基态为 :math:`|011 \rangle`。

    :param basis_state:  ``(n)`` 大小的二进制输入。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_basis_embedding,QMachine
        qm  = QMachine(3)
        vqc_basis_embedding(basis_state=[1,1,0],q_machine=qm)
        print(qm.states)




vqc_angle_embedding
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_angle_embedding(input_feat, wires, q_machine: pyvqnet.qnn.vqc.torch.QMachine, rotation: str = "X")

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
        from pyvqnet.qnn.vqc.torch import vqc_angle_embedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(2)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='X')
        print(qm.states)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Y')
        print(qm.states)
        vqc_angle_embedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Z')
        print(qm.states)



vqc_amplitude_embedding
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_amplitude_embeddingVQC_AmplitudeEmbeddingCircuit(input_feature, q_machine)

    将 :math:`2^n` 特征编码为 :math:`n` 量子比特的振幅向量。该函数别名 `VQC_AmplitudeEmbedding` 。

    :param input_feature: 表示参数的numpy数组。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_amplitude_embedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        vqc_amplitude_embedding(QTensor([3.2,-2,-2,0.3,12,0.1,2,-1]), q_machine=qm)
        print(qm.states)



vqc_iqp_embedding
""""""""""""""""""""""""""""""""""""""""
.. py:function:: pyvqnet.qnn.vqc.vqc_iqp_embedding(input_feat, q_machine: pyvqnet.qnn.vqc.torch.QMachine, rep: int = 1)

    使用IQP线路的对角门将 :math:`n` 特征编码为 :math:`n` 量子比特。该函数别名:  ``VQC_IQPEmbedding`` 。

    编码是由 `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_ 提出。

    通过指定 ``rep`` ,可以重复基本IQP线路。

    :param input_feat: 表示参数的数组。
    :param q_machine: 量子虚拟机设备。
    :param rep: 重复量子线路块次数,默认次数为1。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_iqp_embedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        vqc_iqp_embedding(QTensor([3.2,-2,-2]), q_machine=qm)
        print(qm.states)        



vqc_rotcircuit
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_rotcircuit(q_machine, wire, params)

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
        from pyvqnet.qnn.vqc.torch import vqc_rotcircuit, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        vqc_rotcircuit(q_machine=qm, wire=[1],params=QTensor([2.0,1.5,2.1]))
        print(qm.states)


vqc_crot_circuit
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_crot_circuit(para,control_qubits,rot_wire,q_machine)

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
        from pyvqnet.qnn.vqc.torch import vqc_crot_circuit,QMachine, MeasureAll
        p = QTensor([2, 3, 4.0])
        qm = QMachine(2)
        vqc_crot_circuit(p, 0, 1, qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)




vqc_controlled_hadamard
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_controlled_hadamard(wires, q_machine)

    受控Hadamard逻辑门量子线路。该函数别名:  ``VQC_Controlled_Hadamard`` 。

    .. math:: CH = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
            0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: 量子比特索引列表, 第一位是控制比特, 列表长度为2。
    :param q_machine: 量子虚拟机设备。
    

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_controlled_hadamard,\
            QMachine, MeasureAll

        p = QTensor([0.2, 3, 4.0])
        qm = QMachine(3)
        vqc_controlled_hadamard([1, 0], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_ccz
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_ccz(wires, q_machine)

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
        from pyvqnet.qnn.vqc.torch import vqc_ccz,QMachine, MeasureAll
        p = QTensor([0.2, 3, 4.0])

        qm = QMachine(3)

        vqc_ccz([1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)



vqc_fermionic_single_excitation
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_fermionic_single_excitation(weight, wires, q_machine)

    对泡利矩阵的张量积求幂的耦合簇单激励算子。矩阵形式下式给出:

    .. math::

        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    该函数别名:  ``VQC_FermionicSingleExcitation`` 。

    :param weight:  量子比特p上的参数, 只有一个元素.
    :param wires: 表示区间[r, p]中的量子比特索引子集。最小长度必须为2。第一索引值被解释为r,最后一个索引值被解释为p。
                中间的索引被CNOT门作用,以计算量子位集的奇偶校验。
    :param q_machine: 量子虚拟机设备。

    

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_fermionic_single_excitation,\
            QMachine, MeasureAll
        qm = QMachine(3)
        p0 = QTensor([0.5])

        vqc_fermionic_single_excitation(p0, [1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

 


vqc_fermionic_double_excitation
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_fermionic_double_excitation(weight, wires1, wires2, q_machine)

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

    

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.torch import vqc_fermionic_double_excitation,\
            QMachine, MeasureAll
        qm = QMachine(5)
        p0 = QTensor([0.5])

        vqc_fermionic_double_excitation(p0, [0, 1], [2, 3], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)
 

vqc_uccsd
""""""""""""""""""""""""""""""""""""""""


.. py:function:: pyvqnet.qnn.vqc.torch.vqc_uccsd(weights, wires, s_wires, d_wires, init_state, q_machine)

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
    
    
    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_uccsd, QMachine, MeasureAll
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
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_zfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.torch.QMachine, data_map_func=None, rep: int = 2)

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
        from pyvqnet.qnn.vqc.torch import vqc_zfeaturemap, QMachine, hadamard
        from pyvqnet.tensor import QTensor
        qm = QMachine(3)
        for i in range(3):
            hadamard(q_machine=qm, wires=[i])
        vqc_zfeaturemap(input_feat=QTensor([[0.1,0.2,0.3]]),q_machine = qm)
        print(qm.states)
 

vqc_zzfeaturemap
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_zzfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.torch.QMachine, data_map_func=None, entanglement: Union[str, List[List[int]],Callable[[int], List[int]]] = "full",rep: int = 2)

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
        from pyvqnet.qnn.vqc.torch import vqc_zzfeaturemap, QMachine
        from pyvqnet.tensor import QTensor

        qm = QMachine(3)
        vqc_zzfeaturemap(q_machine=qm, input_feat=QTensor([[0.1,0.2,0.3]]))
        print(qm.states)


vqc_allsinglesdoubles
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_allsinglesdoubles(weights, q_machine: pyvqnet.qnn.vqc.torch.QMachine, hf_state, wires, singles=None, doubles=None)

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
        from pyvqnet.qnn.vqc.torch import vqc_allsinglesdoubles, QMachine

        from pyvqnet.tensor import QTensor
        qubits = 4
        qm = QMachine(qubits)

        vqc_allsinglesdoubles(q_machine=qm, weights=QTensor([0.55, 0.11, 0.53]), 
                              hf_state = QTensor([1,1,0,0]), singles=[[0, 2], [1, 3]], doubles=[[0, 1, 2, 3]], wires=[0,1,2,3])
        print(qm.states)

vqc_basisrotation
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_basisrotation(q_machine: pyvqnet.qnn.vqc.torch.QMachine, wires, unitary_matrix: QTensor, check=False)

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
        from pyvqnet.qnn.vqc.torch import vqc_basisrotation, QMachine
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
                        unitary_matrix=QTensor(umat, dtype=qm.state.dtype))

        print(qm.states)



vqc_quantumpooling_circuit
""""""""""""""""""""""""""""""""""""""""

.. py:function:: pyvqnet.qnn.vqc.torch.vqc_quantumpooling_circuit(ignored_wires, sinks_wires, params, q_machine)

    对数据进行降采样的量子电路。

    为了减少电路中的量子位数量,首先在系统中创建成对的量子位。在最初配对所有量子位之后,将广义2量子位酉元应用于每一对量子位上。并在应用这两个量子位酉元之后,在神经网络的其余部分忽略每对量子位中的一个量子位。

    :param sources_wires: 将被忽略的源量子位索引。
    :param sinks_wires: 将保留的目标量子位索引。
    :param params: 输入参数。
    :param q_machine: 量子虚拟机设备。

    

    Examples:: 

        import pyvqnet

        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.torch import vqc_quantumpooling_circuit, QMachine, MeasureAll
        from pyvqnet import tensor
        p = tensor.full([6], 0.35)
        qm = QMachine(4)
        vqc_quantumpooling_circuit(q_machine=qm,
                                ignored_wires=[0, 1],
                                sinks_wires=[2, 3],
                                params=p)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)


QuantumLayerAdjoint
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.QuantumLayerAdjoint(general_module, use_qpanda=False,name="")


    使用伴随矩阵方式进行梯度计算的可自动微分的变分量子线路层,参考  `Efficient calculation of gradients in classical simulations of variational quantum algorithms <https://arxiv.org/abs/2009.02823>`_ 。

    :param general_module: 一个仅使用 ``pyvqnet.qnn.vqc.torch`` 下量子线路接口搭建的 ``pyvqnet.qnn.vqc.torch.QModule`` 实例。
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
        from pyvqnet.qnn.vqc.torch import QuantumLayerAdjoint, \
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
                    'wires': [1],
                    'observables': ['x'],
                    'coefficient': [1]
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
        batch_y.backward()




TorchHybirdVQCQpanda3QVMLayer
""""""""""""""""""""""""""""""""""""""""

.. py:class:: pyvqnet.qnn.vqc.torch.TorchHybirdVQCQpanda3QVMLayer(vqc_module: Module,qcloud_token: str,pauli_str_dict: Union[List[Dict], Dict, None] = None,shots: int = 1000,dtype: Union[int, None] = None,name: str = "",submit_kwargs: Dict = {},query_kwargs: Dict = {})


    使用torch后端,混合 vqc 和 qpanda3 模拟计算。该层将用户 `forward` 函数定义的VQNet编写的量子线路计算转化为QPanda OriginIR,在QPanda3本地虚拟机或者云端服务上进行前向运行,并在基于自动微分计算线路参数梯度,降低了使用参数漂移法计算的时间复杂度。
    其中 ``vqc_module`` 为用户自定义的量子变分线路模型,其中的QMachine设置 ``save_ir= True`` 。

    :param vqc_module: 带有 forward() 的 vqc_module。
    :param qcloud_token: `str` - 量子机器的类型或用于执行的云令牌。
    :param pauli_str_dict: `dict|list` - 表示量子电路中泡利算子的字典或字典列表。默认值为 None。
    :param shots: `int` - 量子线路测量次数。默认值为 1000。
    :param name: 模块名称。默认值为空字符串。
    :param submit_kwargs: 提交量子电路的附加关键字参数,默认值:
        {"chip_id":pyqpanda.real_chip_type.origin_72,
        "is_amend":True,"is_mapping":True,
        "is_optimization":True,
        "default_task_group_size":200,
        "test_qcloud_fake":True}。
    :param query_kwargs: 查询量子结果的附加关键字参数,默认值:{"timeout":2,"print_query_info":True,"sub_circuits_split_size":1}。
    
    :return: 可以计算量子电路的模块。


    .. warning::

        该类继承于 ``pyvqnet.nn.torch.TorchModule`` 以及 ``pyvqnet.qnn.pq3.HybirdVQCQpandaQVMLayer`` ,可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    .. note::

        pauli_str_dict 不能为 None,并且应与 vqc_module 测量函数中的 obs 相同。
        vqc_module 应具有 QMachine 类型的属性,QMachine 应设置 save_ir=True

    Example::

        import pyvqnet.backends
        import numpy as np
        from pyvqnet.qnn.vqc.torch import QMachine,QModule,RX,RY,\
        RZ,U1,U2,U3,I,S,X1,PauliX,PauliY,PauliZ,SWAP,CZ,\
        RXX,RYY,RZX,RZZ,CR,Toffoli,Hadamard,T,CNOT,MeasureAll
        from pyvqnet.qnn.vqc.torch import HybirdVQCQpanda3QVMLayer
        import pyvqnet

        from pyvqnet import tensor

        import pyvqnet.utils
        pyvqnet.backends.set_backend("torch")
        pyvqnet.utils.set_random_seed(42)

        class QModel(QModule):
            def __init__(self, num_wires, dtype,grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype,grad_mode=grad_mode,save_ir=True)
                self.rx_layer = RX(has_params=True, trainable=False, wires=0)
                self.ry_layer = RY(has_params=True, trainable=False, wires=1)
                self.rz_layer = RZ(has_params=True, trainable=False, wires=1)
                self.u1 = U1(has_params=True,trainable=True,wires=[2])
                self.u2 = U2(has_params=True,trainable=True,wires=[3])
                self.u3 = U3(has_params=True,trainable=True,wires=[1])
                self.i = I(wires=[3])
                self.s = S(wires=[3])
                self.x1 = X1(wires=[3])
                
                self.x = PauliX(wires=[3])
                self.y = PauliY(wires=[3])
                self.z = PauliZ(wires=[3])
                self.swap = SWAP(wires=[2,3])
                self.cz = CZ(wires=[2,3])
                self.cr = CR(has_params=True,trainable=True,wires=[2,3])
                self.rxx = RXX(has_params=True,trainable=True,wires=[2,3])
                self.rzz = RYY(has_params=True,trainable=True,wires=[2,3])
                self.ryy = RZZ(has_params=True,trainable=True,wires=[2,3])
                self.rzx = RZX(has_params=True,trainable=False, wires=[2,3])
                self.toffoli = Toffoli(wires=[2,3,4],use_dagger=True)
                self.h =Hadamard(wires=[1])


                self.tlayer = T(wires=1)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs={'Z0':2,'Y3':3} 
            )

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.i(q_machine=self.qm)
                self.s(q_machine=self.qm)
                self.swap(q_machine=self.qm)
                self.cz(q_machine=self.qm)
                self.x(q_machine=self.qm)
                self.x1(q_machine=self.qm)
                self.y(q_machine=self.qm)

                self.z(q_machine=self.qm)

                self.ryy(q_machine=self.qm)
                self.rxx(q_machine=self.qm)
                self.rzz(q_machine=self.qm)
                self.rzx(q_machine=self.qm,params = x[:,[1]])
                self.cr(q_machine=self.qm)
                self.u1(q_machine=self.qm)
                self.u2(q_machine=self.qm)
                self.u3(q_machine=self.qm)
                self.rx_layer(params = x[:,[0]], q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                self.h(q_machine=self.qm)

                self.ry_layer(params = x[:,[1]], q_machine=self.qm)
                self.tlayer(q_machine=self.qm)
                self.rz_layer(params = x[:,[2]], q_machine=self.qm)
                self.toffoli(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)

                return rlt
            

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        input_x = tensor.broadcast_to(input_x,[2,3])

        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=6, dtype=pyvqnet.kcomplex64)

        l = HybirdVQCQpanda3QVMLayer(qunatum_model,
                                "3047DE8A59764BEDAC9C3282093B16AF1",

                    pauli_str_dict={'Z0':2,'Y3':3},
                    shots = 1000,
                    name="",
            submit_kwargs={"test_qcloud_fake":True},
                    query_kwargs={})

        y = l(input_x)
        print(y)

        y.backward()
        print(input_x.grad)


张量网络后端变分量子线路模块
============================================

基类
--------------------------------------------------

TNQModule
^^^^^^^^^^^^^^^^^^^^^^^^

基于张量网络编写变分量子线路模型需要继承于 ``TNQModule``

.. py:class:: pyvqnet.qnn.vqc.tn.TNQModule(use_jit=False,vectorized_argnums=0,name="")

    在 `torch` 后端下,定义张量网络下量子变分线路模型 `Module` 应该继承的基类。
    该类用于使用张量网络来模块来用语执行量子线路。

    :param use_jit: 开启即时编译功能, 默认为False。
    :param vectorized_argnums: 要被向量化的参数,这些参数应该在同一维共享相同的批次形状,默认值为0
    :param name: 模块名。

    .. note::

        该类以及其派生类仅适用于 ``pyvqnet.backends.set_backend("torch")`` , 不要与默认 ``pyvqnet.nn`` 下的 ``Module`` 混用。

    .. note::
        
        批量化必须搭配TNQModule使用。

    Example::

        import pyvqnet
        from pyvqnet.nn import Parameter
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import TNQModule
        from pyvqnet.qnn.vqc.tn import TNQMachine, RX, RY, CNOT, PauliX, PauliZ,qmeasure,qcircuit,VQC_RotCircuit
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


                def build_circult(weights, xx, nqubits,qm):
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

                build_circult(self.w, x,range(4),self.qm)

                y= qmeasure.MeasureAll(obs={'Z0': 1})(self.qm)
                return y


        x= pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        y.backward()

TNQMachine
^^^^^^^^^^^^^^^^^^^^^^^^

基于张量网络编写变分量子线路设备需要 ``TNQMachine`` 进行初始化。 

.. py:class:: pyvqnet.qnn.vqc.tn.TNQMachine(num_wires, dtype=pyvqnet.kcomplex64,use_mps=False)

    变分量子计算的模拟器类,包含states属性为量子线路的statevectors。

    .. warning::

        该类继承于 ``pyvqnet.nn.tn.TorchModule`` 以及 ``pyvqnet.qnn.QMachine`` 。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入 ``TNQModule`` 的模型中。

    .. warning::
        
        在每次运行一个完整的量子线路之前,必须使用 `pyvqnet.qnn.vqc.tn.TNQMachine.reset_states(batchsize)` 将模拟器里面初态重新初始化,并且广播为
        (batchsize,*) 维度从而适应批量数据训练。

    .. warning::
        
        在张量网络的量子线路中，默认会开启 ``vmap`` 功能，在线路上的逻辑门参数上均为舍弃了批次维度

    .. note::
        
        批量化必须搭配TNQModule使用。

    :param num_wires: 量子比特数。
    :param dtype: 计算数据的数据类型。默认值是pyvqnet。kcomplex64,对应的参数精度为pyvqnet.kfloat32。
    :param use_mps: 是否基于mpscircuit进行模拟, 用于模拟大比特量子线路执行。

    :return: 输出一个TNQMachine对象。

    Example::
        
        import pyvqnet
        from pyvqnet.nn import Parameter
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import TNQModule
        from pyvqnet.qnn.vqc.tn import TNQMachine, RX, RY, CNOT, PauliX, PauliZ,qmeasure,qcircuit,VQC_RotCircuit
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


                def build_circult(weights, xx, nqubits,qm):
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

                build_circult(self.w, x,range(4),self.qm)

                y= qmeasure.MeasureAll(obs={'Z0': 1})(self.qm)
                return y


        x= pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        y.backward()

    .. py:method:: get_states()

        获得张量网络中的states。

变分量子逻辑门模块
--------------------------------------------------


以下 ``pyvqnet.qnn.vqc`` 中的函数接口直接支持 ``torch`` 后端的 ``QTensor`` 进行计算，通过 ``pyvqnet.qnn.vqc.tn`` 下调用使用。

.. csv-table:: 已支持pyvqnet.qnn.vqc接口列表
   :file: ./images/same_apis_from_tn.csv

以下量子线路模块继承于 ``pyvqnet.qnn.vqc.tn.TNQModule``,其中计算使用 ``torch.Tensor`` 进行计算。


.. warning::

    该类以及其派生类仅适用于 ``pyvqnet.backends.set_backend("torch")`` , 不要与默认 ``pyvqnet.nn`` 下的 ``Module`` 混用。

    这些类如果有非参数成员变量 ``_buffers`` ,则其中的数据为 ``torch.Tensor`` 类型。
    这些类如果有参数成员变量 ``_parmeters`` ,则其中的数据为 ``torch.nn.Parameter`` 类型。

I
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个I逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params: 是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 I 逻辑门实例

    Example::
        
        from pyvqnet.qnn.vqc.tn import I,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = I(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



Hadamard
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.Hadamard(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个Hadamard逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 Hadamard 逻辑门实例

    Example::
        
        from pyvqnet.qnn.vqc.tn import Hadamard,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = Hadamard(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



T
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.T(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个T逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 T 逻辑门实例

    Example::
        
        from pyvqnet.qnn.vqc.tn import T,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = T(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



S
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.S(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个S逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 S 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.tn import S,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = S(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



PauliX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.PauliX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliX 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.tn import PauliX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = PauliX(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


PauliY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.PauliY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliY 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.tn import PauliY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = PauliY(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



PauliZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.PauliZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliZ 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.tn import PauliZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = PauliZ(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



X1
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.X1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个X1逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 X1 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.tn import X1,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = X1(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


RX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RX逻辑门类 。


    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import RX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RX(wires=0,has_params=True)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)




RY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import RY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RY(wires=0,has_params=True)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


RZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import RZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RZ(wires=0,has_params=True)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CRX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CRX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import CRX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CRX(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CRY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CRY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import CRY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CRY(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CRZ
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.CRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import CRZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CRZ(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)




U1
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.U1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U1逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U1 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import U1,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = U1(has_params= True, trainable= True, wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


U2
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.U2(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U2逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U2 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import U2,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = U2(has_params= True, trainable= True, wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



U3
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.U3(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U3逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U3 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import U3,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = U3(has_params= True, trainable= True, wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



CNOT
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CNOT(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CNOT逻辑门类,也可称为CX。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CNOT 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import CNOT,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CNOT(wires=[0,1])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import CY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CY(wires=[0,1])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



CZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import CZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CZ(wires=[0,1])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



CR
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CR(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CR逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CR 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import CR,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CR(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



SWAP
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.SWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SWAP逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 SWAP 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import SWAP,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = SWAP(wires=[0,1])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


CSWAP
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.CSWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
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

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CSWAP 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import CSWAP,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = CSWAP(wires=[0,1,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


RXX
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.RXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RXX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RXX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import RXX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RXX(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


RYY
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RYY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RYY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import RYY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RYY(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


RZZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个RZZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import RZZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RZZ(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



RZX
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.RZX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RZX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import RZX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = RZX(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


Toffoli
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.Toffoli(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个Toffoli逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 Toffoli 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import Toffoli,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = Toffoli(wires=[0,2,1])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


IsingXX
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.IsingXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingXX逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingXX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import IsingXX,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = IsingXX(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



IsingYY
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.IsingYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingYY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingYY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import IsingYY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = IsingYY(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


IsingZZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.IsingZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingZZ逻辑门类 。


    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingZZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import IsingZZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = IsingZZ(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


IsingXY
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.IsingXY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingXY逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingXY 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import IsingXY,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = IsingXY(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


PhaseShift
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.PhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PhaseShift逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PhaseShift 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import PhaseShift,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = PhaseShift(has_params= True, trainable= True, wires=1)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


MultiRZ
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.MultiRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个MultiRZ逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 MultiRZ 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import MultiRZ,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = MultiRZ(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



SDG
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.SDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SDG逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 SDG 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.tn import SDG,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = SDG(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)




TDG
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.TDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SDG逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 TDG 逻辑门实例。

    Example::
        
        from pyvqnet.qnn.vqc.tn import TDG,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = TDG(wires=0)
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)



ControlledPhaseShift
^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.ControlledPhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个ControlledPhaseShift逻辑门类 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 ControlledPhaseShift 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc.tn import ControlledPhaseShift,TNQMachine,TNQModule,MeasureAll, rx
        import pyvqnet
        pyvqnet.backends.set_backend("torch")

        class QModel(TNQModule):
            
            def __init__(self, num_wires, dtype,batch_size=2):
                super(QModel, self).__init__()
                self.device = TNQMachine(num_wires)
                self.layer = ControlledPhaseShift(has_params= True, trainable= True, wires=[0,2])
                self.batch_size = batch_size
                self.num_wires = num_wires
                
            def forward(self, x, *args, **kwargs):
                self.device.reset_states(batchsize=self.batch_size)
                for i in range(self.num_wires):
                    rx(self.device, wires=i, params=x[i])
                self.layer(q_machine = self.device)
                y = MeasureAll(obs={'Z0': 1})(self.device)
                return y

        x = pyvqnet.tensor.QTensor([[1,0,0,1],[1,1,0,1]],dtype=pyvqnet.kfloat32,requires_grad=True)
        model = QModel(4,pyvqnet.kcomplex64,2)
        y = model(x)
        print(y)


常见测量接口
--------------------------------------

VQC_Purity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.VQC_Purity(state, qubits_idx, num_wires, use_tn=False)

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
        from pyvqnet.qnn.vqc.tn import TNQMachine, qcircuit, TNQModule,VQC_Purity
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
                return VQC_Purity(self.device.get_states(), [0, 1], num_wires=3, use_tn=True)

        model = QM().toGPU()
        y_tn = model(x)
        x.data.retain_grad()
        y_tn.backward()
        print(y_tn)

VQC_VarMeasure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.VQC_VarMeasure(q_machine, obs)

    提供的可观察量 ``obs`` 的方差。

    :param q_machine: 从pyqpanda get_qstate()获取的量子态
    :param obs: 测量观测量,当前支持Hadamard,I,PauliX,PauliY,PauliZ 几种Observable.

    :return: 计算可观测量方差。

    .. note::

        测量结果一般为[b,1],其中b为 q_machine.reset_states(b)的批处理数量b。

    Example::

        import pyvqnet
        from pyvqnet.qnn.vqc.tn import TNQMachine, qcircuit, VQC_VarMeasure, TNQModule,PauliY
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
        y.backward()
        print(y)

        # [[0.9370641],
        # [0.9516521]]


VQC_DensityMatrixFromQstate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.VQC_DensityMatrixFromQstate(state, indices, use_tn=False)

    计算量子态在一组特定量子比特上的密度矩阵。

    :param state: 一维列表状态向量。 这个列表的大小应该是 ``(2**N,)`` 对于量子比特个数 ``N`` ,qstate 应该从 000 ->111 开始。
    :param indices: 所考虑子系统中的量子比特索引列表。
    :param use_tn: 张量网络后端时改成True, 默认False.
    :return: 大小为“(b, 2**len(indices), 2**len(indices))”的密度矩阵,其中b为 q_machine.reset_states(b)的批处理数量b。

    Example::

        import pyvqnet
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import TNQMachine, qcircuit, VQC_DensityMatrixFromQstate,TNQModule
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
                return VQC_DensityMatrixFromQstate(self.device.get_states(),[0,1],use_tn=True)
            
        model = QM().toGPU()
        y = model(x)
        x.data.retain_grad()
        y.backward()
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


.. py:class:: pyvqnet.qnn.vqc.tn.Probability(wires=None, name="")

    计算量子线路在特定比特上概率测量结果。

    .. warning::
        
        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。

    :param wires: 测量比特的索引,列表、元组或者整数。
    :param name: 模块的名字,默认:""。
    :return: 测量结果,QTensor。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import Probability,rx,ry,cnot,TNQMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
        qm = TNQMachine(4)
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
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.MeasureAll(obs=None, name="")

    计算量子线路的测量结果,支持输入obs为多个或单个泡利算子或哈密顿量。
    例如:

    {\'wires\': [0,  1], \'observables\': [\'x\', \'i\'],\'coefficient\':[0.23,-3.5]}
    或:
    {\'X0\': 0.23}
    或:
    [{\'wires\': [0, 2, 3],\'observables\': [\'X\', \'Y\', \'Z\'],\'coefficient\': [1, 0.5, 0.4]}, {\'wires\': [0, 1, 2],\'observables\': [\'X\', \'Y\', \'Z\'],\'coefficient\': [1, 0.5, 0.4]}]

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param obs: observable。
    :param name: 模块的名字,默认:""。
    :return: 一个 MeasureAll 测量方法实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import MeasureAll,rx,ry,cnot,TNQMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)
        qm = TNQMachine(4)
        qm.reset_states(2)
        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rz(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,2])
        rz(q_machine=qm,wires=3,params=x[:,[1]])
        obs_list = [{
            'wires': [0, 2, 3],
            'observables': ['X', 'Y', 'Z'],
            'coefficient': [1, 0.5, 0.4]
        }, {
            'wires': [0, 1, 2],
            'observables': ['X', 'Y', 'Z'],
            'coefficient': [1, 0.5, 0.4]
        }]
        ma = MeasureAll(obs = obs_list)
        y = ma(q_machine=qm)
        print(y)



Samples
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.Samples(wires=None, obs=None, shots = 1,name="")

    获取特定线路上的带有 shot 的样本结果

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param wires: 样本量子比特索引。默认值: None,根据运行时使用模拟器的所有比特。
    :param obs: 该值只能设为None。
    :param shots: 样本重复次数,默认值: 1。
    :param name: 此模块的名称,默认值: “”。
    :return: 一个 Samples 测量方法实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import Samples,rx,ry,cnot,TNQMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)

        qm = TNQMachine(4)
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
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.HermitianExpval(obs=None, name="")

    计算量子线路某个厄密特量的期望。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param obs: 厄密特量。
    :param name: 模块的名字,默认:""。
    :return: 一个 HermitianExpval 测量方法实例。

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import TNQMachine, rx,ry,\
            RX, RY, CNOT, PauliX, PauliZ, VQC_RotCircuit,HermitianExpval, TNQModule
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
        batch_y.backward()

        print(batch_y)


常见量子线路模版
--------------------------------------------------

VQC_HardwareEfficientAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial = None,dtype=None)

    论文介绍的Hardware Efficient Ansatz的实现: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ 。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
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
        from pyvqnet.qnn.vqc.tn.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ
        from pyvqnet.qnn.vqc.tn import Probability,TNQMachine, TNQModule
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
        y.backward()
        print(y)


VQC_BasicEntanglerTemplate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.VQC_BasicEntanglerTemplate(num_layer=1, num_qubits=1, rotation="RX", initial=None, dtype=None)

    由每个量子位上的单参数单量子位旋转组成的层,后跟一个闭合链或环组合的多个CNOT门。

    CNOT 门环将每个量子位与其邻居连接起来,最后一个量子位被认为是第一个量子位的邻居。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
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
        from pyvqnet.qnn.vqc.tn import TNQModule,\
            VQC_BasicEntanglerTemplate, Probability, TNQMachine
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
        y.backward()
        print(y)



VQC_StronglyEntanglingTemplate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.VQC_StronglyEntanglingTemplate(num_layers=1, num_qubits=1, rotation = "RX", initial = None, dtype: = None)

    由单个量子比特旋转和纠缠器组成的层,参考 `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param num_layers: 重复层数,默认值: 1。
    :param num_qubits: 量子比特数,默认值: 1。
    :param rotation: 要使用的单参数单量子比特门,默认值: `RX`
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数,默认值: None,此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用 float32。
    :return: VQC_BasicEntanglerTemplate 实例


    Example::

        from pyvqnet.nn.torch import TorchModule,Linear,TorchModuleList
        from pyvqnet.qnn.vqc.tn.qcircuit import VQC_StronglyEntanglingTemplate
        from pyvqnet.qnn.vqc.tn import Probability, TNQMachine, TNQModule
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
        y.backward()
        print(y)

VQC_QuantumEmbedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:class:: pyvqnet.qnn.vqc.tn.VQC_QuantumEmbedding(  num_repetitions_input, depth_input, num_unitary_layers, num_repetitions,initial = None,dtype = None,name= "")

    使用 RZ,RY,RZ 创建变分量子电路,将经典数据编码为量子态。
    参考 `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param num_repetitions_input: 子模块中输入编码的重复次数。
    :paramdepth_input: 输入维数。
    :param num_unitary_layers: 变分量子门的重复次数。
    :param num_repetitions: 子模块的重复次数。
    :param initial: 参数初始化值,默认为None
    :param dtype: 参数的类型,默认 None,使用float32.
    :param name: 类的名字

    Example::

        from pyvqnet.qnn.vqc.tn.qcircuit import VQC_QuantumEmbedding
        from pyvqnet.qnn.vqc.tn import TNQMachine, MeasureAll, TNQModule
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
                                                num_repetitions, initial=tensor.full([1],12.0),dtype=pyvqnet.kfloat64)

                self.measure = MeasureAll(obs={f"Z{nq-1}":1})
                self.device = TNQMachine(nq)

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(bz)
                self.ansatz(x,q_machine=self.device)
                return self.measure(q_machine=self.device)

        inputx = tensor.arange(1.0, bz * depth_input + 1).reshape([bz, depth_input])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)


ExpressiveEntanglingAnsatz
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.qnn.vqc.tn.ExpressiveEntanglingAnsatz(type: int, num_wires: int, depth: int, dtype=None, name: str = "")

    论文 `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_ 中的 19 种不同的ansatz。

    .. warning::

        该类继承于 ``pyvqnet.qnn.vqc.tn.QModule`` 以及 ``torch.nn.Module``。
        该类可以作为 ``torch.nn.Module`` 的一个子模块加入torch的模型中。


    :param type: 电路类型从 1 到 19,共19种线路。
    :param num_wires: 量子比特数。
    :param depth: 电路深度。
    :param dtype: 参数的数据类型, 默认值: None, 使用 float32。
    :param name: 名字,默认"".

    :return:
        一个 ExpressiveEntanglingAnsatz 实例

    Example::

        from pyvqnet.qnn.vqc.tn.qcircuit import ExpressiveEntanglingAnsatz
        from pyvqnet.qnn.vqc.tn import Probability, TNQMachine, MeasureAll, TNQModule
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
                    'wires': [1],
                    'observables': ['z'],
                    'coefficient': [1]
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(1)
                self.c1(q_machine = self.qm)
                rlt = self.measure(q_machine=self.qm)
                return rlt
            

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        batch_y.backward()
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
        from pyvqnet.qnn.vqc.tn import vqc_basis_embedding,TNQMachine
        qm  = TNQMachine(3)
        vqc_basis_embedding(basis_state=[1,1,0],q_machine=qm)
        print(qm.get_states())




vqc_angle_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. py:function:: pyvqnet.qnn.vqc.tn.vqc_angle_embedding(input_feat, wires, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, rotation: str = "X")

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
        from pyvqnet.qnn.vqc.tn import vqc_angle_embedding, TNQMachine
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

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_amplitude_embeddingVQC_AmplitudeEmbeddingCircuit(input_feature, q_machine)

    将 :math:`2^n` 特征编码为 :math:`n` 量子比特的振幅向量。该函数别名 `VQC_AmplitudeEmbedding` 。

    :param input_feature: 表示参数的numpy数组。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_amplitude_embedding, TNQMachine
        from pyvqnet.tensor import QTensor
        qm  = TNQMachine(3)
        vqc_amplitude_embedding(QTensor([3.2,-2,-2,0.3,12,0.1,2,-1]), q_machine=qm)
        print(qm.get_states())



vqc_iqp_embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:function:: pyvqnet.qnn.vqc.tn.vqc_iqp_embedding(input_feat, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, rep: int = 1)

    使用IQP线路的对角门将 :math:`n` 特征编码为 :math:`n` 量子比特。该函数别名:  ``VQC_IQPEmbedding`` 。

    编码是由 `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_ 提出。

    通过指定 ``rep`` ,可以重复基本IQP线路。

    :param input_feat: 表示参数的数组。
    :param q_machine: 量子虚拟机设备。
    :param rep: 重复量子线路块次数,默认次数为1。
    

    Example::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_iqp_embedding, TNQMachine
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
        from pyvqnet.qnn.vqc.tn import vqc_rotcircuit, TNQMachine
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
        from pyvqnet.qnn.vqc.tn import vqc_crot_circuit,TNQMachine, MeasureAll
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
    

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_controlled_hadamard,\
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
        from pyvqnet.qnn.vqc.tn import vqc_ccz,TNQMachine, MeasureAll
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

    

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_fermionic_single_excitation,\
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

    

    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.tn import vqc_fermionic_double_excitation,\
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
    
    
    Examples::

        import pyvqnet
        pyvqnet.backends.set_backend("torch")
        from pyvqnet.qnn.vqc.tn import vqc_uccsd, TNQMachine, MeasureAll
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

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_zfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, data_map_func=None, rep: int = 2)

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
        from pyvqnet.qnn.vqc.tn import vqc_zfeaturemap, TNQMachine, hadamard
        from pyvqnet.tensor import QTensor
        qm = TNQMachine(3)
        for i in range(3):
            hadamard(q_machine=qm, wires=[i])
        vqc_zfeaturemap(input_feat=QTensor([[0.1,0.2,0.3]]),q_machine = qm)
        print(qm.get_states())
 

vqc_zzfeaturemap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_zzfeaturemap(input_feat, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, data_map_func=None, entanglement: Union[str, List[List[int]],Callable[[int], List[int]]] = "full",rep: int = 2)

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
        from pyvqnet.qnn.vqc.tn import vqc_zzfeaturemap, TNQMachine
        from pyvqnet.tensor import QTensor

        qm = TNQMachine(3)
        vqc_zzfeaturemap(q_machine=qm, input_feat=QTensor([[0.1,0.2,0.3]]))
        print(qm.get_states())


vqc_allsinglesdoubles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_allsinglesdoubles(weights, q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, hf_state, wires, singles=None, doubles=None)

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
        from pyvqnet.qnn.vqc.tn import vqc_allsinglesdoubles, TNQMachine

        from pyvqnet.tensor import QTensor
        qubits = 4
        qm = TNQMachine(qubits)

        vqc_allsinglesdoubles(q_machine=qm, weights=QTensor([0.55, 0.11, 0.53]), 
                              hf_state = QTensor([1,1,0,0]), singles=[[0, 2], [1, 3]], doubles=[[0, 1, 2, 3]], wires=[0,1,2,3])
        print(qm.get_states())

vqc_basisrotation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.vqc.tn.vqc_basisrotation(q_machine: pyvqnet.qnn.vqc.tn.TNQMachine, wires, unitary_matrix: QTensor, check=False)

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
        from pyvqnet.qnn.vqc.tn import vqc_basisrotation, TNQMachine
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

分布式接口
=================================================

分布式相关功能,当使用 ``torch`` 计算后端时候,封装使用了torch的 ``torch.distributed`` 的接口,
    


.. note::

    请参考 <https://pytorch.org/docs/stable/distributed.html7>`__ 中启动分布式的方法启动。
    当使用CPU上进行分布式,请使用 ``gloo`` 而不是 ``mpi`` 。
    当使用GPU上进行分布式,请使用 ``nccl``。

    :ref:`vqnet_dist` 下VQNet自己实现的分布式接口不适用 ``torch`` 计算后端。

CommController
-------------------------

.. py:class:: pyvqnet.distributed.ControllComm.CommController(backend,rank=None,world_size=None)

    CommController用于控制在cpu、gpu下数据通信的控制器, 通过设置参数 `backend` 来生成cpu(gloo)、gpu(nccl)的控制器。
    这个类会调用 backend,rank,world_size 初始化 ``torch.distributed.init_process_group(backend,rank,world_size)``

    :param backend: 用于生成cpu或者gpu的数据通信控制器,'gloo' 或 'nccl'。
    :param rank: 当前程序所在的进程号。
    :param world_size: 全局所有的进程数量。

    :return:
        CommController 实例。

    Examples::

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
 

    .. py:method:: getRank()
        
        用于获得当前进程的进程号。


        :return: 返回当前进程的进程号。

        Examples::

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
            


    .. py:method:: getSize()
    
        用于获得总共启动的进程数。


        :return: 返回总共进程的数量。

        Examples::

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
            


    .. py:method:: getLocalRank()
        
        在每个进程中通过 ``os.environ['LOCAL_RANK'] = rank`` 获取每个机器的局部进程号。
        需要事先对环境变量 `LOCAL_RANK` 进行设置。

        :return: 当前机器上的当前进程号。

        Examples::

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

 
    .. py:method:: split_group(rankL)
        
        根据入参设置的进程号列表用于划分多个通信组。

        :param rankL: 进程组列表。
        :return: 包含 ``torch.distributed.ProcessGroup`` 的列表

        Examples::

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
                
                local_rank = pp.split_group([[0,1],[2,3]])
                print(local_rank )


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
 


 
    .. py:method:: barrier()
        
        不同进程的同步。

        :return: 同步操作。

        Examples::

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

    .. py:method:: allreduce(tensor, c_op = "avg")
        
        支持对数据作allreduce通信。

        :param tensor: 输入数据.
        :param c_op: 计算方式.

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController,init_group
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
            

 
    .. py:method:: reduce(tensor, root = 0, c_op = "avg")
        
        支持对数据作reduce通信。

        :param tensor: 输入数据。
        :param root: 指定数据返回的节点。
        :param c_op: 计算方式。

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController,init_group
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
 
 
    .. py:method:: broadcast(tensor, root = 0)
        
        将指定进程root上的数据广播到所有进程上。

        :param tensor: 输入数据。
        :param root: 指定的节点。

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController,init_group
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
            

 
    .. py:method:: allgather(tensor)
        
        将所有进程上数据allgather到一起。本接口只支持nccl后端。

        :param tensor: 输入数据。

        Examples::

            from pyvqnet.distributed import get_local_rank,CommController,get_world_size
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


    .. py:method:: send(tensor, dest)
        
        p2p通信接口。

        :param tensor: 输入数据.
        :param dest: 目的进程.

        Examples::

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
 
 
    .. py:method:: recv(tensor, source)
        
        p2p通信接口。

        :param tensor: 输入数据.
        :param source: 接受进程.

        Examples::

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
            

    .. py:method:: allreduce_group(tensor, c_op = "avg", GroupComm = None)
        
        组内allreduce通信接口。

        :param tensor: 输入数据.
        :param c_op: 计算方法.
        :param GroupComm: 通信组, 仅mpi进行组内通信时需要.

        Examples::

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

            
                groups = Comm_OP.split_group([[0,2],[1,3]])
                num = tensor.to_tensor(np.random.rand(1, 5)+get_local_rank()*1000)
                print(f"rank {Comm_OP.getRank()}  {num}")

                Comm_OP.all_reduce_group(num, "sum",groups[0])

                print(f"rank {Comm_OP.getRank()}  {num}")
                num = tensor.to_tensor(np.random.rand(1, 5)-get_local_rank()*100)
                print(f"rank {Comm_OP.getRank()}  {num}")

                Comm_OP.all_reduce_group(num, "sum",groups[0])
                print(f"rank {Comm_OP.getRank()}  {num}")

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
 

    .. py:method:: reduce_group(tensor, root = 0, c_op = "avg", GroupComm = None)
        
        组内reduce通信接口。

        :param tensor: 输入数据.
        :param root: 指定进程号.
        :param c_op: 计算方法.
        :param GroupComm: 通信组, 仅mpi进行组内通信时需要.

        Examples::
            
            from pyvqnet.distributed import get_local_rank,CommController,init_group
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

                group = Comm_OP.split_group([1,3])

                num = tensor.to_tensor(np.random.rand(1, 5)+get_local_rank()*10)
                print(f"before rank {Comm_OP.getRank()}  {num}\n")
                
                Comm_OP.reduce_group(num, 1,"sum",group)
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
            

 
    .. py:method:: broadcast_group(tensor, root = 0, GroupComm = None)
        
        组内broadcast通信接口。

        :param tensor: 输入数据.
        :param root: 指定进程号.
        :param GroupComm: 通信组, 仅mpi进行组内通信时需要.

        Examples::
            
            from pyvqnet.distributed import get_local_rank,CommController,init_group
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

                group = Comm_OP.split_group([2,3])

                num = tensor.to_tensor(np.random.rand(1, 5))+ rank*1000
                print(f"before rank {Comm_OP.getRank()}  {num}")
                
                Comm_OP.broadcast_group(num, 2,group)
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
            

 
    .. py:method:: allgather_group(tensor, GroupComm = None)
        
        组内allgather通信接口。

        :param tensor: 输入数据.
        :param GroupComm: 通信组, 仅mpi进行组内通信时需要.

        Examples::
            
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
                Comm_OP = CommController("gloo", rank=rank, world_size=size)

                group = Comm_OP.split_group([0,2])
                print(f"get_world_size {get_world_size()}")

                num = tensor.QTensor(np.random.rand(5,4)+get_local_rank()*100)
                print(f"before rank {Comm_OP.getRank()}  {num}\n")

                num = Comm_OP.all_gather_group(num,group)
                print(f"after rank {Comm_OP.getRank()}  {num}\n")


                num = tensor.QTensor(np.random.rand(5)+get_local_rank()*100)
                print(f"before rank {Comm_OP.getRank()}  {num}\n")

                num = Comm_OP.all_gather_group(num,group)
                print(f"after rank {Comm_OP.getRank()}  {num}\n")

                num = tensor.QTensor(np.random.rand(3,5,4)+get_local_rank()*100)
                print(f"before rank {Comm_OP.getRank()}  {num}\n")

                num = Comm_OP.all_gather_group(num,group)
                print(f"after rank {Comm_OP.getRank()}  {num}\n")


            if __name__ == "__main__":
                world_size = 3
                processes = []
                mp.set_start_method("spawn")
                for rank in range(world_size):
                    p = mp.Process(target=init_process, args=(rank, world_size ))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            

