XTensor 经典神经网络模块
###########################

以下的经典神经网络模块均支持XTensor自动反向传播计算。当您运行前传函数以后，
需要使用在 ``with pyvqnet.xtensor.autograd.tape()`` 的范围内定义前向计算。这样就可以将需要自动微分的算子纳入计算图中。

.. warning::

    XTensor相关功能属于实验功能，当前只支持经典神经网络计算，与前述介绍的基于QTensor的接口不能混用。
    如需要训练量子机器学习模型，请使用QTensor下相关接口。

接着执行反向函数就获取 ``requires_grad == True`` 的 `XTensor` 的计算梯度。
一个卷积层的简单例子如下:

.. code-block::

    from pyvqnet.xtensor import arange
    from pyvqnet import kfloat32
    from pyvqnet.xtensor import Conv2D,autograd

    # an image feed into two dimension convolution layer
    b = 2        # batch size 
    ic = 2       # input channels
    oc = 2      # output channels
    hw = 4      # input width and heights

    # two dimension convolution layer
    test_conv = Conv2D(ic,oc,(2,2),(2,2),"same")

    # input of shape [b,ic,hw,hw]
    x0 = arange(1,b*ic*hw*hw+1,dtype=kfloat32).reshape([b,ic,hw,hw])
    x0.requires_grad = True
    with autograd.tape():
    #forward function
        x = test_conv(x0)

    #backward function with autograd
    x.backward()
    print(x0.grad)
    """
    [[[[-0.0194679  0.1530238 -0.0194679  0.1530238] 
    [ 0.2553246  0.1616782  0.2553246  0.1616782] 
    [-0.0194679  0.1530238 -0.0194679  0.1530238] 
    [ 0.2553246  0.1616782  0.2553246  0.1616782]]

    [[ 0.0285322  0.1099411  0.0285322  0.1099411] 
    [ 0.3087625 -0.0679072  0.3087625 -0.0679072] 
    [ 0.0285322  0.1099411  0.0285322  0.1099411]
    [ 0.3087625 -0.0679072  0.3087625 -0.0679072]]]


    [[[-0.0194679  0.1530238 -0.0194679  0.1530238]
    [ 0.2553246  0.1616782  0.2553246  0.1616782]
    [-0.0194679  0.1530238 -0.0194679  0.1530238]
    [ 0.2553246  0.1616782  0.2553246  0.1616782]]

    [[ 0.0285322  0.1099411  0.0285322  0.1099411]
    [ 0.3087625 -0.0679072  0.3087625 -0.0679072]
    [ 0.0285322  0.1099411  0.0285322  0.1099411]
    [ 0.3087625 -0.0679072  0.3087625 -0.0679072]]]]
    <XTensor 2x2x4x4 cpu(0) kfloat32>
    """

.. currentmodule:: pyvqnet.xtensor


Module类
******************************************

abstract calculation module


Module
===========================================================

.. py:class:: pyvqnet.xtensor.module.Module

    所有神经网络模块的基类,包括量子模块或经典模块。您的模型也应该是此类的子类,用于 autograd 计算。
    模块还可以包含其他Module类,允许将它们嵌套在树状结构。 您可以将子模块分配为常规属性::

        class Model(Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = pyvqnet.xtensor.Conv2D(1, 20, (5,5))
                self.conv2 = pyvqnet.xtensor.Conv2D(20, 20, (5,5))
            def forward(self, x):
                x = pyvqnet.xtensor.relu(self.conv1(x))
                return pyvqnet.xtensor.relu(self.conv2(x))

    以这种方式分配的子模块将被注册。

forward
===========================================================

.. py:function:: pyvqnet.xtensor.module.Module.forward(x, *args, **kwargs)

    Module类抽象前向计算函数

    :param x: 输入QTensor。
    :param \*args: 非关键字可变参数。
    :param \*\*kwargs: 关键字可变参数。

    :return: 模型输出。

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        import pyvqnet as vq
        from pyvqnet.xtensor import Conv2D
        b = 2
        ic = 3
        oc = 2
        test_conv = Conv2D(ic, oc, (3, 3), (2, 2), "same")
        x0 = XTensor(np.arange(1, b * ic * 5 * 5 + 1).reshape([b, ic, 5, 5]),
                    dtype=vq.kfloat32)
        x = test_conv.forward(x0)
        print(x)

        # [
        # [[[4.3995643, 3.9317808, -2.0707254],
        #  [20.1951981, 21.6946659, 14.2591858],
        #  [38.4702759, 31.9730244, 24.5977650]],
        # [[-17.0607567, -31.5377998, -7.5618000],
        #  [-22.5664024, -40.3876266, -15.1564388],
        #  [-3.1080279, -18.5986233, -8.0648050]]],
        # [[[6.6493244, -13.4840755, -20.2554188],
        #  [54.4235802, 34.4462433, 26.8171902],
        #  [90.2827682, 62.9092331, 51.6892929]],
        # [[-22.3385429, -45.2448578, 5.7101378],
        #  [-32.9464149, -60.9557228, -10.4994345],
        #  [5.9029331, -20.5480480, -0.9379558]]]
        # ]

        
state_dict 
===========================================================

.. py:function:: pyvqnet.xtensor.module.Module.state_dict(destination=None, prefix='')

    返回包含模块整个状态的字典:包括参数和缓存值。
    键是对应的参数和缓存值名称。

    :param destination: 返回保存模型内部模块,参数的字典。
    :param prefix: 使用的参数和缓存值的命名前缀。

    :return: 包含模块整个状态的字典。

    Example::

        from pyvqnet.xtensor import Conv2D
        test_conv = Conv2D(2,3,(3,3),(2,2),"same")
        print(test_conv.state_dict().keys())
        #odict_keys(['weights', 'bias'])


toGPU
===========================================================

.. py:function:: pyvqnet.xtensor.module.Module.toGPU(device: int = DEV_GPU_0)

    将模块和其子模块的参数和缓冲数据移动到指定的 GPU 设备中。

    device 指定存储其内部数据的设备。 当device >= DEV_GPU_0时，数据存储在GPU上。如果您的计算机有多个GPU，
    则可以指定不同的设备来存储数据。例如device = DEV_GPU_1 , DEV_GPU_2, DEV_GPU_3, ... 表示存储在不同序列号的GPU上。
    
    .. note::
        Module在不同GPU上无法进行计算。
        如果您尝试在 ID 超过验证 GPU 最大数量的 GPU 上创建 QTensor，将引发 Cuda 错误。

    :param device: 当前保存QTensor的设备，默认=DEV_GPU_0。device= pyvqnet.DEV_GPU_0，存储在第一个 GPU 中，devcie = DEV_GPU_1，存储在第二个 GPU 中，依此类推
    :return: Module 移动到 GPU 设备。

    Examples::

        from pyvqnet.xtensor import ConvT2D 
        test_conv = ConvT2D(3, 2, (4,4), (2, 2), "same")
        test_conv = test_conv.toGPU()
        print(test_conv.backend)
        #1000


toCPU
===========================================================

.. py:function:: pyvqnet.xtensor.module.Module.toCPU()

    将模块和其子模块的参数和缓冲数据移动到特定的 CPU 设备中。

    :return: Module 移动到 CPU 设备。

    Examples::

        from pyvqnet.xtensor import ConvT2D 
        test_conv = ConvT2D(3, 2, (4,4), (2, 2), "same")
        test_conv = test_conv.toCPU()
        print(test_conv.backend)
        #0


模型参数保存和载入
******************************************

以下接口可以进行模型参数保存到文件中，或从文件中读取参数文件。但请注意，文件中不保存模型结构，需要用户手动构建模型结构。

save_parameters
===========================================================

.. py:function:: pyvqnet.xtensor.storage.save_parameters(obj, f)

    保存模型参数的字典到一个文件。

    :param obj: 需要保存的字典。
    :param f: 保存参数的文件名。

    :return: 无。

    Example::

        from pyvqnet.xtensor import Module,Conv2D,save_parameters
        
        class Net(Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")

            def forward(self, x):
                return super().forward(x)

        model = Net() 
        save_parameters(model.state_dict(),"tmp.model")

load_parameters
===========================================================

.. py:function:: pyvqnet.xtensor.storage.load_parameters(f)

    从文件中载入参数到一个字典中。

    :param f: 保存参数的文件名。

    :return: 保存参数的字典。

    Example::

        from pyvqnet.xtensor import Module, Conv2D,load_parameters,save_parameters

        class Net(Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = Conv2D(input_channels=1,
                                    output_channels=6,
                                    kernel_size=(5, 5),
                                    stride=(1, 1),
                                    padding="valid")

            def forward(self, x):
                return super().forward(x)

        model = Net()
        model1 = Net()  # another Module object
        save_parameters(model.state_dict(), "tmp.model")
        model_para = load_parameters("tmp.model")
        model1.load_state_dict(model_para)

ModuleList
************************************************************************************

.. py:class:: pyvqnet.xtensor.module.ModuleList([pyvqnet.xtensor.module.Module])


    将子模块保存在列表中。 ModuleList 可以像普通的 Python 列表一样被索引， 它包含的Module的内部参数等可以被保存起来。

    :param modules: nn.Modules 列表

    :return: 一个模块列表

    Example::

        from pyvqnet.xtensor import Module, Linear, ModuleList

        class M(Module):
            def __init__(self):
                super(M, self).__init__()
                self.dense = ModuleList([Linear(4, 2), Linear(4, 2)])

            def forward(self, x, *args, **kwargs):
                y = self.dense[0](x) + self.dense[1](x)
                return y

        mm = M()

        print(mm.state_dict())
        """
        OrderedDict([('dense.0.weights', 
        [[ 0.8224208  0.3421015  0.2118234  0.1082053]     
        [-0.8264768  1.1017226 -0.3860411 -1.6656817]]    
        <Parameter 2x4 cpu(0) kfloat32>), ('dense.0.bias', 
        [0.4125615 0.4414732]
        <Parameter 2 cpu(0) kfloat32>), ('dense.1.weights',
        [[ 1.8939902  0.8871605 -0.3880418 -0.4815852]
        [-0.0956827  0.2667428  0.2900301  0.4039476]]
        <Parameter 2x4 cpu(0) kfloat32>), ('dense.1.bias',
        [-0.0544764  0.0289595]
        <Parameter 2 cpu(0) kfloat32>)])
        """


经典神经网络层
******************************************

以下实现了一些经典神经网络层：卷积，转置卷积，池化，归一化，循环神经网络等。


Conv1D
===========================================================

.. py:class:: pyvqnet.xtensor.Conv1D(input_channels:int,output_channels:int,kernel_size:int ,stride:int= 1,padding = "valid",use_bias:bool = True,kernel_initializer = None,bias_initializer =None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    在输入上进行一维卷积运算。 Conv1D模块的输入具有形状(batch_size、input_channels、in_height)。

    :param input_channels: `int` - 输入数据的通道数。
    :param output_channels: `int` - 输出数据的通道数。
    :param kernel_size: `int` - 卷积核的尺寸. 卷积核形状 = [output_channels,input_channels/group,kernel_size,1]。
    :param stride: `int` - 步长, 默认为1。
    :param padding: `str|int` - 填充选项, 它可以是一个字符串 {'valid', 'same'} 或一个整数，给出应用在输入上的填充量。 默认 "valid"。
    :param use_bias: `bool` - 是否使用偏置项, 默认使用。
    :param kernel_initializer: `callable` - 卷积核初始化方法。默认为空,使用kaiming_uniform。
    :param bias_initializer: `callable` - 偏置初始化方法。默认为空,使用kaiming_uniform。
    :param dilation_rate: `int` - 空洞大小,defaults: 1。
    :param group: `int` -  分组卷积的分组数. Default: 1。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 模块的名字,default:""。

    :return: 一维卷积实例。

    .. note::
        ``padding='valid'`` 不进行填充。

        ``padding='same'`` 补零填充输入,输出的out_height 为 = ceil(in_height / stride)。

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import Conv1D
        import pyvqnet
        b= 2
        ic =3
        oc = 2
        test_conv = Conv1D(ic,oc,3,2,"same")
        x0 = XTensor(np.arange(1,b*ic*5*5 +1).reshape([b,ic,25]),dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)
        """
        [[[ 17.637825   26.841843   28.811993   30.782139   32.752293
            34.72244    36.69259    38.66274    40.63289    42.603035
            44.57319    46.543335   36.481537 ]
        6274   106.63289                            707144  -4.07236
        108.60304   110.57319   112.54334   114.5789382  -3.58058381349   116.48363
        118.45379   120.423935   85.522644 ]     
        [ 34.14579    -0.6791078  -0.5807535  -0.46274   106.63289823973  -0.384041                           1349   116.48363
            -0.2856848  -0.1873342  -0.088978    0.0093744   0.107725                           823973  -0.384041
            0.2060831   0.3044413  -1.8352301]]]   093744   0.107725
        <XTensor 2x2x13 cpu(0) kfloat32>
        """

Conv2D
===========================================================

.. py:class:: pyvqnet.xtensor.Conv2D(input_channels:int,output_channels:int,kernel_size:tuple,stride:tuple=(1, 1),padding="valid",use_bias = True,kernel_initializer=None,bias_initializer=None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    在输入上进行二维卷积运算。 Conv2D模块的输入具有形状(batch_size, input_channels, height, width)。

    :param input_channels: `int` - 输入数据的通道数。
    :param output_channels: `int` - 输出数据的通道数。
    :param kernel_size: `tuple|list` - 卷积核的尺寸. 卷积核形状 = [output_channels,input_channels/group,kernel_size,kernel_size]。
    :param stride: `tuple|list` - 步长, 默认为 (1, 1)|[1,1]。
    :param padding: `str|tuple` - 填充选项, 它可以是一个字符串 {'valid', 'same'} 或一个整数元组，给出在两边应用的隐式填充量。 默认 "valid"。
    :param use_bias: `bool` - 是否使用偏置项, 默认使用。
    :param kernel_initializer: `callable` - 卷积核初始化方法。默认为空,使用kaiming_uniform。
    :param bias_initializer: `callable` - 偏置初始化方法。默认为空,使用kaiming_uniform。
    :param dilation_rate: `int` - 空洞大小,defaults: 1。
    :param group: `int` -  分组卷积的分组数. Default: 1。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 模块的名字,default:""。

    :return: 二维卷积实例。

    .. note::
        ``padding='valid'`` 不进行填充。

        ``padding='same'`` 补零填充输入,输出的height 为 = ceil(height / stride), 输出的width 为 = ceil(width / stride)。

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import Conv2D
        import pyvqnet
        b= 2
        ic =3
        oc = 2
        test_conv = Conv2D(ic,oc,(3,3),(2,2),"same")
        x0 = XTensor(np.arange(1,b*ic*5*5+1).reshape([b,ic,5,5]),dtype=pyvqnet.kfloat32)
        x = test_conv.forward(x0)
        print(x)
        """
        [[[[13.091256  16.252321   6.009256 ] 
        [25.834864  29.57479   10.406249 ] 
        [17.177385  27.90065   11.024535 ]]

        [[ 9.705042  11.656831   8.584356 ] 
        [23.186415  29.151287  17.489706 ] 
        [23.500261  23.620876  15.1604395]]]     


        [[[55.944958  66.7173    31.89055  ]       
        [86.231346  99.19392   44.14594  ]       
        [53.740646  83.22319   33.986828 ]]      

        [[44.75199   41.64939   35.985905 ]       
        [54.717422  73.95048   48.546726 ]       
        [33.874504  40.06319   31.02438  ]]]]    
        <XTensor 2x2x3x3 cpu(0) kfloat32>
        """

ConvT2D
===========================================================

.. py:class:: pyvqnet.xtensor.ConvT2D(input_channels,output_channels,kernel_size,stride=[1, 1],padding="valid",use_bias="True", kernel_initializer=None,bias_initializer=None, dilation_rate: int = 1, group: int = 1, dtype = None, name = "")

    在输入上进行二维转置卷积运算。 Conv2D模块的输入具有形状(batch_size, input_channels, height, width)。

    :param input_channels: `int` - 输入数据的通道数。
    :param output_channels: `int` - 输出数据的通道数。
    :param kernel_size: `tuple|list` - 卷积核的尺寸,卷积核形状 = [input_channels,output_channels/group,kernel_size,kernel_size]。 
    :param stride: `tuple|list` - 步长, 默认为 (1, 1)|[1,1]。
    :param padding: `str|tuple` - 填充选项, 它可以是一个字符串 {'valid', 'same'} 或一个整数元组，给出在两边应用的隐式填充量。 默认 "valid"。
    :param use_bias: `bool` - 是否使用偏置项, 默认使用。
    :param kernel_initializer: `callable` - 卷积核初始化方法。默认为空,使用kaiming_uniform。
    :param bias_initializer: `callable` - 偏置项初始化方法。默认为空,使用kaiming_uniform。
    :param dilation_rate: `int` - 空洞大小,defaults: 1。
    :param group: `int` -  分组卷积的分组数. Default: 1。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 模块的名字,default:""。

    :return: 二维转置卷积实例。
    
    .. note::
        ``padding='valid'`` 不进行填充。

        ``padding='same'`` 补零填充输入,输出的height 为 = ceil(height / stride)。

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import ConvT2D
        import pyvqnet
        test_conv = ConvT2D(3, 2, (3, 3), (1, 1), "valid")
        x = XTensor(np.arange(1, 1 * 3 * 5 * 5+1).reshape([1, 3, 5, 5]), dtype=pyvqnet.kfloat32)
        y = test_conv.forward(x)
        print(y)
        """
        
        [[[[  5.4529057  -3.32444     9.211117    9.587392    9.963668
            4.5622444  14.397371 ]
        [ 18.834743   21.769156   36.432068   37.726788   39.021515
            18.737175   16.340559 ]
        [ 29.79763    39.767223   61.934864   63.825333   65.715805
            34.09302    23.981941 ]
        [ 33.684406   45.685955   71.38721    73.27768    75.16815
            39.658592   27.515562 ]
        [ 37.571186   51.604687   80.839554   82.730034   84.6205
            45.224167   31.049183 ]
        [ 33.69648    61.94682    71.845085   73.359276   74.873474
            39.471508   10.021905 ]
        [ 12.103746   23.482103   31.11521    31.710955   32.306698
            19.998552    7.940914 ]]

        [[  4.769257    6.5511374   9.029368    9.207671    9.385972
            3.762906    2.1653163]
        [ -8.366173    0.0307169  -0.3826299  -0.5054388  -0.6282487
            8.602992   -0.3027873]
        [ -9.106487   -4.8349705   1.0091982   0.9871688   0.9651423
            12.1995535   6.483701 ]
        [-11.156897   -5.4630694   0.8990631   0.8770366   0.855011
            14.13983     7.001668 ]
        [-13.207303   -6.09117     0.7889295   0.7669029   0.7448754
            16.080103    7.5196342]
        [-25.585697  -18.799192  -12.708595  -12.908926  -13.109252
            15.557721    7.1425896]
        [ -4.400727   -4.76725     4.1210976   4.2218823   4.322665
            10.08579     9.516866 ]]]]
        <XTensor 1x2x7x7 cpu(0) kfloat32>
        """


AvgPool1D
===========================================================

.. py:class:: pyvqnet.xtensor.AvgPool1D(kernel, stride, padding="valid", name = "")

    对一维输入进行平均池化。输入具有形状(batch_size, input_channels, in_height)。

    :param kernel: 平均池化的窗口大小。
    :param strides: 窗口移动的步长。
    :param padding: 填充选项, "valid" or "same" 或者整数指定填充长度。 默认 "valid"。
    :param name: 模块的名字,default:""。

    :return: 一维平均池化层实例。

    .. note::
        ``padding='valid'`` 不进行填充。

        ``padding='same'`` 补零填充输入,输出的out_height 为 = ceil(in_height / stride)。

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import AvgPool1D
        test_mp = AvgPool1D(3,2,"same")
        x= XTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        # [
        # [[0.3333333, 1.6666666, 3.],
        #  [1.6666666, 2., 1.3333334],
        #  [2.6666667, 2.6666667, 2.3333333],
        #  [2.3333333, 4.3333335, 3.3333333],
        #  [0.3333333, 1.6666666, 4.]]
        # ]
        

MaxPool1D
===========================================================

.. py:class:: pyvqnet.xtensor.MaxPool1D(kernel, stride, padding="valid",name="")

    对一维输入进行最大池化。输入具有形状(batch_size, input_channels, in_height)。

    :param kernel: 最大池化的窗口大小。
    :param strides: 窗口移动的步长。
    :param padding: 填充选项, "valid" or "same" 或者整数指定填充长度。 默认 "valid"。
    :param name: 命名,默认为""。

    :return: 一维最大池化层实例。

    .. note::
        ``padding='valid'`` 不进行填充。

        ``padding='same'`` 补零填充输入,输出的out_height 为 = ceil(in_height / stride)。

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import MaxPool1D
        test_mp = MaxPool1D(3,2,"same")
        x= XTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        #[[[1. 4. 5.]
        #   [3. 3. 3.]
        #   [4. 4. 4.]
        #   [5. 6. 6.]
        #   [1. 5. 7.]]]

AvgPool2D
===========================================================

.. py:class:: pyvqnet.xtensor.AvgPool2D( kernel, stride, padding="valid",name="")

    对二维输入进行平均池化。输入具有形状(batch_size, input_channels, height, width)。

    :param kernel: 平均池化的窗口大小。
    :param strides: 窗口移动的步长。
    :param padding: 填充选项, "valid" or "same" 或包含2个整数的元组，整数为两个维度上的填充长度。 默认 "valid"。
    :param name: 命名,默认为""。

    :return: 二维平均池化层实例。

    .. note::
        ``padding='valid'`` 不进行填充。

        ``padding='same'`` 补零填充输入,输出的height 为 = ceil(height / stride), 输出的width 为 = ceil(width / stride)。

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import AvgPool2D
        test_mp = AvgPool2D((2,2),(2,2),"valid")
        x= XTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        #[[[[1.5  1.75]
        #    [3.75 3.  ]]]]
        

MaxPool2D
===========================================================

.. py:class:: pyvqnet.xtensor.MaxPool2D(kernel, stride, padding="valid",name="")

    对二维输入进行最大池化。输入具有形状(batch_size, input_channels, height, width)。

    :param kernel: 最大池化的窗口大小。
    :param strides: 窗口移动的步长。
    :param padding: 填充选项, "valid" or "same" 或包含2个整数的元组，整数为两个维度上的填充长度。 默认 "valid"。
    :param name: 命名,默认为""。

    :return: 二维最大池化层实例。

    .. note::
        ``padding='valid'`` 不进行填充。

        ``padding='same'`` 补零填充输入,输出的height 为 = ceil(height / stride), 输出的width 为 = ceil(width / stride)。


    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import MaxPool2D
        test_mp = MaxPool2D((2,2),(2,2),"valid")
        x= XTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,1,5,5]),requires_grad=True)

        y= test_mp.forward(x)
        print(y)
        # [[[[3. 4.]
        #    [5. 6.]]]]
        

Embedding
===========================================================

.. py:class:: pyvqnet.xtensor.Embedding(num_embeddings, embedding_dim, weight_initializer=xavier_normal, dtype=None, name: str = "")

    该模块通常用于存储词嵌入并使用索引检索它们。模块的输入是索引列表,输出是对应的词嵌入。 

    :param num_embeddings: `int` - 嵌入字典的大小。
    :param embedding_dim: `int` - 每个嵌入向量的大小
    :param weight_initializer: `callable` - 参数初始化方式,默认正态分布。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 嵌入层的命名,默认为""。

    :return: a Embedding 实例。

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import Embedding
        import pyvqnet
        vlayer = Embedding(30,3)
        x = XTensor(np.arange(1,25).reshape([2,3,2,2]),dtype= pyvqnet.kint64)
        y = vlayer(x)
        print(y)

        # [
        # [[[[-0.3168081, 0.0329394, -0.2934906],
        #  [0.1057295, -0.2844988, -0.1687456]],
        # [[-0.2382513, -0.3642318, -0.2257225],
        #  [0.1563180, 0.1567665, 0.3038477]]],
        # [[[-0.4131152, -0.0564500, -0.2804018],
        #  [-0.2955172, -0.0009581, -0.1641144]],
        # [[0.0692555, 0.1094901, 0.4099118],
        #  [0.4348361, 0.0304361, -0.0061203]]],
        # [[[-0.3310401, -0.1836129, 0.1098949],
        #  [-0.1840732, 0.0332474, -0.0261806]],
        # [[-0.1489778, 0.2519453, 0.3299376],
        #  [-0.1942692, -0.1540277, -0.2335350]]]],
        # [[[[-0.2620637, -0.3181309, -0.1857461],
        #  [-0.0878164, -0.4180320, -0.1831555]],
        # [[-0.0738970, -0.1888980, -0.3034399],
        #  [0.1955448, -0.0409723, 0.3023460]]],
        # [[[0.2430045, 0.0880465, 0.4309453],
        #  [-0.1796514, -0.1432367, -0.1253638]],
        # [[-0.5266719, 0.2386262, -0.0329155],
        #  [0.1033449, -0.3442690, -0.0471130]]],
        # [[[-0.5336705, -0.1939755, -0.3000667],
        #  [0.0059001, 0.5567381, 0.1926173]],
        # [[-0.2385869, -0.3910453, 0.2521235],
        #  [-0.0246447, -0.0241158, -0.1402829]]]]
        # ]
        


BatchNorm2d
===========================================================

.. py:class:: pyvqnet.xtensor.BatchNorm2d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5,beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")
    
    在 4D 输入(B、C、H、W)上应用批归一化。参照论文
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ 。
    
    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    其中 :math:`\gamma` 和 :math:`\beta` 为待训练参数。在训练期间,该层会继续运行估计其计算的均值和方差,然后在评估期间用于归一化。平均方差均值保持默认动量 0.1。

    .. note::

        当使用 `with autograd.tape()` 时候，BatchNorm2d进入train模式,使用local_mean,local_variance,gamma,beta进行批归一化。
        
        当不使用以上代码时候，BatchNorm2d使用eval模式，使用缓存的global_mean,global_variance,gamma,beta进行批归一化。


    :param channel_num: `int` - 输入通道数。
    :param momentum: `float` - 计算指数加权平均时的动量,默认为 0.1。
    :param beta_initializer: `callable` - beta的初始化方式,默认全零初始化。
    :param gamma_initializer: `callable` - gamma的的初始化方式,默认全一初始化。
    :param epsilon: `float` - 数值稳定参数, 默认 1e-5。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 批归一化层命名,默认为""。

    :return: 二维批归一化层实例。

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import BatchNorm2d,autograd

        b = 2
        ic = 2
        test_conv = BatchNorm2d(ic)

        x = XTensor(np.arange(1, 17).reshape([b, ic, 4, 1]))
        x.requires_grad = True
        with autograd.tape():
            y = test_conv.forward(x)
        print(y)

        # [
        # [[[-1.3242440],
        #  [-1.0834724],
        #  [-0.8427007],
        #  [-0.6019291]],
        # [[-1.3242440],
        #  [-1.0834724],
        #  [-0.8427007],
        #  [-0.6019291]]],
        # [[[0.6019291],
        #  [0.8427007],
        #  [1.0834724],
        #  [1.3242440]],
        # [[0.6019291],
        #  [0.8427007],
        #  [1.0834724],
        #  [1.3242440]]]
        # ]
        

BatchNorm1d
===========================================================

.. py:class:: pyvqnet.xtensor.BatchNorm1d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5, beta_initializer=zeros, gamma_initializer=ones, dtype=None, name="")

    在 2D 输入 (B,C) 上进行批归一化操作。 参照论文
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ 。
    
    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    其中 :math:`\gamma` 和 :math:`\beta` 为待训练参数。在训练期间,该层会继续运行估计其计算的均值和方差,然后在评估期间用于归一化。平均方差均值保持默认动量 0.1。


    .. note::

        当使用 `with autograd.tape()` 时候，BatchNorm2d进入train模式,使用local_mean,local_variance,gamma,beta进行批归一化。
        
        当不使用以上代码时候，BatchNorm2d使用eval模式，使用缓存的global_mean,global_variance,gamma,beta进行批归一化。

    :param channel_num: `int` - 输入通道数。
    :param momentum: `float` - 计算指数加权平均时的动量,默认为 0.1。
    :param beta_initializer: `callable` - beta的初始化方式,默认全零初始化。
    :param gamma_initializer: `callable` - gamma的的初始化方式,默认全一初始化。
    :param epsilon: `float` - 数值稳定性常数,默认为 1e-5。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    
    :param name: 批归一化层命名,默认为""。

    :return: 一维批归一化层实例。

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import BatchNorm1d,autograd

        test_conv = BatchNorm1d(4)

        x = XTensor(np.arange(1, 17).reshape([4, 4]))
        with autograd.tape():
            y = test_conv.forward(x)
        print(y)

        # [
        # [-1.3416405, -1.3416405, -1.3416405, -1.3416405],
        # [-0.4472135, -0.4472135, -0.4472135, -0.4472135],
        # [0.4472135, 0.4472135, 0.4472135, 0.4472135],
        # [1.3416405, 1.3416405, 1.3416405, 1.3416405]
        # ]

LayerNormNd
===========================================================

.. py:class:: pyvqnet.xtensor.LayerNormNd(normalized_shape: list, epsilon: float = 1e-5, affine: bool = True, dtype=None, name="")

    在任意输入的后D个维度上进行层归一化。具体方式如论文所述:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    对于像 (B,C,H,W,D) 这样的输入， ``norm_shape`` 可以是 [C,H,W,D],[H,W,D],[W,D] 或 [D] .

    :param norm_shape: `float` - 标准化形状。
    :param epsilon: `float` - 数值稳定性常数，默认为 1e-5。
    :param affine: `bool` - 是否使用应用仿射变换，默认为 True。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字， 默认为""。

    :return: 一个 LayerNormNd 类

    Example::

        import numpy as np
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import LayerNormNd
        ic = 4
        test_conv = LayerNormNd([2,2])
        x = XTensor(np.arange(1,17).reshape([2,2,2,2]))
        y = test_conv.forward(x)
        print(y)
        # [
        # [[[-1.3416355, -0.4472118],
        #  [0.4472118, 1.3416355]],
        # [[-1.3416355, -0.4472118],
        #  [0.4472118, 1.3416355]]],
        # [[[-1.3416355, -0.4472118],
        #  [0.4472118, 1.3416355]],
        # [[-1.3416355, -0.4472118],
        #  [0.4472118, 1.3416355]]]
        # ]

LayerNorm2d
===========================================================

.. py:class:: pyvqnet.xtensor.LayerNorm2d(norm_size:int, epsilon:float = 1e-5,  affine: bool = True, dtype=None, name="")

    在 4D 输入上进行层归一化。具体方式如论文所述:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    平均值和标准差是在除去第一个维度以外的剩余维度数据上计算的。对于像 (B,C,H,W) 这样的输入, ``norm_size`` 应该等于 C * H * W。

    :param norm_size: `float` - 归一化大小,应该等于 C * H * W。
    :param epsilon: `float` - 数值稳定性常数,默认为 1e-5。
    :param affine: `bool` - 是否使用应用仿射变换，默认为 True。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字， 默认为""。

    :return: 二维层归一化实例。

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import LayerNorm2d
        ic = 4
        test_conv = LayerNorm2d(8)
        x = XTensor(np.arange(1,17).reshape([2,2,4,1]))
        y = test_conv.forward(x)
        print(y)

        # [
        # [[[-1.5275238],
        #  [-1.0910884],
        #  [-0.6546531],
        #  [-0.2182177]],
        # [[0.2182177],
        #  [0.6546531],
        #  [1.0910884],
        #  [1.5275238]]],
        # [[[-1.5275238],
        #  [-1.0910884],
        #  [-0.6546531],
        #  [-0.2182177]],
        # [[0.2182177],
        #  [0.6546531],
        #  [1.0910884],
        #  [1.5275238]]]
        # ]
        

LayerNorm1d
===========================================================

.. py:class:: pyvqnet.xtensor.LayerNorm1d(norm_size:int, epsilon:float = 1e-5, affine: bool = True, dtype=None, name="")
    
    在 2D 输入上进行层归一化。具体方式如论文所述:
    `Layer Normalization <https://arxiv.org/abs/1607.06450>`__。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    均值和标准差是在最后一个维度大小上计算的,其中“norm_size” 是 ``norm_size`` 的值。

    :param norm_size: `float` - 归一化大小,应该等于最后一维大小。
    :param epsilon: `float` - 数值稳定性常数,默认为 1e-5。
    :param affine: `bool` - 是否使用应用仿射变换，默认为 True。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字， 默认为""。

    :return: 一维层归一化实例。

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import LayerNorm1d
        test_conv = LayerNorm1d(4)
        x = XTensor(np.arange(1,17).reshape([4,4]))
        y = test_conv.forward(x)
        print(y)

        # [
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355],
        # [-1.3416355, -0.4472118, 0.4472118, 1.3416355]
        # ]
        

Linear
===========================================================

.. py:class:: pyvqnet.xtensor.Linear(input_channels, output_channels, weight_initializer=None, bias_initializer=None,use_bias=True, dtype=None, name: str = "")

    线性模块(全连接层)。
    :math:`y = x*A + b`

    :param input_channels: `int` - 输入数据通道数。
    :param output_channels: `int` - 输出数据通道数。
    :param weight_initializer: `callable` - 权重初始化函数,默认为空,使用he_uniform。
    :param bias_initializer: `callable` - 偏置初始化参数,默认为空,使用he_uniform。
    :param use_bias: `bool` - 是否使用偏置项, 默认使用。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 线性层的命名,默认为""。

    :return: 线性层实例。

    Example::

        import numpy as np
        import pyvqnet
        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import Linear
        c1 =2
        c2 = 3
        cin = 7
        cout = 5
        n = Linear(cin,cout)
        input = XTensor(np.arange(1,c1*c2*cin+1).reshape((c1,c2,cin)),dtype=pyvqnet.kfloat32)
        y = n.forward(input)
        print(y)

        # [
        # [[4.3084583, -1.9228780, -0.3428757, 1.2840536, -0.5865945],
        #  [9.8339605, -5.5135884, -3.1228657, 4.3025794, -4.1492314],
        #  [15.3594627, -9.1042995, -5.9028554, 7.3211040, -7.7118683]],
        # [[20.8849659, -12.6950111, -8.6828451, 10.3396301, -11.2745066],
        #  [26.4104652, -16.2857227, -11.4628344, 13.3581581, -14.8371439],
        #  [31.9359703, -19.8764324, -14.2428246, 16.3766804, -18.3997803]]
        # ]
        


Dropout
===========================================================

.. py:class:: pyvqnet.xtensor.Dropout(dropout_rate = 0.5)

    Dropout 模块。dropout 模块将一些单元的输出随机设置为零,同时根据给定的 dropout_rate 概率升级其他单元。

    .. note::

        当使用 `with autograd.tape()` 时候，Dropout进入train模式,将一些单元的输出随机设置为零。
        
        当不使用以上代码时候，时候，Dropout进入train模式使用eval模式，原样输出。

    :param dropout_rate: `float` - 神经元被设置为零的概率。
    :param name: 这个模块的名字， 默认为""。

    :return: Dropout实例。

    Example::

        import numpy as np
        from pyvqnet.xtensor import Dropout
        from pyvqnet.xtensor import XTensor
        b = 2
        ic = 2
        x = XTensor(np.arange(-1 * ic * 2 * 2,
                            (b - 1) * ic * 2 * 2).reshape([b, ic, 2, 2]))
        
        droplayer = Dropout(0.5)
        
        y = droplayer(x)
        print(y)
        """
        [[[[-0. -0.]
        [-0. -0.]]

        [[-0. -0.]
        [-0. -0.]]]


        [[[ 0.  0.]
        [ 0.  6.]]

        [[ 8. 10.]
        [ 0. 14.]]]]
        <XTensor 2x2x2x2 cpu(0) kfloat32>
        """

Pixel_Shuffle 
===============
.. py:class:: pyvqnet.xtensor.Pixel_Shuffle(upscale_factors)

    重新排列形状为：(\*, C * r^2, H, W)  的张量
    到形状为 (\*, C, H * r, W * r) 的张量，其中 r 是尺度变换因子。

    :param upscale_factors: 增加尺度变换的因子

    :return:
            Pixel_Shuffle 模块

    Example::

        from pyvqnet.xtensor import Pixel_Shuffle
        from pyvqnet.xtensor import ones
        ps = Pixel_Shuffle(3)
        inx = ones([5,2,3,18,4,4])
        inx.requires_grad=  True
        y = ps(inx)
        print(y.shape)
        #[5, 2, 3, 2, 12, 12]

Pixel_Unshuffle 
===============
.. py:class:: pyvqnet.xtensor.Pixel_Unshuffle(downscale_factors)

    通过重新排列元素来反转 Pixel_Shuffle 操作. 将 (*, C, H * r, W * r) 形状的张量变化为 (*, C * r^2, H, W) ，其中 r 是缩小因子。

    :param downscale_factors: 增加尺度变换的因子

    :return:
            Pixel_Unshuffle 模块

    Example::

        from pyvqnet.xtensor import Pixel_Unshuffle
        from pyvqnet.xtensor import ones
        ps = Pixel_Unshuffle(3)
        inx = ones([5, 2, 3, 2, 12, 12])
        inx.requires_grad = True
        y = ps(inx)
        print(y.shape)
        #[5, 2, 3, 18, 4, 4]


GRU
===============

.. py:class:: pyvqnet.xtensor.GRU(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    门控循环单元 (GRU) 模块。支持多层堆叠，双向配置。单层单向GRU的计算公式如下:

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    :param input_size: 输入特征维度。
    :param hidden_size:  隐藏特征维度。
    :param num_layers: 堆叠GRU层数， 默认: 1。
    :param batch_first: 如果为 True， 则输入形状为 [batch_size,seq_len,feature_dim]，
     如果为 False， 则输入形状为 [seq_len,batch_size,feature_dim]，默认为 True。
    :param use_bias: 如果为 False，该模块不适用偏置项，默认: True。
    :param bidirectional: 如果为 True, 变为双向GRU， 默认: False。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字， 默认为""。

    :return: GRU 实例

    Example::

        from pyvqnet.xtensor import GRU
        import pyvqnet.xtensor as tensor

        rnn2 = GRU(4, 6, 2, batch_first=False, bidirectional=True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])

        output, hn = rnn2(input, h0)
        print(output)
        print(hn)
        # [[[-0.3525755 -0.2587337  0.149786   0.461374   0.1449795  0.4734624
        #    -0.0581852 -0.3912893 -0.2553828  0.5960887  0.6639917  0.2783016]
        #   [-0.3525755 -0.2587337  0.149786   0.461374   0.1449795  0.4734624
        #    -0.0581852 -0.3912893 -0.2553828  0.5960887  0.6639917  0.2783016]
        #   [-0.3525755 -0.2587337  0.149786   0.461374   0.1449796  0.4734623
        #    -0.0581852 -0.3912893 -0.2553828  0.5960887  0.6639917  0.2783016]]

        #  [[-0.5718066 -0.2159877 -0.2978141  0.1441837 -0.0761072  0.3096682
        #     0.0115526 -0.2295886 -0.1917011  0.5739203  0.5765471  0.3173765]
        #   [-0.5718066 -0.2159877 -0.2978141  0.1441837 -0.0761072  0.3096682
        #     0.0115526 -0.2295886 -0.1917011  0.5739203  0.5765471  0.3173765]
        #   [-0.5718066 -0.2159877 -0.2978141  0.1441837 -0.0761071  0.3096681
        #     0.0115526 -0.2295886 -0.1917011  0.5739203  0.5765471  0.3173765]]

        #  [[-0.5341613 -0.0807438 -0.2697436 -0.0872668 -0.2630565  0.2262028
        #     0.1108652 -0.0301746 -0.0453528  0.5952781  0.5141098  0.3779774]
        #   [-0.5341613 -0.0807438 -0.2697436 -0.0872668 -0.2630565  0.2262028
        #     0.1108652 -0.0301746 -0.0453528  0.5952781  0.5141098  0.3779774]
        #   [-0.5341613 -0.0807438 -0.2697436 -0.0872668 -0.2630565  0.2262028
        #     0.1108652 -0.0301746 -0.0453528  0.5952781  0.5141098  0.3779774]]

        #  [[-0.4394189  0.0851144 -0.0413915 -0.2225544 -0.4341614  0.1345865
        #     0.241122   0.2335991  0.1918445  0.6587436  0.4840603  0.4863765]
        #   [-0.4394189  0.0851144 -0.0413915 -0.2225544 -0.4341614  0.1345865
        #     0.241122   0.2335991  0.1918445  0.6587436  0.4840603  0.4863765]
        #   [-0.4394189  0.0851144 -0.0413915 -0.2225544 -0.4341614  0.1345865
        #     0.241122   0.2335991  0.1918445  0.6587436  0.4840603  0.4863765]]

        #  [[-0.335566   0.2576246  0.2983787 -0.3105901 -0.5880742 -0.0203008
        #     0.4682454  0.5807121  0.5288119  0.7821091  0.5577921  0.6762444]
        #   [-0.335566   0.2576246  0.2983787 -0.3105901 -0.5880742 -0.0203008
        #     0.4682454  0.5807121  0.5288119  0.7821091  0.5577921  0.6762444]
        #   [-0.335566   0.2576246  0.2983787 -0.3105901 -0.5880742 -0.0203008
        #     0.4682454  0.5807121  0.5288119  0.7821091  0.5577921  0.6762444]]]
        # <XTensor 5x3x12 cpu(0) kfloat32>

        # [[[ 0.0388437 -0.0772902 -0.3355273 -0.143957   0.0520131 -0.323649 ]
        #   [ 0.0388437 -0.0772902 -0.3355273 -0.143957   0.0520131 -0.323649 ]
        #   [ 0.0388437 -0.0772902 -0.3355273 -0.143957   0.0520131 -0.323649 ]]

        #  [[-0.4715263 -0.5740314 -0.6212391 -0.20578    0.8116962 -0.1011644]
        #   [-0.4715263 -0.5740314 -0.6212391 -0.20578    0.8116962 -0.1011644]
        #   [-0.4715263 -0.5740314 -0.6212391 -0.20578    0.8116962 -0.1011644]]

        #  [[-0.335566   0.2576246  0.2983787 -0.3105901 -0.5880742 -0.0203008]
        #   [-0.335566   0.2576246  0.2983787 -0.3105901 -0.5880742 -0.0203008]
        #   [-0.335566   0.2576246  0.2983787 -0.3105901 -0.5880742 -0.0203008]]

        #  [[-0.0581852 -0.3912893 -0.2553828  0.5960887  0.6639917  0.2783016]
        #   [-0.0581852 -0.3912893 -0.2553828  0.5960887  0.6639917  0.2783016]
        #   [-0.0581852 -0.3912893 -0.2553828  0.5960887  0.6639917  0.2783016]]]
        # <XTensor 4x3x6 cpu(0) kfloat32>

RNN 
===============

.. py:class:: pyvqnet.xtensor.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    循环神经网络(RNN)模块，使用 :math:`\tanh` 或 :math:`\text{relu}` 作为激活函数。支持双向，多层配置。
    单层单向RNN计算公式如下:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    如果 :attr:`nonlinearity` 是 ``'relu'``, 则 :math:`\text{relu}` 将替代 :math:`\tanh`。

    :param input_size: 输入特征维度。
    :param hidden_size:  隐藏特征维度。
    :param num_layers: 堆叠RNN层数， 默认: 1。
    :param nonlinearity: 非线性激活函数，默认为 ``'tanh'``。
    :param batch_first: 如果为 True， 则输入形状为 [batch_size,seq_len,feature_dim]，
     如果为 False， 则输入形状为 [seq_len,batch_size,feature_dim]，默认为 True。
    :param use_bias: 如果为 False， 该模块不适用偏置项，默认: True。
    :param bidirectional: 如果为 True，变为双向RNN，默认: False。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字， 默认为""。

    :return: RNN 实例

    Example::

        from pyvqnet.xtensor import RNN
        import pyvqnet.xtensor as tensor

        rnn2 = RNN(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        output, hn = rnn2(input, h0)
        print(output)
        print(hn)

        # [[[ 0.2618747  0.5377525  0.5129814 -0.0242058 -0.6311338  0.655068   
        #    -0.7402378 -0.1209883 -0.2462614  0.1552387  0.433101  -0.3321311] 
        #   [ 0.2618747  0.5377525  0.5129814 -0.0242058 -0.6311338  0.655068   
        #    -0.7402378 -0.1209883 -0.2462614  0.1552387  0.433101  -0.3321311] 
        #   [ 0.2618747  0.5377525  0.5129814 -0.0242058 -0.6311338  0.655068   
        #    -0.7402378 -0.1209883 -0.2462614  0.1552387  0.4331009 -0.3321312]]

        #  [[ 0.4478737  0.8829155 -0.0466572 -0.1412439 -0.0311011  0.3848146  
        #    -0.710574   0.0830743 -0.2802465 -0.0228663  0.1831353 -0.3086829] 
        #   [ 0.4478737  0.8829155 -0.0466572 -0.1412439 -0.0311011  0.3848146
        #    -0.710574   0.0830743 -0.2802465 -0.0228663  0.1831353 -0.3086829]
        #   [ 0.4478737  0.8829155 -0.0466572 -0.1412439 -0.0311011  0.3848146
        #    -0.710574   0.0830743 -0.2802465 -0.0228663  0.1831353 -0.3086829]]

        #  [[ 0.581092   0.8708823  0.2848003 -0.154836  -0.4118715  0.5057767
        #    -0.8474038  0.0595496 -0.5158566 -0.1731871  0.3361979 -0.2265194]
        #   [ 0.581092   0.8708823  0.2848003 -0.154836  -0.4118715  0.5057767
        #    -0.8474038  0.0595496 -0.5158566 -0.1731871  0.3361979 -0.2265194]
        #   [ 0.581092   0.8708823  0.2848004 -0.154836  -0.4118715  0.5057766
        #    -0.8474038  0.0595496 -0.5158566 -0.1731871  0.3361979 -0.2265194]]

        #  [[ 0.5710331  0.8946801 -0.0285869 -0.032192  -0.2297462  0.4527371
        #    -0.7243505  0.2147861 -0.3519893 -0.1745383 -0.5063711 -0.7420927]
        #   [ 0.5710331  0.8946801 -0.0285869 -0.032192  -0.2297462  0.4527371
        #    -0.7243505  0.2147861 -0.3519893 -0.1745383 -0.5063711 -0.7420927]
        #   [ 0.5710331  0.8946801 -0.0285869 -0.032192  -0.2297462  0.4527371
        #    -0.7243505  0.2147861 -0.3519893 -0.1745383 -0.5063711 -0.7420927]]

        #  [[ 0.6136605  0.8351185 -0.0997602 -0.4209205  0.0391619  0.6528097
        #    -0.4387358 -0.9216538 -0.3471112 -0.9059284  0.9011658 -0.6876704]
        #   [ 0.6136605  0.8351185 -0.0997602 -0.4209205  0.0391619  0.6528097
        #    -0.4387358 -0.9216538 -0.3471112 -0.9059284  0.9011658 -0.6876704]
        #   [ 0.6136605  0.8351185 -0.0997602 -0.4209205  0.0391619  0.6528097
        #    -0.4387358 -0.9216538 -0.3471112 -0.9059284  0.9011658 -0.6876704]]]
        # <XTensor 5x3x12 cpu(0) kfloat32>

        # [[[-0.1595494 -0.4154228  0.6991262  0.634046   0.6960769 -0.4966849]
        #   [-0.1595494 -0.4154228  0.6991262  0.634046   0.6960769 -0.4966849]
        #   [-0.1595494 -0.4154228  0.6991262  0.634046   0.6960769 -0.4966849]]

        #  [[ 0.2330922 -0.7264591 -0.7571693  0.3780781  0.4115383  0.7572027]
        #   [ 0.2330922 -0.7264591 -0.7571693  0.3780781  0.4115383  0.7572027]
        #   [ 0.2330922 -0.7264591 -0.7571693  0.3780781  0.4115383  0.7572027]]

        #  [[ 0.6136605  0.8351185 -0.0997602 -0.4209205  0.0391619  0.6528097]
        #   [ 0.6136605  0.8351185 -0.0997602 -0.4209205  0.0391619  0.6528097]
        #   [ 0.6136605  0.8351185 -0.0997602 -0.4209205  0.0391619  0.6528097]]

        #  [[-0.7402378 -0.1209883 -0.2462614  0.1552387  0.433101  -0.3321311]
        #   [-0.7402378 -0.1209883 -0.2462614  0.1552387  0.433101  -0.3321311]
        #   [-0.7402378 -0.1209883 -0.2462614  0.1552387  0.4331009 -0.3321312]]]
        # <XTensor 4x3x6 cpu(0) kfloat32>

LSTM
===========================================================

.. py:class:: pyvqnet.xtensor.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    长短期记忆(LSTM)模块。支持双向LSTM， 堆叠多层LSTM等配置。单层单向LSTM计算公式如下:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    :param input_size: 输入特征维度。
    :param hidden_size:  隐藏特征维度。
    :param num_layers: 堆叠LSTM层数，默认: 1。
    :param batch_first: 如果为 True，则输入形状为 [batch_size,seq_len,feature_dim]，
     如果为 False, 则输入形状为 [seq_len,batch_size,feature_dim]，默认为 True。
    :param use_bias: 如果为 False，该模块不适用偏置项， 默认: True。
    :param bidirectional: 如果为 True，变为双向LSTM， 默认: False。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字， 默认为""。

    :return: LSTM 实例

    Example::

        from pyvqnet.xtensor import LSTM
        import pyvqnet.xtensor as tensor

        rnn2 = LSTM(4, 6, 2, batch_first=False, bidirectional = True)

        input = tensor.ones([5, 3, 4])
        h0 = tensor.ones([4, 3, 6])
        c0 = tensor.ones([4, 3, 6])
        output, (hn, cn) = rnn2(input, (h0, c0))

        print(output)
        print(hn)
        print(cn)

        """
        [[[ 0.0952653  0.2589843  0.2486762  0.1287942  0.1021227  0.1911968
        -0.0414362 -0.3521166  0.2783869  0.2294379 -0.0897322 -0.0056688]
        [ 0.0952653  0.2589843  0.2486762  0.1287942  0.1021227  0.1911968
        -0.0414362 -0.3521166  0.2783869  0.2294379 -0.0897322 -0.0056688]
        [ 0.0952653  0.2589843  0.2486762  0.1287942  0.1021227  0.1911968
        -0.0414362 -0.3521166  0.2783869  0.2294379 -0.0897322 -0.0056688]]

        [[ 0.085497   0.1925259  0.3265719 -0.0792181  0.0310715  0.1398373
        -0.0023391 -0.3675225  0.3450475  0.2388555 -0.0579962 -0.0536785]
        [ 0.085497   0.1925259  0.3265719 -0.0792181  0.0310715  0.1398373
        -0.0023391 -0.3675225  0.3450475  0.2388555 -0.0579962 -0.0536785]
        [ 0.085497   0.1925259  0.3265719 -0.0792181  0.0310715  0.1398373
        -0.0023391 -0.3675225  0.3450475  0.2388555 -0.0579962 -0.0536785]]

        [[ 0.0788613  0.1431215  0.3656158 -0.1339753  0.0101364  0.0694154
            0.0299196 -0.329112   0.4189231  0.2440841  0.0078633 -0.091286 ]
        [ 0.0788613  0.1431215  0.3656158 -0.1339753  0.0101364  0.0694154
            0.0299196 -0.329112   0.4189231  0.2440841  0.0078633 -0.091286 ]
        [ 0.0788613  0.1431215  0.3656158 -0.1339753  0.0101364  0.0694154
            0.0299196 -0.329112   0.4189231  0.2440841  0.0078633 -0.091286 ]]

        [[ 0.0730032  0.1006287  0.3992919 -0.1526794 -0.0077587  0.0049352
            0.0684287 -0.2034638  0.4807869  0.2429611  0.1062173 -0.100478 ]
        [ 0.0730032  0.1006287  0.3992919 -0.1526794 -0.0077587  0.0049352
            0.0684287 -0.2034638  0.4807869  0.2429611  0.1062173 -0.100478 ]
        [ 0.0730032  0.1006287  0.3992919 -0.1526794 -0.0077587  0.0049352
            0.0684287 -0.2034638  0.4807869  0.2429611  0.1062173 -0.100478 ]]

        [[ 0.0421946  0.0219641  0.4280343 -0.1918173 -0.0422104 -0.0659961
            0.1292751  0.0901535  0.5008832  0.1749451  0.2660664  0.0133356]
        [ 0.0421946  0.0219641  0.4280343 -0.1918173 -0.0422104 -0.0659961
            0.1292751  0.0901535  0.5008832  0.1749451  0.2660664  0.0133356]
        [ 0.0421946  0.0219641  0.4280343 -0.1918173 -0.0422104 -0.0659961
            0.1292751  0.0901535  0.5008832  0.1749451  0.2660664  0.0133356]]]
        <XTensor 5x3x12 cpu(0) kfloat32>

        [[[-0.1111576 -0.0410911 -0.1360135 -0.2142551  0.3879747 -0.4244705]
        [-0.1111576 -0.0410911 -0.1360135 -0.2142551  0.3879747 -0.4244705]
        [-0.1111576 -0.0410911 -0.1360135 -0.2142551  0.3879747 -0.4244705]]

        [[ 0.2689943  0.3063364 -0.2738754  0.2168426 -0.4891826  0.2454725]
        [ 0.2689943  0.3063364 -0.2738754  0.2168426 -0.4891826  0.2454725]
        [ 0.2689943  0.3063364 -0.2738754  0.2168426 -0.4891826  0.2454725]]

        [[ 0.0421946  0.0219641  0.4280343 -0.1918173 -0.0422104 -0.0659961]
        [ 0.0421946  0.0219641  0.4280343 -0.1918173 -0.0422104 -0.0659961]
        [ 0.0421946  0.0219641  0.4280343 -0.1918173 -0.0422104 -0.0659961]]

        [[-0.0414362 -0.3521166  0.2783869  0.2294379 -0.0897322 -0.0056688]
        [-0.0414362 -0.3521166  0.2783869  0.2294379 -0.0897322 -0.0056688]
        [-0.0414362 -0.3521166  0.2783869  0.2294379 -0.0897322 -0.0056688]]]
        <XTensor 4x3x6 cpu(0) kfloat32>

        [[[-0.1973301 -0.2033432 -0.4812729 -0.3186268  0.7220416 -0.6785325]
        [-0.1973301 -0.2033432 -0.4812729 -0.3186268  0.7220416 -0.6785325]
        [-0.1973301 -0.2033432 -0.4812729 -0.3186268  0.7220416 -0.6785325]]

        [[ 1.6790413  0.4572753 -0.4843928  0.3902704 -0.9931879  0.4607614]
        [ 1.6790413  0.4572753 -0.4843928  0.3902704 -0.9931879  0.4607614]
        [ 1.6790413  0.4572753 -0.4843928  0.3902704 -0.9931879  0.4607614]]

        [[ 0.0753609  0.0477858  0.7334676 -0.3433677 -0.1412639 -0.1365461]
        [ 0.0753609  0.0477858  0.7334676 -0.3433677 -0.1412639 -0.1365461]
        [ 0.0753609  0.0477858  0.7334676 -0.3433677 -0.1412639 -0.1365461]]

        [[-0.104733  -0.6896871  0.4874932  0.4806345 -0.167628  -0.0125997]
        [-0.104733  -0.6896871  0.4874932  0.4806345 -0.167628  -0.0125997]
        [-0.104733  -0.6896871  0.4874932  0.4806345 -0.167628  -0.0125997]]]
        <XTensor 4x3x6 cpu(0) kfloat32>
        """


Dynamic_GRU
===========================================================

.. py:class:: pyvqnet.xtensor.Dynamic_GRU(input_size,hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")

    将多层门控循环单元 (GRU) RNN 应用于动态长度输入序列。

    第一个输入应该是定义了可变长度的批处理序列输入
    通过 ``xtensor.PackedSequence`` 类。
    ``xtensor.PackedSequence`` 类可以构造为
    连续调用下一个函数: ``pad_sequence`` 、 ``pack_pad_sequence``。

    Dynamic_GRU 的第一个输出也是一个 ``xtensor.PackedSequence`` 类，
    可以使用 ``xtensor.pad_pack_sequence`` 将其解压缩为普通 XTensor。

    对于输入序列中的每个元素，每一层计算以下公式：

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}


    :param input_size: 输入特征维度。
    :param hidden_size: 隐藏的特征维度。
    :param num_layers: 循环层数。 默认值：1
    :param batch_first: 如果为 True，输入形状提供为 [批大小,序列长度,特征维度]。如果为 False，输入形状提供为 [序列长度,批大小,特征维度]，默认为 True。
    :param use_bias: 如果为False，则该层不使用偏置权重b_ih和b_hh。 默认值：True。
    :param bidirectional: 如果为真，则成为双向 GRU。 默认值：False。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字， 默认为""。

    :return: 一个 Dynamic_GRU 类

    Example::

        from pyvqnet.xtensor import Dynamic_GRU
        import pyvqnet.xtensor as tensor
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
        print(seq_unpacked)
        print(lens_unpacked)

        # [[[ 0.872566   0.8611314 -0.5047759  0.9130142]
        #   [ 0.5690175  0.9443005 -0.3432685  0.9585502]
        #   [ 0.8512535  0.8650243 -0.487494   0.9192616]]

        #  [[ 0.7886224  0.7501922 -0.4578349  0.919861 ]
        #   [ 0.         0.         0.         0.       ]
        #   [ 0.712998   0.749097  -0.2539869  0.9405512]]

        #  [[ 0.7242796  0.6568378 -0.4209562  0.9258339]
        #   [ 0.         0.         0.         0.       ]
        #   [ 0.         0.         0.         0.       ]]

        #  [[ 0.6601164  0.5651093 -0.2040557  0.9421862]
        #   [ 0.         0.         0.         0.       ]
        #   [ 0.         0.         0.         0.       ]]]
        # <XTensor 4x3x4 cpu(0) kfloat32>
        # [4 1 2]

Dynamic_RNN 
===============

.. py:class:: pyvqnet.xtensor.Dynamic_RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    将循环神经网络 RNN 应用于动态长度输入序列。

    第一个输入应该是定义了可变长度的批处理序列输入
    通过 ``tensor.PackedSequence`` 类。
    ``tensor.PackedSequence`` 类可以构造为
    连续调用下一个函数: ``pad_sequence`` 、 ``pack_pad_sequence``。

    Dynamic_RNN 的第一个输出也是一个 ``tensor.PackedSequence`` 类，
    可以使用 ``tensor.pad_pack_sequence`` 将其解压缩为普通 XTensor。

    循环神经网络(RNN)模块，使用 :math:`\tanh` 或 :math:`\text{ReLU}` 作为激活函数。支持双向，多层配置。
    单层单向RNN计算公式如下:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    如果 :attr:`nonlinearity` 是 ``'relu'``, 则 :math:`\text{ReLU}` 将替代 :math:`\tanh`。

    :param input_size: 输入特征维度。
    :param hidden_size:  隐藏特征维度。
    :param num_layers: 堆叠RNN层数， 默认: 1。
    :param nonlinearity: 非线性激活函数，默认为 ``'tanh'``。
    :param batch_first: 如果为 True， 则输入形状为 [批大小,序列长度,特征维度]，
     如果为 False， 则输入形状为 [序列长度,批大小,特征维度]，默认为 True。
    :param use_bias: 如果为 False， 该模块不适用偏置项，默认: True。
    :param bidirectional: 如果为 True，变为双向RNN，默认: False。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字， 默认为""。

    :return: Dynamic_RNN 实例

    Example::

        from pyvqnet.xtensor import Dynamic_RNN
        import pyvqnet.xtensor as tensor
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
        print(seq_unpacked)
        print(lens_unpacked)

        """
        [[[ 2.283666   2.1524734  0.         0.8799834]
        [ 2.9422705  2.6565394  0.         0.8055274]
        [ 2.9554236  2.1205678  0.         1.1741859]]

        [[ 6.396565   3.8327866  0.         2.6239884]
        [ 0.         0.         0.         0.       ]
        [ 7.37332    4.7455616  0.         2.6786256]]

        [[12.521921   5.239943   0.         4.62357  ]
        [ 0.         0.         0.         0.       ]
        [ 0.         0.         0.         0.       ]]

        [[19.627499   8.675274   0.         6.6746845]
        [ 0.         0.         0.         0.       ]
        [ 0.         0.         0.         0.       ]]]
        <XTensor 4x3x4 cpu(0) kfloat32>
        [4 1 2]
        """

Dynamic_LSTM
===========================================================

.. py:class:: pyvqnet.xtensor.Dynamic_LSTM(input_size, hidden_size, num_layers=1, batch_first=True, use_bias=True, bidirectional=False, dtype=None, name: str = "")


    将长短期记忆(LSTM) RNN 应用于动态长度输入序列。

    第一个输入应该是定义了可变长度的批处理序列输入
    通过 ``tensor.PackedSequence`` 类。
    ``tensor.PackedSequence`` 类可以构造为
    连续调用下一个函数: ``pad_sequence`` 、 ``pack_pad_sequence``。

    Dynamic_LSTM 的第一个输出也是一个 ``tensor.PackedSequence`` 类，
    可以使用 ``tensor.pad_pack_sequence`` 将其解压缩为普通 XTensor。

    循环神经网络(RNN)模块，使用 :math:`\tanh` 或 :math:`\text{ReLU}` 作为激活函数。支持双向，多层配置。
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

    :param input_size: 输入特征维度。
    :param hidden_size:  隐藏特征维度。
    :param num_layers: 堆叠LSTM层数，默认: 1。
    :param batch_first: 如果为 True，则输入形状为 [批大小,序列长度,特征维度]，
     如果为 False, 则输入形状为 [序列长度,批大小,特征维度]，默认为 True。
    :param use_bias: 如果为 False，该模块不适用偏置项， 默认: True。
    :param bidirectional: 如果为 True，变为双向LSTM， 默认: False。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: 这个模块的名字， 默认为""。

    :return: Dynamic_LSTM 实例

    Example::

        from pyvqnet.xtensor import Dynamic_LSTM
        import pyvqnet.xtensor as tensor

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

        print(seq_unpacked)
        print(lens_unpacked)
        """
        [[[ 0.1970974  0.2246606  0.2627596 -0.080385 ] 
        [ 0.2071671  0.2119733  0.2301395 -0.2693036] 
        [ 0.1106544  0.3478935  0.4335948  0.378578 ]]

        [[ 0.1176731 -0.0304668  0.2993484  0.0920533] 
        [ 0.1386266 -0.0483974  0.2384422 -0.1798031] 
        [ 0.         0.         0.         0.       ]]

        [[ 0.0798466 -0.1468595  0.4139522  0.3376699] 
        [ 0.1303781 -0.1537685  0.2934605  0.0475375] 
        [ 0.         0.         0.         0.       ]]

        [[ 0.         0.         0.         0.       ]
        [ 0.0958745 -0.2243107  0.4114271  0.3248508]
        [ 0.         0.         0.         0.       ]]]
        <XTensor 4x3x4 cpu(0) kfloat32>
        [3 4 1]
        """


损失函数层
******************************************

以下为神经网络常用的损失层。

    .. note::

            请注意，跟pytorch等框架不同的是，以下loss函数的前向函数中，第一个参数为标签，第二个参数为预测值。

MeanSquaredError
===========================================================

.. py:class:: pyvqnet.xtensor.MeanSquaredError(name="")

    计算输入 :math:`x` 和目标值 :math:`y` 之间的均方根误差。

    若平方根误差可由如下函数描述:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    :math:`x` 和 :math:`y` 是任意形状的 XTensor , 总 :math:`n` 个元素的均方根误差由下式计算。

    .. math::
        \ell(x, y) =
            \operatorname{mean}(L)

    :param name: 这个模块的名字， 默认为""。
    :return: 一个均方根误差实例。

    均方根误差前向计算函数的所需参数:

        x: :math:`(N, *)` 预测值,其中 :math:`*` 表示任意维度。

        y: :math:`(N, *)`, 目标值, 和输入一样维度的 XTensor 。


    .. note::

            请注意，跟pytorch等框架不同的是，以下MeanSquaredError函数的前向函数中，第一个参数为目标值，第二个参数为预测值。

    Example::

        from pyvqnet.xtensor import XTensor, kfloat64
        from pyvqnet.xtensor import MeanSquaredError
        y = XTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
                    dtype=kfloat64)
        x = XTensor([[0.1, 0.05, 0.7, 0, 0.05, 0.1, 0, 0, 0, 0]],
                    
                    dtype=kfloat64)

        loss_result = MeanSquaredError()
        result = loss_result(y, x)
        print(result)
        # [0.0115000]
        


BinaryCrossEntropy
===========================================================

.. py:class:: pyvqnet.xtensor.BinaryCrossEntropy(name="")

    测量目标和输入之间的平均二元交叉熵损失。

    未做平均运算的二元交叉熵如下式:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    若 :math:`N` 为批的大小,则平均二元交叉熵.

    .. math::
        \ell(x, y) = \operatorname{mean}(L)

    :param name: 这个模块的名字， 默认为""。
    :return: 一个平均二元交叉熵实例。

    平均二元交叉熵误差前向计算函数的所需参数:

        x: :math:`(N, *)` 预测值,其中 :math:`*` 表示任意维度。

        y: :math:`(N, *)`, 目标值,和输入一样维度的 XTensor 。

    .. note::

            请注意，跟pytorch等框架不同的是，BinaryCrossEntropy函数的前向函数中，第一个参数为目标值，第二个参数为预测值。

    Example::

        from pyvqnet.xtensor import XTensor
        from pyvqnet.xtensor import BinaryCrossEntropy
        x = XTensor([[0.3, 0.7, 0.2], [0.2, 0.3, 0.1]] )
        y = XTensor([[0.0, 1.0, 0], [0.0, 0, 1]] )

        loss_result = BinaryCrossEntropy()
        result = loss_result(y, x)
        print(result)
        # [0.6364825]
        

CategoricalCrossEntropy
===========================================================

.. py:class:: pyvqnet.xtensor.CategoricalCrossEntropy(name="")

    该损失函数将 LogSoftmax 和 NLLLoss 同时计算的平均分类交叉熵。

    损失函数计算方式如下,其中 class 为目标值的对应分类标签:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: 这个模块的名字， 默认为""。
    :return: 平均分类交叉熵实例。

    误差前向计算函数的所需参数:

        x: :math:`(N, *)` 预测值,其中 :math:`*` 表示任意维度。

        y: :math:`(N, *)`, 目标值,和输入一样维度的 XTensor 。

    .. note::

            请注意，跟pytorch等框架不同的是，CategoricalCrossEntropy函数的前向函数中，第一个参数为目标值，第二个参数为预测值。

    Example::

        from pyvqnet.xtensor import XTensor,kfloat32,kint64
        from pyvqnet.xtensor import CategoricalCrossEntropy
        x = XTensor([[1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5]], dtype=kfloat32)
        y = XTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
        dtype=kfloat32)
        loss_result = CategoricalCrossEntropy()
        result = loss_result(y, x)
        print(result)
        # [3.7852428]

SoftmaxCrossEntropy
===========================================================

.. py:class:: pyvqnet.xtensor.SoftmaxCrossEntropy(name="")

    该损失函数将 LogSoftmax 和 NLLLoss 同时计算的平均分类交叉熵,并具有更高的数值稳定性。

    损失函数计算方式如下,其中 class 为目标值的对应分类标签:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: 这个模块的名字， 默认为""。
    :return: 一个Softmax交叉熵损失函数实例

    误差前向计算函数的所需参数:

        x: :math:`(N, *)` 预测值,其中 :math:`*` 表示任意维度。

        y: :math:`(N, *)`, 目标值,和输入一样维度的 XTensor 。

    .. note::

            请注意，跟pytorch等框架不同的是，SoftmaxCrossEntropy函数的前向函数中，第一个参数为目标值，第二个参数为预测值。

    Example::

        from pyvqnet.xtensor import XTensor, kfloat32, kint64
        from pyvqnet.xtensor import SoftmaxCrossEntropy
        x = XTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
                    dtype=kfloat32)
        y = XTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]],
                    dtype=kfloat32)
        loss_result = SoftmaxCrossEntropy()
        result = loss_result(y, x)
        print(result)

        # [3.7852478]

NLL_Loss
===========================================================

.. py:class:: pyvqnet.xtensor.NLL_Loss(name="")

    平均负对数似然损失。 对C个类别的分类问题很有用。

    `x` 是模型给出的概率形式的似然量。其尺寸可以是 :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` 。 `y` 是损失函数期望的真值，包含 :math:`[0, C-1]` 的类别索引。

    .. math::

        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = -  
            \sum_{n=1}^N \frac{1}{N}x_{n,y_n} \quad

    :param name: 这个模块的名字， 默认为""。
    :return: 一个NLL_Loss损失函数实例

    误差前向计算函数的所需参数:

        x: :math:`(N, *)`,损失函数的输出预测值，可以为多维变量。

        y: :math:`(N, *)`,损失函数目标值。

    .. note::

            请注意，跟pytorch等框架不同的是，NLL_Loss函数的前向函数中，第一个参数为目标值，第二个参数为预测值。

    Example::

        from pyvqnet.xtensor import XTensor, kint64
        from pyvqnet.xtensor import NLL_Loss

        x = XTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ]).reshape([1, 3, 1, 5])

        x.requires_grad = True
        y = XTensor([[[2, 1, 0, 0, 2]]])

        loss_result = NLL_Loss()
        result = loss_result(y, x)
        print(result)
        #[-0.6187226]

CrossEntropyLoss
===========================================================

.. py:class:: pyvqnet.xtensor.CrossEntropyLoss(name="")

    该函数计算LogSoftmax以及NLL_Loss在一起的损失。

    `x` 是包含未做归一化的输出.它的尺寸可以为 :math:`(C)` , :math:`(N, C)` 二维或 :math:`(N, C, d_1, d_2, ..., d_K)` 多维。

    损失函数的公式如下,其中 class 为目标值的对应分类标签:

    .. math::
        \text{loss}(x, y) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :param name: 这个模块的名字， 默认为""。
    :return: 一个CrossEntropyLoss损失函数实例

    误差前向计算函数的所需参数:

        x: :math:`(N, *)`,损失函数的输出，可以为多维变量。

        y: :math:`(N, *)`,损失函数期望的真值。

    .. note::

            请注意，跟pytorch等框架不同的是，CrossEntropyLoss函数的前向函数中，第一个参数为目标值，第二个参数为预测值。

    Example::

        from pyvqnet.xtensor import XTensor, kfloat32
        from pyvqnet.xtensor import CrossEntropyLoss
        x = XTensor([
            0.9476322568516703, 0.226547421131723, 0.5944201443911326,
            0.42830868492969476, 0.76414068655387, 0.00286059168094277,
            0.3574236812873617, 0.9096948856639084, 0.4560809854582528,
            0.9818027091583286, 0.8673569904602182, 0.9860275114020933,
            0.9232667066664217, 0.303693313961628, 0.8461034903175555
        ]).reshape([1, 3, 1, 5])
        
        x.requires_grad = True
        y = XTensor([[[2, 1, 0, 0, 2]]], dtype=kfloat32)

        loss_result = CrossEntropyLoss()
        result = loss_result(y, x)
        print(result)

        #[1.1508200]


激活函数
******************************************



sigmoid
===========================================================
.. py:function:: pyvqnet.xtensor.sigmoid(x)

    Sigmoid激活函数层。

    .. math::
        \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    :param x: 输入。

    :return: Sigmoid激活函数结果。

    Examples::

        from pyvqnet.xtensor import sigmoid
        from pyvqnet.xtensor import XTensor

        y = sigmoid(XTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.7310586, 0.8807970, 0.9525741, 0.9820138]


softplus
===========================================================
.. py:class:: pyvqnet.xtensor.softplus(x)

    Softplus激活函数层。

    .. math::
        \text{Softplus}(x) = \log(1 + \exp(x))

    :param x: 输入。

    :return: softplus激活函数层结果。

    Examples::

        from pyvqnet.xtensor import softplus
        from pyvqnet.xtensor import XTensor
        y = softplus(XTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [1.3132616, 2.1269281, 3.0485873, 4.0181499]
        

softsign
===========================================================
.. py:class:: pyvqnet.xtensor.softsign(x)

    Softsign 激活函数层。

    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

    :param x: 输入。

    :return: Softsign 激活函数结果。

    Examples::

        from pyvqnet.xtensor import softsign
        from pyvqnet.xtensor import XTensor
        y = softsign(XTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.5000000, 0.6666667, 0.7500000, 0.8000000]
        


softmax
===========================================================
.. py:class:: pyvqnet.xtensor.softmax(x,axis:int = -1)

    Softmax 激活函数层。

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    :param x: 输入。
    :param axis: 计算的维度(最后一个轴为-1),默认值 = -1。


    :return: Softmax 激活函数层结果。

    Examples::

        from pyvqnet.xtensor import softmax
        from pyvqnet.xtensor import XTensor

        y = softmax(XTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.0320586, 0.0871443, 0.2368828, 0.6439142]
        

hard_sigmoid
===========================================================
.. py:class:: pyvqnet.xtensor.hard_sigmoid(x)

    HardSigmoid 激活函数层。

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{ if } x \le -3, \\
            1 & \text{ if } x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}

    :param x: 输入。

    :return: HardSigmoid 激活函数层结果。

    Examples::

        from pyvqnet.xtensor import hard_sigmoid
        from pyvqnet.xtensor import XTensor

        y = hard_sigmoid(XTensor([1.0, 2.0, 3.0, 4.0]))
        print(y)

        # [0.6666667, 0.8333334, 1., 1.]
        

relu
===========================================================
.. py:class:: pyvqnet.xtensor.relu(x)

    ReLu 整流线性单元激活函数层。

    .. math::
        \text{ReLu}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        0, & \text{ if } x \leq 0
        \end{cases}


    :param x: 输入。

    :return: ReLu 激活函数层实例。

    Examples::

        from pyvqnet.xtensor import relu
        from pyvqnet.xtensor import XTensor

        y = relu(XTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [0., 2., 0., 4.]
        


leaky_relu
===========================================================
.. py:class:: pyvqnet.xtensor.leaky_relu(x, alpha:float=0.01)

    LeakyReLu 带泄露的修正线性单元激活函数层。

    .. math::
        \text{LeakyRelu}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \alpha * x, & \text{ otherwise }
        \end{cases}

    :param x: 输入。
    :param alpha: LeakyRelu 系数,默认:0.01。


    :return: LeakyReLu 激活函数层结果。

    Examples::

        from pyvqnet.xtensor import leaky_relu
        from pyvqnet.xtensor import XTensor
        y = leaky_relu(XTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.0100000, 2., -0.0300000, 4.]
        


elu
===========================================================
.. py:class:: pyvqnet.xtensor.elu(x, alpha:float=1)

    ELU 指数线性单位激活函数层。

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    :param x: 输入。
    :param alpha: ELU 系数,默认:1。

    :return: ELU 激活函数层结果。

    Examples::

        from pyvqnet.xtensor import elu
        from pyvqnet.xtensor import XTensor

        y = elu(XTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.6321205, 2., -0.9502130, 4.]
        
         
tanh
===========================================================
.. py:class:: pyvqnet.xtensor.tanh(x)

    Tanh双曲正切激活函数.

    .. math::
        \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    :param x: 输入。

    :return: Tanh 激活函数层结果。

    Examples::

        from pyvqnet.xtensor import tanh
        from pyvqnet.xtensor import XTensor
        y = tanh(XTensor([-1, 2.0, -3, 4.0]))
        print(y)

        # [-0.7615942, 0.9640276, -0.9950548, 0.9993293]
        

优化器模块
******************************************


Optimizer
===========================================================
.. py:class:: pyvqnet.xtensor.optimizer.Optimizer( params, lr=0.01)

    所有优化器的基类。

    :param params: 需要优化的模型参数列表。
    :param lr: 学习率,默认值:0.01。

step
===========================================================
.. py:method:: pyvqnet.xtensor.optimizer.Optimizer.step()

    使用对应优化器的更新方法进行参数更新。

Adadelta
===========================================================
.. py:class:: pyvqnet.xtensor.optimizer.Adadelta( params, lr=0.01, beta=0.99, epsilon=1e-8)

    ADADELTA: An Adaptive Learning Rate Method。
    
    参考:https://arxiv.org/abs/1212.5701。

    .. math::

        E(g_t^2) &= \beta * E(g_{t-1}^2) + (1-\beta) * g^2\\
        Square\_avg &= \sqrt{ ( E(dx_{t-1}^2) + \epsilon ) / ( E(g_t^2) + \epsilon ) }\\
        E(dx_t^2) &= \beta * E(dx_{t-1}^2) + (1-\beta) * (-g*square\_avg)^2 \\
        param\_new &= param - lr * Square\_avg

    :param params: 需要优化的模型参数列表。
    :param lr: 学习率(默认值:0.01)。
    :param beta: 用于计算平方梯度的运行平均值(默认值:0.99)。
    :param epsilon: 添加到分母以提高数值稳定性的常数(默认值:1e-8)。

    :return: 一个 Adadelta 优化器。

    Example::

        import numpy as np
        from pyvqnet.xtensor import Adadelta,Linear,Module,autograd
        from pyvqnet.xtensor import XTensor

        class model(Module):
            def __init__(self, name=""):
                super().__init__(name)
                self.a1 = Linear(4,3)
            def forward(self, x, *args, **kwargs):
                return self.a1(x)

        MM = model()
        opti = Adadelta(MM.parameters(),lr=100)

        for i in range(1, 3):
            w = np.arange(24).reshape(1, 2, 3, 4).astype(np.float64)
            param = XTensor(w)
            param.requires_grad = True
            with autograd.tape():
                y = MM(param)
            y.backward()
            
            opti.step()
            print(MM.a1.weights)
        """
        [[ 0.7224208  0.2421015  0.1118234  0.0082053]
        [-0.9264768  1.0017226 -0.4860411 -1.7656817]
        [ 0.282856   1.7939901  0.7871605 -0.4880418]]
        <Parameter 3x4 cpu(0) kfloat32>

        [[ 0.6221698  0.1418506  0.0115724 -0.0920456]
        [-1.0267278  0.9014716 -0.586292  -1.8659327]
        [ 0.1826051  1.6937392  0.6869095 -0.5882927]]
        <Parameter 3x4 cpu(0) kfloat32>
        """

Adagrad
===========================================================
.. py:class:: pyvqnet.xtensor.optimizer.Adagrad( params, lr=0.01, epsilon=1e-8)

    Adagrad自适应梯度优化器。
    
    参考:https://databricks.com/glossary/adagrad。

    .. math::
        \begin{align}
        moment\_new &= moment + g * g\\param\_new 
        &= param - \frac{lr * g}{\sqrt{moment\_new} + \epsilon}
        \end{align}

    :param params: 需要优化的模型参数列表。
    :param lr: 学习率(默认值:0.01)。
    :param epsilon: 添加到分母以提高数值稳定性的常数(默认值:1e-8)。
    :return: 一个 Adagrad 优化器。

    Example::

        import numpy as np
        from pyvqnet.xtensor import Adagrad,Linear,Module,autograd
        from pyvqnet.xtensor import XTensor

        class model(Module):
            def __init__(self, name=""):
                super().__init__(name)
                self.a1 = Linear(4,3)
            def forward(self, x, *args, **kwargs):
                return self.a1(x)

        MM = model()
        opti = Adagrad(MM.parameters(),lr=100)

        for i in range(1, 3):
            w = np.arange(24).reshape(1, 2, 3, 4)
            param = XTensor(w)
            param.requires_grad = True
            with autograd.tape():
                y = MM(param)
            y.backward()
            opti.step()
            print(MM.a1.weights)
        """
        [[ -99.17758  -99.6579   -99.78818  -99.89179]
        [-100.82648  -98.89828 -100.38604 -101.66568]
        [ -99.61714  -98.10601  -99.11284 -100.38804]]
        <Parameter 3x4 cpu(0) kfloat32>

        [[-169.88826 -170.36858 -170.49886 -170.60248]
        [-171.53716 -169.60895 -171.09671 -172.37637]
        [-170.32782 -168.81668 -169.82352 -171.09872]]
        <Parameter 3x4 cpu(0) kfloat32>
        """


Adam
===========================================================
.. py:class:: pyvqnet.xtensor.optimizer.Adam( params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8,amsgrad: bool = False)

    Adam优化器,它可以使用一阶矩估计动态调整每个参数的学习率和梯度的二阶矩估计。
    
    参考:https://arxiv.org/abs/1412.6980。

    .. math::
        t = t + 1 
    .. math::
        moment\_1\_new=\beta1∗moment\_1+(1−\beta1)g
    .. math::
        moment\_2\_new=\beta2∗moment\_2+(1−\beta2)g*g
    .. math::
        lr = lr*\frac{\sqrt{1-\beta2^t}}{1-\beta1^t}

    如果参数 amsgrad 为 True

    .. math::
        moment\_2\_max = max(moment\_2\_max,moment\_2)
    .. math::
        param\_new=param-lr*\frac{moment\_1}{\sqrt{moment\_2\_max}+\epsilon} 

    否则

    .. math::
        param\_new=param-lr*\frac{moment\_1}{\sqrt{moment\_2}+\epsilon} 

    :param params: 需要优化的模型参数列表。
    :param lr: 学习率(默认值:0.01)。
    :param beta1: 用于计算梯度及其平方的运行平均值的系数(默认值:0.9)。
    :param beta2: 用于计算梯度及其平方的运行平均值的系数(默认值:0.999)。
    :param epsilon: 添加到分母以提高数值稳定性的常数(默认值:1e-8)。
    :param amsgrad: 是否使用该算法的 AMSGrad 变体(默认值:False)。
    :return: 一个 Adam 优化器。

    Example::

        import numpy as np
        from pyvqnet.xtensor import Adam,Linear,Module,autograd
        from pyvqnet.xtensor import XTensor

        class model(Module):
            def __init__(self, name=""):
                super().__init__(name)
                self.a1 = Linear(4,3)
            def forward(self, x, *args, **kwargs):
                return self.a1(x)

        MM = model()
        opti = Adam(MM.parameters(),lr=100)

        for i in range(1, 3):
            w = np.arange(24).reshape(1, 2, 3, 4)
            param = XTensor(w)
            param.requires_grad = True
            with autograd.tape():
                y = MM(param)
            y.backward()
            
            opti.step()
            print(MM.a1.weights)
        """
        [[ -99.17759   -99.6579    -99.788185  -99.8918  ]
        [-100.826485  -98.89828  -100.38605  -101.66569 ]
        [ -99.61715   -98.10601   -99.11285  -100.38805 ]]
        <Parameter 3x4 cpu(0) kfloat32>

        [[-199.17725 -199.65756 -199.78784 -199.89145]
        [-200.82614 -198.89795 -200.38571 -201.66534]
        [-199.61682 -198.10568 -199.1125  -200.3877 ]]
        """

Adamax
===========================================================
.. py:class:: pyvqnet.xtensor.optimizer.Adamax(params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

    实现 Adamax 优化器(基于无穷范数的 Adam 变体)。
    
    参考:https://arxiv.org/abs/1412.6980。

    .. math::
        \\t = t + 1
    .. math::
        moment\_new=\beta1∗moment+(1−\beta1)g
    .. math::
        norm\_new = \max{(\beta1∗norm+\epsilon, \left|g\right|)}
    .. math::
        lr = \frac{lr}{1-\beta1^t}
    .. math::
        param\_new = param − lr*\frac{moment\_new}{norm\_new}\\

    :param params: 需要优化的模型参数列表。
    :param lr: 学习率(默认值:0.01)。
    :param beta1: 用于计算梯度及其平方的运行平均值的系数(默认值:0.9)。
    :param beta2: 用于计算梯度及其平方的运行平均值的系数(默认值:0.999)。
    :param epsilon: 添加到分母以提高数值稳定性的常数(默认值:1e-8)。

    :return: 一个 Adamax 优化器。

    Example::

        import numpy as np
        from pyvqnet.xtensor import Adamax,Linear,Module,autograd
        from pyvqnet.xtensor import XTensor

        class model(Module):
            def __init__(self, name=""):
                super().__init__(name)
                self.a1 = Linear(4,3)
            def forward(self, x, *args, **kwargs):
                return self.a1(x)

        MM = model()
        opti = Adamax(MM.parameters(),lr=100)

        for i in range(1, 3):
            w = np.arange(24).reshape(1, 2, 3, 4)
            param = XTensor(w)
            param.requires_grad = True
            with autograd.tape():
                y = MM(param)
            y.backward()
            
            opti.step()
            print(MM.a1.weights)
        """
        [[ -99.17758   -99.6579    -99.788185  -99.89179 ]
        [-100.82648   -98.89828  -100.38605  -101.66568 ]
        [ -99.61714   -98.10601   -99.11285  -100.38804 ]]
        <Parameter 3x4 cpu(0) kfloat32>

        [[-199.17758 -199.6579  -199.7882  -199.89178]
        [-200.82648 -198.89827 -200.38605 -201.66568]
        [-199.61714 -198.106   -199.11285 -200.38803]]
        <Parameter 3x4 cpu(0) kfloat32>
        """
        
RMSProp
===========================================================
.. py:class:: pyvqnet.xtensor.optimizer.RMSProp( params, lr=0.01, beta=0.99, epsilon=1e-8)
    
    RMSprop 均方根传播算法优化器。
    
    参考:https://arxiv.org/pdf/1308.0850v5.pdf。

    .. math::
        s_{t+1} = s_{t} + (1 - \beta)*(g)^2

    .. math::
        param_new = param -  \frac{g}{\sqrt{s_{t+1}} + epsilon}


    :param params: 需要优化的模型参数列表。
    :param lr: 学习率(默认值:0.01)。
    :param beta: 用于计算梯度及其平方的运行平均值的系数(默认值:0.99)。
    :param epsilon: 添加到分母以提高数值稳定性的常数(默认值:1e-8)。

    :return: 一个 RMSProp 优化器。

    Example::

        import numpy as np
        from pyvqnet.xtensor import RMSProp,Linear,Module,autograd
        from pyvqnet.xtensor import XTensor

        class model(Module):
            def __init__(self, name=""):
                super().__init__(name)
                self.a1 = Linear(4,3)
            def forward(self, x, *args, **kwargs):
                return self.a1(x)

        MM = model()
        opti = RMSProp(MM.parameters(),lr=100)

        for i in range(1, 3):
            w = np.arange(24).reshape(1, 2, 3, 4)
            param = XTensor(w)
            param.requires_grad = True
            with autograd.tape():
                y = MM(param)
            y.backward()
            
            opti.step()
            print(MM.a1.weights)
        """
        [[ -999.17804  -999.6584   -999.78864  -999.8923 ]
        [-1000.82697  -998.89874 -1000.38654 -1001.6662 ]
        [ -999.6176   -998.1065   -999.11334 -1000.38855]]
        <Parameter 3x4 cpu(0) kfloat32>

        [[-1708.0596 -1708.54   -1708.6702 -1708.7738]
        [-1709.7085 -1707.7803 -1709.2681 -1710.5477]
        [-1708.4991 -1706.988  -1707.9949 -1709.27  ]]
        <Parameter 3x4 cpu(0) kfloat32>
        """

SGD
===========================================================
.. py:class:: pyvqnet.xtensor.optimizer.SGD(params, lr=0.01,momentum=0, nesterov=False)

    随机梯度下降优化器。
    
    参考:https://en.wikipedia.org/wiki/Stochastic_gradient_descent。

    .. math::

        \\param\_new=param-lr*g\\

    :param params: 需要优化的模型参数列表。
    :param lr: 学习率(默认值:0.01)。
    :param momentum: 动量因子(默认值:0)。
    :param nesterov: 启用 Nesterov 动量 (默认: False)。

    :return: 一个 SGD 优化器。
    
    Example::

        import numpy as np
        from pyvqnet.xtensor import SGD,Linear,Module,autograd
        from pyvqnet.xtensor import XTensor

        class model(Module):
            def __init__(self, name=""):
                super().__init__(name)
                self.a1 = Linear(4,3)
            def forward(self, x, *args, **kwargs):
                return self.a1(x)

        MM = model()
        opti = SGD(MM.parameters(),lr=100,momentum=0.2)

        for i in range(1, 3):
            w = np.arange(24).reshape(1, 2, 3, 4)
            param = XTensor(w)
            param.requires_grad = True
            with autograd.tape():
                y = MM(param)
            y.backward()
            
            opti.step()
            print(MM.a1.weights)
        """
        [[-5999.1777 -6599.6577 -7199.788  -7799.8916]
        [-6000.8267 -6598.8984 -7200.386  -7801.6655]
        [-5999.617  -6598.106  -7199.113  -7800.388 ]]
        <Parameter 3x4 cpu(0) kfloat32>

        [[-13199.178 -14519.658 -15839.788 -17159.89 ]
        [-13200.826 -14518.898 -15840.387 -17161.666]
        [-13199.617 -14518.105 -15839.113 -17160.389]]
        <Parameter 3x4 cpu(0) kfloat32>
        """


本源量子云接口
******************************************

自2.12.0版本起，当用户同时安装了3.8.2.3以上pyQpanda，可使用QuantumBatchAsyncQcloudLayer 调用真实芯片或本地CPU虚拟机进行模拟。




QuantumBatchAsyncQcloudLayer
===========================================
.. py:class:: pyvqnet.xtensor.qcloud.QuantumBatchAsyncQcloudLayer(origin_qprog_func,qcloud_token,para_num,num_qubits,num_cubits,pauli_str_dict=None,shots=1000,initializer=None,dtype=None,name="",diff_method="parameter_shift",submit_kwargs={},query_kwargs={})
    
    用于本源量子计算机的量子计算的变分线路训练模块，使用pyqpanda QCLOUD从版本3.8.2.2开始。它提交参数化量子电路到真实芯片并获取测量结果。
    
    :param origin_qprog_func: 由QPanda构建的可调用量子电路函数。
    :param qcloud_token: str - 量子机器的类型或者执行的云令牌。
    :param para_num: int - 参数的数量；参数是一维的。
    :param num_qubits: int - 量子电路中的量子比特数。
    :param num_cubits: int - 量子电路中用于测量的经典比特数。
    :param pauli_str_dict: dict|list - 表示量子电路中Pauli算符的字典或字典列表。默认为None。
    :param shots: int - 测量次数。默认为1000。
    :param initializer: 参数值的初始化器。默认为None。
    :param dtype: 参数的数据类型。默认为None，即使用默认数据类型。
    :param name: 模块的名称。默认为空字符串。
    :param diff_method: 用于梯度计算的微分方法。默认为"parameter_shift"。
    :param submit_kwargs: 提交量子电路的额外关键字参数，默认为{"chip_id":pyqpanda.real_chip_type.origin_72,"is_amend":True,"is_mapping":True,"is_optimization": True,"default_task_group_size":200,"test_qcloud_fake":True}。
    :param query_kwargs: 查询量子结果的额外关键字参数，默认为{"timeout":2,"print_query_info":True,"sub_circuits_split_size":1}。
    
    :return: 一个能够计算量子电路的模块。

    .. note::

        submit_kwargs 的 `test_qcloud_fake` 默认为 True, 调用本地模拟。如果设为 False,则提交真机计算。
    Example::

        from pyqpanda import *

        import pyqpanda as pq
        from pyvqnet.xtensor.qcloud import QuantumBatchAsyncQcloudLayer
        from pyvqnet.xtensor.autograd import tape
        from pyvqnet.xtensor import arange,XTensor,ones,ones_like


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
                cir.insert(pq.RZ(m_qlist[2],param[2]))
                cir.insert(pq.RZ(m_qlist[2],param[3]))
                cir.insert(pq.RZ(m_qlist[1],param[4]))

                cir.insert(pq.H(m_qlist[2]))
                m_prog.insert(cir)


                return m_prog

        l = QuantumBatchAsyncQcloudLayer(qfun,
                    "302e020100301006072a8648ce3d020106052b8104001c041730150201010410def6ef7286d4a2fd143ea10e2de4638f/12570",
                    5,
                    6,
                    6,
                    pauli_str_dict=[{'Z0 X1':1,'Y2':1},{'Y2':1},{'Z0 X1':1,'Y2':1,'X2':1}],#{'Z0 X1':1,'Y2':1},#,
                    shots = 1000,
                    initializer=None,
                    dtype=None,
                    name="",
                    diff_method="parameter_shift",
                    submit_kwargs={"test_qcloud_fake":True},
                    query_kwargs={})


        x = XTensor([[0.56,1.2],[0.56,1.2],[0.56,1.2]],requires_grad= True)

        with tape():
            y = l(x)

        print(y)
        y.backward(ones_like(y))

        print(x.grad)
        print(l.m_para.grad)

        # [[-0.2554    -0.2038    -1.1429999]
        #  [-0.2936    -0.2082    -1.127    ]
        #  [-0.3144    -0.1812    -1.1208   ]]
        # <XTensor 3x3 cpu(0) kfloat32>

        # [[ 0.0241    -0.6001   ]
        #  [-0.0017    -0.5624   ]
        #  [ 0.0029999 -0.6071001]]
        # <XTensor 3x2 cpu(0) kfloat32>

        # [-1.5474    -1.0477002 -4.5562    -4.6365    -1.7573001]
        # <XTensor 5 cpu(0) kfloat32>
