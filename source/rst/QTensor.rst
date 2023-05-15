QTensor 模块
==============

VQNet量子机器学习所使用的数据结构QTensor的python接口介绍。QTensor支持常用的多维张量的操作，例如创建函数，数学函数，逻辑函数，矩阵变换等。



QTensor's 函数与属性
----------------------------------


__init__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: QTensor.__init__(data, requires_grad=False, nodes=None, device=0, dtype = None, name = "")

    具有动态计算图构造和自动微分的张量。

    :param data: 输入数据，可以是 _core.Tensor 或numpy 数组。
    :param requires_grad: 是否应该跟踪张量的梯度，默认为 False。
    :param nodes: 计算图中的后继者列表，默认为无。
    :param device: 储存在哪个设备上，默认0，在CPU上。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param name: QTensor的名字,default:""。
    :return: 输出 QTensor。

    .. note::
            QTensor 内部数据类型dtype支持kbool,kuint8,kint8,kint16,kint32,kint64,kfloat32,kfloat64,kcomplex64,kcomplex128.

            分别代表C++的 bool,uint8_t,int8_t,int16_t,int32_t,int64_t,float,double,complex<float>,complex<double>.

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.dtype import *
        import numpy as np

        t1 = QTensor(np.ones([2,3]))
        t2 =  QTensor([2,3,4j,5])
        t3 =  QTensor([[[2,3,4,5],[2,3,4,5]]],dtype=kbool)
        print(t1)
        print(t2)
        print(t3)
        # [[1. 1. 1.]
        #  [1. 1. 1.]]
        # [2.+0.j 3.+0.j 0.+4.j 5.+0.j]
        # [[[ True  True  True  True]
        #   [ True  True  True  True]]]


ndim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:attribute:: QTensor.ndim

    返回张量的维度的个数。
        
    :return: 张量的维度的个数。

    Example::

        from pyvqnet.tensor import QTensor

        a = QTensor([2, 3, 4, 5], requires_grad=True)
        print(a.ndim)

        # 1
    
shape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:attribute:: QTensor.shape

    返回张量的维度
    
    :return: 一个列表存有张量的维度

    Example::

        from pyvqnet.tensor import QTensor

        a = QTensor([2, 3, 4, 5], requires_grad=True)
        print(a.shape)

        # [4]

size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:attribute:: QTensor.size

    返回张量的元素个数。
    
    :return: 张量的元素个数。

    Example::

        from pyvqnet.tensor import QTensor

        a = QTensor([2, 3, 4, 5], requires_grad=True)
        print(a.size)

        # 4

numel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.numel

    返回张量的元素个数。
    
    :return: 张量的元素个数。

    Example::

        from pyvqnet.tensor import QTensor

        a = QTensor([2, 3, 4, 5], requires_grad=True)
        print(a.numel())

        # 4


dtype
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:attribute:: QTensor.dtype

    返回张量的数据类型。

    QTensor 内部数据类型dtype支持kbool=0,kuint8=1,kint8=2,kint16=3,kint32=4,kint64=5,kfloat32=6,kfloat64=7,kcomplex64=8,kcomplex128=9.

    :return: 张量的数据类型。

    Example::

        from pyvqnet.tensor import QTensor

        a = QTensor([2, 3, 4, 5])
        print(a.dtype)
        # 4


zero_grad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.zero_grad()

    将张量的梯度设置为零。将在优化过程中被优化器使用。

    :return: 无。

    Example::

        from pyvqnet.tensor import QTensor
        t3 = QTensor([2, 3, 4, 5], requires_grad=True)
        t3.zero_grad()
        print(t3.grad)
        # [0.0000000, 0.0000000, 0.0000000, 0.0000000]
        

backward
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.backward(grad=None)

    利用反向传播算法，计算当前张量所在的计算图中的所有需计算梯度的张量的梯度。

    :return: 无

    Example::

        from pyvqnet.tensor import QTensor

        target = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0.2]], requires_grad=True)
        y = 2*target + 3
        y.backward()
        print(target.grad)
        #[[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]]

to_numpy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.to_numpy()

    将张量的数据拷贝到一个numpy.ndarray里面。

    :return: 一个新的 numpy.ndarray 包含 QTensor 数据

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t3 = QTensor([2, 3, 4, 5], requires_grad=True)
        t4 = t3.to_numpy()
        print(t4)

        # [2. 3. 4. 5.]

item
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.item()

    从只包含单个元素的 QTensor 返回唯一的元素。

    :return: 元素值

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.ones([1])
        print(t.item())

        # 1.0

argmax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.argmax(*kargs)

    返回输入 QTensor 中所有元素的最大值的索引，或返回 QTensor 按某一维度的最大值的索引。

    :param dim: 计算argmax的轴，只接受单个维度。 如果 dim == None，则返回输入张量中所有元素的最大值的索引。有效的 dim 范围是 [-R, R)，其中 R 是输入的 ndim。 当 dim < 0 时，它的工作方式与 dim + R 相同。
    :param keepdims: 输出 QTensor 是否保留了最大值索引操作的轴，默认是False。

    :return: 输入 QTensor 中最大值的索引。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        a = QTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092, 0.5419, -0.2993, 0.3195]])
        flag = a.argmax()
        print(flag)
        
        # [0.0000000]

        flag_0 = a.argmax([0], True)
        print(flag_0)

        # [
        # [0.0000000, 3.0000000, 0.0000000, 3.0000000]
        # ]

        flag_1 = a.argmax([1], True)
        print(flag_1)

        # [
        # [0.0000000],
        # [2.0000000],
        # [0.0000000],
        # [1.0000000]
        # ]

argmin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.argmin(*kargs)

    返回输入 QTensor 中所有元素的最小值的索引，或返回 QTensor 按某一维度的最小值的索引。

    :param dim: 计算argmax的轴，只接受单个维度。 如果 dim == None，则返回输入张量中所有元素的最小值的索引。有效的 dim 范围是 [-R, R)，其中 R 是输入的 ndim。 当 dim < 0 时，它的工作方式与 dim + R 相同。
    :param keepdims: 输出 QTensor 是否保留了最小值索引操作的轴，默认是False。

    :return: 输入 QTensor 中最小值的索引。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        a = QTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092, 0.5419, -0.2993, 0.3195]])
        flag = a.argmin()
        print(flag)

        # [12.0000000]

        flag_0 = a.argmin([0], True)
        print(flag_0)

        # [
        # [3.0000000, 2.0000000, 2.0000000, 1.0000000]
        # ]

        flag_1 = a.argmin([1], False)
        print(flag_1)

        # [2.0000000, 3.0000000, 1.0000000, 0.0000000]

        

fill\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.fill_(v)

    为当前张量填充特定值，该函数改变原张量的内部数据。

    :param v: 填充值。

    :return: 无。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        shape = [2, 3]
        value = 42
        t = tensor.zeros(shape)
        t.fill_(value)
        print(t)

        # [
        # [42.0000000, 42.0000000, 42.0000000],
        # [42.0000000, 42.0000000, 42.0000000]
        # ]


all
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.all()

    判断张量内数据是否全为全零。

    :return: 返回True，如果全为非0;否则返回False。

    Example::

        from pyvqnet.tensor import tensor

        shape = [2, 3]
        t = tensor.zeros(shape)
        t.fill_(1.0)
        flag = t.all()
        print(flag)

        # True

any
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.any()

    判断张量内数据是否有任意元素不为0。

    :return: 返回True，如果有任意元素不为0;否则返回False。

    Example::

        from pyvqnet.tensor import tensor

        shape = [2, 3]
        t = tensor.ones(shape)
        t.fill_(1.0)
        flag = t.any()
        print(flag)

        # True


fill_rand_binary\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.fill_rand_binary_(v=0.5)

    用从二项分布中随机采样的值填充 QTensor 。

    如果二项分布后随机生成的数据大于二值化阈值 v ，则设置 QTensor 对应位置的元素值为1，否则为0。

    :param v: 二值化阈值，默认0.5。

    :return: 无。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = QTensor(a)
        t.fill_rand_binary_(2)
        print(t)

        # [
        # [1.0000000, 1.0000000, 1.0000000],
        # [1.0000000, 1.0000000, 1.0000000]
        # ]

fill_rand_signed_uniform\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.fill_rand_signed_uniform_(v=1)

    用从有符号均匀分布中随机采样的值填充 QTensor 。用缩放因子 v 对生成的随机采样的值进行缩放。

    :param v: 缩放因子，默认1。

    :return: 无。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = QTensor(a)
        value = 42

        t.fill_rand_signed_uniform_(value)
        print(t)

        # [
        # [12.8852444, 4.4327269, 4.8489408],
        # [-24.3309803, 26.8036957, 39.4903450]
        # ]


fill_rand_uniform\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.fill_rand_uniform_(v=1)

    用从均匀分布中随机采样的值填充 QTensor 。用缩放因子 v 对生成的随机采样的值进行缩放。

    :param v: 缩放因子，默认1。

    :return: 无。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = QTensor(a)
        value = 42
        t.fill_rand_uniform_(value)
        print(t)

        # [
        # [20.0404720, 14.4064417, 40.2955666],
        # [5.5692234, 26.2520485, 35.3326073]
        # ]


fill_rand_normal\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.fill_rand_normal_(m=0, s=1, fast_math=True)

    生成均值为 m 和方差 s 产生正态分布元素，并填充到张量中。

    :param m: 均值，默认0。
    :param s: 方差，默认1。
    :param fast_math: 是否使用快速方法产生高斯分布，默认True。

    :return: 无。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = QTensor(a)
        t.fill_rand_normal_(2, 10, True)
        print(t)

        # [
        # [-10.4446531    4.9158096   2.9204607],
        # [ -7.2682705   8.1267328    6.2758742 ],
        # ]


QTensor.transpose
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.transpose(new_dims=None)

    反转张量的轴。如果 new_dims = None，则反转所有轴。

    :param new_dims: 列表形式储存的新的轴顺序。

    :return:  新的 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2, 2, 3]).astype(np.float32)
        t = QTensor(a)
        rlt = t.transpose([2,0,1])
        print(rlt)
        # [
        # [[0.0000000, 3.0000000],
        #  [6.0000000, 9.0000000]],
        # [[1.0000000, 4.0000000],
        #  [7.0000000, 10.0000000]],
        # [[2.0000000, 5.0000000],
        #  [8.0000000, 11.0000000]]
        # ]
        


transpose\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.transpose_(new_dims=None)

    反转张量的轴。如果 new_dims = None，则反转所有轴。该接口改变当前张量自己的轴顺序。

    :param new_dims: 列表形式储存的新的轴顺序。

    :return: 无。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2, 2, 3]).astype(np.float32)
        t = QTensor(a)
        t.transpose_([2, 0, 1])
        print(t)

        # [
        # [[0.0000000, 3.0000000],
        #  [6.0000000, 9.0000000]],
        # [[1.0000000, 4.0000000],
        #  [7.0000000, 10.0000000]],
        # [[2.0000000, 5.0000000],
        #  [8.0000000, 11.0000000]]
        # ]
        


QTensor.reshape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.reshape(new_shape)

    改变 QTensor 的形状，返回一个新的张量。

    :param new_shape: 新的形状。

    :return: 新形状的 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t = QTensor(a)
        reshape_t = t.reshape([C, R])
        print(reshape_t)
        # [
        # [0.0000000, 1.0000000, 2.0000000],
        # [3.0000000, 4.0000000, 5.0000000],
        # [6.0000000, 7.0000000, 8.0000000],
        # [9.0000000, 10.0000000, 11.0000000]
        # ]
        

reshape\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.reshape_(new_shape)

    改变当前 QTensor 的形状。

    :param new_shape: 新的形状。

    :return: 无。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t = QTensor(a)
        t.reshape_([C, R])
        print(t)

        # [
        # [0.0000000, 1.0000000, 2.0000000],
        # [3.0000000, 4.0000000, 5.0000000],
        # [6.0000000, 7.0000000, 8.0000000],
        # [9.0000000, 10.0000000, 11.0000000]
        # ]


getdata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.getdata()

    返回一个numpy.ndarray 储存当前 QTensor 的数据。

    :return: 包含当前 QTensor 数据的numpy.ndarray。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.ones([3, 4])
        a = t.getdata()
        print(a)

        # [[1. 1. 1. 1.]
        #  [1. 1. 1. 1.]
        #  [1. 1. 1. 1.]]

__getitem__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.__getitem__()

    支持对 QTensor 使用 切片索引，下标，或使用 QTensor 作为高级索引访问输入。该操作返回一个新的 QTensor 。

    通过冒号 ``:``  分隔切片参数 start:stop:step 来进行切片操作，其中 start、stop、step 均可缺省。

    针对1-D QTensor ，则仅有单个轴上的索引或切片。

    针对2-D及以上的 QTensor ，则会有多个轴上的索引或切片。

    使用 QTensor 作为 索引，则进行高级索引，请参考numpy中 `高级索引 <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ 部分。

    若作为索引的 QTensor 为逻辑运算的结果，则进行 布尔数组索引。

    .. note:: a[3][4][1] 形式的索引暂不支持, 使用 a[3,4,1] 形式代替。
                ``Ellipsis`` `...` 暂不支持 。

    :param item: 以 pyslice , 整数, QTensor 构成切片索引。

    :return: 新的 QTensor。

    Example::

        from pyvqnet.tensor import tensor, QTensor
        aaa = tensor.arange(1, 61)
        aaa.reshape_([4, 5, 3])
        print(aaa[0:2, 3, :2])
        # [
        # [10.0000000, 11.0000000],
        #  [25.0000000, 26.0000000]
        # ]
        print(aaa[3, 4, 1])
        #[59.0000000]
        print(aaa[:, 2, :])
        # [
        # [7.0000000, 8.0000000, 9.0000000],    
        #  [22.0000000, 23.0000000, 24.0000000],
        #  [37.0000000, 38.0000000, 39.0000000],
        #  [52.0000000, 53.0000000, 54.0000000] 
        # ]
        print(aaa[2])
        # [
        # [31.0000000, 32.0000000, 33.0000000], 
        #  [34.0000000, 35.0000000, 36.0000000],
        #  [37.0000000, 38.0000000, 39.0000000],
        #  [40.0000000, 41.0000000, 42.0000000],
        #  [43.0000000, 44.0000000, 45.0000000]
        # ]
        print(aaa[0:2, ::3, 2:])
        # [
        # [[3.0000000],
        #  [12.0000000]],
        # [[18.0000000],
        #  [27.0000000]]
        # ]
        a = tensor.ones([2, 2])
        b = QTensor([[1, 1], [0, 1]])
        b = b > 0
        c = a[b]
        print(c)
        #[1.0000000, 1.0000000, 1.0000000]
        tt = tensor.arange(1, 56 * 2 * 4 * 4 + 1).reshape([2, 8, 4, 7, 4])
        tt.requires_grad = True
        index_sample1 = tensor.arange(0, 3).reshape([3, 1])
        index_sample2 = QTensor([0, 1, 0, 2, 3, 2, 2, 3, 3]).reshape([3, 3])
        gg = tt[:, index_sample1, 3:, index_sample2, 2:]
        print(gg)
        # [
        # [[[[87.0000000, 88.0000000]],
        # [[983.0000000, 984.0000000]]],
        # [[[91.0000000, 92.0000000]],
        # [[987.0000000, 988.0000000]]],
        # [[[87.0000000, 88.0000000]],
        # [[983.0000000, 984.0000000]]]],
        # [[[[207.0000000, 208.0000000]],
        # [[1103.0000000, 1104.0000000]]],
        # [[[211.0000000, 212.0000000]],
        # [[1107.0000000, 1108.0000000]]],
        # [[[207.0000000, 208.0000000]],
        # [[1103.0000000, 1104.0000000]]]],
        # [[[[319.0000000, 320.0000000]],
        # [[1215.0000000, 1216.0000000]]],
        # [[[323.0000000, 324.0000000]],
        # [[1219.0000000, 1220.0000000]]],
        # [[[323.0000000, 324.0000000]],
        # [[1219.0000000, 1220.0000000]]]]
        # ]

__setitem__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.__setitem__()

    支持对 QTensor 使用 切片索引，下标，或使用 QTensor 作为高级索引修改输入。该操作对输入原地进行修改 。

    通过冒号 ``:``  分隔切片参数 start:stop:step 来进行切片操作，其中 start、stop、step 均可缺省。

    针对1-D QTensor，则仅有单个轴上的索引或切片。

    针对2-D及以上的 QTensor ，则会有多个轴上的索引或切片。

    使用 QTensor 作为 索引，则进行高级索引，请参考numpy中 `高级索引 <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ 部分。

    若作为索引的 QTensor 为逻辑运算的结果，则进行 布尔数组索引。

    .. note:: a[3][4][1] 形式的索引暂不支持, 使用 a[3,4,1] 形式代替。
                ``Ellipsis`` `...` 暂不支持 。

    :param item: 以 pyslice , 整数, QTensor 构成切片索引。

    :return: 无。

    Example::

        from pyvqnet.tensor import tensor
        aaa = tensor.arange(1, 61)
        aaa.reshape_([4, 5, 3])
        vqnet_a2 = aaa[3, 4, 1]
        aaa[3, 4, 1] = tensor.arange(10001,
                                        10001 + vqnet_a2.size).reshape(vqnet_a2.shape)
        print(aaa)
        # [
        # [[1.0000000, 2.0000000, 3.0000000],    
        #  [4.0000000, 5.0000000, 6.0000000],    
        #  [7.0000000, 8.0000000, 9.0000000],    
        #  [10.0000000, 11.0000000, 12.0000000], 
        #  [13.0000000, 14.0000000, 15.0000000]],
        # [[16.0000000, 17.0000000, 18.0000000], 
        #  [19.0000000, 20.0000000, 21.0000000], 
        #  [22.0000000, 23.0000000, 24.0000000], 
        #  [25.0000000, 26.0000000, 27.0000000], 
        #  [28.0000000, 29.0000000, 30.0000000]],
        # [[31.0000000, 32.0000000, 33.0000000], 
        #  [34.0000000, 35.0000000, 36.0000000],
        #  [37.0000000, 38.0000000, 39.0000000],
        #  [40.0000000, 41.0000000, 42.0000000],
        #  [43.0000000, 44.0000000, 45.0000000]],
        # [[46.0000000, 47.0000000, 48.0000000],
        #  [49.0000000, 50.0000000, 51.0000000],
        #  [52.0000000, 53.0000000, 54.0000000],
        #  [55.0000000, 56.0000000, 57.0000000],
        #  [58.0000000, 10001.0000000, 60.0000000]]
        # ]
        aaa = tensor.arange(1, 61)
        aaa.reshape_([4, 5, 3])
        vqnet_a3 = aaa[:, 2, :]
        aaa[:, 2, :] = tensor.arange(10001,
                                        10001 + vqnet_a3.size).reshape(vqnet_a3.shape)
        print(aaa)
        # [
        # [[1.0000000, 2.0000000, 3.0000000],
        #  [4.0000000, 5.0000000, 6.0000000],
        #  [10001.0000000, 10002.0000000, 10003.0000000],
        #  [10.0000000, 11.0000000, 12.0000000],
        #  [13.0000000, 14.0000000, 15.0000000]],
        # [[16.0000000, 17.0000000, 18.0000000],
        #  [19.0000000, 20.0000000, 21.0000000],
        #  [10004.0000000, 10005.0000000, 10006.0000000],
        #  [25.0000000, 26.0000000, 27.0000000],
        #  [28.0000000, 29.0000000, 30.0000000]],
        # [[31.0000000, 32.0000000, 33.0000000],
        #  [34.0000000, 35.0000000, 36.0000000],
        #  [10007.0000000, 10008.0000000, 10009.0000000],
        #  [40.0000000, 41.0000000, 42.0000000],
        #  [43.0000000, 44.0000000, 45.0000000]],
        # [[46.0000000, 47.0000000, 48.0000000],
        #  [49.0000000, 50.0000000, 51.0000000],
        #  [10010.0000000, 10011.0000000, 10012.0000000],
        #  [55.0000000, 56.0000000, 57.0000000],
        #  [58.0000000, 59.0000000, 60.0000000]]
        # ]
        aaa = tensor.arange(1, 61)
        aaa.reshape_([4, 5, 3])
        vqnet_a4 = aaa[2, :]
        aaa[2, :] = tensor.arange(10001,
                                    10001 + vqnet_a4.size).reshape(vqnet_a4.shape)
        print(aaa)
        # [
        # [[1.0000000, 2.0000000, 3.0000000],
        #  [4.0000000, 5.0000000, 6.0000000],
        #  [7.0000000, 8.0000000, 9.0000000],
        #  [10.0000000, 11.0000000, 12.0000000],
        #  [13.0000000, 14.0000000, 15.0000000]],
        # [[16.0000000, 17.0000000, 18.0000000],
        #  [19.0000000, 20.0000000, 21.0000000],
        #  [22.0000000, 23.0000000, 24.0000000],
        #  [25.0000000, 26.0000000, 27.0000000],
        #  [28.0000000, 29.0000000, 30.0000000]],
        # [[10001.0000000, 10002.0000000, 10003.0000000],
        #  [10004.0000000, 10005.0000000, 10006.0000000],
        #  [10007.0000000, 10008.0000000, 10009.0000000],
        #  [10010.0000000, 10011.0000000, 10012.0000000],
        #  [10013.0000000, 10014.0000000, 10015.0000000]],
        # [[46.0000000, 47.0000000, 48.0000000],
        #  [49.0000000, 50.0000000, 51.0000000],
        #  [52.0000000, 53.0000000, 54.0000000],
        #  [55.0000000, 56.0000000, 57.0000000],
        #  [58.0000000, 59.0000000, 60.0000000]]
        # ]
        aaa = tensor.arange(1, 61)
        aaa.reshape_([4, 5, 3])
        vqnet_a5 = aaa[0:2, ::2, 1:2]
        aaa[0:2, ::2,
            1:2] = tensor.arange(10001,
                                    10001 + vqnet_a5.size).reshape(vqnet_a5.shape)
        print(aaa)
        # [
        # [[1.0000000, 10001.0000000, 3.0000000],
        #  [4.0000000, 5.0000000, 6.0000000],
        #  [7.0000000, 10002.0000000, 9.0000000],
        #  [10.0000000, 11.0000000, 12.0000000],
        #  [13.0000000, 10003.0000000, 15.0000000]],
        # [[16.0000000, 10004.0000000, 18.0000000],
        #  [19.0000000, 20.0000000, 21.0000000],
        #  [22.0000000, 10005.0000000, 24.0000000],
        #  [25.0000000, 26.0000000, 27.0000000],
        #  [28.0000000, 10006.0000000, 30.0000000]],
        # [[31.0000000, 32.0000000, 33.0000000],
        #  [34.0000000, 35.0000000, 36.0000000],
        #  [37.0000000, 38.0000000, 39.0000000],
        #  [40.0000000, 41.0000000, 42.0000000],
        #  [43.0000000, 44.0000000, 45.0000000]],
        # [[46.0000000, 47.0000000, 48.0000000],
        #  [49.0000000, 50.0000000, 51.0000000],
        #  [52.0000000, 53.0000000, 54.0000000],
        #  [55.0000000, 56.0000000, 57.0000000],
        #  [58.0000000, 59.0000000, 60.0000000]]
        # ]
        a = tensor.ones([2, 2])
        b = tensor.QTensor([[1, 1], [0, 1]])
        b = b > 0
        x = tensor.QTensor([1001, 2001, 3001])

        a[b] = x
        print(a)
        # [
        # [1001.0000000, 2001.0000000],
        #  [1.0000000, 3001.0000000]
        # ]


创建函数
-----------------------------

.. _ones:

ones
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.ones(shape,device=0,dtype-None)

    创建元素全一的 QTensor 。

    :param shape: 数据的形状。
    :param device: 储存在哪个设备上，默认0，在CPU上。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 返回新的 QTensor 。

    Example::

        from pyvqnet.tensor import tensor

        x = tensor.ones([2, 3])
        print(x)

        # [
        # [1.0000000, 1.0000000, 1.0000000],
        # [1.0000000, 1.0000000, 1.0000000]
        # ]

ones_like
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.ones_like(t: pyvqnet.tensor.QTensor,device=0,dtype=None)

    创建元素全一的 QTensor ,形状和输入的 QTensor 一样。

    :param t: 输入 QTensor 。
    :param device: 储存在哪个设备上，默认0，在CPU上。
    :param dtype: 参数的数据类型，defaults：None,跟输入的dtype一样。

    :return: 新的全一  QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.ones_like(t)
        print(x)

        # [1.0000000, 1.0000000, 1.0000000]


full
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.full(shape, value, device: int = 0,dtype=None)

    创建一个指定形状的 QTensor 并用特定值填充它。

    :param shape: 要创建的张量形状。
    :param value: 填充的值。
    :param device: 储存在哪个设备上，默认0，在CPU上。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 输出新 QTensor 。 

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        shape = [2, 3]
        value = 42
        t = tensor.full(shape, value)
        print(t)
        # [
        # [42.0000000, 42.0000000, 42.0000000],
        # [42.0000000, 42.0000000, 42.0000000]
        # ]


full_like
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.full_like(t, value, device: int = 0,dtype=None)

    创建一个形状和输入一样的 QTensor,所有元素填充 value 。

    :param t: 输入 QTensor 。
    :param value: 填充 QTensor 的值。
    :param device: 储存在哪个设备上，默认0，在CPU上。
    :param dtype: 参数的数据类型，defaults：None,跟输入的dtype一样。

    :return: 输出 QTensor。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        a = tensor.randu([3,5])
        value = 42
        t = tensor.full_like(a, value)
        print(t)
        # [
        # [42.0000000, 42.0000000, 42.0000000, 42.0000000, 42.0000000],    
        # [42.0000000, 42.0000000, 42.0000000, 42.0000000, 42.0000000],    
        # [42.0000000, 42.0000000, 42.0000000, 42.0000000, 42.0000000]     
        # ]
        

zeros
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.zeros(shape, device: int = 0,dtype=None)

    创建输入形状大小的全零 QTensor 。

    :param shape: 输入形状。
    :param device: 储存在哪个设备上，默认0，在CPU上。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.zeros([2, 3, 4])
        print(t)
        # [
        # [[0.0000000, 0.0000000, 0.0000000, 0.0000000],
        #  [0.0000000, 0.0000000, 0.0000000, 0.0000000],
        #  [0.0000000, 0.0000000, 0.0000000, 0.0000000]],
        # [[0.0000000, 0.0000000, 0.0000000, 0.0000000],
        #  [0.0000000, 0.0000000, 0.0000000, 0.0000000],
        #  [0.0000000, 0.0000000, 0.0000000, 0.0000000]]
        # ]
        

zeros_like
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.zeros_like(t: pyvqnet.tensor.QTensor, device: int = 0,dtype=None)

    创建一个形状和输入一样的 QTensor,所有元素为0 。

    :param t: 输入参考 QTensor 。
    :param device: 储存在哪个设备上，默认0，在CPU上。
    :param dtype: 参数的数据类型，defaults：None,跟输入的dtype一样。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.zeros_like(t)
        print(x)

        # [0.0000000, 0.0000000, 0.0000000]
        


arange
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.arange(start, end, step=1, device: int = 0,dtype=None,requires_grad=False)

    创建一个在给定间隔内具有均匀间隔值的一维 QTensor 。

    :param start: 间隔开始。
    :param end: 间隔结束。
    :param step: 值之间的间距，默认为1。
    :param device: 要使用的设备，默认 = 0 ，使用 CPU 设备。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param requires_grad: 是否计算梯度，默认为False。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.arange(2, 30, 4)
        print(t)

        # [ 2.0000000,  6.0000000, 10.0000000, 14.0000000, 18.0000000, 22.0000000, 26.0000000]
        

linspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.linspace(start, end, num, device: int = 0,dtype=None,requires_grad= False)

    创建一维 QTensor ，其中的元素为区间 start 和 end 上均匀间隔的共 num 个值。

    :param start: 间隔开始。
    :param end: 间隔结束。
    :param num: 间隔的个数。
    :param device: 要使用的设备，默认 = 0 ，使用 CPU 设备。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param requires_grad: 是否计算梯度，默认为False。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        start, stop, num = -2.5, 10, 10
        t = tensor.linspace(start, stop, num)
        print(t)
        #[-2.5000000, -1.1111112, 0.2777777, 1.6666665, 3.0555553, 4.4444442, 5.8333330, 7.2222219, 8.6111107, 10.0000000]

logspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.logspace(start, end, num, base, device: int = 0,dtype=None, requires_grad)

    在对数刻度上创建具有均匀间隔值的一维 QTensor。

    :param start: ``base ** start`` 是起始值
    :param end: ``base ** end`` 是序列的最终值
    :param num: 要生成的样本数
    :param base: 对数刻度的基数
    :param device: 要使用的设备，默认 = 0 ，使用 CPU 设备。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param requires_grad: 是否计算梯度，默认为False。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        start, stop, steps, base = 0.1, 1.0, 5, 10.0
        t = tensor.logspace(start, stop, steps, base)
        print(t)

        # [1.2589254, 2.1134889, 3.5481336, 5.9566211, 10.0000000]
        

eye
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.eye(size, offset: int = 0, device: int = 0,dtype=None)

    创建一个 size x size 的 QTensor，对角线上为 1，其他地方为 0。

    :param size: 要创建的（正方形）QTensor 的大小。
    :param offset: 对角线的索引：0（默认）表示主对角线，正值表示上对角线，负值表示下对角线。
    :param device: 要使用的设备，默认 = 0 ，使用 CPU 设备。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        size = 3
        t = tensor.eye(size)
        print(t)

        # [
        # [1.0000000, 0.0000000, 0.0000000],
        # [0.0000000, 1.0000000, 0.0000000],
        # [0.0000000, 0.0000000, 1.0000000]
        # ]
        

diag
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.diag(t, k: int = 0,requires_grad=False)

    构造对角矩阵。

    输入一个 2-D QTensor，则返回一个与此相同的新张量，除了
    选定对角线中的元素以外的元素设置为零。

    :param t: 输入 QTensor。
    :param k: 偏移量（主对角线为 0，正数为向上偏移，负数为向下偏移），默认为0。
    :param requires_grad: 是否计算梯度，默认为False。

    :return: 输出 QTensor。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(16).reshape(4, 4).astype(np.float32)
        t = QTensor(a)
        for k in range(-3, 4):
            u = tensor.diag(t,k=k)
            print(u)


        # [[ 0.  0.  0.  0.]
        #  [ 0.  0.  0.  0.]
        #  [ 0.  0.  0.  0.]
        #  [12.  0.  0.  0.]]
        # [[ 0.  0.  0.  0.]
        #  [ 0.  0.  0.  0.]
        #  [ 8.  0.  0.  0.]
        #  [ 0. 13.  0.  0.]]
        # [[ 0.  0.  0.  0.]
        #  [ 4.  0.  0.  0.]
        #  [ 0.  9.  0.  0.]
        #  [ 0.  0. 14.  0.]]
        # [[ 0.  0.  0.  0.]
        #  [ 0.  5.  0.  0.]
        #  [ 0.  0. 10.  0.]
        #  [ 0.  0.  0. 15.]]
        # [[ 0.  1.  0.  0.]
        #  [ 0.  0.  6.  0.]
        #  [ 0.  0.  0. 11.]
        #  [ 0.  0.  0.  0.]]
        # [[0. 0. 2. 0.]
        #  [0. 0. 0. 7.]
        #  [0. 0. 0. 0.]
        #  [0. 0. 0. 0.]]
        # [[0. 0. 0. 3.]
        #  [0. 0. 0. 0.]
        #  [0. 0. 0. 0.]
        #  [0. 0. 0. 0.]]


randu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.randu(shape, min=0.0,max=1.0, device: int = 0, dtype=None, requires_grad=False)

    创建一个具有均匀分布随机值的 QTensor 。

    :param shape: 要创建的 QTensor 的形状。
    :param min: 分布的下限，默认: 0。
    :param max: 分布的上线，默认: 1。
    :param device: 要使用的设备，默认 = 0 ，使用 CPU 设备。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param requires_grad: 是否计算梯度，默认为False。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        shape = [2, 3]
        t = tensor.randu(shape)
        print(t)

        # [
        # [0.0885886, 0.9570093, 0.8304565],
        # [0.6055251, 0.8721224, 0.1927866]
        # ]
        

randn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.randn(shape, mean=0.0,std=1.0, device: int = 0, dtype=None, requires_grad=False)

    创建一个具有正态分布随机值的 QTensor 。

    :param shape: 要创建的 QTensor 的形状。
    :param mean: 分布的均值，默认: 0。
    :param max: 分布的方差，默认: 1。
    :param device: 要使用的设备，默认 = 0 ，使用 CPU 设备。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。
    :param requires_grad: 是否计算梯度，默认为False。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        shape = [2, 3]
        t = tensor.randn(shape)
        print(t)

        # [
        # [-0.9529880, -0.4947567, -0.6399882],
        # [-0.6987777, -0.0089036, -0.5084590]
        # ]


multinomial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.multinomial(t, num_samples)

    返回一个张量，其中每行包含 num_samples 个索引采样，来自位于张量输入的相应行中的多项式概率分布。
    
    :param t: 输入概率分布,仅支持浮点数。
    :param num_samples: 采样样本。

    :return:
         输出采样索引

    Examples::

        from pyvqnet import tensor
        weights = tensor.QTensor([0.1,10, 3, 1]) 
        idx = tensor.multinomial(weights,3)
        print(idx)

        from pyvqnet import tensor
        weights = tensor.QTensor([0,10, 3, 2.2,0.0]) 
        idx = tensor.multinomial(weights,3)
        print(idx)

        # [1 0 3]
        # [1 3 2]

triu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.triu(t, diagonal=0)

    返回输入 t 的上三角矩阵，其余部分被设为0。

    :param t: 输入 QTensor。
    :param diagonal: 偏移量（主对角线为 0，正数为向上偏移，负数为向下偏移），默认=0。

    :return: 输出 QTensor。

    Examples::

        from pyvqnet.tensor import tensor
        a = tensor.arange(1.0, 2 * 6 * 5 + 1.0).reshape([2, 6, 5])
        u = tensor.triu(a, 1)
        print(u)
        # [
        # [[0.0000000, 2.0000000, 3.0000000, 4.0000000, 5.0000000],       
        #  [0.0000000, 0.0000000, 8.0000000, 9.0000000, 10.0000000],      
        #  [0.0000000, 0.0000000, 0.0000000, 14.0000000, 15.0000000],     
        #  [0.0000000, 0.0000000, 0.0000000, 0.0000000, 20.0000000],      
        #  [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000],       
        #  [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000]],      
        # [[0.0000000, 32.0000000, 33.0000000, 34.0000000, 35.0000000],   
        #  [0.0000000, 0.0000000, 38.0000000, 39.0000000, 40.0000000],    
        #  [0.0000000, 0.0000000, 0.0000000, 44.0000000, 45.0000000],     
        #  [0.0000000, 0.0000000, 0.0000000, 0.0000000, 50.0000000],      
        #  [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000],       
        #  [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000]]       
        # ]

tril
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tril(t, diagonal=0)

    返回输入 t 的下三角矩阵，其余部分被设为0。

    :param t: 输入 QTensor。
    :param diagonal: 偏移量（主对角线为 0，正数为向上偏移，负数为向下偏移），默认=0。

    :return: 输出 QTensor。

    Examples::

        from pyvqnet.tensor import tensor
        a = tensor.arange(1.0, 2 * 6 * 5 + 1.0).reshape([12, 5])
        u = tensor.tril(a, 1)
        print(u)
        # [
        # [1.0000000, 2.0000000, 0.0000000, 0.0000000, 0.0000000],      
        #  [6.0000000, 7.0000000, 8.0000000, 0.0000000, 0.0000000],     
        #  [11.0000000, 12.0000000, 13.0000000, 14.0000000, 0.0000000], 
        #  [16.0000000, 17.0000000, 18.0000000, 19.0000000, 20.0000000],
        #  [21.0000000, 22.0000000, 23.0000000, 24.0000000, 25.0000000],
        #  [26.0000000, 27.0000000, 28.0000000, 29.0000000, 30.0000000],
        #  [31.0000000, 32.0000000, 33.0000000, 34.0000000, 35.0000000],
        #  [36.0000000, 37.0000000, 38.0000000, 39.0000000, 40.0000000],
        #  [41.0000000, 42.0000000, 43.0000000, 44.0000000, 45.0000000],
        #  [46.0000000, 47.0000000, 48.0000000, 49.0000000, 50.0000000],
        #  [51.0000000, 52.0000000, 53.0000000, 54.0000000, 55.0000000],
        #  [56.0000000, 57.0000000, 58.0000000, 59.0000000, 60.0000000]
        # ]

数学函数
-----------------------------


floor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.floor(t)

    返回一个新的 QTensor，其中元素为输入 QTensor 的向下取整。

    :param t: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.floor(t)
        print(u)

        # [-2.0000000, -2.0000000, -2.0000000, -2.0000000, -1.0000000, -1.0000000, -1.0000000, -1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000]

ceil
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.ceil(t)

    返回一个新的 QTensor，其中元素为输入 QTensor 的向上取整。

    :param t: 输入 QTensor 。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.ceil(t)
        print(u)

        # [-2.0000000, -1.0000000, -1.0000000, -1.0000000, -1.0000000, -0.0000000, -0.0000000, -0.0000000, 0.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000, 2.0000000, 2.0000000, 2.0000000]

round
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.round(t)

    返回一个新的 QTensor，其中元素为输入 QTensor 的四舍五入到最接近的整数.

    :param t: 输入 QTensor 。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.arange(-2.0, 2.0, 0.4)
        u = tensor.round(t)
        print(u)

        # [-2.0000000, -2.0000000, -1.0000000, -1.0000000, -0.0000000, -0.0000000, 0.0000000, 1.0000000, 1.0000000, 2.0000000]

sort
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.sort(t, axis: int, descending=False, stable=True)

    按指定轴对输入 QTensor 进行排序。

    :param t: 输入 QTensor 。
    :param axis: 排序使用的轴。
    :param descending: 如果是True，进行降序排序，否则使用升序排序。默认为升序。
    :param stable: 是否使用稳定排序，默认为稳定排序。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
        A = QTensor(a)
        AA = tensor.sort(A,1,False)
        print(AA)

        # [
        # [0.0000000, 1.0000000, 2.0000000, 4.0000000, 6.0000000, 7.0000000, 8.0000000, 8.0000000],
        # [2.0000000, 5.0000000, 5.0000000, 8.0000000, 9.0000000, 9.0000000, 9.0000000, 9.0000000],
        # [1.0000000, 2.0000000, 5.0000000, 5.0000000, 5.0000000, 6.0000000, 7.0000000, 7.0000000]
        # ]

argsort
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.argsort(t, axis: int, descending=False, stable=True)

    对输入变量沿给定轴进行排序，输出排序好的数据的相应索引。

    :param t: 输入 QTensor 。
    :param axis: 排序使用的轴。
    :param descending: 如果是True，进行降序排序，否则使用升序排序。默认为升序。
    :param stable: 是否使用稳定排序，默认为稳定排序。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
        A = QTensor(a)
        bb = tensor.argsort(A,1,False)
        print(bb)

        # [
        # [4.0000000, 0.0000000, 1.0000000, 7.0000000, 5.0000000, 3.0000000, 2.0000000, 6.0000000], 
        #  [3.0000000, 0.0000000, 7.0000000, 6.0000000, 2.0000000, 1.0000000, 4.0000000, 5.0000000],
        #  [4.0000000, 7.0000000, 5.0000000, 0.0000000, 2.0000000, 1.0000000, 3.0000000, 6.0000000]
        # ]

topK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.topK(t, k, axis=-1, if_descent=True)

    返回给定输入张量沿给定维度的 k 个最大元素。

    如果 if_descent 为 False，则返回 k 个最小元素。

    :param t: 输入 QTensor 。
    :param k: 取排序后的 k 的个数。
    :param axis: 要排序的维度。默认 = -1，最后一个轴。
    :param if_descent: 排序使用升序还是降序，默认降序。

    :return: 新的 QTensor 。

    Examples::

        from pyvqnet.tensor import tensor, QTensor
        x = QTensor([
            24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
            3., 6., 11.
        ])
        x.reshape_([2, 5, 1, 2])
        x.requires_grad = True
        y = tensor.topK(x, 3, 1)
        print(y)
        # [
        # [[[24.0000000, 15.0000000]],
        # [[15.0000000, 13.0000000]],
        # [[11.0000000, 8.0000000]]],
        # [[[24.0000000, 13.0000000]],
        # [[15.0000000, 11.0000000]],
        # [[7.0000000, 8.0000000]]]
        # ]

argtopK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.argtopK(t, k, axis=-1, if_descent=True)

    返回给定输入张量沿给定维度的 k 个最大元素的索引。

    如果 if_descent 为 False，则返回 k 个最小元素的索引。

    :param t: 输入 QTensor 。
    :param k: 取排序后的 k 的个数。
    :param axis: 要排序的维度。默认 = -1，最后一个轴。
    :param if_descent: 排序使用升序还是降序，默认降序。

    :return: 新的 QTensor 。

    Examples::

        from pyvqnet.tensor import tensor, QTensor
        x = QTensor([
            24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
            3., 6., 11.
        ])
        x.reshape_([2, 5, 1, 2])
        x.requires_grad = True
        y = tensor.argtopK(x, 3, 1)
        print(y)
        # [
        # [[[0.0000000, 4.0000000]],
        # [[1.0000000, 0.0000000]],
        # [[3.0000000, 2.0000000]]],
        # [[[0.0000000, 0.0000000]],
        # [[1.0000000, 4.0000000]],
        # [[3.0000000, 2.0000000]]]
        # ]


add
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.add(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    两个 QTensor 按元素相加。等价于t1 + t2。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。
    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.add(t1, t2)
        print(x)

        # [5.0000000, 7.0000000, 9.0000000]

sub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.sub(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    两个 QTensor 按元素相减。等价于t1 - t2。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。
    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.sub(t1, t2)
        print(x)

        # [-3.0000000, -3.0000000, -3.0000000]

mul
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.mul(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    两个 QTensor 按元素相乘。等价于t1 * t2。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。
    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.mul(t1, t2)
        print(x)

        # [4.0000000, 10.0000000, 18.0000000]

divide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.divide(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    两个 QTensor 按元素相除。等价于t1 / t2。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。
    :return:  输出 QTensor 。


    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.divide(t1, t2)
        print(x)

        # [0.2500000, 0.4000000, 0.5000000]

sums
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.sums(t: pyvqnet.tensor.QTensor, axis: Optional[int] = None, keepdims=False)

    对输入的 QTensor 按 axis 设定的轴计算元素和，如果 axis 是None，则返回所有元素和。

    :param t: 输入 QTensor 。
    :param axis: 用于求和的轴，默认为None。
    :param keepdims: 输出张量是否保留了减小的维度。默认为False。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor(([1, 2, 3], [4, 5, 6]))
        x = tensor.sums(t)
        print(x)

        # [21.0000000]

cumsum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.cumsum(t, axis=-1)

    返回维度轴中输入元素的累积总和。

    :param t: 输入 QTensor 。
    :param axis: 计算的轴，默认 -1，使用最后一个轴。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor, QTensor
        t = QTensor(([1, 2, 3], [4, 5, 6]))
        x = tensor.cumsum(t,-1)
        print(x)
        # [
        # [1.0000000, 3.0000000, 6.0000000], 
        # [4.0000000, 9.0000000, 15.0000000]
        # ]


mean
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.mean(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

    对输入的 QTensor 按 axis 设定的轴计算元素的平均，如果 axis 是None，则返回所有元素平均。

    :param t: 输入 QTensor ,需要是浮点数或者复数。
    :param axis: 用于求平均的轴，默认为None。
    :param keepdims: 输出张量是否保留了减小的维度。默认为False。
    :return: 输出 QTensor 或 均值。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[1, 2, 3], [4, 5, 6.0]])
        x = tensor.mean(t, axis=1)
        print(x)

        # [2. 5.]

median
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.median(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

    对输入的 QTensor 按 axis 设定的轴计算元素的平均，如果 axis 是None，则返回所有元素平均。

    :param t: 输入 QTensor 。
    :param axis: 用于求平均的轴，默认为None。
    :param keepdims: 输出张量是否保留了减小的维度。默认为False。
    :return: 输出 QTensor 或 中值。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1.5219, -1.5212,  0.2202]])
        median_a = tensor.median(a)
        print(median_a)

        # [0.2202000]

        b = QTensor([[0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
                    [0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
                    [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
                    [1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
        median_b = tensor.median(b,1, False)
        print(median_b)

        # [-0.3982000, 0.2269999, 0.2487999, 0.4742000]

std
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.std(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False, unbiased=True)

    对输入的 QTensor 按 axis 设定的轴计算元素的标准差，如果 axis 是None，则返回所有元素标准差。

    :param t: 输入 QTensor 。
    :param axis: 用于求标准差的轴，默认为None。
    :param keepdims: 输出张量是否保留了减小的维度。默认为False。
    :param unbiased: 是否使用贝塞尔修正,默认使用。
    :return: 输出 QTensor 或 标准差。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[-0.8166, -1.3802, -0.3560]])
        std_a = tensor.std(a)
        print(std_a)

        # [0.5129624]

        b = QTensor([[0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
                    [0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
                    [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
                    [1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
        std_b = tensor.std(b, 1, False, False)
        print(std_b)

        # [0.6593542, 0.5583112, 0.3206565, 1.1103367]

var
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.var(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False, unbiased=True)

    对输入的 QTensor 按 axis 设定的轴计算元素的方差，如果 axis 是None，则返回所有元素方差。

    :param t: 输入 QTensor 。
    :param axis: 用于求方差的轴，默认为None。
    :param keepdims: 输出张量是否保留了减小的维度。默认为False。
    :param unbiased: 是否使用贝塞尔修正,默认使用。
    :return: 输出 QTensor 或方差。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[-0.8166, -1.3802, -0.3560]])
        a_var = tensor.var(a)
        print(a_var)

        # [0.2631305]

matmul
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.matmul(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    二维矩阵点乘或3、4维张量进行批矩阵乘法.

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。
    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        t1 = tensor.ones([2,3])
        t1.requires_grad = True
        t2 = tensor.ones([3,4])
        t2.requires_grad = True
        t3  = tensor.matmul(t1,t2)
        t3.backward(tensor.ones_like(t3))
        print(t1.grad)

        # [
        # [4.0000000, 4.0000000, 4.0000000],
        #  [4.0000000, 4.0000000, 4.0000000]
        # ]

        print(t2.grad)

        # [
        # [2.0000000, 2.0000000, 2.0000000, 2.0000000],
        #  [2.0000000, 2.0000000, 2.0000000, 2.0000000],
        #  [2.0000000, 2.0000000, 2.0000000, 2.0000000]
        # ]

kron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    计算 :attr:`input` 和 :attr:`other` 的 Kronecker 积，用 :math:`\otimes` 表示。

    如果 :attr:`input` 是一个 :math:`(a_0 \times a_1 \times \dots \times a_n)` 张量并且 :attr:`other` 是一个
     :math:`(b_0 \times b_1 \times \dots \times b_n)` 张量，结果将是
     :math:`(a_0*b_0 \times a_1*b_1 \times \dots \times a_n*b_n)` 张量，包含以下条目：

     .. math::
         (\text{input} \otimes \text{other})_{k_0, k_1, \dots, k_n} =
             \text{input}_{i_0, i_1, \dots, i_n} * \text{other}_{j_0, j_1, \dots, j_n},

     其中 :math:`k_t = i_t * b_t + j_t` 为 :math:`0 \leq t \leq n`。
     如果一个张量的维数少于另一个，它将被解压缩，直到它具有相同的维数。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。
    :return:  输出 QTensor 。

    Example::

        from pyvqnet import tensor
        a = tensor.arange(1,1+ 24).reshape([2,1,2,3,2])
        b = tensor.arange(1,1+ 24).reshape([6,4])


        c = tensor.kron(a,b)
        print(c)

        # [[[[[  1.   2.   3.   4.   2.   4.   6.   8.]
        #     [  5.   6.   7.   8.  10.  12.  14.  16.]
        #     [  9.  10.  11.  12.  18.  20.  22.  24.]
        #     [ 13.  14.  15.  16.  26.  28.  30.  32.]
        #     [ 17.  18.  19.  20.  34.  36.  38.  40.]
        #     [ 21.  22.  23.  24.  42.  44.  46.  48.]
        #     [  3.   6.   9.  12.   4.   8.  12.  16.]
        #     [ 15.  18.  21.  24.  20.  24.  28.  32.]
        #     [ 27.  30.  33.  36.  36.  40.  44.  48.]
        #     [ 39.  42.  45.  48.  52.  56.  60.  64.]
        #     [ 51.  54.  57.  60.  68.  72.  76.  80.]
        #     [ 63.  66.  69.  72.  84.  88.  92.  96.]
        #     [  5.  10.  15.  20.   6.  12.  18.  24.]
        #     [ 25.  30.  35.  40.  30.  36.  42.  48.]
        #     [ 45.  50.  55.  60.  54.  60.  66.  72.]
        #     [ 65.  70.  75.  80.  78.  84.  90.  96.]
        #     [ 85.  90.  95. 100. 102. 108. 114. 120.]
        #     [105. 110. 115. 120. 126. 132. 138. 144.]]

        #    [[  7.  14.  21.  28.   8.  16.  24.  32.]
        #     [ 35.  42.  49.  56.  40.  48.  56.  64.]
        #     [ 63.  70.  77.  84.  72.  80.  88.  96.]
        #     [ 91.  98. 105. 112. 104. 112. 120. 128.]
        #     [119. 126. 133. 140. 136. 144. 152. 160.]
        #     [147. 154. 161. 168. 168. 176. 184. 192.]
        #     [  9.  18.  27.  36.  10.  20.  30.  40.]
        #     [ 45.  54.  63.  72.  50.  60.  70.  80.]
        #     [ 81.  90.  99. 108.  90. 100. 110. 120.]
        #     [117. 126. 135. 144. 130. 140. 150. 160.]
        #     [153. 162. 171. 180. 170. 180. 190. 200.]
        #     [189. 198. 207. 216. 210. 220. 230. 240.]
        #     [ 11.  22.  33.  44.  12.  24.  36.  48.]
        #     [ 55.  66.  77.  88.  60.  72.  84.  96.]
        #     [ 99. 110. 121. 132. 108. 120. 132. 144.]
        #     [143. 154. 165. 176. 156. 168. 180. 192.]
        #     [187. 198. 209. 220. 204. 216. 228. 240.]
        #     [231. 242. 253. 264. 252. 264. 276. 288.]]]]



        #  [[[[ 13.  26.  39.  52.  14.  28.  42.  56.]
        #     [ 65.  78.  91. 104.  70.  84.  98. 112.]
        #     [117. 130. 143. 156. 126. 140. 154. 168.]
        #     [169. 182. 195. 208. 182. 196. 210. 224.]
        #     [221. 234. 247. 260. 238. 252. 266. 280.]
        #     [273. 286. 299. 312. 294. 308. 322. 336.]
        #     [ 15.  30.  45.  60.  16.  32.  48.  64.]
        #     [ 75.  90. 105. 120.  80.  96. 112. 128.]
        #     [135. 150. 165. 180. 144. 160. 176. 192.]
        #     [195. 210. 225. 240. 208. 224. 240. 256.]
        #     [255. 270. 285. 300. 272. 288. 304. 320.]
        #     [315. 330. 345. 360. 336. 352. 368. 384.]
        #     [ 17.  34.  51.  68.  18.  36.  54.  72.]
        #     [ 85. 102. 119. 136.  90. 108. 126. 144.]
        #     [153. 170. 187. 204. 162. 180. 198. 216.]
        #     [221. 238. 255. 272. 234. 252. 270. 288.]
        #     [289. 306. 323. 340. 306. 324. 342. 360.]
        #     [357. 374. 391. 408. 378. 396. 414. 432.]]

        #    [[ 19.  38.  57.  76.  20.  40.  60.  80.]
        #     [ 95. 114. 133. 152. 100. 120. 140. 160.]
        #     [171. 190. 209. 228. 180. 200. 220. 240.]
        #     [247. 266. 285. 304. 260. 280. 300. 320.]
        #     [323. 342. 361. 380. 340. 360. 380. 400.]
        #     [399. 418. 437. 456. 420. 440. 460. 480.]
        #     [ 21.  42.  63.  84.  22.  44.  66.  88.]
        #     [105. 126. 147. 168. 110. 132. 154. 176.]
        #     [189. 210. 231. 252. 198. 220. 242. 264.]
        #     [273. 294. 315. 336. 286. 308. 330. 352.]
        #     [357. 378. 399. 420. 374. 396. 418. 440.]
        #     [441. 462. 483. 504. 462. 484. 506. 528.]
        #     [ 23.  46.  69.  92.  24.  48.  72.  96.]
        #     [115. 138. 161. 184. 120. 144. 168. 192.]
        #     [207. 230. 253. 276. 216. 240. 264. 288.]
        #     [299. 322. 345. 368. 312. 336. 360. 384.]
        #     [391. 414. 437. 460. 408. 432. 456. 480.]
        #     [483. 506. 529. 552. 504. 528. 552. 576.]]]]]


reciprocal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.reciprocal(t)

    计算输入 QTensor 的倒数。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(1, 10, 1)
        u = tensor.reciprocal(t)
        print(u)

        #[1.0000000, 0.5000000, 0.3333333, 0.2500000, 0.2000000, 0.1666667, 0.1428571, 0.1250000, 0.1111111]

sign
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.sign(t)

    对输入 t 中每个元素进行正负判断，并且输出正负判断值：1代表正，-1代表负，0代表零。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。


    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-5, 5, 1)
        u = tensor.sign(t)
        print(u)

        # [-1.0000000, -1.0000000, -1.0000000, -1.0000000, -1.0000000, 0.0000000, 1.0000000, 1.0000000, 1.0000000, 1.0000000]

neg
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.neg(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的相反数并返回。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.neg(t)
        print(x)

        # [-1.0000000, -2.0000000, -3.0000000]

trace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.trace(t, k: int = 0)

    返回二维矩阵的迹。

    :param t: 输入 QTensor 。
    :param k: 偏移量（主对角线为 0，正数为向上偏移，负数为向下偏移），默认为0。

    :return: 输入二维矩阵的对角线元素之和。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.randn([4,4])
        for k in range(-3, 4):
            u=tensor.trace(t,k=k)
            print(u)

        # 0.07717618346214294
        # -1.9287869930267334
        # 0.6111435890197754
        # 2.8094992637634277
        # 0.6388946771621704
        # -1.3400784730911255
        # 0.26980453729629517

exp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.exp(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的自然数e为底指数。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.exp(t)
        print(x)

        # [2.7182817, 7.3890562, 20.0855369]

acos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.acos(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的反余弦。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(36).reshape(2,6,3).astype(np.float32)
        a =a/100
        A = QTensor(a,requires_grad = True)
        y = tensor.acos(A)
        print(y)

        # [
        # [[1.5707964, 1.5607961, 1.5507950],
        #  [1.5407919, 1.5307857, 1.5207754],
        #  [1.5107603, 1.5007390, 1.4907107],
        #  [1.4806744, 1.4706289, 1.4605733],
        #  [1.4505064, 1.4404273, 1.4303349],
        #  [1.4202280, 1.4101057, 1.3999666]],
        # [[1.3898098, 1.3796341, 1.3694384],
        #  [1.3592213, 1.3489819, 1.3387187],
        #  [1.3284305, 1.3181161, 1.3077742],
        #  [1.2974033, 1.2870022, 1.2765695],
        #  [1.2661036, 1.2556033, 1.2450669],
        #  [1.2344928, 1.2238795, 1.2132252]]
        # ]

asin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.asin(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的反正弦。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-1, 1, .5)
        u = tensor.asin(t)
        print(u)

        #[-1.5707964, -0.5235988, 0.0000000, 0.5235988]

atan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.atan(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的反正切。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.arange(-1, 1, .5)
        u = tensor.atan(t)
        print(u)

        # [-0.7853981, -0.4636476, 0.0000000, 0.4636476]

sin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.sin(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的正弦。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sin(t)
        print(x)

        # [0.8414709, 0.9092974, 0.1411200]

cos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.cos(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的余弦。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.cos(t)
        print(x)

        # [0.5403022, -0.4161468, -0.9899924]

tan 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tan(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的正切。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.tan(t)
        print(x)

        # [1.5574077, -2.1850397, -0.1425465]

tanh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tanh(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的双曲正切。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.tanh(t)
        print(x)

        # [0.7615941, 0.9640275, 0.9950547]

sinh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.sinh(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的双曲正弦。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sinh(t)
        print(x)

        # [1.1752011, 3.6268603, 10.0178747]

cosh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.cosh(t: pyvqnet.tensor.QTensor)

    计算输入 t 每个元素的双曲余弦。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.cosh(t)
        print(x)

        # [1.5430806, 3.7621955, 10.0676622]

power
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.power(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    第一个 QTensor 的元素计算第二个 QTensor 的幂指数。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。
    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 4, 3])
        t2 = QTensor([2, 5, 6])
        x = tensor.power(t1, t2)
        print(x)

        # [1.0000000, 1024.0000000, 729.0000000]

abs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.abs(t: pyvqnet.tensor.QTensor)

    计算输入 QTensor 的每个元素的绝对值。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, -2, 3])
        x = tensor.abs(t)
        print(x)

        # [1.0000000, 2.0000000, 3.0000000]

log
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.log(t: pyvqnet.tensor.QTensor)

    计算输入 QTensor 的每个元素的自然对数值。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.log(t)
        print(x)

        # [0.0000000, 0.6931471, 1.0986123]

log_softmax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.log_softmax(t, axis=-1)

    顺序计算在轴axis上的softmax函数以及log函数的结果。

    :param t: 输入 QTensor 。
    :param axis: 用于求softmax的轴，默认为-1。

    :return: 输出 QTensor。

    Example::

        from pyvqnet import tensor
        output = tensor.arange(1,13).reshape([3,2,2])
        t = tensor.log_softmax(output,1)
        print(t)
        # [
        # [[-2.1269281, -2.1269281],
        #  [-0.1269280, -0.1269280]],
        # [[-2.1269281, -2.1269281],
        #  [-0.1269280, -0.1269280]],
        # [[-2.1269281, -2.1269281],
        #  [-0.1269280, -0.1269280]]
        # ]

sqrt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.sqrt(t: pyvqnet.tensor.QTensor)

    计算输入 QTensor 的每个元素的平方根值。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sqrt(t)
        print(x)

        # [1.0000000, 1.4142135, 1.7320507]

square
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.square(t: pyvqnet.tensor.QTensor)

    计算输入 QTensor 的每个元素的平方值。

    :param t: 输入 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.square(t)
        print(x)

        # [1.0000000, 4.0000000, 9.0000000]

frobenius_norm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.frobenius_norm(t: QTensor, axis: int = None, keepdims=False):

    对输入的 QTensor 按 axis 设定的轴计算张量的F范数，如果 axis 是None，则返回所有元素F范数。

    :param t: 输入 QTensor 。
    :param axis: 用于求F范数的轴，默认为None。
    :param keepdims: 输出张量是否保留了减小的维度。默认为False。
    :return: 输出 QTensor 或 F范数值。


    Example::

        from pyvqnet.tensor import tensor,QTensor
        t = QTensor([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]],
                    [[13., 14., 15.], [16., 17., 18.]]])
        t.requires_grad = True
        result = tensor.frobenius_norm(t, -2, False)
        print(result)
        # [
        # [4.1231055, 5.3851647, 6.7082038],
        #  [12.2065554, 13.6014709, 15.0000000],
        #  [20.6155281, 22.0227146, 23.4307499]
        # ]


逻辑函数
--------------------------

maximum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.maximum(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    计算两个 QTensor 的逐元素中的较大值。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([6, 4, 3])
        t2 = QTensor([2, 5, 7])
        x = tensor.maximum(t1, t2)
        print(x)

        # [6.0000000, 5.0000000, 7.0000000]

minimum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.minimum(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    计算两个 QTensor 的逐元素中的较小值。

    :param t1: 第一个 QTensor 。
    :param t2: 第二个 QTensor 。

    :return:  输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([6, 4, 3])
        t2 = QTensor([2, 5, 7])
        x = tensor.minimum(t1, t2)
        print(x)

        # [2.0000000, 4.0000000, 3.0000000]

min
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.min(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

    对输入的 QTensor 按 axis 设定的轴计算元素的最小值，如果 axis 是None，则返回所有元素的最小值。

    :param t: 输入 QTensor 。
    :param axis: 用于求最小值的轴，默认为None。
    :param keepdims: 输出张量是否保留了减小的维度。默认为False。

    :return: 输出 QTensor 或浮点数。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.min(t, axis=1, keepdims=True)
        print(x)

        # [
        # [1.0000000],
        #  [4.0000000]
        # ]

max
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.max(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)

    对输入的 QTensor 按 axis 设定的轴计算元素的最大值，如果 axis 是None，则返回所有元素的最大值。

    :param t: 输入 QTensor 。
    :param axis: 用于求最大值的轴，默认为None。
    :param keepdims: 输出张量是否保留了减小的维度。默认为False。
    
    :return: 输出 QTensor 或浮点数。


    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.max(t, axis=1, keepdims=True)
        print(x)

        # [
        # [3.0000000],
        #  [6.0000000]
        # ]

clip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.clip(t: pyvqnet.tensor.QTensor, min_val, max_val)

    将输入的所有元素进行剪裁，使得输出元素限制在[min_val, max_val]。

    :param t: 输入 QTensor 。
    :param min_val:  裁剪下限值。
    :param max_val:  裁剪上限值。
    :return:  output QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([2, 4, 6])
        x = tensor.clip(t, 3, 8)
        print(x)

        # [3.0000000, 4.0000000, 6.0000000]


where
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.where(condition: pyvqnet.tensor.QTensor, t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)


    根据条件返回从 t1 或 t2 中选择的元素。

    :param condition: 判断条件 QTensor,需要是kbool数据类型 。
    :param t1: 如果满足条件，则从中获取元素。
    :param t2: 如果条件不满足，则从中获取元素。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.where(t1 < 2, t1, t2)
        print(x)

        # [1.0000000, 5.0000000, 6.0000000]

nonzero
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.nonzero(t)

    返回一个包含非零元素索引的 QTensor 。

    :param t: 输入 QTensor 。
    :return: 输出 QTensor 包含非零元素的索引。

    Example::
    
        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[0.6, 0.0, 0.0, 0.0],
                                    [0.0, 0.4, 0.0, 0.0],
                                    [0.0, 0.0, 1.2, 0.0],
                                    [0.0, 0.0, 0.0,-0.4]])
        t = tensor.nonzero(t)
        print(t)
        # [
        # [0.0000000, 0.0000000],
        # [1.0000000, 1.0000000],
        # [2.0000000, 2.0000000],
        # [3.0000000, 3.0000000]
        # ]

isfinite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.isfinite(t)

    逐元素判断输入是否为Finite （既非 +/-INF 也非 +/-NaN ）。

    :param t: 输入 QTensor 。
    :return: 输出 QTensor , 其中对应位置元素满足条件时返回True，否则返回False。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isfinite(t)
        print(flag)

        #[ True False  True False False]

isinf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.isinf(t)

    逐元素判断输入的每一个值是否为 +/-INF 。

    :param t: 输入 QTensor 。
    :return: 输出 QTensor , 其中对应位置元素满足条件时返回True，否则返回False。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isinf(t)
        print(flag)

        # [False  True False  True False]

isnan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.isnan(t)

    逐元素判断输入的每一个值是否为 +/-NaN 。

    :param t: 输入 QTensor 。
    :return: 输出 QTensor , 其中对应位置元素满足条件时返回True，否则返回False。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isnan(t)
        print(flag)

        # [False False False False  True]

isneginf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.isneginf(t)

    逐元素判断输入的每一个值是否为 -INF 。

    :param t: 输入 QTensor 。
    :return: 输出 QTensor , 其中对应位置元素满足条件时返回True，否则返回False。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isneginf(t)
        print(flag)

        # [[False False False  True False]

isposinf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.isposinf(t)

    逐元素判断输入的每一个值是否为 +INF 。

    :param t: 输入 QTensor 。
    :return: 输出 QTensor , 其中对应位置元素满足条件时返回True，否则返回False。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isposinf(t)
        print(flag)

        # [False  True False False False]

logical_and
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.logical_and(t1, t2)

    对两个输入进行逐元素逻辑与操作，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_and(a,b)
        print(flag)

        # [False False  True False]

logical_or
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.logical_or(t1, t2)

    对两个输入进行逐元素逻辑或操作，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_or(a,b)
        print(flag)

        # [ True  True  True False]

logical_not
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.logical_not(t)

    对输入进行逐元素逻辑非操作，其中对应位置元素满足条件时返回True，否则返回False。

    :param t: 输入 QTensor 。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        flag = tensor.logical_not(a)
        print(flag)

        # [ True False False  True]

logical_xor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.logical_xor(t1, t2)

    对两个输入进行逐元素逻辑异或操作，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_xor(a,b)
        print(flag)

        # [ True  True False False]

greater
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.greater(t1, t2)

    逐元素比较 t1 是否大于 t2 ，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.greater(a,b)
        print(flag)

        # [[False  True]
        #  [False False]]

greater_equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.greater_equal(t1, t2)

    逐元素比较 t1 是否大于等于 t2 ，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.greater_equal(a,b)
        print(flag)

        #[[ True  True]
        # [False  True]]

less
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.less(t1, t2)

    逐元素比较 t1 是否小于 t2 ，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.less(a,b)
        print(flag)

        #[[False False]
        # [ True False]]

less_equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.less_equal(t1, t2)

    逐元素比较 t1 是否小于等于 t2 ，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.less_equal(a,b)
        print(flag)


        # [[ True False]
        #  [ True  True]]

equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.equal(t1, t2)

    逐元素比较 t1 是否等于 t2 ，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.equal(a,b)
        print(flag)

        #[[ True False]
        # [False  True]]

not_equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.not_equal(t1, t2)

    逐元素比较 t1 是否不等于 t2 ，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 QTensor 。
    :param t2: 输入 QTensor 。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.not_equal(a,b)
        print(flag)

        #[[False  True]
        # [ True False]]

矩阵操作
--------------------------

broadcast
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.broadcast(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    受到某些限制，较小的阵列在整个更大的阵列，以便它们具有兼容的形状。该接口可对入参张量进行自动微分。

    参考https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t1: 输入 QTensor 1
    :param t2: 输入 QTensor 2

    :return t11: 具有新的广播形状 t1。
    :return t22: 具有新广播形状的 t2。

    Example::

        from pyvqnet.tensor import *
        t1 = ones([5,4])
        t2 = ones([4])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)


        t1 = ones([5,4])
        t2 = ones([1])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)


        t1 = ones([5,4])
        t2 = ones([2,1,4])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)


        # [5, 4]
        # [5, 4]
        # [5, 4]
        # [5, 4]
        # [2, 5, 4]
        # [2, 5, 4]

select
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.select(t: pyvqnet.tensor.QTensor, index)

    输入字符串形式的索引位置，获取该索引下的数据切片，返回一个新的 QTensor 。
    
    :param t: 输入 QTensor 。
    :param index: 一个字符串包含切片的索引。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        t = QTensor(np.arange(1,25).reshape(2,3,4))
              
        indx = [":", "0", ":"]        
        t.requires_grad = True
        t.zero_grad()
        ts = tensor.select(t,indx)
        
        print(ts)  
        # [
        # [[1.0000000, 2.0000000, 3.0000000, 4.0000000]],
        # [[13.0000000, 14.0000000, 15.0000000, 16.0000000]]
        # ]

concatenate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.concatenate(args: list, axis=1)

    对 args 内的多个 QTensor 沿 axis 轴进行联结，返回一个新的 QTensor 。

    :param args: 包含输入 QTensor 。
    :param axis: 要连接的维度。 必须介于 0 和输入张量的最大维数之间。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        x = QTensor([[1, 2, 3],[4,5,6]], requires_grad=True)
        y = 1-x
        x = tensor.concatenate([x,y],1)
        print(x)

        # [
        # [1.0000000, 2.0000000, 3.0000000, 0.0000000, -1.0000000, -2.0000000],
        # [4.0000000, 5.0000000, 6.0000000, -3.0000000, -4.0000000, -5.0000000]
        # ]
        
        

stack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.stack(QTensors: list, axis) 

    沿新轴 axis 堆叠输入的 QTensors ，返回一个新的 QTensor。

    :param QTensors: 包含输入 QTensor 。
    :param axis: 要堆叠的维度。 必须介于 0 和输入张量的最大维数之间。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t11 = QTensor(a)
        t22 = QTensor(a)
        t33 = QTensor(a)
        rlt1 = tensor.stack([t11,t22,t33],2)
        print(rlt1)
        
        # [
        # [[0.0000000, 0.0000000, 0.0000000],
        #  [1.0000000, 1.0000000, 1.0000000],
        #  [2.0000000, 2.0000000, 2.0000000],
        #  [3.0000000, 3.0000000, 3.0000000]],
        # [[4.0000000, 4.0000000, 4.0000000],
        #  [5.0000000, 5.0000000, 5.0000000],
        #  [6.0000000, 6.0000000, 6.0000000],
        #  [7.0000000, 7.0000000, 7.0000000]],
        # [[8.0000000, 8.0000000, 8.0000000],
        #  [9.0000000, 9.0000000, 9.0000000],
        #  [10.0000000, 10.0000000, 10.0000000],
        #  [11.0000000, 11.0000000, 11.0000000]]
        # ]
                

permute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.permute(t: pyvqnet.tensor.QTensor, dim: list)

    根据输入的 dim 的顺序，改变t 的轴的顺序。如果 dims = None，则按顺序反转 t 的轴。

    :param t: 输入 QTensor 。
    :param dim: 维度的新顺序（整数列表）。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2,2,3]).astype(np.float32)
        t = QTensor(a)
        tt = tensor.permute(t,[2,0,1])
        print(tt)
        
        # [
        # [[0.0000000, 3.0000000],
        #  [6.0000000, 9.0000000]],
        # [[1.0000000, 4.0000000],
        #  [7.0000000, 10.0000000]],
        # [[2.0000000, 5.0000000],
        #  [8.0000000, 11.0000000]]
        # ]
                
        

transpose
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.transpose(t: pyvqnet.tensor.QTensor, dim: list)

    根据输入的 dim 的顺序，改变t 的轴的顺序。如果 dims = None，则按顺序反转 t 的轴。该函数功能与 permute 一致。

    :param t: 输入 QTensor 。
    :param dim: 维度的新顺序（整数列表）。

    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2,2,3]).astype(np.float32)
        t = QTensor(a)
        tt = tensor.transpose(t,[2,0,1])
        print(tt)

        # [
        # [[0.0000000, 3.0000000],
        #  [6.0000000, 9.0000000]],
        # [[1.0000000, 4.0000000],
        #  [7.0000000, 10.0000000]],
        # [[2.0000000, 5.0000000],
        #  [8.0000000, 11.0000000]]
        # ]
        

tile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tile(t: pyvqnet.tensor.QTensor, reps: list)


    通过按照 reps 给出的次数复制输入 QTensor 。
    
    如果 reps 的长度为 d，则结果 QTensor 的维度大小为 max(d, t.ndim)。如果 t.ndim < d，则通过从起始维度插入新轴，将 t 扩展为 d 维度。
    
    因此形状 (3,) 数组被提升为 (1, 3) 用于 2-D 复制，或形状 (1, 1, 3) 用于 3-D 复制。如果 t.ndim > d，reps 通过插入 1 扩展为 t.ndim。

    因此，对于形状为 (2, 3, 4, 5) 的 t，(4, 3) 的 reps 被视为 (1, 1, 4, 3)。

    :param t: 输入 QTensor 。
    :param reps: 每个维度的重复次数。
    :return: 一个新的 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        import numpy as np
        a = np.arange(6).reshape(2,3).astype(np.float32)
        A = QTensor(a)
        reps = [2,2]
        B = tensor.tile(A,reps)
        print(B)

        # [
        # [0.0000000, 1.0000000, 2.0000000, 0.0000000, 1.0000000, 2.0000000],
        # [3.0000000, 4.0000000, 5.0000000, 3.0000000, 4.0000000, 5.0000000],
        # [0.0000000, 1.0000000, 2.0000000, 0.0000000, 1.0000000, 2.0000000],
        # [3.0000000, 4.0000000, 5.0000000, 3.0000000, 4.0000000, 5.0000000]
        # ]
        

squeeze
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.squeeze(t: pyvqnet.tensor.QTensor, axis: int = - 1)

    删除 axis 指定的轴，该轴的维度为1。如果 axis = None ，则将输入所有长度为1的维度删除。

    :param t: 输入 QTensor 。
    :param axis: 要压缩的轴，默认为None。 
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(6).reshape(1,6,1).astype(np.float32)
        A = QTensor(a)
        AA = tensor.squeeze(A,0)
        print(AA)

        # [
        # [0.0000000],
        # [1.0000000],
        # [2.0000000],
        # [3.0000000],
        # [4.0000000],
        # [5.0000000]
        # ]
        

unsqueeze
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.unsqueeze(t: pyvqnet.tensor.QTensor, axis: int = 0)

    在axis 指定的维度上插入一个维度为的1的轴，返回一个新的 QTensor 。

    :param t: 输入 QTensor 。
    :param axis: 要插入维度的位置，默认为0。 
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(24).reshape(2,1,1,4,3).astype(np.float32)
        A = QTensor(a)
        AA = tensor.unsqueeze(A,1)
        print(AA)

        # [
        # [[[[[0.0000000, 1.0000000, 2.0000000],
        #  [3.0000000, 4.0000000, 5.0000000],
        #  [6.0000000, 7.0000000, 8.0000000],
        #  [9.0000000, 10.0000000, 11.0000000]]]]],
        # [[[[[12.0000000, 13.0000000, 14.0000000],
        #  [15.0000000, 16.0000000, 17.0000000],
        #  [18.0000000, 19.0000000, 20.0000000],
        #  [21.0000000, 22.0000000, 23.0000000]]]]]
        # ]
        

swapaxis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.swapaxis(t, axis1: int, axis2: int)

    交换输入 t 的 第 axis1 和 axis 维度。

    :param t: 输入 QTensor 。
    :param axis1: 要交换的第一个轴。
    :param axis2:  要交换的第二个轴。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        import numpy as np
        a = np.arange(24).reshape(2,3,4).astype(np.float32)
        A = QTensor(a)
        AA = tensor.swapaxis(A, 2, 1)
        print(AA)

        # [
        # [[0.0000000, 4.0000000, 8.0000000],
        #  [1.0000000, 5.0000000, 9.0000000],
        #  [2.0000000, 6.0000000, 10.0000000],
        #  [3.0000000, 7.0000000, 11.0000000]],
        # [[12.0000000, 16.0000000, 20.0000000],
        #  [13.0000000, 17.0000000, 21.0000000],
        #  [14.0000000, 18.0000000, 22.0000000],
        #  [15.0000000, 19.0000000, 23.0000000]]
        # ]

masked_fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.masked_fill(t, mask, value)

    在 mask == 1 的位置，用值 value 填充输入。
    mask的形状必须与输入的 QTensor 的形状是可广播的。

    :param t: 输入 QTensor。
    :param mask: 掩码 QTensor,必须是kbool。
    :param value: 填充值。
    :return: 一个 QTensor。

    Examples::

        from pyvqnet.tensor import tensor
        import numpy as np
        a = tensor.ones([2, 2, 2, 2])
        mask = np.random.randint(0, 2, size=4).reshape([2, 2])
        b = tensor.QTensor(mask==1)
        c = tensor.masked_fill(a, b, 13)
        print(c)
        # [
        # [[[1.0000000, 1.0000000],  
        #  [13.0000000, 13.0000000]],
        # [[1.0000000, 1.0000000],   
        #  [13.0000000, 13.0000000]]],
        # [[[1.0000000, 1.0000000],
        #  [13.0000000, 13.0000000]],
        # [[1.0000000, 1.0000000],
        #  [13.0000000, 13.0000000]]]
        # ]

flatten
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.flatten(t: pyvqnet.tensor.QTensor, start: int = 0, end: int = - 1)

    将输入 t 从 start 到 end 的连续维度展平。

    :param t: 输入 QTensor 。
    :param start: 展平开始的轴，默认 = 0，从第一个轴开始。
    :param end: 展平结束的轴，默认 = -1，以最后一个轴结束。
    :return: 输出 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.flatten(t)
        print(x)

        # [1.0000000, 2.0000000, 3.0000000]


reshape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.reshape(t: pyvqnet.tensor.QTensor,new_shape)

    改变 QTensor 的形状，返回一个新的张量。

    :param t: 输入 QTensor 。
    :param new_shape: 新的形状。

    :return: 新形状的 QTensor 。

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t = QTensor(a)
        reshape_t = tensor.reshape(t, [C, R])
        print(reshape_t)
        # [
        # [0.0000000, 1.0000000, 2.0000000],
        # [3.0000000, 4.0000000, 5.0000000],
        # [6.0000000, 7.0000000, 8.0000000],
        # [9.0000000, 10.0000000, 11.0000000]
        # ]

flip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.flip(t, flip_dims)

    沿指定轴反转QTensor，返回一个新的张量。

    :param t: 输入 QTensor 。
    :param flip_dims: 需要翻转的轴或轴列表。

    :return: 新形状的 QTensor 。

    Example::

        from pyvqnet import tensor
        t = tensor.arange(1, 3 * 2 *2 * 2 + 1).reshape([3, 2, 2, 2])
        t.requires_grad = True
        y = tensor.flip(t, [0, -1])
        print(y)
        # [
        # [[[18.0000000, 17.0000000], 
        #  [20.0000000, 19.0000000]], 
        # [[22.0000000, 21.0000000],  
        #  [24.0000000, 23.0000000]]],
        # [[[10.0000000, 9.0000000],  
        #  [12.0000000, 11.0000000]], 
        # [[14.0000000, 13.0000000],  
        #  [16.0000000, 15.0000000]]],
        # [[[2.0000000, 1.0000000],   
        #  [4.0000000, 3.0000000]],   
        # [[6.0000000, 5.0000000],    
        #  [8.0000000, 7.0000000]]]   
        # ]

gather
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.gather(t, dim, index)

    沿由“dim”指定的轴收集值。

    对于 3-D 张量，输出由以下指定：

    .. math::

         out[i][j][k] = t[index[i][j][k]][j][k] , 如果 dim == 0 \\

         out[i][j][k] = t[i][index[i][j][k]][k] , 如果 dim == 1 \\

         out[i][j][k] = t[i][j][index[i][j][k]] , 如果 dim == 2 \\

    :param t: 输入 QTensor。
    :param dim: 聚集轴。
    :param index: 索引QTensor，应该与输入具有相同的维度大小。

    :return: 聚集的结果

    Example::

        from pyvqnet.tensor import gather,QTensor,tensor
        import numpy as np
        np.random.seed(25)
        npx = np.random.randn( 3, 4,6)
        npindex = np.array([2,3,1,2,1,2,3,0,2,3,1,2,3,2,0,1]).reshape([2,2,4]).astype(np.int64)

        x1 = QTensor(npx)
        indices1 =  QTensor(npindex)
        x1.requires_grad = True
        y1 = gather(x1,1,indices1)
        y1.backward(tensor.arange(0,y1.numel()).reshape(y1.shape))

        print(y1)
        # [
        # [[2.1523438, -0.4196777, -2.0527344, -1.2460938],
        #  [-0.6201172, -1.3349609, 2.2949219, -0.5913086]],
        # [[0.2170410, -0.7055664, 1.6074219, -1.9394531],
        #  [0.2430420, -0.6333008, 0.5332031, 0.3881836]]
        # ]


scatter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.scatter(input, dim, index,src)


    将张量 src 中的所有值写入 indices 张量中指定的索引处的 input 中。

    对于 3-D 张量，输出由以下指定：

    .. math::

        input[indices[i][j][k]][j][k] = src[i][j][k] , 如果 dim == 0 \\
        input[i][indices[i][j][k]][k] = src[i][j][k] , 如果 dim == 1 \\
        input[i][j][indices[i][j][k]] = src[i][j][k] , 如果 dim == 2 \\

    :param input: 输入QTensor。
    :param dim: 散点轴。
    :param indices: 索引QTensor，应该和输入有相同的维度大小。
    :param src: 要散布的源张量。

    Example::

        from pyvqnet.tensor import scatter, QTensor
        import numpy as np
        np.random.seed(25)
        npx = np.random.randn(3, 2, 4, 2)
        npindex = np.array([2, 3, 1, 2, 1, 2, 3, 0, 2, 3, 1, 2, 3, 2, 0,
                            1]).reshape([2, 2, 4, 1]).astype(np.int64)
        x1 = QTensor(npx)
        npsrc = QTensor(np.full_like(npindex, 200), dtype=x1.dtype)
        npsrc.requires_grad = True
        indices1 = QTensor(npindex)
        y1 = scatter(x1, 2, indices1, npsrc)
        print(y1)

        # [[[[  0.2282731   1.0268903]
        #    [200.         -0.5911815]
        #    [200.         -0.2223257]
        #    [200.          1.8379046]]

        #   [[200.          0.8685831]
        #    [200.         -0.2323119]
        #    [200.         -1.3346615]
        #    [200.         -1.2460893]]]


        #  [[[  1.2022723  -1.0499416]
        #    [200.         -0.4196777]
        #    [200.         -2.5944874]
        #    [200.          0.6808889]]

        #   [[200.         -1.9762536]
        #    [200.         -0.2908697]
        #    [200.          1.9826261]
        #    [200.         -1.839905 ]]]


        #  [[[  1.6076708   0.3882919]
        #    [  0.3997321   0.4054766]
        #    [  0.2170018  -0.6334391]
        #    [  0.2466215  -1.9395455]]

        #   [[  0.1140596  -1.8853414]
        #    [  0.2430805  -0.7054807]
        #    [  0.3646276  -0.5029522]
        #    [ -0.2257515  -0.5655377]]]]

broadcast_to
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.broadcast_to(t, ref)

    受到某些约束，数组 t 被“广播”到参考形状，以便它们具有兼容的形状。

    https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t: 输入QTensor
    :param ref: 参考形状。
    
    :return: 新广播的 t 的 QTensor。

    Example::

        from pyvqnet.tensor.tensor import QTensor
        from pyvqnet.tensor import *
        ref = [2,3,4]
        a = ones([4])
        b = tensor.broadcast_to(a,ref)
        print(b.shape)
        #[2, 3, 4]



实用函数
-----------------------------


to_tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.to_tensor(x)

    将输入数值或 numpy.ndarray 等转换为 QTensor 。

    :param x: 整数、浮点数、布尔数、复数、或 numpy.ndarray
    :return: 输出 QTensor

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.to_tensor(10.0)
        print(t)

        # [10.0000000]
        

pad_sequence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.pad_sequence(qtensor_list, batch_first=False, padding_value=0)

    用 ``padding_value`` 填充可变长度张量列表。 ``pad_sequence`` 沿新维度堆叠张量列表，并将它们填充到相等的长度。
    输入是列表大小为 ``L x *`` 的序列。 L 是可变长度。

    :param qtensor_list: `list[QTensor]`- 可变长度序列列表。
    :param batch_first: 'bool' - 如果为真，输出将是 ``批大小 x 最长序列长度 x *`` ，否则为 ``最长序列长度 x 批大小 x *`` 。 默认值: False。
    :param padding_value: 'float' - 填充值。 默认值：0。

    :return:
        如果 batch_first 为 ``False``，则张量大小为 ``批大小 x 最长序列长度 x *``。
        否则张量的大小为 ``最长序列长度 x 批大小 x *`` 。

    Examples::

        from pyvqnet.tensor import tensor
        a = tensor.ones([4, 2,3])
        b = tensor.ones([1, 2,3])
        c = tensor.ones([2, 2,3])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], True)

        print(y)
        # [
        # [[[1.0000000, 1.0000000, 1.0000000],
        #  [1.0000000, 1.0000000, 1.0000000]],
        # [[1.0000000, 1.0000000, 1.0000000],
        #  [1.0000000, 1.0000000, 1.0000000]],
        # [[1.0000000, 1.0000000, 1.0000000],
        #  [1.0000000, 1.0000000, 1.0000000]],
        # [[1.0000000, 1.0000000, 1.0000000],
        #  [1.0000000, 1.0000000, 1.0000000]]],
        # [[[1.0000000, 1.0000000, 1.0000000],
        #  [1.0000000, 1.0000000, 1.0000000]],
        # [[0.0000000, 0.0000000, 0.0000000],
        #  [0.0000000, 0.0000000, 0.0000000]],
        # [[0.0000000, 0.0000000, 0.0000000],
        #  [0.0000000, 0.0000000, 0.0000000]],
        # [[0.0000000, 0.0000000, 0.0000000],
        #  [0.0000000, 0.0000000, 0.0000000]]],
        # [[[1.0000000, 1.0000000, 1.0000000],
        #  [1.0000000, 1.0000000, 1.0000000]],
        # [[1.0000000, 1.0000000, 1.0000000],
        #  [1.0000000, 1.0000000, 1.0000000]],
        # [[0.0000000, 0.0000000, 0.0000000],
        #  [0.0000000, 0.0000000, 0.0000000]],
        # [[0.0000000, 0.0000000, 0.0000000],
        #  [0.0000000, 0.0000000, 0.0000000]]]
        # ]


pad_packed_sequence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.pad_packed_sequence(sequence, batch_first=False, padding_value=0, total_length=None)

    填充一批打包的可变长度序列。它是 `pack_pad_sequence` 的逆操作。
    当  ``batch_first`` 是 True，它将返回  ``B x T x *`` 形状的张量，否则返回  ``T x B x *``。
    其中 `T` 为序列最长长度, `B` 为批处理大小。



    :param sequence: 'QTensor' - 待处理数据。
    :param batch_first: 'bool' - 如果为 ``True`` ，批处理将是输入的第一维。 默认值：False。
    :param padding_value: 'bool' - 填充值。默认:0。
    :param total_length: 'bool' - 如果不是 ``None`` ，输出将被填充到长度 :attr:`total_length`。 默认值：None。
    :return:
        包含填充序列的张量元组，以及批次中每个序列的长度列表。批次元素将按照最初的顺序重新排序。

    Examples::

        from pyvqnet.tensor import tensor
        a = tensor.ones([4, 2,3])
        b = tensor.ones([2, 2,3])
        c = tensor.ones([1, 2,3])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], True)
        seq_len = [4, 2, 1]
        data = tensor.pack_pad_sequence(y,
                                seq_len,
                                batch_first=True,
                                enforce_sorted=True)

        seq_unpacked, lens_unpacked = tensor.pad_packed_sequence(data, batch_first=True)
        print(seq_unpacked)
        # [
        # [[1.0000000, 1.0000000],
        #  [1.0000000, 1.0000000],
        #  [1.0000000, 1.0000000]],
        # [[1.0000000, 1.0000000],
        #  [0.0000000, 0.0000000],
        #  [1.0000000, 1.0000000]],
        # [[1.0000000, 1.0000000],
        #  [0.0000000, 0.0000000],
        #  [0.0000000, 0.0000000]],
        # [[1.0000000, 1.0000000],
        #  [0.0000000, 0.0000000],
        #  [0.0000000, 0.0000000]]
        # ]
        print(lens_unpacked)
        # [4 1 2]


pack_pad_sequence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.pack_pad_sequence(input, lengths, batch_first=False, enforce_sorted=True)

    打包一个包含可变长度填充序列的张量。
    如果 batch_first 是 True, `input` 的形状应该为 [批大小,长度,*]，否则形状 [长度，批大小,*]。

    对于未排序的序列，使用 ``enforce_sorted`` 是 False。 如果 :attr:`enforce_sorted` 是 ``True``，序列应该按长度降序排列。

    :param input: 'QTensor' - 填充的可变长度序列。
    :param lengths: 'list' - 每个批次的序列长度。
    :param batch_first: 'bool' - 如果 ``True``，则输入预期为 ``B x T x *``
        格式，默认：False。
    :param enforce_sorted: 'bool' - 如果 ``True``，输入应该是
        包含按长度降序排列的序列。 如果 ``False``，输入将无条件排序。 默认值：True。

    :return: 一个 :class:`PackedSequence` 对象。

    Examples::

        from pyvqnet.tensor import tensor
        a = tensor.ones([4, 2,3])
        b = tensor.ones([2, 2,3])
        c = tensor.ones([1, 2,3])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], True)
        seq_len = [4, 2, 1]
        data = tensor.pack_pad_sequence(y,
                                seq_len,
                                batch_first=True,
                                enforce_sorted=False)
        print(data.data)
        print(data.batch_sizes)
        print(data.sort_indice)
        print(data.unsorted_indice)

        # [
        # [[1.0000000, 1.0000000, 1.0000000],
        #  [1.0000000, 1.0000000, 1.0000000]],
        # [[1.0000000, 1.0000000, 1.0000000],
        #  [1.0000000, 1.0000000, 1.0000000]],
        # [[1.0000000, 1.0000000, 1.0000000],
        #  [1.0000000, 1.0000000, 1.0000000]],
        # [[1.0000000, 1.0000000, 1.0000000],
        #  [1.0000000, 1.0000000, 1.0000000]],
        # [[1.0000000, 1.0000000, 1.0000000],
        #  [1.0000000, 1.0000000, 1.0000000]],
        # [[1.0000000, 1.0000000, 1.0000000],
        #  [1.0000000, 1.0000000, 1.0000000]],
        # [[1.0000000, 1.0000000, 1.0000000],
        #  [1.0000000, 1.0000000, 1.0000000]]
        # ]
        # [3, 2, 1, 1]
        # [0, 2, 1]
        # [0, 2, 1]