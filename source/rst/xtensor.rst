XTensor 模块
###########################

XTensor 是VQNet借助算子自动并行方式对张量计算进行加速的功能接口，接口支持CPU/GPU下经典计算，API定义与原先XTensor基本一致。

.. warning::

    XTensor相关功能属于实验功能，当前只支持经典神经网络计算，与前述介绍的基于QTensor的接口不能混用。
    如需要训练量子机器学习模型，请使用QTensor下相关接口。


例如下例中，使用reshape对a进行循环计算，由于这些reshape计算之间没有前后依赖关系，可以天然的进行并行计算。所以对于该例子中的100次reshape计算是自动异步计算的，达到加速的目的。

    Example::

        from pyvqnet.xtensor import xtensor,reshape
        a = xtensor([2, 3, 4, 5])
        for i in range(100):
            y = reshape(a,(2,2))


XTensor's 函数与属性
******************************************


ndim
===========================================================


.. py:attribute:: XTensor.ndim

    返回张量的维度的个数。
        
    :return: 张量的维度的个数。

    Example::

        from pyvqnet.xtensor import xtensor

        a = xtensor([2, 3, 4, 5])
        print(a.ndim)

        # 1
    
shape
===========================================================

.. py:attribute:: XTensor.shape

    返回张量的维度
    
    :return: 一个元组存有张量的维度

    Example::

        from pyvqnet.xtensor import xtensor

        a = xtensor([2, 3, 4, 5])
        print(a.shape)

        # (4)

size
===========================================================

.. py:attribute:: XTensor.size

    返回张量的元素个数。
    
    :return: 张量的元素个数。

    Example::

        from pyvqnet.xtensor import xtensor

        a = xtensor([2, 3, 4, 5])
        print(a.size)

        # 4

numel
===========================================================

.. py:method:: XTensor.numel

    返回张量的元素个数。
    
    :return: 张量的元素个数。

    Example::

        from pyvqnet.xtensor import xtensor

        a = xtensor([2, 3, 4, 5])
        print(a.numel())

        # 4

device
===========================================================

.. py:attribute:: XTensor.device

    返回XTensor存放的硬件位置。

    XTensor 硬件位置支持CPU device=0, 第一个GPU device=1000, 第2个GPU device=1001, ... 第10个GPU device=1009。

    :return: 张量的硬件位置。

    Example::

        from pyvqnet.xtensor import xtensor

        a = xtensor([2, 3, 4, 5])
        print(a.device)
        # 0

dtype
===========================================================

.. py:attribute:: XTensor.dtype

    返回张量的数据类型。

    XTensor 内部数据类型dtype支持kbool = 0, kuint8 = 1, kint8 = 2,kint32 = 4,
    kint64 = 5, kfloat32 = 6, kfloat64 = 7。如果使用列表进行初始化，默认为kfloat32。

    :return: 张量的数据类型。

    Example::

        from pyvqnet.xtensor import xtensor

        a = xtensor([2, 3, 4, 5])
        print(a.dtype)
        # 4

requires_grad
===========================================================

.. py:attribute:: XTensor.requires_grad

    设置和获取该XTensor是否需要计算梯度。

    .. note::

        XTensor 如果希望计算梯度，需要显式地设置requires_grad = True。

    Example::

        from pyvqnet.xtensor import xtensor

        a = xtensor([2, 3, 4, 5.0])
        a.requires_grad = True
        print(a.grad)


backward
===========================================================

.. py:method:: XTensor.backward(grad=None)

    利用反向传播算法，计算当前张量所在的计算图中的所有需计算梯度的张量的梯度。

    .. note::

        对于xtensor下的接口，需要使用 `with autograd.tape()` 将所有希望进行自动微分的操作纳入其中，并且这些操作不包含in-place的操作，例如：
        a+=1, a[:]=1, 也不包含数据的复制，例如toGPU(),toCPU()等。

    :return: 无

    Example::

        from pyvqnet.xtensor import xtensor,autograd

        target = xtensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0.2]])
        target.requires_grad=True
        with autograd.tape():
            y = 2*target + 3
            y.backward()
        print(target.grad)
        #[[2. 2. 2. 2. 2. 2. 2. 2. 2. 2.]]

to_numpy
===========================================================

.. py:method:: XTensor.to_numpy()

    将张量的数据拷贝到一个numpy.ndarray里面。

    :return: 一个新的 numpy.ndarray 包含 XTensor 数据

    Example::

        from pyvqnet.xtensor import xtensor
        t3 = xtensor([2, 3, 4, 5])
        t4 = t3.to_numpy()
        print(t4)

        # [2. 3. 4. 5.]

item
===========================================================

.. py:method:: XTensor.item()

    从只包含单个元素的 XTensor 返回唯一的元素。

    :return: 元素值

    Example::

        from pyvqnet.xtensor import ones

        t = ones([1])
        print(t.item())

        # 1.0

argmax
===========================================================

.. py:method:: XTensor.argmax(*kargs)

    返回输入 XTensor 中所有元素的最大值的索引，或返回 XTensor 按某一维度的最大值的索引。

    :param dim: 计算argmax的轴，只接受单个维度。 如果 dim == None，则返回输入张量中所有元素的最大值的索引。有效的 dim 范围是 [-R, R)，其中 R 是输入的 ndim。 当 dim < 0 时，它的工作方式与 dim + R 相同。
    :param keepdims: 输出 XTensor 是否保留了最大值索引操作的轴，默认是False。

    :return: 输入 XTensor 中最大值的索引。

    Example::

        from pyvqnet.xtensor import XTensor
        a = XTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092, 0.5419, -0.2993, 0.3195]])

        flag = a.argmax()
        print(flag)
        
        # [0.]

        flag_0 = a.argmax(0, True)
        print(flag_0)

        # [
        # [0., 3., 0., 3.]
        # ]

        flag_1 = a.argmax(1, True)
        print(flag_1)

        # [
        # [0.],
        # [2.],
        # [0.],
        # [1.]
        # ]

argmin
===========================================================

.. py:method:: XTensor.argmin(*kargs)

    返回输入 XTensor 中所有元素的最小值的索引，或返回 XTensor 按某一维度的最小值的索引。

    :param dim: 计算argmax的轴，只接受单个维度。 如果 dim == None，则返回输入张量中所有元素的最小值的索引。有效的 dim 范围是 [-R, R)，其中 R 是输入的 ndim。 当 dim < 0 时，它的工作方式与 dim + R 相同。
    :param keepdims: 输出 XTensor 是否保留了最小值索引操作的轴，默认是False。

    :return: 输入 XTensor 中最小值的索引。

    Example::

        
        from pyvqnet.xtensor import XTensor
        a = XTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                    [-0.7401, -0.8805, -0.3402, -1.1936],
                    [0.4907, -1.3948, -1.0691, -0.3132],
                    [-1.6092, 0.5419, -0.2993, 0.3195]])
        flag = a.argmin()
        print(flag)

        # [12.]

        flag_0 = a.argmin(0, True)
        print(flag_0)

        # [
        # [3., 2., 2., 1.]
        # ]

        flag_1 = a.argmin(1, False)
        print(flag_1)

        # [2., 3., 1., 0.]

all
===========================================================

.. py:method:: XTensor.all()

    判断张量内数据是否全为全零。

    :return: 返回True，如果全为非0;否则返回False。

    Example::

        import pyvqnet.xtensor as xtensor
        shape = [2, 3]
        t = xtensor.full(shape,1)
        flag = t.all()
        print(flag)

        #True
        #<XTensor  cpu(0) kbool>

any
===========================================================

.. py:method:: XTensor.any()

    判断张量内数据是否有任意元素不为0。

    :return: 返回True，如果有任意元素不为0;否则返回False。

    Example::

        import pyvqnet.xtensor as xtensor
        shape = [2, 3]
        t = xtensor.full(shape,1)
        flag = t.any()
        print(flag)

        #True
        #<XTensor  cpu(0) kbool>


fill_rand_binary\_
===========================================================

.. py:method:: XTensor.fill_rand_binary_(v=0.5)

    用从二项分布中随机采样的值填充 XTensor 。

    如果二项分布后随机生成的数据大于二值化阈值 v ，则设置 XTensor 对应位置的元素值为1，否则为0。

    :param v: 二值化阈值，默认0.5。

    :return: 无。

    Example::

        
        from pyvqnet.xtensor import XTensor
        import numpy as np
        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = XTensor(a)
        t.fill_rand_binary_(2)
        print(t)

        # [
        # [1., 1., 1.],
        # [1., 1., 1.]
        # ]

fill_rand_signed_uniform\_
===========================================================

.. py:method:: XTensor.fill_rand_signed_uniform_(v=1)

    用从有符号均匀分布中随机采样的值填充 XTensor 。用缩放因子 v 对生成的随机采样的值进行缩放。

    :param v: 缩放因子，默认1。

    :return: 无。

    Example::

        
        from pyvqnet.xtensor import XTensor
        import numpy as np
        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = XTensor(a)
        value = 42

        t.fill_rand_signed_uniform_(value)
        print(t)

        # [[ 4.100334   7.7989464 18.075905 ]
        #  [28.918327   8.632122  30.067429 ]]
        # <XTensor 2x3 cpu(0) kfloat32>


fill_rand_uniform\_
===========================================================

.. py:method:: XTensor.fill_rand_uniform_(v=1)

    用从均匀分布中随机采样的值填充 XTensor 。用缩放因子 v 对生成的随机采样的值进行缩放。

    :param v: 缩放因子，默认1。

    :return: 无。

    Example::

        
        from pyvqnet.xtensor import XTensor
        import numpy as np
        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = XTensor(a)
        value = 42
        t.fill_rand_uniform_(value)
        print(t)

        # [[23.050167 24.899473 30.037952]
        #  [35.459164 25.316061 36.033714]]
        # <XTensor 2x3 cpu(0) kfloat32>


fill_rand_normal\_
===========================================================

.. py:method:: XTensor.fill_rand_normal_(m=0, s=1)

    生成均值为 m 和方差 s 产生正态分布元素，并填充到张量中。

    :param m: 均值，默认0。
    :param s: 方差，默认1。

    :return: 无。

    Example::

        
        from pyvqnet.xtensor import XTensor
        import numpy as np
        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = XTensor(a)
        t.fill_rand_normal_(2, 10)
        print(t)

        # [[13.630787   6.838046   4.9956346]
        #  [ 3.5302546 -9.688148  17.580711 ]]
        # <XTensor 2x3 cpu(0) kfloat32>


XTensor.transpose
===========================================================

.. py:method:: XTensor.transpose(*axes)

    反转张量的轴。如果 new_dims = None，则反转所有轴。

    :param axes: 列表形式储存的新的轴顺序。

    :return:  新的 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2, 2, 3]).astype(np.float32)
        t = XTensor(a)
        rlt = t.transpose([2,0,1])
        print(rlt)

        rlt = t.transpose()
        print(rlt)
        """
        [[[ 0.  3.]
        [ 6.  9.]]

        [[ 1.  4.]
        [ 7. 10.]]

        [[ 2.  5.]
        [ 8. 11.]]]
        <XTensor 3x2x2 cpu(0) kfloat32>

        [[[ 0.  6.]
        [ 3.  9.]]

        [[ 1.  7.]
        [ 4. 10.]]

        [[ 2.  8.]
        [ 5. 11.]]]
        <XTensor 3x2x2 cpu(0) kfloat32>
        """

XTensor.reshape
===========================================================

.. py:method:: XTensor.reshape(new_shape)

    改变 XTensor 的形状，返回一个新的张量。

    :param new_shape: 新的形状。

    :return: 新形状的 XTensor 。

    Example::

        
        from pyvqnet.xtensor import XTensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C)
        t = XTensor(a)
        reshape_t = t.reshape([C, R])
        print(reshape_t)
        # [
        # [0., 1., 2.],
        # [3., 4., 5.],
        # [6., 7., 8.],
        # [9., 10., 11.]
        # ]


getdata
===========================================================

.. py:method:: XTensor.getdata()

    返回一个numpy.ndarray 浅拷贝表示XTensor中的数据，如果原数据在GPU上，则会首先返回CPU上的XTensor复制的ndarray视图。

    :return: 包含当前 XTensor 数据的numpy.ndarray浅拷贝。

    Example::

        import pyvqnet.xtensor  as xtensor
        t = xtensor.ones([3, 4])
        a = t.getdata()
        print(a)

        # [[1. 1. 1. 1.]
        #  [1. 1. 1. 1.]
        #  [1. 1. 1. 1.]]

__getitem__
===========================================================

.. py:method:: XTensor.__getitem__()

    支持对 XTensor 使用切片索引，下标，或使用 XTensor 作为高级索引访问输入。该操作返回一个新的 XTensor 。

    通过冒号 ``:``  分隔切片参数 start:stop:step 来进行切片操作，其中 start、stop、step 均可缺省。

    针对1-D XTensor ，则仅有单个轴上的索引或切片。

    针对2-D及以上的 XTensor ，则会有多个轴上的索引或切片。

    使用 XTensor 作为 索引，则进行高级索引，请参考numpy中 `高级索引 <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ 部分。

    若作为索引的 XTensor 为逻辑运算的结果，则进行 布尔数组索引。

    :param item: 以 pyslice , 整数, XTensor 构成切片索引。

    :return: 新的 XTensor。

    Example::

        import pyvqnet.xtensor as tensor
        from pyvqnet.xtensor import XTensor
        aaa = tensor.arange(1, 61).reshape([4, 5, 3])

        print(aaa[0:2, 3, :2])

        print(aaa[3, 4, 1])

        print(aaa[3][4][1])

        print(aaa[:, 2, :])

        print(aaa[2])

        print(aaa[0:2, ::3, 2:])

        a = tensor.ones([2, 2])
        b = XTensor([[1, 1], [0, 1]])
        b = b > 0
        c = a[b]
        print(c)

        tt = tensor.arange(1, 56 * 2 * 4 * 4 + 1).reshape([2, 8, 4, 7, 4])
        tt.requires_grad = True
        index_sample1 = tensor.arange(0, 3).reshape([3, 1])
        index_sample2 = XTensor([0, 1, 0, 2, 3, 2, 2, 3, 3]).reshape([3, 3])
        gg = tt[:, index_sample1, 3:, index_sample2, 2:]
        """
        [[10. 11.]
        [25. 26.]]
        <XTensor 2x2 cpu(0) kfloat32>

        [59.]
        <XTensor 1 cpu(0) kfloat32>

        [59.]
        <XTensor 1 cpu(0) kfloat32>

        [[ 7.  8.  9.]
        [22. 23. 24.]
        [37. 38. 39.]
        [52. 53. 54.]]
        <XTensor 4x3 cpu(0) kfloat32>

        [[31. 32. 33.]
        [34. 35. 36.]
        [37. 38. 39.]
        [40. 41. 42.]
        [43. 44. 45.]]
        <XTensor 5x3 cpu(0) kfloat32>

        [[[ 3.]
        [12.]]

        [[18.]
        [27.]]]
        <XTensor 2x2x1 cpu(0) kfloat32>

        [1. 1. 1.]
        <XTensor 3 cpu(0) kfloat32>

        [[[[[  87.   88.]]

        [[ 983.  984.]]]


        [[[  91.   92.]]

        [[ 987.  988.]]]


        [[[  87.   88.]]

        [[ 983.  984.]]]]



        [[[[ 207.  208.]]

        [[1103. 1104.]]]


        [[[ 211.  212.]]

        [[1107. 1108.]]]


        [[[ 207.  208.]]

        [[1103. 1104.]]]]



        [[[[ 319.  320.]]

        [[1215. 1216.]]]


        [[[ 323.  324.]]

        [[1219. 1220.]]]


        [[[ 323.  324.]]

        [[1219. 1220.]]]]]
        <XTensor 3x3x2x1x2 cpu(0) kfloat32>
        """

__setitem__
===========================================================

.. py:method:: XTensor.__setitem__()

    支持对 XTensor 使用切片索引，下标，或使用 XTensor 作为高级索引修改输入。该操作对输入原地进行修改 。

    通过冒号 ``:``  分隔切片参数 start:stop:step 来进行切片操作，其中 start、stop、step 均可缺省。

    针对1-D XTensor，则仅有单个轴上的索引或切片。

    针对2-D及以上的 XTensor ，则会有多个轴上的索引或切片。

    使用 XTensor 作为 索引，则进行高级索引，请参考numpy中 `高级索引 <https://docs.scipy.org/doc/numpy-1.10.1/reference/arrays.indexing.html>`_ 部分。

    若作为索引的 XTensor 为逻辑运算的结果，则进行 布尔数组索引。

    :param item: 以 pyslice , 整数, XTensor 构成切片索引。

    :return: 无。

    Example::

        import pyvqnet.xtensor as tensor
        aaa = tensor.arange(1, 61)
        aaa = aaa.reshape([4, 5, 3])
        vqnet_a2 = aaa[3, 4, 1]
        aaa[3, 4, 1] = tensor.arange(10001,
                                        10001 + vqnet_a2.size).reshape(vqnet_a2.shape)
        print(aaa)
        # [
        # [[1., 2., 3.],    
        #  [4., 5., 6.],    
        #  [7., 8., 9.],    
        #  [10., 11., 12.], 
        #  [13., 14., 15.]],
        # [[16., 17., 18.], 
        #  [19., 20., 21.], 
        #  [22., 23., 24.], 
        #  [25., 26., 27.], 
        #  [28., 29., 30.]],
        # [[31., 32., 33.], 
        #  [34., 35., 36.],
        #  [37., 38., 39.],
        #  [40., 41., 42.],
        #  [43., 44., 45.]],
        # [[46., 47., 48.],
        #  [49., 50., 51.],
        #  [52., 53., 54.],
        #  [55., 56., 57.],
        #  [58., 10001., 60.]]
        # ]
        aaa = tensor.arange(1, 61)
        aaa = aaa.reshape([4, 5, 3])
        vqnet_a3 = aaa[:, 2, :]
        aaa[:, 2, :] = tensor.arange(10001,
                                        10001 + vqnet_a3.size).reshape(vqnet_a3.shape)
        print(aaa)
        # [
        # [[1., 2., 3.],
        #  [4., 5., 6.],
        #  [10001., 10002., 10003.],
        #  [10., 11., 12.],
        #  [13., 14., 15.]],
        # [[16., 17., 18.],
        #  [19., 20., 21.],
        #  [10004., 10005., 10006.],
        #  [25., 26., 27.],
        #  [28., 29., 30.]],
        # [[31., 32., 33.],
        #  [34., 35., 36.],
        #  [10007., 10008., 10009.],
        #  [40., 41., 42.],
        #  [43., 44., 45.]],
        # [[46., 47., 48.],
        #  [49., 50., 51.],
        #  [10010., 10011., 10012.],
        #  [55., 56., 57.],
        #  [58., 59., 60.]]
        # ]
        aaa = tensor.arange(1, 61)
        aaa = aaa.reshape([4, 5, 3])
        vqnet_a4 = aaa[2, :]
        aaa[2, :] = tensor.arange(10001,
                                    10001 + vqnet_a4.size).reshape(vqnet_a4.shape)
        print(aaa)
        # [
        # [[1., 2., 3.],
        #  [4., 5., 6.],
        #  [7., 8., 9.],
        #  [10., 11., 12.],
        #  [13., 14., 15.]],
        # [[16., 17., 18.],
        #  [19., 20., 21.],
        #  [22., 23., 24.],
        #  [25., 26., 27.],
        #  [28., 29., 30.]],
        # [[10001., 10002., 10003.],
        #  [10004., 10005., 10006.],
        #  [10007., 10008., 10009.],
        #  [10010., 10011., 10012.],
        #  [10013., 10014., 10015.]],
        # [[46., 47., 48.],
        #  [49., 50., 51.],
        #  [52., 53., 54.],
        #  [55., 56., 57.],
        #  [58., 59., 60.]]
        # ]
        aaa = tensor.arange(1, 61)
        aaa = aaa.reshape([4, 5, 3])
        vqnet_a5 = aaa[0:2, ::2, 1:2]
        aaa[0:2, ::2,
            1:2] = tensor.arange(10001,
                                    10001 + vqnet_a5.size).reshape(vqnet_a5.shape)
        print(aaa)
        # [
        # [[1., 10001., 3.],
        #  [4., 5., 6.],
        #  [7., 10002., 9.],
        #  [10., 11., 12.],
        #  [13., 10003., 15.]],
        # [[16., 10004., 18.],
        #  [19., 20., 21.],
        #  [22., 10005., 24.],
        #  [25., 26., 27.],
        #  [28., 10006., 30.]],
        # [[31., 32., 33.],
        #  [34., 35., 36.],
        #  [37., 38., 39.],
        #  [40., 41., 42.],
        #  [43., 44., 45.]],
        # [[46., 47., 48.],
        #  [49., 50., 51.],
        #  [52., 53., 54.],
        #  [55., 56., 57.],
        #  [58., 59., 60.]]
        # ]
        a = tensor.ones([2, 2])
        b = tensor.XTensor([[1, 1], [0, 1]])
        b = b > 0
        x = tensor.XTensor([1001, 2001, 3001])

        a[b] = x
        print(a)
        # [
        # [1001., 2001.],
        #  [1., 3001.]
        # ]


GPU
===========================================================

.. py:function:: XTensor.GPU(device: int = DEV_GPU_0)

    复制XTensor数据到指定的GPU设备,返回一个新的XTensor

    device 指定存储其内部数据的设备。 当device >= DEV_GPU_0时，数据存储在GPU上。 
    如果您的计算机有多个 GPU，您可以指定不同的设备来存储数据。 例如，device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... 表示存储在具有不同序列号的GPU上。

    .. note::
        XTensor在不同GPU上无法进行计算。
        如果您尝试在 ID 超过验证 GPU 最大数量的 GPU 上创建 XTensor，将引发 Cuda 错误。
        注意，该接口会断开当前已构建的计算图。

    :param device: 当前保存XTensor的设备，默认=DEV_GPU_0，
     device = pyvqnet.DEV_GPU_0，存储在第一个 GPU 中，devcie = DEV_GPU_1，
     存储在第二个 GPU 中，依此类推。

    :return: XTensor 复制到 GPU 设备。

    Examples::

        from pyvqnet.xtensor import XTensor
        a = XTensor([2])
        b = a.GPU()
        print(b.device)
        #1000

CPU
===========================================================

.. py:function:: XTensor.CPU()

    复制XTensor到特定的CPU设备,返回一个新的XTensor

    .. note::
        XTensor在不同硬件上无法进行计算。
        注意，该接口会断开当前已构建的计算图。

    :return: XTensor 复制到 CPU 设备。

    Examples::

        from pyvqnet.xtensor import XTensor
        a = XTensor([2])
        b = a.CPU()
        print(b.device)
        # 0

toGPU
===========================================================

.. py:function:: XTensor.toGPU(device: int = DEV_GPU_0)

    移动XTensor到指定的GPU设备

    device 指定存储其内部数据的设备。 当device >= DEV_GPU时，数据存储在GPU上。
     如果您的计算机有多个 GPU，您可以指定不同的设备来存储数据。 
     例如，device = DEV_GPU_1, DEV_GPU_2, DEV_GPU_3, ... 表示存储在具有不同序列号的GPU上。

    .. note::
        XTensor在不同GPU上无法进行计算。
        如果您尝试在 ID 超过验证 GPU 最大数量的 GPU 上创建 XTensor，将引发 Cuda 错误。
        注意，该接口会断开当前已构建的计算图。

    :param device: 当前保存XTensor的设备，默认=DEV_GPU_0。device = pyvqnet.DEV_GPU_0，存储在第一个 GPU 中，devcie = DEV_GPU_1，存储在第二个 GPU 中，依此类推。
    :return: 当前XTensor。

    Examples::

        from pyvqnet.xtensor import XTensor
        a = XTensor([2])
        a = a.toGPU()
        print(a.device)
        #1000


toCPU
===========================================================

.. py:function:: XTensor.toCPU()

    移动XTensor到特定的GPU设备

    .. note::
        XTensor在不同硬件上无法进行计算。
        注意，该接口会断开当前已构建的计算图。

    :return: 当前XTensor。

    Examples::

        from pyvqnet.xtensor import XTensor
        a = XTensor([2])
        b = a.toCPU()
        print(b.device)
        # 0


isGPU
===========================================================

.. py:function:: XTensor.isGPU()

    该 XTensor 的数据是否存储在 GPU 主机内存上。

    :return: 该 XTensor 的数据是否存储在 GPU 主机内存上。

    Examples::
    
        from pyvqnet.xtensor import XTensor
        a = XTensor([2])
        a = a.isGPU()
        print(a)
        # False

isCPU
===========================================================

.. py:function:: XTensor.isCPU()

    该 XTensor 的数据是否存储在 CPU 主机内存上。

    :return: 该 XTensor 的数据是否存储在 CPU 主机内存上。

    Examples::
    
        from pyvqnet.xtensor import XTensor
        a = XTensor([2])
        a = a.isCPU()
        print(a)
        # True


创建函数
***********************

ones
===========================================================

.. py:function:: pyvqnet.xtensor.ones(shape,device=None,dtype=None)

    创建元素全一的 XTensor 。

    :param shape: 数据的形状。
    :param device: 储存在哪个设备上，默认: None，在CPU上。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 返回新的 XTensor 。

    Example::

        from pyvqnet.xtensor import ones

        x = ones([2, 3])
        print(x)

        # [
        # [1., 1., 1.],
        # [1., 1., 1.]
        # ]

ones_like
===========================================================

.. py:function:: pyvqnet.xtensor.ones_like(t: pyvqnet.xtensor.XTensor)

    创建元素全一的 XTensor ,形状和输入的 XTensor 一样。

    :param t: 输入 XTensor 。

    :return: 新的全一  XTensor 。

    Example::

        
        from pyvqnet.xtensor import XTensor,ones_like
        t = XTensor([1, 2, 3])
        x = ones_like(t)
        print(x)

        # [1., 1., 1.]


full
===========================================================

.. py:function:: pyvqnet.xtensor.full(shape, value, device=None, dtype=None)

    创建一个指定形状的 XTensor 并用特定值填充它。

    :param shape: 要创建的张量形状。
    :param value: 填充的值。
    :param device: 储存在哪个设备上，默认: None，在CPU上。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 输出新 XTensor 。 

    Example::

        
        from pyvqnet.xtensor import XTensor,full
        shape = [2, 3]
        value = 42
        t = full(shape, value)
        print(t)
        # [
        # [42., 42., 42.],
        # [42., 42., 42.]
        # ]


full_like
===========================================================

.. py:function:: pyvqnet.xtensor.full_like(t, value)

    创建一个形状和输入一样的 XTensor,所有元素填充 value 。

    :param t: 输入 XTensor 。
    :param value: 填充 XTensor 的值。

    :return: 输出 XTensor。

    Example::

        
        from pyvqnet.xtensor import XTensor,full_like,randu
        a =  randu([3,5])
        value = 42
        t =  full_like(a, value)
        print(t)
        # [
        # [42., 42., 42., 42., 42.],    
        # [42., 42., 42., 42., 42.],    
        # [42., 42., 42., 42., 42.]     
        # ]
        

zeros
===========================================================

.. py:function:: pyvqnet.xtensor.zeros(shape, device=None,dtype=None)

    创建输入形状大小的全零 XTensor 。

    :param shape: 输入形状。
    :param device: 储存在哪个设备上，默认: None，在CPU上。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 输出 XTensor 。

    Example::

        
        from pyvqnet.xtensor import XTensor,zeros
        t = zeros([2, 3, 4])
        print(t)
        # [
        # [[0., 0., 0., 0.],
        #  [0., 0., 0., 0.],
        #  [0., 0., 0., 0.]],
        # [[0., 0., 0., 0.],
        #  [0., 0., 0., 0.],
        #  [0., 0., 0., 0.]]
        # ]
        

zeros_like
===========================================================

.. py:function:: pyvqnet.xtensor.zeros_like(t: pyvqnet.xtensor.XTensor)

    创建一个形状和输入一样的 XTensor,所有元素为0 。

    :param t: 输入参考 XTensor 。

    :return: 输出 XTensor 。

    Example::

        
        from pyvqnet.xtensor import XTensor,zeros_like
        t = XTensor([1, 2, 3])
        x = zeros_like(t)
        print(x)

        # [0., 0., 0.]
        


arange
===========================================================

.. py:function:: pyvqnet.xtensor.arange(start, end, step=1, device=None,dtype=None)

    创建一个在给定间隔内具有均匀间隔值的一维 XTensor 。

    :param start: 间隔开始。
    :param end: 间隔结束。
    :param step: 值之间的间距，默认为1。
    :param device: 要使用的设备，默认 = None，使用 CPU 设备。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import arange
        t =  arange(2, 30, 4)
        print(t)

        # [ 2.,  6., 10., 14., 18., 22., 26.]
        

linspace
===========================================================

.. py:function:: pyvqnet.xtensor.linspace(start, end, num, device=None,dtype=None)

    创建一维 XTensor ，其中的元素为区间 start 和 end 上均匀间隔的共 num 个值。

    :param start: 间隔开始。
    :param end: 间隔结束。
    :param num: 间隔的个数。
    :param device: 要使用的设备，默认: None ，使用 CPU 设备。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 输出 XTensor 。

    Example::

        
        from pyvqnet.xtensor import XTensor,linspace
        start, stop, num = -2.5, 10, 10
        t = linspace(start, stop, num)
        print(t)
        #[-2.5000000, -1.1111112, 0.2777777, 1.6666665, 3.0555553, 4.4444442, 5.8333330, 7.2222219, 8.6111107, 10.]

logspace
===========================================================

.. py:function:: pyvqnet.xtensor.logspace(start, end, num, base, device=None,dtype=None)

    在对数刻度上创建具有均匀间隔值的一维 XTensor。

    :param start: ``base ** start`` 是起始值
    :param end: ``base ** end`` 是序列的最终值
    :param num: 要生成的样本数
    :param base: 对数刻度的基数
    :param device: 要使用的设备，默认: None ，使用 CPU 设备。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor,logspace
        start, stop, steps, base = 0.1, 1.0, 5, 10.0
        t = logspace(start, stop, steps, base)
        print(t)

        # [1.2589254, 2.1134889, 3.5481336, 5.9566211, 10.]
        

eye
===========================================================

.. py:function:: pyvqnet.xtensor.eye(size, offset: int = 0, device=None,dtype=None)

    创建一个 size x size 的 XTensor，对角线上为 1，其他地方为 0。

    :param size: 要创建的（正方形）XTensor 的大小。
    :param offset: 对角线的索引：0（默认）表示主对角线，正值表示上对角线，负值表示下对角线。
    :param device: 要使用的设备，默认: None ，使用 CPU 设备。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 输出 XTensor 。

    Example::

        import pyvqnet.xtensor as tensor
        size = 3
        t = tensor.eye(size)
        print(t)

        # [
        # [1., 0., 0.],
        # [0., 1., 0.],
        # [0., 0., 1.]
        # ]
        

diag
===========================================================

.. py:function:: pyvqnet.xtensor.diag(t, k: int = 0)

    构造对角矩阵。

    输入一个 2-D XTensor，则返回一个与此相同的新张量，除了
    选定对角线中的元素以外的元素设置为零。

    :param t: 输入 XTensor。
    :param k: 偏移量（主对角线为 0，正数为向上偏移，负数为向下偏移），默认为0。

    :return: 输出 XTensor。

    Example::

        
        from pyvqnet.xtensor import XTensor,diag
        import numpy as np
        a = np.arange(16).reshape(4, 4).astype(np.float32)
        t = XTensor(a)
        for k in range(-3, 4):
            u = diag(t,k=k)
            print(u)


randu
===========================================================

.. py:function:: pyvqnet.xtensor.randu(shape, min=0.0,max=1.0, device=None, dtype=None)

    创建一个具有均匀分布随机值的 XTensor 。

    :param shape: 要创建的 XTensor 的形状。
    :param min: 分布的下限，默认: 0。
    :param max: 分布的上线，默认: 1。
    :param device: 要使用的设备，默认: None ，使用 CPU 设备。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 输出 XTensor 。

    Example::

        
        from pyvqnet.xtensor import XTensor, randu
        shape = [2, 3]
        t =  randu(shape)
        print(t)

        # [
        # [0.0885886, 0.9570093, 0.8304565],
        # [0.6055251, 0.8721224, 0.1927866]
        # ]
        

randn
===========================================================

.. py:function:: pyvqnet.xtensor.randn(shape, mean=0.0,std=1.0, device=None, dtype=None)

    创建一个具有正态分布随机值的 XTensor 。

    :param shape: 要创建的 XTensor 的形状。
    :param mean: 分布的均值，默认: 0。
    :param max: 分布的方差，默认: 1。
    :param device: 要使用的设备，默认: None ，使用 CPU 设备。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 输出 XTensor 。

    Example::

        
        from pyvqnet.xtensor import XTensor,randn
        shape = [2, 3]
        t = randn(shape)
        print(t)

        # [
        # [-0.9529880, -0.4947567, -0.6399882],
        # [-0.6987777, -0.0089036, -0.5084590]
        # ]


multinomial
===========================================================

.. py:function:: pyvqnet.xtensor.multinomial(t, num_samples)

    返回一个张量，其中每行包含 num_samples 个索引采样，来自位于张量输入的相应行中的多项式概率分布。
    
    :param t: 输入概率分布,仅支持浮点数。
    :param num_samples: 采样样本。

    :return:
         输出采样索引

    Examples::

        import pyvqnet.xtensor as tensor
        weights = tensor.XTensor([0.1,10, 3, 1]) 
        idx = tensor.multinomial(weights,3)
        print(idx)

        weights = tensor.XTensor([0,10, 3, 2.2,0.0]) 
        idx = tensor.multinomial(weights,3)
        print(idx)


triu
===========================================================

.. py:function:: pyvqnet.xtensor.triu(t, diagonal=0)

    返回输入 t 的上三角矩阵，其余部分被设为0。

    :param t: 输入 XTensor。
    :param diagonal: 偏移量（主对角线为 0，正数为向上偏移，负数为向下偏移），默认=0。

    :return: 输出 XTensor。

    Examples::

        import pyvqnet.xtensor as tensor
        
        a = tensor.arange(1.0, 2 * 6 * 5 + 1.0).reshape([2, 6, 5])
        u = tensor.triu(a, 1)
        print(u)
        # [
        # [[0., 2., 3., 4., 5.],       
        #  [0., 0., 8., 9., 10.],      
        #  [0., 0., 0., 14., 15.],     
        #  [0., 0., 0., 0., 20.],      
        #  [0., 0., 0., 0., 0.],       
        #  [0., 0., 0., 0., 0.]],      
        # [[0., 32., 33., 34., 35.],   
        #  [0., 0., 38., 39., 40.],    
        #  [0., 0., 0., 44., 45.],     
        #  [0., 0., 0., 0., 50.],      
        #  [0., 0., 0., 0., 0.],       
        #  [0., 0., 0., 0., 0.]]       
        # ]

tril
===========================================================

.. py:function:: pyvqnet.xtensor.tril(t, diagonal=0)

    返回输入 t 的下三角矩阵，其余部分被设为0。


    :param t: 输入 XTensor。
    :param diagonal: 偏移量（主对角线为 0，正数为向上偏移，负数为向下偏移），默认=0。

    :return: 输出 XTensor。

    Examples::

        import pyvqnet.xtensor as tensor
        a = tensor.arange(1.0, 2 * 6 * 5 + 1.0).reshape([12, 5])
        u = tensor.tril(a, 1)
        print(u)
        # [
        # [1., 2., 0., 0., 0.],      
        #  [6., 7., 8., 0., 0.],     
        #  [11., 12., 13., 14., 0.], 
        #  [16., 17., 18., 19., 20.],
        #  [21., 22., 23., 24., 25.],
        #  [26., 27., 28., 29., 30.],
        #  [31., 32., 33., 34., 35.],
        #  [36., 37., 38., 39., 40.],
        #  [41., 42., 43., 44., 45.],
        #  [46., 47., 48., 49., 50.],
        #  [51., 52., 53., 54., 55.],
        #  [56., 57., 58., 59., 60.]
        # ]

数学函数
***********************


floor
===========================================================

.. py:function:: pyvqnet.xtensor.floor(t)

    返回一个新的 XTensor，其中元素为输入 XTensor 的向下取整。

    :param t: 输入 XTensor 。

    :return: 输出 XTensor 。

    Example::


        import pyvqnet.xtensor as tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.floor(t)
        print(u)

        # [-2., -2., -2., -2., -1., -1., -1., -1., 0., 0., 0., 0., 1., 1., 1., 1.]

ceil
===========================================================

.. py:function:: pyvqnet.xtensor.ceil(t)

    返回一个新的 XTensor，其中元素为输入 XTensor 的向上取整。

    :param t: 输入 XTensor 。
    :return: 输出 XTensor 。

    Example::

        import pyvqnet.xtensor as tensor

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.ceil(t)
        print(u)

        # [-2., -1., -1., -1., -1., -0., -0., -0., 0., 1., 1., 1., 1., 2., 2., 2.]

round
===========================================================

.. py:function:: pyvqnet.xtensor.round(t)

    返回一个新的 XTensor，其中元素为输入 XTensor 的四舍五入到最接近的整数.

    :param t: 输入 XTensor 。
    :return: 输出 XTensor 。

    Example::

        import pyvqnet.xtensor as tensor

        t = tensor.arange(-2.0, 2.0, 0.4)
        u = tensor.round(t)
        print(u)

        # [-2., -2., -1., -1., -0., -0., 0., 1., 1., 2.]

sort
===========================================================

.. py:function:: pyvqnet.xtensor.sort(t, axis=None, descending=False, stable=True)

    按指定轴对输入 XTensor 进行排序。

    :param t: 输入 XTensor 。
    :param axis: 排序使用的轴。
    :param descending: 如果是True，进行降序排序，否则使用升序排序。默认为升序。
    :param stable: 是否使用稳定排序，默认为稳定排序。
    :return: 输出 XTensor 。

    Example::

        
        import pyvqnet.xtensor as tensor
        import numpy as np
        a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
        A = tensor.xtensor(a)
        AA = tensor.sort(A,1,False)
        print(AA)

        # [
        # [0., 1., 2., 4., 6., 7., 8., 8.],
        # [2., 5., 5., 8., 9., 9., 9., 9.],
        # [1., 2., 5., 5., 5., 6., 7., 7.]
        # ]

argsort
===========================================================

.. py:function:: pyvqnet.xtensor.argsort(t, axis = None, descending=False, stable=True)

    对输入变量沿给定轴进行排序，输出排序好的数据的相应索引。

    :param t: 输入 XTensor 。
    :param axis: 排序使用的轴。
    :param descending: 如果是True，进行降序排序，否则使用升序排序。默认为升序。
    :param stable: 是否使用稳定排序，默认为稳定排序。
    :return: 输出 XTensor 。

    Example::

        
        import pyvqnet.xtensor as tensor
        import numpy as np
        a = np.random.randint(10, size=24).reshape(3,8)
        A =tensor.XTensor(a)
        bb = tensor.argsort(A,1,False)
        print(bb)

        # [
        # [4., 0., 1., 7., 5., 3., 2., 6.], 
        #  [3., 0., 7., 6., 2., 1., 4., 5.],
        #  [4., 7., 5., 0., 2., 1., 3., 6.]
        # ]

topK
===========================================================

.. py:function:: pyvqnet.xtensor.topK(t, k, axis=-1, if_descent=True)

    返回给定输入张量沿给定维度的 k 个最大元素。

    如果 if_descent 为 False，则返回 k 个最小元素。

    :param t: 输入 XTensor 。
    :param k: 取排序后的 前k 的个数。
    :param axis: 要排序的维度。默认 = -1，最后一个轴。
    :param if_descent: 排序使用升序还是降序，默认降序。

    :return: 新的 XTensor 。

    Examples::

        import pyvqnet.xtensor as tensor
        from pyvqnet.xtensor import XTensor
        x = XTensor([
            24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
            3., 6., 11.
        ])
        x = x.reshape([2, 5, 1, 2])
        x.requires_grad = True
        y = tensor.topK(x, 3, 1)
        print(y)
        # [
        # [[[24., 15.]],
        # [[15., 13.]],
        # [[11., 8.]]],
        # [[[24., 13.]],
        # [[15., 11.]],
        # [[7., 8.]]]
        # ]

argtopK
===========================================================

.. py:function:: pyvqnet.xtensor.argtopK(t, k, axis=-1, if_descent=True)

    返回给定输入张量沿给定维度的 k 个最大元素的索引。

    如果 if_descent 为 False，则返回 k 个最小元素的索引。

    :param t: 输入 XTensor 。
    :param k: 取排序后的 k 的个数。
    :param axis: 要排序的维度。默认 = -1，最后一个轴。
    :param if_descent: 排序使用升序还是降序，默认降序。

    :return: 新的 XTensor 。

    Examples::

        import pyvqnet.xtensor as tensor
        from pyvqnet.xtensor import XTensor
        x = XTensor([
            24., 13., 15., 4., 3., 8., 11., 3., 6., 15., 24., 13., 15., 3., 3., 8., 7.,
            3., 6., 11.
        ])
        x = x.reshape([2, 5, 1, 2])
        x.requires_grad = True
        y = tensor.argtopK(x, 3, 1)
        print(y)
        # [
        # [[[0., 4.]],
        # [[1., 0.]],
        # [[3., 2.]]],
        # [[[0., 0.]],
        # [[1., 4.]],
        # [[3., 2.]]]
        # ]


add
===========================================================

.. py:function:: pyvqnet.xtensor.add(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    两个 XTensor 按元素相加。等价于t1 + t2。

    :param t1: 第一个 XTensor 。
    :param t2: 第二个 XTensor 。
    :return:  输出 XTensor 。

    Example::

        
        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = XTensor([1, 2, 3])
        t2 = XTensor([4, 5, 6])
        x = tensor.add(t1, t2)
        print(x)

        # [5., 7., 9.]

sub
===========================================================

.. py:function:: pyvqnet.xtensor.sub(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    两个 XTensor 按元素相减。等价于t1 - t2。

    :param t1: 第一个 XTensor 。
    :param t2: 第二个 XTensor 。
    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = XTensor([1, 2, 3])
        t2 = XTensor([4, 5, 6])
        x = tensor.sub(t1, t2)
        print(x)

        # [-3., -3., -3.]

mul
===========================================================

.. py:function:: pyvqnet.xtensor.mul(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    两个 XTensor 按元素相乘。等价于t1 * t2。

    :param t1: 第一个 XTensor 。
    :param t2: 第二个 XTensor 。
    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = XTensor([1, 2, 3])
        t2 = XTensor([4, 5, 6])
        x = tensor.mul(t1, t2)
        print(x)

        # [4., 10., 18.]

divide
===========================================================

.. py:function:: pyvqnet.xtensor.divide(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    两个 XTensor 按元素相除。等价于t1 / t2。

    :param t1: 第一个 XTensor 。
    :param t2: 第二个 XTensor 。
    :return:  输出 XTensor 。


    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = XTensor([1, 2, 3])
        t2 = XTensor([4, 5, 6])
        x = tensor.divide(t1, t2)
        print(x)

        # [0.2500000, 0.4000000, 0.5000000]

sums
===========================================================

.. py:function:: pyvqnet.xtensor.sums(t: pyvqnet.xtensor.XTensor, axis: int = None, keepdims=False)

    对输入的 XTensor 按 axis 设定的轴计算元素和，如果 axis 是None，则返回所有元素和。

    :param t: 输入 XTensor 。
    :param axis: 用于求和的轴，默认为None。
    :param keepdims: 输出张量是否保留了减小的维度。默认为False。
    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.sums(t)
        print(x)

        # [21.]

cumsum
===========================================================

.. py:function:: pyvqnet.xtensor.cumsum(t, axis=-1)

    返回维度轴中输入元素的累积总和。

    :param t: 输入 XTensor 。
    :param axis: 计算的轴，默认 -1，使用最后一个轴。
    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.cumsum(t,-1)
        print(x)
        """
        [[ 1.  3.  6.]
        [ 4.  9. 15.]]
        <XTensor 2x3 cpu(0) kfloat32>
        """


mean
===========================================================

.. py:function:: pyvqnet.xtensor.mean(t: pyvqnet.xtensor.XTensor, axis=None, keepdims=False)

    对输入的 XTensor 按 axis 设定的轴计算元素的平均，如果 axis 是None，则返回所有元素平均。

    :param t: 输入 XTensor ,需要是浮点数或者复数。
    :param axis: 用于求平均的轴，默认为None。
    :param keepdims: 输出张量是否保留了减小的维度。默认为False。
    :return: 输出 XTensor 或 均值。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([[1, 2, 3], [4, 5, 6.0]])
        x = tensor.mean(t, axis=1)
        print(x)

        # [2. 5.]

median
===========================================================

.. py:function:: pyvqnet.xtensor.median(t: pyvqnet.xtensor.XTensor, axis=None, keepdims=False)

    对输入的 XTensor 按 axis 设定的轴计算元素的平均，如果 axis 是None，则返回所有元素平均。

    :param t: 输入 XTensor 。
    :param axis: 用于求平均的轴，默认为None。
    :param keepdims: 输出张量是否保留了减小的维度。默认为False。
    :return: 输出 XTensor 或 中值。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([[1, 2, 3], [4, 5, 6.0]])
        x = tensor.mean(t, axis=1)
        print(x)
        #[2.5]
        a = XTensor([[1.5219, -1.5212,  0.2202]])
        median_a = tensor.median(a)
        print(median_a)

        # [0.2202000]

        b = XTensor([[0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
                    [0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
                    [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
                    [1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
        median_b = tensor.median(b,1, False)
        print(median_b)

        # [-0.3982000, 0.2269999, 0.2487999, 0.4742000]

std
===========================================================

.. py:function:: pyvqnet.xtensor.std(t: pyvqnet.xtensor.XTensor, axis=None, keepdims=False, unbiased=True)

    对输入的 XTensor 按 axis 设定的轴计算元素的标准差，如果 axis 是None，则返回所有元素标准差。

    :param t: 输入 XTensor 。
    :param axis: 用于求标准差的轴，默认为None。
    :param keepdims: 输出张量是否保留了减小的维度。默认为False。
    :param unbiased: 是否使用贝塞尔修正,默认使用。
    :return: 输出 XTensor 或 标准差。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([[-0.8166, -1.3802, -0.3560]])
        std_a = tensor.std(a)
        print(std_a)

        # [0.5129624]

        b = XTensor([[0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
                    [0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
                    [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
                    [1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
        std_b = tensor.std(b, 1, False, False)
        print(std_b)

        # [0.6593542, 0.5583112, 0.3206565, 1.1103367]

var
===========================================================

.. py:function:: pyvqnet.xtensor.var(t: pyvqnet.xtensor.XTensor, axis=None, keepdims=False, unbiased=True)

    对输入的 XTensor 按 axis 设定的轴计算元素的方差，如果 axis 是None，则返回所有元素方差。

    :param t: 输入 XTensor 。
    :param axis: 用于求方差的轴，默认为None。
    :param keepdims: 输出张量是否保留了减小的维度。默认为False。
    :param unbiased: 是否使用贝塞尔修正,默认使用。
    :return: 输出 XTensor 或方差。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([[-0.8166, -1.3802, -0.3560]])
        a_var = tensor.var(a)
        print(a_var)

        # [0.2631305]

matmul
===========================================================

.. py:function:: pyvqnet.xtensor.matmul(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    二维矩阵点乘或3、4维张量进行批矩阵乘法.

    :param t1: 第一个 XTensor 。
    :param t2: 第二个 XTensor 。
    :return:  输出 XTensor 。

    Example::

        import pyvqnet.xtensor as tensor
        from pyvqnet.xtensor import autograd
        t1 = tensor.ones([2,3])
        t1.requires_grad = True
        t2 = tensor.ones([3,4])
        t2.requires_grad = True
        with autogard.tape():
            t3  = tensor.matmul(t1,t2)
            t3.backward(tensor.ones_like(t3))
        print(t1.grad)

        # [
        # [4., 4., 4.],
        #  [4., 4., 4.]
        # ]

        print(t2.grad)

        # [
        # [2., 2., 2., 2.],
        #  [2., 2., 2., 2.],
        #  [2., 2., 2., 2.]
        # ]

kron
===========================================================

.. py:function:: pyvqnet.xtensor.kron(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    计算 ``t1`` 和  ``t2`` 的 Kronecker 积，用 :math:`\otimes` 表示。

    如果 ``t1`` 是一个 :math:`(a_0 \times a_1 \times \dots \times a_n)` 张量并且 ``t2`` 是一个 :math:`(b_0 \times b_1 \times \dots \times b_n)` 张量，结果将是 :math:`(a_0*b_0 \times a_1*b_1 \times \dots \times a_n*b_n)` 张量，包含以下条目：

     .. math::
         (\text{input} \otimes \text{other})_{k_0, k_1, \dots, k_n} =
             \text{input}_{i_0, i_1, \dots, i_n} * \text{other}_{j_0, j_1, \dots, j_n},

     其中 :math:`k_t = i_t * b_t + j_t` 为 :math:`0 \leq t \leq n`。
     如果一个张量的维数少于另一个，它将被解压缩，直到它具有相同的维数。

    :param t1: 第一个 XTensor 。
    :param t2: 第二个 XTensor 。
    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
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
===========================================================

.. py:function:: pyvqnet.xtensor.reciprocal(t)

    计算输入 XTensor 的倒数。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = tensor.arange(1, 10, 1)
        u = tensor.reciprocal(t)
        print(u)

        #[1., 0.5000000, 0.3333333, 0.2500000, 0.2000000, 0.1666667, 0.1428571, 0.1250000, 0.1111111]

sign
===========================================================

.. py:function:: pyvqnet.xtensor.sign(t)

    对输入 t 中每个元素进行正负判断，并且输出正负判断值：1代表正，-1代表负，0代表零。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = tensor.arange(-5, 5, 1)
        u = tensor.sign(t)
        print(u)

        # [-1., -1., -1., -1., -1., 0., 1., 1., 1., 1.]

neg
===========================================================

.. py:function:: pyvqnet.xtensor.neg(t: pyvqnet.xtensor.XTensor)

    计算输入 t 每个元素的相反数并返回。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.neg(t)
        print(x)

        # [-1., -2., -3.]

trace
===========================================================

.. py:function:: pyvqnet.xtensor.trace(t, k: int = 0)

    返回二维矩阵的迹。

    :param t: 输入 XTensor 。
    :param k: 偏移量（主对角线为 0，正数为向上偏移，负数为向下偏移），默认为0。

    :return: 输入二维矩阵的对角线元素之和。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        t = tensor.randn([4,4])
        for k in range(-3, 4):
            u=tensor.trace(t,k=k)
            print(u)


exp
===========================================================

.. py:function:: pyvqnet.xtensor.exp(t: pyvqnet.xtensor.XTensor)

    计算输入 t 每个元素的自然数e为底指数。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.exp(t)
        print(x)

        # [2.7182817, 7.3890562, 20.0855369]

acos
===========================================================

.. py:function:: pyvqnet.xtensor.acos(t: pyvqnet.xtensor.XTensor)

    计算输入 t 每个元素的反余弦。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        import numpy as np
        a = np.arange(36).reshape(2,6,3).astype(np.float32)
        a =a/100
        A = XTensor(a)
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
===========================================================

.. py:function:: pyvqnet.xtensor.asin(t: pyvqnet.xtensor.XTensor)

    计算输入 t 每个元素的反正弦。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        t = tensor.arange(-1, 1, .5)
        u = tensor.asin(t)
        print(u)

        #[-1.5707964, -0.5235988, 0., 0.5235988]

atan
===========================================================

.. py:function:: pyvqnet.xtensor.atan(t: pyvqnet.xtensor.XTensor)

    计算输入 t 每个元素的反正切。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = tensor.arange(-1, 1, .5)
        u = tensor.atan(t)
        print(u)

        # [-0.7853981, -0.4636476, 0., 0.4636476]

sin
===========================================================

.. py:function:: pyvqnet.xtensor.sin(t: pyvqnet.xtensor.XTensor)

    计算输入 t 每个元素的正弦。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.sin(t)
        print(x)

        # [0.8414709, 0.9092974, 0.1411200]

cos
===========================================================

.. py:function:: pyvqnet.xtensor.cos(t: pyvqnet.xtensor.XTensor)

    计算输入 t 每个元素的余弦。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.cos(t)
        print(x)

        # [0.5403022, -0.4161468, -0.9899924]

tan 
===========================================================

.. py:function:: pyvqnet.xtensor.tan(t: pyvqnet.xtensor.XTensor)

    计算输入 t 每个元素的正切。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.tan(t)
        print(x)

        # [1.5574077, -2.1850397, -0.1425465]

tanh
===========================================================

.. py:function:: pyvqnet.xtensor.tanh(t: pyvqnet.xtensor.XTensor)

    计算输入 t 每个元素的双曲正切。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.tanh(t)
        print(x)

        # [0.7615941, 0.9640275, 0.9950547]

sinh
===========================================================

.. py:function:: pyvqnet.xtensor.sinh(t: pyvqnet.xtensor.XTensor)

    计算输入 t 每个元素的双曲正弦。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.sinh(t)
        print(x)

        # [1.1752011, 3.6268603, 10.0178747]

cosh
===========================================================

.. py:function:: pyvqnet.xtensor.cosh(t: pyvqnet.xtensor.XTensor)

    计算输入 t 每个元素的双曲余弦。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.cosh(t)
        print(x)

        # [1.5430806, 3.7621955, 10.0676622]

power
===========================================================

.. py:function:: pyvqnet.xtensor.power(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    第一个 XTensor 的元素计算第二个 XTensor 的幂指数。

    :param t1: 第一个 XTensor 。
    :param t2: 第二个 XTensor 。
    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = XTensor([1, 4, 3])
        t2 = XTensor([2, 5, 6])
        x = tensor.power(t1, t2)
        print(x)

        # [1., 1024., 729.]

abs
===========================================================

.. py:function:: pyvqnet.xtensor.abs(t: pyvqnet.xtensor.XTensor)

    计算输入 XTensor 的每个元素的绝对值。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, -2, 3])
        x = tensor.abs(t)
        print(x)

        # [1., 2., 3.]

log
===========================================================

.. py:function:: pyvqnet.xtensor.log(t: pyvqnet.xtensor.XTensor)

    计算输入 XTensor 的每个元素的自然对数值。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.log(t)
        print(x)

        # [0., 0.6931471, 1.0986123]

log_softmax
===========================================================

.. py:function:: pyvqnet.xtensor.log_softmax(t, axis=-1)

    顺序计算在轴axis上的softmax函数以及log函数的结果。

    :param t: 输入 XTensor 。
    :param axis: 用于求softmax的轴，默认为-1。

    :return: 输出 XTensor。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
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
===========================================================

.. py:function:: pyvqnet.xtensor.sqrt(t: pyvqnet.xtensor.XTensor)

    计算输入 XTensor 的每个元素的平方根值。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.sqrt(t)
        print(x)

        # [1., 1.4142135, 1.7320507]

square
===========================================================

.. py:function:: pyvqnet.xtensor.square(t: pyvqnet.xtensor.XTensor)

    计算输入 XTensor 的每个元素的平方值。

    :param t: 输入 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.square(t)
        print(x)

        # [1., 4., 9.]

frobenius_norm
===========================================================

.. py:function:: pyvqnet.xtensor.frobenius_norm(t: XTensor, axis: int = None, keepdims=False):

    对输入的 XTensor 按 axis 设定的轴计算张量的F范数，如果 axis 是None，则返回所有元素F范数。

    :param t: 输入 XTensor 。
    :param axis: 用于求F范数的轴，默认为None。
    :param keepdims: 输出张量是否保留了减小的维度。默认为False。
    :return: 输出 XTensor 或 F范数值。


    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]],
                    [[13., 14., 15.], [16., 17., 18.]]])
        t.requires_grad = True
        result = tensor.frobenius_norm(t, -2, False)
        print(result)
        # [
        # [4.1231055, 5.3851647, 6.7082038],
        #  [12.2065554, 13.6014709, 15.],
        #  [20.6155281, 22.0227146, 23.4307499]
        # ]


逻辑函数
***********************

maximum
===========================================================

.. py:function:: pyvqnet.xtensor.maximum(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    计算两个 XTensor 的逐元素中的较大值。

    :param t1: 第一个 XTensor 。
    :param t2: 第二个 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = XTensor([6, 4, 3])
        t2 = XTensor([2, 5, 7])
        x = tensor.maximum(t1, t2)
        print(x)

        # [6., 5., 7.]

minimum
===========================================================

.. py:function:: pyvqnet.xtensor.minimum(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    计算两个 XTensor 的逐元素中的较小值。

    :param t1: 第一个 XTensor 。
    :param t2: 第二个 XTensor 。

    :return:  输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = XTensor([6, 4, 3])
        t2 = XTensor([2, 5, 7])
        x = tensor.minimum(t1, t2)
        print(x)

        # [2., 4., 3.]

min
===========================================================

.. py:function:: pyvqnet.xtensor.min(t: pyvqnet.xtensor.XTensor, axis=None, keepdims=False)

    对输入的 XTensor 按 axis 设定的轴计算元素的最小值，如果 axis 是None，则返回所有元素的最小值。

    :param t: 输入 XTensor 。
    :param axis: 用于求最小值的轴，默认为None。
    :param keepdims: 输出张量是否保留了减小的维度。默认为False。

    :return: 输出 XTensor 或浮点数。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.min(t, axis=1, keepdims=True)
        print(x)

        # [
        # [1.],
        #  [4.]
        # ]

max
===========================================================

.. py:function:: pyvqnet.xtensor.max(t: pyvqnet.xtensor.XTensor, axis=None, keepdims=False)

    对输入的 XTensor 按 axis 设定的轴计算元素的最大值，如果 axis 是None，则返回所有元素的最大值。

    :param t: 输入 XTensor 。
    :param axis: 用于求最大值的轴，默认为None。
    :param keepdims: 输出张量是否保留了减小的维度。默认为False。
    
    :return: 输出 XTensor 或浮点数。


    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.max(t, axis=1, keepdims=True)
        print(x)

        # [
        # [3.],
        #  [6.]
        # ]

clip
===========================================================

.. py:function:: pyvqnet.xtensor.clip(t: pyvqnet.xtensor.XTensor, min_val, max_val)

    将输入的所有元素进行剪裁，使得输出元素限制在[min_val, max_val]。

    :param t: 输入 XTensor 。
    :param min_val:  裁剪下限值。
    :param max_val:  裁剪上限值。
    :return:  output XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([2, 4, 6])
        x = tensor.clip(t, 3, 8)
        print(x)

        # [3., 4., 6.]


where
===========================================================

.. py:function:: pyvqnet.xtensor.where(condition: pyvqnet.xtensor.XTensor, t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)


    根据条件返回从 t1 或 t2 中选择的元素。

    :param condition: 判断条件 XTensor,需要是kbool数据类型 。
    :param t1: 如果满足条件，则从中获取元素。
    :param t2: 如果条件不满足，则从中获取元素。

    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = XTensor([1, 2, 3])
        t2 = XTensor([4, 5, 6])
        x = tensor.where(t1 < 2, t1, t2)
        print(x)

        # [1., 5., 6.]

nonzero
===========================================================

.. py:function:: pyvqnet.xtensor.nonzero(t)

    返回一个包含非零元素索引的 XTensor 。

    :param t: 输入 XTensor 。
    :return: 输出 XTensor 包含非零元素的索引。

    Example::
    
        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([[0.6, 0.0, 0.0, 0.0],
                                    [0.0, 0.4, 0.0, 0.0],
                                    [0.0, 0.0, 1.2, 0.0],
                                    [0.0, 0.0, 0.0,-0.4]])
        t = tensor.nonzero(t)
        print(t)
        # [
        # [0., 0.],
        # [1., 1.],
        # [2., 2.],
        # [3., 3.]
        # ]

isfinite
===========================================================

.. py:function:: pyvqnet.xtensor.isfinite(t)

    逐元素判断输入是否为Finite （既非 +/-INF 也非 +/-NaN ）。

    :param t: 输入 XTensor 。
    :return: 输出 XTensor , 其中对应位置元素满足条件时返回True，否则返回False。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        t = XTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isfinite(t)
        print(flag)

        #[ True False  True False False]

isinf
===========================================================

.. py:function:: pyvqnet.xtensor.isinf(t)

    逐元素判断输入的每一个值是否为 +/-INF 。

    :param t: 输入 XTensor 。
    :return: 输出 XTensor , 其中对应位置元素满足条件时返回True，否则返回False。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        t = XTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isinf(t)
        print(flag)

        # [False  True False  True False]

isnan
===========================================================

.. py:function:: pyvqnet.xtensor.isnan(t)

    逐元素判断输入的每一个值是否为 +/-NaN 。

    :param t: 输入 XTensor 。
    :return: 输出 XTensor , 其中对应位置元素满足条件时返回True，否则返回False。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        t = XTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isnan(t)
        print(flag)

        # [False False False False  True]

isneginf
===========================================================

.. py:function:: pyvqnet.xtensor.isneginf(t)

    逐元素判断输入的每一个值是否为 -INF 。

    :param t: 输入 XTensor 。
    :return: 输出 XTensor , 其中对应位置元素满足条件时返回True，否则返回False。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        t = XTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isneginf(t)
        print(flag)

        # [False False False  True False]

isposinf
===========================================================

.. py:function:: pyvqnet.xtensor.isposinf(t)

    逐元素判断输入的每一个值是否为 +INF 。

    :param t: 输入 XTensor 。
    :return: 输出 XTensor , 其中对应位置元素满足条件时返回True，否则返回False。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        t = XTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isposinf(t)
        print(flag)

        # [False  True False False False]

logical_and
===========================================================

.. py:function:: pyvqnet.xtensor.logical_and(t1, t2)

    对两个输入进行逐元素逻辑与操作，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 XTensor 。
    :param t2: 输入 XTensor 。

    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([0, 1, 10, 0])
        b = XTensor([4, 0, 1, 0])
        flag = tensor.logical_and(a,b)
        print(flag)

        # [False False  True False]

logical_or
===========================================================

.. py:function:: pyvqnet.xtensor.logical_or(t1, t2)

    对两个输入进行逐元素逻辑或操作，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 XTensor 。
    :param t2: 输入 XTensor 。

    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([0, 1, 10, 0])
        b = XTensor([4, 0, 1, 0])
        flag = tensor.logical_or(a,b)
        print(flag)

        # [ True  True  True False]

logical_not
===========================================================

.. py:function:: pyvqnet.xtensor.logical_not(t)

    对输入进行逐元素逻辑非操作，其中对应位置元素满足条件时返回True，否则返回False。

    :param t: 输入 XTensor 。
    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([0, 1, 10, 0])
        flag = tensor.logical_not(a)
        print(flag)

        # [ True False False  True]

logical_xor
===========================================================

.. py:function:: pyvqnet.xtensor.logical_xor(t1, t2)

    对两个输入进行逐元素逻辑异或操作，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 XTensor 。
    :param t2: 输入 XTensor 。

    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([0, 1, 10, 0])
        b = XTensor([4, 0, 1, 0])
        flag = tensor.logical_xor(a,b)
        print(flag)

        # [ True  True False False]

greater
===========================================================

.. py:function:: pyvqnet.xtensor.greater(t1, t2)

    逐元素比较 t1 是否大于 t2 ，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 XTensor 。
    :param t2: 输入 XTensor 。

    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([[1, 2], [3, 4]])
        b = XTensor([[1, 1], [4, 4]])
        flag = tensor.greater(a,b)
        print(flag)

        # [[False  True]
        #  [False False]]

greater_equal
===========================================================

.. py:function:: pyvqnet.xtensor.greater_equal(t1, t2)

    逐元素比较 t1 是否大于等于 t2 ，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 XTensor 。
    :param t2: 输入 XTensor 。

    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([[1, 2], [3, 4]])
        b = XTensor([[1, 1], [4, 4]])
        flag = tensor.greater_equal(a,b)
        print(flag)

        #[[ True  True]
        # [False  True]]

less
===========================================================

.. py:function:: pyvqnet.xtensor.less(t1, t2)

    逐元素比较 t1 是否小于 t2 ，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 XTensor 。
    :param t2: 输入 XTensor 。

    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([[1, 2], [3, 4]])
        b = XTensor([[1, 1], [4, 4]])
        flag = tensor.less(a,b)
        print(flag)

        #[[False False]
        # [ True False]]

less_equal
===========================================================

.. py:function:: pyvqnet.xtensor.less_equal(t1, t2)

    逐元素比较 t1 是否小于等于 t2 ，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 XTensor 。
    :param t2: 输入 XTensor 。

    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        a = XTensor([[1, 2], [3, 4]])
        b = XTensor([[1, 1], [4, 4]])
        flag = tensor.less_equal(a,b)
        print(flag)


        # [[ True False]
        #  [ True  True]]

equal
===========================================================

.. py:function:: pyvqnet.xtensor.equal(t1, t2)

    逐元素比较 t1 是否等于 t2 ，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 XTensor 。
    :param t2: 输入 XTensor 。

    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([[1, 2], [3, 4]])
        b = XTensor([[1, 1], [4, 4]])
        flag = tensor.equal(a,b)
        print(flag)

        #[[ True False]
        # [False  True]]

not_equal
===========================================================

.. py:function:: pyvqnet.xtensor.not_equal(t1, t2)

    逐元素比较 t1 是否不等于 t2 ，其中对应位置元素满足条件时返回True，否则返回False。

    :param t1: 输入 XTensor 。
    :param t2: 输入 XTensor 。

    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        a = XTensor([[1, 2], [3, 4]])
        b = XTensor([[1, 1], [4, 4]])
        flag = tensor.not_equal(a,b)
        print(flag)

        #[[False  True]
        # [ True False]]

矩阵操作
***********************

broadcast
===========================================================

.. py:function:: pyvqnet.xtensor.broadcast(t1: pyvqnet.xtensor.XTensor, t2: pyvqnet.xtensor.XTensor)

    受到某些限制，较小的阵列在整个更大的阵列，以便它们具有兼容的形状。该接口可对入参张量进行自动微分。

    参考https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t1: 输入 XTensor 1
    :param t2: 输入 XTensor 2

    :return t11: 具有新的广播形状 t1。
    :return t22: 具有新广播形状的 t2。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t1 = tensor.ones([5, 4])
        t2 = tensor.ones([4])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)

        t1 = tensor.ones([5, 4])
        t2 = tensor.ones([1])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)

        t1 = tensor.ones([5, 4])
        t2 = tensor.ones([2, 1, 4])

        t11, t22 = tensor.broadcast(t1, t2)

        print(t11.shape)
        print(t22.shape)


        # [5, 4]
        # [5, 4]
        # [5, 4]
        # [5, 4]
        # [2, 5, 4]
        # [2, 5, 4]


concatenate
===========================================================

.. py:function:: pyvqnet.xtensor.concatenate(args: list, axis=0)

    对 args 内的多个 XTensor 沿 axis 轴进行联结，返回一个新的 XTensor 。

    :param args: 包含输入 XTensor 。
    :param axis: 要连接的维度，默认：0。
    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        x = XTensor([[1, 2, 3],[4,5,6]])
        y = 1-x
        x = tensor.concatenate([x,y],1)
        print(x)

        # [
        # [1., 2., 3., 0., -1., -2.],
        # [4., 5., 6., -3., -4., -5.]
        # ]
        
        

stack
===========================================================

.. py:function:: pyvqnet.xtensor.stack(XTensors: list, axis=0) 

    沿新轴 axis 堆叠输入的 XTensors ，返回一个新的 XTensor。

    :param XTensors: 包含输入 XTensor 。
    :param axis: 要堆叠的维度，默认：0。
    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t11 = XTensor(a)
        t22 = XTensor(a)
        t33 = XTensor(a)
        rlt1 = tensor.stack([t11,t22,t33],2)
        print(rlt1)
        
        # [
        # [[0., 0., 0.],
        #  [1., 1., 1.],
        #  [2., 2., 2.],
        #  [3., 3., 3.]],
        # [[4., 4., 4.],
        #  [5., 5., 5.],
        #  [6., 6., 6.],
        #  [7., 7., 7.]],
        # [[8., 8., 8.],
        #  [9., 9., 9.],
        #  [10., 10., 10.],
        #  [11., 11., 11.]]
        # ]
                

permute
===========================================================

.. py:function:: pyvqnet.xtensor.permute(t: pyvqnet.xtensor.XTensor, *axes)

    根据输入的 axes 的顺序，改变t 的轴的顺序。如果 axes = None，则按顺序反转 t 的轴。

    :param t: 输入 XTensor 。
    :param axes: 维度的新顺序，默认：None,反转。
    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2,2,3]).astype(np.float32)
        t = XTensor(a)
        tt = tensor.permute(t,[2,0,1])
        print(tt)
        
        # [
        # [[0., 3.],
        #  [6., 9.]],
        # [[1., 4.],
        #  [7., 10.]],
        # [[2., 5.],
        #  [8., 11.]]
        # ]
                
        

transpose
===========================================================

.. py:function:: pyvqnet.xtensor.transpose(t: pyvqnet.xtensor.XTensor, *axes)

    根据输入的 axes 的顺序，改变t 的轴的顺序。如果 axes = None，则按顺序反转 t 的轴。

    :param t: 输入 XTensor 。
    :param axes: 维度的新顺序，默认：None,反转。

    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape([2,2,3]).astype(np.float32)
        t = XTensor(a)
        tt = tensor.transpose(t,[2,0,1])
        print(tt)

        # [
        # [[0., 3.],
        #  [6., 9.]],
        # [[1., 4.],
        #  [7., 10.]],
        # [[2., 5.],
        #  [8., 11.]]
        # ]
        

tile
===========================================================

.. py:function:: pyvqnet.xtensor.tile(t: pyvqnet.xtensor.XTensor, reps: list)


    通过按照 reps 给出的次数复制输入 XTensor 。
    
    如果 reps 的长度为 d，则结果 XTensor 的维度大小为 max(d, t.ndim)。如果 t.ndim < d，则通过从起始维度插入新轴，将 t 扩展为 d 维度。
    
    因此形状 (3,) 数组被提升为 (1, 3) 用于 2-D 复制，或形状 (1, 1, 3) 用于 3-D 复制。如果 t.ndim > d，reps 通过插入 1 扩展为 t.ndim。

    因此，对于形状为 (2, 3, 4, 5) 的 t，(4, 3) 的 reps 被视为 (1, 1, 4, 3)。

    :param t: 输入 XTensor 。
    :param reps: 每个维度的重复次数。
    :return: 一个新的 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        import numpy as np
        a = np.arange(6).reshape(2,3).astype(np.float32)
        A = XTensor(a)
        reps = [2,2]
        B = tensor.tile(A,reps)
        print(B)

        # [
        # [0., 1., 2., 0., 1., 2.],
        # [3., 4., 5., 3., 4., 5.],
        # [0., 1., 2., 0., 1., 2.],
        # [3., 4., 5., 3., 4., 5.]
        # ]
        

squeeze
===========================================================

.. py:function:: pyvqnet.xtensor.squeeze(t: pyvqnet.xtensor.XTensor, axis: int = - 1)

    删除 axis 指定的轴，该轴的维度为1。如果 axis = None ，则将输入所有长度为1的维度删除。

    :param t: 输入 XTensor 。
    :param axis: 要压缩的轴，默认为None。 
    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        import numpy as np
        a = np.arange(6).reshape(1,6,1).astype(np.float32)
        A = XTensor(a)
        AA = tensor.squeeze(A,0)
        print(AA)

        # [
        # [0.],
        # [1.],
        # [2.],
        # [3.],
        # [4.],
        # [5.]
        # ]
        

unsqueeze
===========================================================

.. py:function:: pyvqnet.xtensor.unsqueeze(t: pyvqnet.xtensor.XTensor, axis: int = 0)

    在axis 指定的维度上插入一个维度为的1的轴，返回一个新的 XTensor 。

    :param t: 输入 XTensor 。
    :param axis: 要插入维度的位置，默认为0。 
    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        import numpy as np
        a = np.arange(24).reshape(2,1,1,4,3).astype(np.float32)
        A = XTensor(a)
        AA = tensor.unsqueeze(A,1)
        print(AA)

        # [
        # [[[[[0., 1., 2.],
        #  [3., 4., 5.],
        #  [6., 7., 8.],
        #  [9., 10., 11.]]]]],
        # [[[[[12., 13., 14.],
        #  [15., 16., 17.],
        #  [18., 19., 20.],
        #  [21., 22., 23.]]]]]
        # ]
        

swapaxis
===========================================================

.. py:function:: pyvqnet.xtensor.swapaxis(t, axis1: int, axis2: int)

    交换输入 t 的 第 axis1 和 axis 维度。

    :param t: 输入 XTensor 。
    :param axis1: 要交换的第一个轴。
    :param axis2:  要交换的第二个轴。
    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor

        import numpy as np
        a = np.arange(24).reshape(2,3,4).astype(np.float32)
        A = XTensor(a)
        AA = tensor.swapaxis(A, 2, 1)
        print(AA)

        # [
        # [[0., 4., 8.],
        #  [1., 5., 9.],
        #  [2., 6., 10.],
        #  [3., 7., 11.]],
        # [[12., 16., 20.],
        #  [13., 17., 21.],
        #  [14., 18., 22.],
        #  [15., 19., 23.]]
        # ]

masked_fill
===========================================================

.. py:function:: pyvqnet.xtensor.masked_fill(t, mask, value)

    在 mask == 1 的位置，用值 value 填充输入。
    mask的形状必须与输入的 XTensor 的形状是可广播的。

    :param t: 输入 XTensor。
    :param mask: 掩码 XTensor,必须是kbool。
    :param value: 填充值。
    :return: 一个 XTensor。

    Examples::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        import numpy as np
        a = tensor.ones([2, 2, 2, 2])
        mask = np.random.randint(0, 2, size=4).reshape([2, 2])
        b = tensor.XTensor(mask==1)
        c = tensor.masked_fill(a, b, 13)
        print(c)
        # [
        # [[[1., 1.],  
        #  [13., 13.]],
        # [[1., 1.],   
        #  [13., 13.]]],
        # [[[1., 1.],
        #  [13., 13.]],
        # [[1., 1.],
        #  [13., 13.]]]
        # ]

flatten
===========================================================

.. py:function:: pyvqnet.xtensor.flatten(t: pyvqnet.xtensor.XTensor, start: int = 0, end: int = - 1)

    将输入 t 从 start 到 end 的连续维度展平。

    :param t: 输入 XTensor 。
    :param start: 展平开始的轴，默认 = 0，从第一个轴开始。
    :param end: 展平结束的轴，默认 = -1，以最后一个轴结束。
    :return: 输出 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = XTensor([1, 2, 3])
        x = tensor.flatten(t)
        print(x)

        # [1., 2., 3.]


reshape
===========================================================

.. py:function:: pyvqnet.xtensor.reshape(t: pyvqnet.xtensor.XTensor,new_shape)

    改变 XTensor 的形状，返回一个新的张量。

    :param t: 输入 XTensor 。
    :param new_shape: 新的形状。

    :return: 新形状的 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        import numpy as np
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t = XTensor(a)
        reshape_t = tensor.reshape(t, [C, R])
        print(reshape_t)
        # [
        # [0., 1., 2.],
        # [3., 4., 5.],
        # [6., 7., 8.],
        # [9., 10., 11.]
        # ]

flip
===========================================================

.. py:function:: pyvqnet.xtensor.flip(t, flip_dims)

    沿指定轴反转XTensor，返回一个新的张量。

    :param t: 输入 XTensor 。
    :param flip_dims: 需要翻转的轴或轴列表。

    :return: 新形状的 XTensor 。

    Example::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        t = tensor.arange(1, 3 * 2 *2 * 2 + 1).reshape([3, 2, 2, 2])
        t.requires_grad = True
        y = tensor.flip(t, [0, -1])
        print(y)
        # [
        # [[[18., 17.], 
        #  [20., 19.]], 
        # [[22., 21.],  
        #  [24., 23.]]],
        # [[[10., 9.],  
        #  [12., 11.]], 
        # [[14., 13.],  
        #  [16., 15.]]],
        # [[[2., 1.],   
        #  [4., 3.]],   
        # [[6., 5.],    
        #  [8., 7.]]]   
        # ]

broadcast_to
===========================================================

.. py:function:: pyvqnet.xtensor.broadcast_to(t, ref)

    受到某些约束，数组 t 被“广播”到参考形状，以便它们具有兼容的形状。

    https://numpy.org/doc/stable/user/basics.broadcasting.html

    :param t: 输入XTensor
    :param ref: 参考形状。
    
    :return: 新广播的 t 的 XTensor。

    Example::

        import pyvqnet.xtensor as tensor
        ref = [2,3,4]
        a = tensor.ones([4])
        b = tensor.broadcast_to(a,ref)
        print(b.shape)
        #[2, 3, 4]



实用函数
***********************


to_xtensor
===========================================================

.. py:function:: pyvqnet.xtensor.to_xtensor(x,device=None,dtype=None)

    将输入数值或 numpy.ndarray 等转换为 XTensor 。

    :param x: 整数、浮点数、布尔数、复数、或 numpy.ndarray
    :param device: 储存在哪个设备上，默认: None，在CPU上。
    :param dtype: 参数的数据类型，defaults：None，使用默认数据类型:kfloat32,代表32位浮点数。

    :return: 输出 XTensor

    Example::

        import pyvqnet.xtensor as tensor
        t = tensor.to_xtensor(10.0)
        print(t)

        # [10.]
        

pad_sequence
===========================================================

.. py:function:: pyvqnet.xtensor.pad_sequence(qtensor_list, batch_first=False, padding_value=0)

    用 ``padding_value`` 填充可变长度张量列表。 ``pad_sequence`` 沿新维度堆叠张量列表，并将它们填充到相等的长度。
    输入是列表大小为 ``L x *`` 的序列。 L 是可变长度。

    :param qtensor_list: `list[XTensor]`- 可变长度序列列表。
    :param batch_first: 'bool' - 如果为真，输出将是 ``批大小 x 最长序列长度 x *`` ，否则为 ``最长序列长度 x 批大小 x *`` 。 默认值: False。
    :param padding_value: 'float' - 填充值。 默认值：0。

    :return:
        如果 batch_first 为 ``False``，则张量大小为 ``批大小 x 最长序列长度 x *``。
        否则张量的大小为 ``最长序列长度 x 批大小 x *`` 。

    Examples::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        a = tensor.ones([4, 2,3])
        b = tensor.ones([1, 2,3])
        c = tensor.ones([2, 2,3])
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        y = tensor.pad_sequence([a, b, c], True)

        print(y)
        # [
        # [[[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[1., 1., 1.],
        #  [1., 1., 1.]]],
        # [[[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[0., 0., 0.],
        #  [0., 0., 0.]],
        # [[0., 0., 0.],
        #  [0., 0., 0.]],
        # [[0., 0., 0.],
        #  [0., 0., 0.]]],
        # [[[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[1., 1., 1.],
        #  [1., 1., 1.]],
        # [[0., 0., 0.],
        #  [0., 0., 0.]],
        # [[0., 0., 0.],
        #  [0., 0., 0.]]]
        # ]


pad_packed_sequence
===========================================================

.. py:function:: pyvqnet.xtensor.pad_packed_sequence(sequence, batch_first=False, padding_value=0, total_length=None)

    填充一批打包的可变长度序列。它是 `pack_pad_sequence` 的逆操作。
    当  ``batch_first`` 是 True，它将返回  ``B x T x *`` 形状的张量，否则返回  ``T x B x *``。
    其中 `T` 为序列最长长度, `B` 为批处理大小。



    :param sequence: 'XTensor' - 待处理数据。
    :param batch_first: 'bool' - 如果为 ``True`` ，批处理将是输入的第一维。 默认值：False。
    :param padding_value: 'bool' - 填充值。默认:0。
    :param total_length: 'bool' - 如果不是 ``None`` ，输出将被填充到长度 :attr:`total_length`。 默认值：None。
    :return:
        包含填充序列的张量元组，以及批次中每个序列的长度列表。批次元素将按照最初的顺序重新排序。

    Examples::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
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
        # [[[[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[1. 1. 1.]
        #    [1. 1. 1.]]]


        #  [[[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]]


        #  [[[1. 1. 1.]
        #    [1. 1. 1.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]

        #   [[0. 0. 0.]
        #    [0. 0. 0.]]]]
        print(lens_unpacked)
        # [4, 2, 1]


pack_pad_sequence
===========================================================

.. py:function:: pyvqnet.xtensor.pack_pad_sequence(input, lengths, batch_first=False, enforce_sorted=True)

    打包一个包含可变长度填充序列的张量。
    如果 batch_first 是 True, `input` 的形状应该为 [批大小,长度,*]，否则形状 [长度，批大小,*]。

    对于未排序的序列，使用 ``enforce_sorted`` 是 False。 如果 :attr:`enforce_sorted` 是 ``True``，序列应该按长度降序排列。

    :param input: 'XTensor' - 填充的可变长度序列。
    :param lengths: 'list' - 每个批次的序列长度。
    :param batch_first: 'bool' - 如果 ``True``，则输入预期为 ``B x T x *``
        格式，默认：False。
    :param enforce_sorted: 'bool' - 如果 ``True``，输入应该是
        包含按长度降序排列的序列。 如果 ``False``，输入将无条件排序。 默认值：True。

    :return: 一个 :class:`PackedSequence` 对象。

    Examples::

        from pyvqnet.xtensor import XTensor
        import pyvqnet.xtensor as tensor
        a = tensor.ones([4, 2,3])
        c = tensor.ones([1, 2,3])
        b = tensor.ones([2, 2,3])
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

        # [[[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]

        #  [[1. 1. 1.]
        #   [1. 1. 1.]]]

        print(data.batch_sizes)
        # [3, 2, 1, 1]