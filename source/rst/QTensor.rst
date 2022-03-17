QTensor 模块
==============

VQNet量子机器学习所使用的数据结构QTensor的python接口文档。QTensor支持常用的多维矩阵的操作包括创建函数，数学函数，逻辑函数，矩阵变换等。



QTensor's 函数与属性
----------------------------------


__init__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: QTensor.__init__(data, requires_grad=False, nodes=None, DEVICE=0)

    Wrapper of data structure with dynamic computational graph construction and automatic differentiation.

    :param data: _core.Tensor or numpy array which represents a QTensor
    :param requires_grad: should tensor's gradient be tracked, defaults to False
    :param nodes: list of successors in the computational graph, defaults to None
    :param DEVICE: current device to save QTensor ,default = 0
    :return: output QTensor

    .. note::
            QTensor only accepts float number as input. It supports at most 7 significant digits.

    Example::


        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        from pyvqnet._core import Tensor as CoreTensor
        t1 = QTensor(np.ones([2,3]))
        t2 = QTensor(CoreTensor.ones([2,3]))
        t3 =  QTensor([2,3,4,5])
        t4 =  QTensor([[[2,3,4,5],[2,3,4,5]]])
        print(t1)

        # [
        # [1.0000000, 1.0000000, 1.0000000],
        # [1.0000000, 1.0000000, 1.0000000]
        # ]

        print(t2)

        # [
        # [1.0000000, 1.0000000, 1.0000000],
        # [1.0000000, 1.0000000, 1.0000000]
        # ]

        print(t3)

        #[2.0000000, 3.0000000, 4.0000000, 5.0000000]

        print(t4)

        # [
        # [[2.0000000, 3.0000000, 4.0000000, 5.0000000],
        #  [2.0000000, 3.0000000, 4.0000000, 5.0000000]]
        # ]


ndim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.ndim

    Return number of dimensions
        
    :return: number of dimensions

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([2, 3, 4, 5], requires_grad=True)
        print(a.ndim)

        # 1
    
shape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.shape

    Return the shape of the QTensor.
    
    :return: value of shape

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([2, 3, 4, 5], requires_grad=True)
        print(a.shape)

        # [4]

size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.size

    Return the number of elements in the QTensor.
    
    :return: number of elements

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([2, 3, 4, 5], requires_grad=True)
        print(a.size)

        # 4

zero_grad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.zero_grad()

    Sets gradient to zero. Will be used by optimizer in the optimization process.

    :return: None

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t3 = QTensor([2, 3, 4, 5], requires_grad=True)
        t3.zero_grad()
        print(t3.grad)
        # [0.0000000, 0.0000000, 0.0000000, 0.0000000]
        

backward
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.backward(grad=None)

    Computes the gradient of current QTensor .

    :return: None

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        target = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], requires_grad=True)
        y = 2*target + 3
        y.backward()
        print(target.grad)
        # [
        # [2.0000000, 2.0000000, 2.0000000, 2.0000000, 2.0000000, 2.0000000, 2.0000000, 2.0000000, 2.0000000, 2.0000000]
        # ]

to_numpy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.to_numpy()

    Copy self data to a new numpy.array.

    :return: a new numpy.array contains QTensor data

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

    Return the only element from in the QTensor. Raises ‘RuntimeError’ if QTensor has more than 1 element.

    :return: only data of this object

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.ones([1])
        print(t.item())

        # 1.0

argmax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.argmax(*kargs)

    Return the indices of the maximum value of all elements in the input QTensor,or returns the indices of the maximum values of a QTensor across a dimension.

    :param dim: dim ([int]]) – the dimension to reduce,only accepts single axis. if dim == None, returns the indices of the maximum value of all elements in the input tensor.The valid dim range is [-R, R), where R is input's ndim. when dim < 0, it works the same way as dim + R.
    :param keepdims:   whether the output QTensor has dim retained or not.
    :return: the indices of the maximum value in the input QTensor.

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

    Return the indices of the minimum value of all elements in the input QTensor,or returns the indices of the minimum values of a QTensor across a dimension.

    :param dim: dim ([int]]) – the dimension to reduce,only accepts single axis. if dim == None, returns the indices of the maximum value of all elements in the input tensor.The valid dim range is [-R, R), where R is input's ndim. when dim < 0, it works the same way as dim + R.
    :param keepdims:    whether the output QTensor has dim retained or not.
    :return: the indices of the minimum value in the input QTensor.

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

     Fill the QTensor with the specified value.

    :param v: a scalar value
    :return: None

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

    Return true if all QTensor value is non-zero.

    :return: True,if all QTensor value is non-zero.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        shape = [2, 3]
        t = tensor.zeros(shape)
        t.fill_(1.0)
        flag = t.all()
        print(flag)

        # True

any
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.any()

    Return true if any QTensor value is non-zero.

    :return: True,if any QTensor value is non-zero.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        shape = [2, 3]
        t = tensor.ones(shape)
        t.fill_(1.0)
        flag = t.any()
        print(flag)

        # True


fill_rand_binary\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.fill_rand_binary_(v=0.5)

    Fills a QTensor with values randomly sampled from a binomial distribution.

    If the data generated randomly after binomial distribution is greater than Binarization threshold, then the number of corresponding positions of the QTensor is set to 1, otherwise 0.

    :param v: Binarization threshold
    :return: None

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

    Fills a QTensor with values randomly sampled from a signed uniform distribution.

    Scale factor of the values generated by the signed uniform distribution.

    :param v: a scalar value
    :return: None

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

    Fills a QTensor with values randomly sampled from a uniform distribution

    Scale factor of the values generated by the uniform distribution.

    :param v: a scalar value
    :return: None

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

    Fills a QTensor with values randomly sampled from a normal distribution
    Mean of the normal distribution. Standard deviation of the normal distribution.
    Whether to use or not the fast math mode.

    :param m: mean of the normal distribution
    :param s: standard deviation of the normal distribution
    :param fast_math: True if use fast-math
    :return: None

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

    Reverse or permute the axes of an array.if new_dims = None, revsers the dim.

    :param new_dims: the new order of the dimensions (list of integers).
    :return:  result QTensor.

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

    Reverse or permute the axes of an array inplace.if new_dims = None, revsers the dim.

    :param new_dims: the new order of the dimensions (list of integers).
    :return: None.

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

    Change the tensor’s shape ,return a new QTensor.

    :param new_shape: the new shape (list of integers)
    :return: a new QTensor

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

    Change the current object’s shape.

    :param new_shape: the new shape (list of integers)
    :return: None

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

    Get the tensor’s data as a NumPy array.

    :return: a NumPy array

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.ones([3, 4])
        a = t.getdata()
        print(a)

        # [[1. 1. 1. 1.]
        #  [1. 1. 1. 1.]
        #  [1. 1. 1. 1.]]


创建函数
-----------------------------


ones
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.ones(shape)

    Return one-tensor with the input shape.

    :param shape: input shape
    :return: output QTensor with the input shape.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        x = tensor.ones([2, 3])
        print(x)

        # [
        # [1.0000000, 1.0000000, 1.0000000],
        # [1.0000000, 1.0000000, 1.0000000]
        # ]

ones_like
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.ones_like(t: pyvqnet.tensor.QTensor)

    Return one-tensor with the same shape as the input QTensor.

    :param t: input QTensor
    :return: output QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.ones_like(t)
        print(x)

        # [1.0000000, 1.0000000, 1.0000000]
        


full
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.full(shape, value, dev: int = 0)

    Create a QTensor of the specified shape and fill it with value.

    :param shape: shape of the QTensor to create
    :param value: value to fill the QTensor with
    :param dev: device to use,default = 0 ,use cpu device.
    :return: output QTensor

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

.. py:function:: pyvqnet.tensor.full_like(t, value, dev: int = 0)

    Create a QTensor of the specified shape and fill it with value.

    :param t: input QTensor
    :param value: value to fill the QTensor with.
    :param dev: device to use,default = 0 ,use cpu device.
    :return: output QTensor

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

.. py:function:: pyvqnet.tensor.zeros(shape)

    Return zero-tensor of the input shape.

    :param shape: shape of tensor
    :return: output QTensor

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

.. py:function:: pyvqnet.tensor.zeros_like(t: pyvqnet.tensor.QTensor)

    Return zero-tensor with the same shape as the input QTensor.

    :param t: input QTensor
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.zeros_like(t)
        print(x)

        # [0.0000000, 0.0000000, 0.0000000]
        


arange
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.arange(start, end, step, dev: int = 0)

    Create a 1D QTensor with evenly spaced values within a given interval.

    :param start: start of interval
    :param end: end of interval
    :param step: spacing between values
    :param dev: device to use,default = 0 ,use cpu device.
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = tensor.arange(2, 30, 4)
        print(t)

        # [ 2.0000000,  6.0000000, 10.0000000, 14.0000000, 18.0000000, 22.0000000, 26.0000000]
        

linspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.linspace(start, end, steps, dev: int = 0)

    Create a 1D QTensor with evenly spaced values within a given interval.

    :param start: starting value
    :param end: end value
    :param steps: number of samples to generate
    :param dev: device to use,default = 0 ,use cpu device.
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        start, stop, steps = -2.5, 10, 10
        t = tensor.linspace(start, stop, steps)
        print(t)
        #[-2.5000000, -1.1111112, 0.2777777, 1.6666665, 3.0555553, 4.4444442, 5.8333330, 7.2222219, 8.6111107, 10.0000000]

logspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.logspace(start, end, steps, base, dev: int = 0)

    Create a 1D QTensor with evenly spaced values on a log scale.

    :param start: ``base ** start`` is the starting value
    :param end: ``base ** end`` is the final value of the sequence
    :param steps: number of samples to generate
    :param base: the base of the log space
    :param dev: device to use,default = 0 ,use cpu device.
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        start, stop, steps, base = 0.1, 1.0, 5, 10.0
        t = tensor.logspace(start, stop, steps, base)
        print(t)

        # [1.2589254, 2.1134889, 3.5481336, 5.9566211, 10.0000000]
        

eye
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.eye(size, offset: int = 0, dev: int = 0)

    Create a size x size QTensor with ones on the diagonal and zeros elsewhere.

    :param size: size of the (square) QTensor to create
    :param offset: Index of the diagonal: 0 (the default) refers to the main diagonal, a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
    :param dev: device to use,default = 0 ,use cpu device.
    :return: output QTensor

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

.. py:function:: pyvqnet.tensor.diag(t, k: int = 0)

    Select diagonal elements or construct a diagonal QTensor.

    If input is 2-D QTensor,returns a new tensor which is the same as this one, except that
    elements other than those in the selected diagonal are set to zero.

    If v is a 1-D QTensor, return a 2-D QTensor with v on the k-th diagonal.

    :param t: input QTensor
    :param k: offset (0 for the main diagonal, positive for the nth diagonal above the main one, negative for the nth diagonal below the main one)
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        a = np.arange(16).reshape(4, 4).astype(np.float32)
        t = QTensor(a)
        for k in range(-3, 4):
            u = tensor.diag(t,k=k)
            print(u)

        # [
        # [0.0000000, 0.0000000, 0.0000000, 0.0000000],
        # [0.0000000, 0.0000000, 0.0000000, 0.0000000],
        # [0.0000000, 0.0000000, 0.0000000, 0.0000000],
        # [12.0000000, 0.0000000, 0.0000000, 0.0000000]
        # ]

        # [
        # [0.0000000, 0.0000000, 0.0000000, 0.0000000],
        # [0.0000000, 0.0000000, 0.0000000, 0.0000000],
        # [8.0000000, 0.0000000, 0.0000000, 0.0000000],
        # [0.0000000, 13.0000000, 0.0000000, 0.0000000]
        # ]

        # [
        # [0.0000000, 0.0000000, 0.0000000, 0.0000000],
        # [4.0000000, 0.0000000, 0.0000000, 0.0000000],
        # [0.0000000, 9.0000000, 0.0000000, 0.0000000],
        # [0.0000000, 0.0000000, 14.0000000, 0.0000000]
        # ]

        # [
        # [0.0000000, 0.0000000, 0.0000000, 0.0000000],
        # [0.0000000, 5.0000000, 0.0000000, 0.0000000],
        # [0.0000000, 0.0000000, 10.0000000, 0.0000000],
        # [0.0000000, 0.0000000, 0.0000000, 15.0000000]
        # ]

        # [
        # [0.0000000, 1.0000000, 0.0000000, 0.0000000],
        # [0.0000000, 0.0000000, 6.0000000, 0.0000000],
        # [0.0000000, 0.0000000, 0.0000000, 11.0000000],
        # [0.0000000, 0.0000000, 0.0000000, 0.0000000]
        # ]

        # [
        # [0.0000000, 0.0000000, 2.0000000, 0.0000000],
        # [0.0000000, 0.0000000, 0.0000000, 7.0000000],
        # [0.0000000, 0.0000000, 0.0000000, 0.0000000],
        # [0.0000000, 0.0000000, 0.0000000, 0.0000000]
        # ]

        # [
        # [0.0000000, 0.0000000, 0.0000000, 3.0000000],
        # [0.0000000, 0.0000000, 0.0000000, 0.0000000],
        # [0.0000000, 0.0000000, 0.0000000, 0.0000000],
        # [0.0000000, 0.0000000, 0.0000000, 0.0000000]
        # ]


        

randu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.randu(shape, dev: int = 0)

    Create a QTensor with uniformly distributed random values.

    :param shape: shape of the QTensor to create
    :param dev: device to use,default = 0 ,use cpu device.
    :return: output QTensor

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

.. py:function:: pyvqnet.tensor.randn(shape, dev: int = 0)

    Create a QTensor with normally distributed random values.

    :param shape: shape of the QTensor to create
    :param dev: device to use,default = 0 ,use cpu device.
    :return: output QTensor

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
        



数学函数
-----------------------------


floor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.floor(t)

    Return a new QTensor with the floor of the elements of input, the largest integer less than or equal to each element.

    :param t: input Qtensor
    :return: output QTensor

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

    Return a new QTensor with the ceil of the elements of input, the smallest integer greater than or equal to each element.

    :param t: input Qtensor
    :return: output QTensor

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

    Round QTensor values to the nearest integer.

    :param t: input QTensor
    :return: output QTensor

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

    Sort QTensor along the axis

    :param t: input QTensor
    :param axis: sort axis
    :param descending: sort order if desc
    :param stable:  Whether to use stable sorting or not
    :return: output QTensor

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

    Return an array of indices of the same shape as input that index data along the given axis in sorted order.

    :param t: input QTensor
    :param axis: sort axis
    :param descending: sort order if desc
    :param stable:  Whether to use stable sorting or not
    :return: output QTensor

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
        
add
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.add(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Element-wise adds two QTensors .

    :param t1: first QTensor
    :param t2: second QTensor
    :return:  output QTensor

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

    Element-wise subtracts two QTensors.


    :param t1: first QTensor
    :param t2: second QTensor
    :return:  output QTensor

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

    Element-wise multiplies two QTensors.

    :param t1: first QTensor
    :param t2: second QTensor
    :return:  output QTensor


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

    Element-wise divides two QTensors.


    :param t1: first QTensor
    :param t2: second QTensor
    :return:  output QTensor


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

    Sums all the elements in QTensor along given axis.if axis = None, sums all the elements in QTensor. 

    :param t: input QTensor
    :param axis: axis used to sum,defaults to None
    :param keepdims:  whether the output tensor has dim retained or not. - defaults to False
    :return:  output QTensor


    Example::


        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor(([1, 2, 3], [4, 5, 6]))
        x = tensor.sums(t)
        print(x)

        # [21.0000000]

mean
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.mean(t: pyvqnet.tensor.QTensor, axis=None, keepdims=False)


    Obtain the mean values in the QTensor along the axis.

    :param t:  the input QTensor.
    :param axis:  the dimension to reduce.
    :param keepdims:  whether the output QTensor has dim retained or not,defaults to False.
    :return: returns the mean value of the input QTensor.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.mean(t, axis=1)
        print(x)

        # [2.0000000, 5.0000000]

median
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.median(t: pyvqnet.tensor.QTensor, *kargs)

    Obtain the median value in the QTensor.

    :param t: the input QTensor.
    :param dim:  the dimension to reduce.
    :param keepdims:  whether the output QTensor has dim retained or not,defaults to False.

    :return: Return the median of the values in input.

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
        median_b = tensor.median(b,[1], False)
        print(median_b)

        # [-0.3982000, 0.2269999, 0.2487999, 0.4742000]

std
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.std(t: pyvqnet.tensor.QTensor, *kargs)

    Obtain the standard variance value in the QTensor.


    :param t:  the input QTensor.
    :param dim:  the dimension to reduce.
    :param keepdims:  whether the output QTensor has dim retained or not,default False.
    :param unbiased:  whether to use Bessel’s correction,default True.
    :return: Return the standard variance of the values in input.

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
        std_b = tensor.std(b, [1], False, False)
        print(std_b)

        # [0.6593542, 0.5583112, 0.3206565, 1.1103367]

var
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.var(t: pyvqnet.tensor.QTensor, *kargs)

    Obtain the variance in the QTensor.


    :param t:  the input QTensor.
    :param dim:  the dimension to reduce.
    :param keepdims:  whether the output QTensor has dim retained or not,default False
    :param unbiased:  whether to use Bessel’s correction,default True.


    :return: Obtain the variance in the QTensor.

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

    Matrix multiplications of two 2d or 4d matrix.

    :param t1: first QTensor
    :param t2: second QTensor
    :return:  output QTensor

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


reciprocal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.reciprocal(t)

    Compute the element-wise reciprocal of the QTensor.

    :param t: input QTensor
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(1, 10, 1)
        u = tensor.reciprocal(t)
        print(u)

        #[1.0000000, 0.5000000, 0.3333333, 0.2500000, 0.2000000, 0.1666667, 0.1428571, 0.1250000, 0.1111111]

sign
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.sign(t)

    Return a new QTensor with the signs of the elements of input.The sign function returns -1 if t < 0, 0 if t==0, 1 if t > 0.

    :param t: input QTensor
    :return: output QTensor


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

    Unary negation of QTensor elements.

    :param t: input QTensor
    :return:  output QTensor

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

    Return the sum of the elements of the diagonal of the input 2-D matrix.

    :param t: input 2-D QTensor
    :param k: offset (0 for the main diagonal, positive for the nth
        diagonal above the main one, negative for the nth diagonal below the
        main one)
    :return: the sum of the elements of the diagonal of the input 2-D matrix.

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

    Applies exponential function to all the elements of the input QTensor.

    :param t: input QTensor
    :return:  output QTensor

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

    Compute the element-wise inverse cosine of the QTensor. 

    :param t: input QTensor
    :return: output QTensor

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

    Compute the element-wise inverse sine of the QTensor.

    :param t: input QTensor
    :return: output QTensor

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

    Compute the element-wise inverse tangent of the QTensor.

    :param t: input QTensor
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = tensor.arange(-1, 1, .5)
        u = tensor.atan(t)
        print(u)

        # [-0.7853981, -0.4636476, 0.0000000, 0.4636476]

sin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.sin(t: pyvqnet.tensor.QTensor)

    Applies sine function to all the elements of the input QTensor.

    :param t: input QTensor
    :return:  output QTensor

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

    Applies cosine function to all the elements of the input QTensor.

    :param t: input QTensor
    :return:  output QTensor

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

    Applies tangent function to all the elements of the input QTensor.

    :param t: input QTensor
    :return:  output QTensor

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

    Applies hyperbolic tangent function to all the elements of the input QTensor.

    :param t: input QTensor
    :return:  output QTensor

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

    Applies hyperbolic sine function to all the elements of the input QTensor.


    :param t: input QTensor
    :return:  output QTensor

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

    Applies hyperbolic cosine function to all the elements of the input QTensor.

    :param t: input QTensor
    :return:  output QTensor

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

    Raises first QTensor to the power of second QTensor.

    :param t1: first QTensor
    :param t2: second QTensor
    :return:  output QTensor

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

    Applies abs function to all the elements of the input QTensor.

    :param t: input QTensor
    :return:  output QTensor

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

    Applies log (ln) function to all the elements of the input QTensor.

    :param t: input QTensor
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.log(t)
        print(x)

        # [0.0000000, 0.6931471, 1.0986123]

sqrt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.sqrt(t: pyvqnet.tensor.QTensor)

    Applies sqrt function to all the elements of the input QTensor.


    :param t: input QTensor
    :return:  output QTensor

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

    Applies square function to all the elements of the input QTensor.


    :param t: input QTensor
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.square(t)
        print(x)

        # [1.0000000, 4.0000000, 9.0000000]

逻辑函数
--------------------------

maximum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.maximum(t1: pyvqnet.tensor.QTensor, t2: pyvqnet.tensor.QTensor)

    Element-wise maximum of two tensor.


    :param t1: first QTensor
    :param t2: second QTensor
    :return:  output QTensor

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

    Element-wise minimum of two tensor.


    :param t1: first QTensor
    :param t2: second QTensor
    :return:  output QTensor

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

    Return min elements of the input QTensor alongside given axis.
    if axis == None, return the min value of all elements in tensor.

    :param t: input QTensor
    :param axis: axis used to min, defaults to None
    :param keepdims:  whether the output tensor has dim retained or not. - defaults to False
    :return: output QTensor or float

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

    Return max elements of the input QTensor alongside given axis.
    if axis == None, return the max value of all elements in tensor.

    :param t: input QTensor
    :param axis: axis used to max,defaults to None
    :param keepdims:  whether the output tensor has dim retained or not. - defaults to False
    :return: output QTensor or float

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

    Clips input QTensor to minimum and maximum value.

    :param t: input QTensor
    :param min_val:  minimum value
    :param max_val:  maximum value
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([2, 4, 6])
        x = tensor.clip(t, 3, 8)
        print(x)

        # [3.0000000, 4.0000000, 6.0000000]


where
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.where(condition: pyvqnet.tensor.QTensor, t1: Optional[pyvqnet.tensor.QTensor] = None, t2: Optional[pyvqnet.tensor.QTensor] = None)

    Return elements chosen from t1 or t2 depending on condition.

    :param condition: condition tensor
    :param t1: QTensor from which to take elements if condition is met, defaults to None
    :param t2: QTensor from which to take elements if condition is not met, defaults to None
    :return: output QTensor

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

.. py:function:: pyvqnet.tensor.nonzero(A)

    Return a QTensor containing the indices of nonzero elements.


    :param A: input QTensor
    :return: output QTensor contains indices of nonzero elements.

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

.. py:function:: pyvqnet.tensor.isfinite(A)

    Test element-wise for finiteness (not infinity or not Not a Number).

    :param A: input QTensor
    :return: output QTensor with each elements presents 1, if the QTensor value is isfinite. else 0.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isfinite(t)
        print(flag)

        # [1.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000]

isinf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.isinf(A)

    Test element-wise for positive or negative infinity.

    :param A: input QTensor
    :return: output QTensor with each elements presents 1, if the QTensor value is isinf. else 0.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isinf(t)
        print(flag)

        # [0.0000000, 1.0000000, 0.0000000, 1.0000000, 0.0000000]

isnan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.isnan(A)

    Test element-wise for Nan.

    :param A: input QTensor
    :return: output QTensor with each elements presents 1, if the QTensor value is isnan. else 0.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isnan(t)
        print(flag)

        # [0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000]

isneginf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.isneginf(A)

    Test element-wise for negative infinity.

    :param A: input QTensor
    :return: output QTensor with each elements presents 1, if the QTensor value is isneginf. else 0.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isneginf(t)
        print(flag)

        # [0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000]

isposinf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.isposinf(A)

    Test element-wise for positive infinity.

    :param A: input QTensor
    :return: output QTensor with each elements presents 1, if the QTensor value is isposinf. else 0.

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isposinf(t)
        print(flag)

        # [0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000]

logical_and
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.logical_and(A, B)

    Compute the truth value of ``A`` and ``B`` element-wise.If logicial calculation result is False, it presents 0,else 1.

    :param A: input QTensor
    :param B: input QTensor
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_and(a,b)
        print(flag)

        # [0.0000000, 0.0000000, 1.0000000, 0.0000000]

logical_or
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.logical_or(A, B)

    Compute the truth value of ``A or B`` element-wise.If logicial calculation result is False, it presents 0,else 1.

    :param A: input QTensor
    :param B: input QTensor
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_or(a,b)
        print(flag)

        # [1.0000000, 1.0000000, 1.0000000, 0.0000000]

logical_not
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.logical_not(A)

    Compute the truth value of ``not A`` element-wise.If logicial calculation result is False, it presents 0,else 1.

    :param A: input QTensor
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        flag = tensor.logical_not(a)
        print(flag)

        # [1.0000000, 0.0000000, 0.0000000, 1.0000000]

logical_xor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.logical_xor(A, B)

    Compute the truth value of ``A xor B`` element-wise.If logicial calculation result is False, it presents 0,else 1.

    :param A: input QTensor
    :param B: input QTensor
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_xor(a,b)
        print(flag)

        # [1.0000000, 1.0000000, 0.0000000, 0.0000000]

greater
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.greater(A, B)

    Return the truth value of ``A > B`` element-wise.


    :param A: input QTensor
    :param B: input QTensor
    :return: output QTensor that is 1 where A is greater than B and False elsewhere 

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.greater(a,b)
        print(flag)

        # [
        # [0.0000000, 1.0000000],
        #  [0.0000000, 0.0000000]
        # ]

greater_equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.greater_equal(A, B)

    Return the truth value of ``A >= B`` element-wise.

    :param A: input QTensor
    :param B: input QTensor
    :return: output QTensor that is 1 where A is greater than or equal to B and False elsewhere

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.greater_equal(a,b)
        print(flag)

        # [
        # [1.0000000, 1.0000000],
        #  [0.0000000, 1.0000000]
        # ]

less
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.less(A, B)

    Return the truth value of ``A < B`` element-wise.


    :param A: input QTensor
    :param B: input QTensor
    :return: output QTensor that is 1 where A is less than B and False elsewhere

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.less(a,b)
        print(flag)

        # [
        # [0.0000000, 0.0000000],
        #  [1.0000000, 0.0000000]
        # ]

less_equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.less_equal(A, B)

    Return the truth value of ``A <= B`` element-wise.


    :param A: input QTensor
    :param B: input QTensor
    :return: output QTensor that is 1 where A is less than or equal to B and False elsewhere

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.less_equal(a,b)
        print(flag)

        # [
        # [1.0000000, 0.0000000],
        #  [1.0000000, 1.0000000]
        # ]

equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.equal(A, B)

    Return the truth value of ``B == A`` element-wise.


    :param A: input QTensor
    :param B: input QTensor
    :return: output QTensor that is 1 where A is equal to B and False elsewhere

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.equal(a,b)
        print(flag)

        # [
        # [1.0000000, 0.0000000],
        #  [0.0000000, 1.0000000]
        # ]

not_equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.not_equal(A, B)

    Return the truth value of ``B != A`` element-wise.

    :param A: input QTensor
    :param B: input QTensor
    :return: output QTensor that is 1 where A is not equal to B and False elsewhere

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.not_equal(a,b)
        print(flag)

        # [
        # [0.0000000, 1.0000000],
        #  [1.0000000, 0.0000000]
        # ]

矩阵操作
--------------------------

select
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.select(t: pyvqnet.tensor.QTensor, index)

    Return QTensor in the QTensor at the given axis. following operation get same result's value.
    
    :param t: input QTensor
    :param index: a string contains output dim  
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        import numpy as np
        t = QTensor(np.arange(1,25).reshape(2,3,4))
              
        indx = [":", "0", ":"]        
        t.requires_grad = True
        t.zero_grad()
        ts = tensor.select(t,indx)
        ts.backward(tensor.ones(ts.shape))
        print(ts)  
        # [
        # [[1.0000000, 2.0000000, 3.0000000, 4.0000000]],
        # [[13.0000000, 14.0000000, 15.0000000, 16.0000000]]
        # ]

concatenate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.concatenate(args: list, axis=1)

    Concatenate with channel, i.e. concatenate C of QTensor shape (N,C,H,W)

    :param args: list consist of input QTensors
    :param axis: dimension to concatenate. Has to be between 0 and the number of dimensions of concatenate tensors.
    :return: output QTensor

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

    Join a sequence of arrays along a new axis,return a new QTensor.

    :param QTensors: list contains QTensors
    :param axis: dimension to insert. Has to be between 0 and the number of dimensions of stacked tensors.
    :return: output QTensor

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

    Reverse or permute the axes of an array.if dims = None, reverse the dim.

    :param t: input QTensor
    :param dim: the new order of the dimensions (list of integers).
    :return: output QTensor.

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

    Transpose the axes of an array.if dim = None, reverse the dim. This function is same as permute.

    :param t: input QTensor
    :param dim: the new order of the dimensions (list of integers).
    :return: output QTensor.

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

    Construct a QTensor by repeating QTensor the number of times given by reps.

    If reps has length d, the result QTensor will have dimension of max(d, t.ndim).

    If t.ndim < d, t is expanded to be d-dimensional by inserting new axes from start dimension.
    So a shape (3,) array is promoted to (1, 3) for 2-D replication, or shape (1, 1, 3) for 3-D replication.

    If t.ndim > d, reps is expanded to t.ndim by inserting 1’s to it.

    Thus for an t of shape (2, 3, 4, 5), a reps of (4, 3) is treated as (1, 1, 4, 3).

    :param t: input QTensor
    :param reps: the number of repetitions per dimension.
    :return: a new QTensor

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

    Remove axes of length one .

    :param t: input QTensor
    :param axis: squeeze axis,if axis = -1 ,squeeze all the dimensions that have size of 1.
    :return: output QTensor

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

    Return a new QTensor with a dimension of size one inserted at the specified position.

    :param t: input QTensor
    :param axis: unsqueeze axis,which will insert dimension.
    :return: output QTensor

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

    Interchange two axes of an array.The given dimensions axis1 and axis2 are swapped.

    :param t: input QTensor
    :param axis1: First axis.
    :param axis2:  Destination position for the original axis. These must also be unique
    :return: output QTensor

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
        

flatten
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.flatten(t: pyvqnet.tensor.QTensor, start: int = 0, end: int = - 1)

    Flatten QTensor from dim start to dim end.

    :param t: input QTensor
    :param start: dim start,default = 0,start from first dim.
    :param end: dim end,default = -1,end with last dim.
    :return:  output QTensor

    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.flatten(t)
        print(x)

        # [1.0000000, 2.0000000, 3.0000000]
        

实用函数
-----------------------------


to_tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.to_tensor(x)

    Convert input array to Qtensor if it isn't already.

    :param x: integer,float or numpy.array
    :return: output QTensor

    Example::

        from pyvqnet.tensor import tensor

        t = tensor.to_tensor(10.0)
        print(t)

        # [10.0000000]
        
