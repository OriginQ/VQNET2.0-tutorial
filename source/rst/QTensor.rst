QTensor 模块
==============

VQNet量子机器学习所使用的数据结构QTensor的python接口文档。QTensor支持常用的多维矩阵的操作包括创建函数，数学函数，逻辑函数，矩阵变换等。



QTensor's 函数与属性
----------------------------------


__init__
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: QTensor.__init__(data, requires_grad=False, nodes=None, DEVICE=0)

    Wrapper of data structure with dynamic computational graph construction and automatic differentiation.

    :param data: _core.Tensor or numpy array which represents a tensor
    :param requires_grad: should tensor's gradient be tracked, defaults to False
    :param nodes: list of successors in the computational graph, defaults to None
    :param DEVICE: current device to save QTensor ,default = 0
    :return: A QTensor

    Example::

        from pyvqnet._core import Tensor as CoreTensor
        t = QTensor(np.ones([2,3]))
        t2 = QTensor(CoreTensor.ones([2,3]))
        t3 =  QTensor([2,3,4,5])
        t4 =  QTensor([[[2,3,4,5],[2,3,4,5]]])
        print(t)
        print(t2)
        print(t4)
        print(t3)

ndim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:property:: QTensor.ndim

    return number of dimensions
        
    :return: number of dimensions
    
shape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:property:: QTensor.shape

    Returns the shape of the tensor.
    
    :return: number of shape

size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:property:: QTensor.size

    Returns the number of elements in the tensor.
    
    :return: number of elements

zero_grad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.zero_grad()

    Sets gradient to zero. Will be used by optimizer in the optimization process.

    Example::

        t3  =  QTensor([2,3,4,5],requires_grad = True)
        t3.zero_grad()
        print(t3.grad)

backward
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.backward(grad=None)

to_numpy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.to_numpy()

    copy self.data to a new np.array.

    Example::

        t3  =  QTensor([2,3,4,5],requires_grad = True)
        t4 = t3.to_numpy()

   :return: a new np.array contains QTensor data

item
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.item()

    Returns the only element from in the tensor. ## Raises ‘RuntimeError’ if tensor has more than 1 element.

   :return: only data of this object

argmax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.argmax(*kargs)

    Returns the indices of the maximum value of all elements in the input tensor,or Returns the indices of the maximum values of a tensor across a dimension.

    :param dim: dim ([int]]) – the dimension to reduce. if dim == None, returns the indices of the maximum value of all elements in the input tensor
    :param keepdim:  keepdim (bool) – whether the output tensor has dim retained or not.
    :return: the indices of the maximum value in the input tensor.

    Example::

        a = QTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                   [-0.7401, -0.8805, -0.3402, -1.1936],
                   [0.4907, -1.3948, -1.0691, -0.3132],
                   [-1.6092, 0.5419, -0.2993, 0.3195]])
        flag = a.argmax()
        #[0.000000]
        flag_0 = a.argmax([0], True)
        #[
        #[0.000000, 3.000000, 0.000000, 3.000000]
        #]
        flag_1 = a.argmax(a[1], True)

argmin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.argmin(*kargs)

    Returns the indices of the minimum value of all elements in the input tensor,or Returns the indices of the minimum values of a tensor across a dimension.

    :param dim: dim ([int]]) – the dimension to reduce. if dim == None, returns the indices of the minimum value of all elements in the input tensor
    :param keepdim:   keepdim (bool) – whether the output tensor has dim retained or not.
    :return: the indices of the minimum value in the input tensor.

    Example::

        a = QTensor([[1.3398, 0.2663, -0.2686, 0.2450],
                   [-0.7401, -0.8805, -0.3402, -1.1936],
                   [0.4907, -1.3948, -1.0691, -0.3132],
                   [-1.6092, 0.5419, -0.2993, 0.3195]])
        flag = a.argmin()

        flag_0 = a.argmin([0], True)

        flag_1 = a.argmin(a[1], False)

fill\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.fill_(v)

     Fill the tensor with the specified value.

    :param v: a scalar value
    :return: None

    Example::

        shape = [2, 3]
        value = 42
        t = tensor.zeros(shape)
        t.fill_(value)

all
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.all()

    Return if all tensor value is non-zero.

    :return: True,if all tensor value is non-zero.

    Example::

        shape = [2, 3]
        t = tensor.zeros(shape)
        t.fill_(1.0)
        flag = t.all()

any
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.any()

    Return if any tensor value is non-zero.

    :return: True,if any tensor value is non-zero.

    Example::

        shape = [2, 3]
        t = tensor.ones(shape)
        t.fill_(1.0)
        flag = t.any()

fill_rand_binary\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.fill_rand_binary_(v=0.5)

    Fills a tensor with values randomly sampled from a binary distribution

    Binarization threshold. 1 if rnd() >= t, 0 otherwise

    :param v: threshold a scalar value 1 if rnd() >= t, 0 otherwise
    :return: A new tensor

    Example::

        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = QTensor(a)
        t.fill_rand_binary_(2)

fill_rand_signed_uniform\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.fill_rand_signed_uniform_(v=1)

    Fills a tensor with values randomly sampled from a signed uniform distribution

    Scale factor of the values generated by the signed uniform distribution.

    :param v: a scalar value
    :return: A new tensor

    Example::

        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = QTensor(a)
        value = 42

        t.fill_rand_signed_uniform_(value)

fill_rand_uniform\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.fill_rand_uniform_(v=1)

    Fills a tensor with values randomly sampled from a uniform distribution

    Scale factor of the values generated by the uniform distribution.

    :param v: a scalar value
    :return: A new tensor

    Example::

        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = QTensor(a)
        value = 42
        t.fill_rand_uniform_(value)

fill_rand_normal\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.fill_rand_normal_(m=0, s=1, fast_math=True)

    Fills a tensor with values randomly sampled from a normal distribution Mean of the normal distribution.
    Standard deviation of the normal distribution. Whether to use or not the fast math mode.

    :param m: mean a scalar value
    :param s: std a scalar value
    :param fast_math:  a bool value
    :return: A new tensor

    Example::

        a = np.arange(6).reshape(2, 3).astype(np.float32)
        t = QTensor(a)
        t.fill_rand_normal_(2, 10, True)

QTensor.transpose
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.transpose(new_dims=None)

    Reverse or permute the axes of an array.if new_dims = None, revsers the dim.

    :param new_dims: the new order of the dimensions (list of integers).
    :return: result tensor.

    Example::

        R, C = 3, 4
        a = np.arange(R * C).reshape([2,2,3]).astype(np.float32)
        t = QTensor(a)
        reshape_t = t.transpose([2,0,1])

transpose\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.transpose_(new_dims=None)

    Reverse or permute the axes of an array inplace.if new_dims = None, revsers the dim.

    :param new_dims: the new order of the dimensions (list of integers).
    :return: None.

    Example::

        R, C = 3, 4
        a = np.arange(R * C).reshape([2,2,3]).astype(np.float32)
        t = QTensor(a)
        reshape_t = t.transpose_([2,0,1])

QTensor.reshape
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.reshape(new_shape)

    Change the tensor’s shape ,return a new Tensor.

    :param new_shape: the new shape (list of integers)
    :return: new Tensor

    Example::

        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t = QTensor(a)
        reshape_t = t.reshape([C, R])
        print(reshape_t)
        -------------------------------------------
            [
            [0.000000, 1.000000, 2.000000],

            [3.000000, 4.000000, 5.000000],

            [6.000000, 7.000000, 8.000000],

            [9.000000, 10.000000, 11.000000]
            ]
        ------------------------------------------

reshape\_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.reshape_(new_shape)

    Change the current object’s shape.

    :param new_shape: the new shape (list of integers)
    :return: None

    Example::

        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t = QTensor(a)
        t.reshape_([C, R])
        print(t)
        -------------------------------------------
            [
            [0.000000, 1.000000, 2.000000],

            [3.000000, 4.000000, 5.000000],

            [6.000000, 7.000000, 8.000000],

            [9.000000, 10.000000, 11.000000]
            ]
        ------------------------------------------

getdata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: QTensor.getdata()

    Get the tensor’s data as a NumPy array.

    :return: a NumPy array

    Example::

        t = tensor.ones([3, 4])
        a = t.getdata()
          ----------------
          [[1. 1. 1. 1.]
           [1. 1. 1. 1.]
           [1. 1. 1. 1.]]
          ----------------

创建函数
-----------------------------


ones
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.ones(shape)

    Return one-tensor with the input shape.

    :param t: ‘QTensor’ - input parameter
    :return: QTensor with the input shape.

    Example::

        from vqnet.tensor import tensor
        from vqnet.tensor.tensor import QTensor
        x = tensor.ones([2,3])

ones_like
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.ones_like(t: pyvqnet.tensor.tensor.QTensor)

    Return one-tensor with the same shape as the input tensor.

    :param t: ‘QTensor’ - input parameter
    :return: QTensor

    Example::

        t = QTensor([1, 2, 3])
        x = tensor.ones_like(t)

full
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.full(shape, value, dev: int = 0)

    Create a tensor of the specified shape and fill it with value.

    :param shape: shape of the tensor to create
    :param dev: device to use,default = 0
    :param value: value to fill the tensor with
    :return: QTensor

    Example::

        shape = [2, 3]
        value = 42
        t = tensor.full(shape, value)

full_like
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.full_like(t, value, dev: int = 0)

    Create a tensor of the specified shape and fill it with value.

    :param t: input qtensor
    :param dev: device to use,default = 0
    :param value: value to fill the tensor with
    :return: QTensor

    Example::

        a = tensor.randu([3,5])
        value = 42
        t = tensor.full_like(a, value)

zeros
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.zeros(shape)

    Return zero-tensor of the passed shape.

    :param shape: shape of tensor
    :return: QTensor

    Example::

        t = tensor.zeros([2, 3, 4])

zeros_like
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.zeros_like(t: pyvqnet.tensor.tensor.QTensor)

    Return zero-tensor with the same shape as the input tensor.

    :param t: ‘QTensor’ - input parameter
    :return: QTensor

    Example::

        t = QTensor([1, 2, 3])
        x = tensor.zeros_like(t)

arange
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.arange(start, end, step=1, dev: int = 0)

    Create a 1D tensor with evenly spaced values within a given interval.

    :param start: start of interval
    :param end: end of interval
    :param step: spacing between values
    :param dev: device to use,default = 0
    :return: QTensor

    Example::

        t = tensor.arange(2, 30,4)
        print(t)

linspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.linspace(start, end, steps: int, dev: int = 0)

    Create a 1D tensor with evenly spaced values within a given interval.

    :param start: starting value
    :param end: end value
    :param steps: number of samples to generate
    :param dev:  device to use,default = 0
    :return: QTensor

    Example::

        start, stop, num = -2.5, 10, 10
        t = tensor.linspace(start, stop, num)

logspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.logspace(start, end, steps, base, dev: int = 0)

    Create a 1D tensor with evenly spaced values on a log scale.

    :param start: base ** start is the starting value
    :param end: base ** end is the final value of the sequence
    :param steps: number of samples to generate
    :param base: the base of the log space
    :param dev:  device to use,default = 0
    :return: QTensor

    Example::

        start, stop, num, base = 0.1, 1.0, 5, 10.0
        t = tensor.logspace(start, stop, num, base)

eye
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.eye(size, offset: int = 0, dev: int = 0)

    Create a size x size tensor with ones on the diagonal and zeros elsewhere.

    :param size: size of the (square) tensor to create
    :param dev: device to use,default = 0
    :return: QTensor

    Example::

        size = 3
        t = tensor.eye(size)

diag
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.diag(t, k: int = 0)

    Select diagonal elements.

    Returns a new tensor which is the same as this one,
    except that elements other than those in the selected diagonal are set to zero.

    :param t: input tensor
    :param k: offset (0 for the main diagonal, positive for the nth diagonal above the main one, negative for the nth diagonal below the main one)
    :return: QTensor

    Example::

        a = np.arange(16).reshape(4, 4).astype(np.float32)
        t = QTensor(a)
        for k in range(-3, 4):
            u = tensor.diag(t,k=k)
            print(u)

randu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.randu(shape, dev: int = 0)

    Create a tensor with uniformly distributed random values.

    :param shape: shape of the tensor to create
    :param dev: device to use,default = 0
    :return: QTensor

    Example::

        shape = [2, 3]
        t = tensor.randu(shape)

randn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.randn(shape, dev: int = 0)

    Create a tensor with normally distributed random values.

    :param shape: shape of the tensor to create
    :param dev: device to use,default = 0
    :return: QTensor

    Example::

        shape = [2, 3]
        t = tensor.randn(shape)

数学函数
-----------------------------


floor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.floor(t)

    Compute the element-wise floor (largest integer i such that i <= x)
    of the tensor.

    :param t: input qtensor
    :return: A QTensor

    Examples::

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.floor(t)

ceil
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.ceil(t)

    Compute the element-wise ceiling (smallest integer i such that i >= x)
    of the tensor.

    :param t: input qtensor
    :return: A QTensor

    Examples::

        t = tensor.arange(-2.0, 2.0, 0.25)
        u = tensor.ceil(t)

round
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.round(t)

    Round tensor values to the nearest integer.

    :parma t: input tensor
    :return: A QTensor

    Examples::

        t = tensor.arange(-2.0, 2.0, 0.4)
        u = tensor.round(t)

sort
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.sort(t, axis: int, descending=False, stable=True)

    sort tensor along the axis

    :param t: input tensor
    :param axis: sort axis
    :param descending: sort order if desc
    :param stable:  Whether to use stable sorting or not
    :return: A QTensor

    Examples::

        a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
        A = QTensor(a)
        AA = tensor.sort(A,1,False)

argsort
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.argsort(t, axis: int, descending=False, stable=True)

    sort tensor along the axis

    :param t: input tensor
    :param axis: sort axis
    :param descending: sort order if desc
    :param stable:  Whether to use stable sorting or not
    :return: QTensor

    Examples::

        a = np.random.randint(10, size=24).reshape(3,8).astype(np.float32)
        A = QTensor(a)
        bb = tensor.argsort(A,1,False)

add
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.add(t1: pyvqnet.tensor.tensor.QTensor, t2: pyvqnet.tensor.tensor.QTensor)

    Element-wise Adds two tensors .

    :param t1: 'QTensor' - first tensor
    :param t2: 'QTensor' - second tensor
    :return:  QTensor

    Example::


        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.add(t1, t2)

sub
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.sub(t1: pyvqnet.tensor.tensor.QTensor, t2: pyvqnet.tensor.tensor.QTensor)

    Element-wise subtracts two tensors.


    :param t1: 'QTensor' - first tensor
    :param t2: 'QTensor' - second tensor
    :return:  QTensor

    Example::

        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.sub(t1, t2)

mul
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.mul(t1: pyvqnet.tensor.tensor.QTensor, t2: pyvqnet.tensor.tensor.QTensor)

    Element-wise multiplies two tensors.

    :param t1: 'QTensor' - first tensor
    :param t2: 'QTensor' - second tensor
    :return:  QTensor


    Example::

        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.mul(t1, t2)

divide
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.divide(t1: pyvqnet.tensor.tensor.QTensor, t2: pyvqnet.tensor.tensor.QTensor)

    Element-wise divides two tensors.


    :param t1: 'QTensor' - first tensor
    :param t2: 'QTensor' - second tensor
    :return:  QTensor


    Example::

        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.divide(t1, t2)

sums
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.sums(t: pyvqnet.tensor.tensor.QTensor, axis: Optional[int] = None, keepdims=False)

    Sums all the elements in tensor along given axis

    :param t: 'QTensor'
    :param axis: 'int' - defaults to None
    :param keepdims: 'bool' - defaults to False
    :return:  QTensor


    Example::

        t = QTensor(([1, 2, 3], [4, 5, 6]))
        x = tensor.sums(t)

mean
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.mean(t: pyvqnet.tensor.tensor.QTensor, axis=None, keepdims=False)

    Obtain the mean of all the values in the tensor.

    mean(input)

    :param input: input (Tensor) – the input tensor.

    Obtain the mean values in the tensor along the axis.

    mean(input, dim, keepdim=False)

    :param input: input (Tensor) – the input tensor.
    :param dim: dim ([int]) – the dimension to reduce.
    :param keepdim: keepdim (bool) – whether the output tensor has dim retained or not.
    :return: returns the mean value of the input tensor.

    Example::

        t = QTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.mean(t, axis=1)

median
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.median(t: pyvqnet.tensor.tensor.QTensor, *kargs)

    Obtain the median value of all the elements in the tensor.

    median(input)

    :param input: input (Tensor) – the input tensor.

    median(input, dim, keepdim=False)

    :param input: input (Tensor) – the input tensor.
    :param dim: dim ([int]) – the dimension to reduce.
    :param keepdim: keepdim (bool) – whether the output tensor has dim retained or not.

    :return: Returns the median of the values in input.

    Examples::

        a = QTensor([[1.5219, -1.5212,  0.2202]])
        median_a = tensor.median(a)
        print(median_a)

        b = QTensor([[0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
                    [0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
                    [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
                    [1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
        median_b = tensor.median(b,[1], False)
        print(median_b)

std
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.std(t: pyvqnet.tensor.tensor.QTensor, *kargs)

    Obtain the median value of all the elements in the tensor.

    std(input, unbiased)

    :param input: input (Tensor) – the input tensor.

    :param unbiased: unbiased (bool) – whether to use Bessel’s correction.

    std(input, dim, keepdim=False, unbiased)

    :param input: input (Tensor) – the input tensor.
    :param dim: dim ([int]) – the dimension to reduce.
    :param keepdim: keepdim (bool) – whether the output tensor has dim retained or not.
    :param unbiased: unbiased (bool) – whether to use Bessel’s correction.
    :return: Returns the median of the values in input.

    Examples::

        a = QTensor([[-0.8166, -1.3802, -0.3560]])
        std_a = tensor.std(a)
        print(std_a)

        b = QTensor([[0.2505, -0.3982, -0.9948,  0.3518, -1.3131],
                    [0.3180, -0.6993,  1.0436,  0.0438,  0.2270],
                    [-0.2751,  0.7303,  0.2192,  0.3321,  0.2488],
                    [1.0778, -1.9510,  0.7048,  0.4742, -0.7125]])
        std_b = tensor.std(b, [1], False, False)
        print(std_b)

var
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.var(t: pyvqnet.tensor.tensor.QTensor, *kargs)

    Obtain the product of all the values in the tensor.

    var(input, unbiased)

    :param input: input (Tensor) – the input tensor.

    :param unbiased: unbiased (bool) – whether to use Bessel’s correction.

    var(input, dim, keepdim=False, unbiased)

    :param input: input (Tensor) – the input tensor.
    :param dim: dim ([int]) – the dimension to reduce.
    :param keepdim: keepdim (bool) – whether the output tensor has dim retained or not.
    :param unbiased: unbiased (bool) – whether to use Bessel’s correction.


    :return: Returns the product of all elements in the input tensor.

    Examples::

        a = QTensor([[-0.8166, -1.3802, -0.3560]])
        a_var = tensor.var(a)
        print(a_var)

matmul
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.matmul(t1: pyvqnet.tensor.tensor.QTensor, t2: pyvqnet.tensor.tensor.QTensor)

    Element-wise multiplies two tensors.

    :param t1: 'QTensor' - first tensor
    :param t2: 'QTensor' - second tensor
    :return:  QTensor


    Example::

        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.mul(t1, t2)

reciprocal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.reciprocal(t)

    Compute the element-wise reciprocal of the tensor.

    :parma t: input tensor
    :return: A QTensor

    Examples::

        t = tensor.arange(1, 10, 1)
        u = tensor.reciprocal(t)

sign
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.sign(t)

    Compute the element-wise sign (-1 if x < 0, 0 if x == 0, 1 if x > 0)
    of the tensor.

    :parma t: input tensor
    :return: A QTensor


    Examples::

        t = tensor.arange(-5, 5, 1)
        u = tensor.sign(t)

neg
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.neg(t: pyvqnet.tensor.tensor.QTensor)

    Unary negation of tensor elements.

    :param t: 'QTensor' - input tensor
    :return:  QTensor

    Example::

        t = QTensor([1, 2, 3])
        x = tensor.neg(t)

trace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.trace(t, k: int = 0)

    Sum diagonal elements.

    :param t: 'QTensor' - input tensor
    :param k: offset (0 for the main diagonal, positive for the nth
        diagonal above the main one, negative for the nth diagonal below the
        main one)
    :return: float

    Examples::

        t = tensor.randn([4,4])
        for k in range(-3, 4):
            u=t.trace(k=k)

exp
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.exp(t: pyvqnet.tensor.tensor.QTensor)

    Applies exp function to all the elements of the input tensor.

    :param t: 'QTensor' - input tensor
    :return:  QTensor

    Example::

        t = QTensor([1, 2, 3])
        x = tensor.exp(t)

acos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.acos(t: pyvqnet.tensor.tensor.QTensor)

    Compute the element-wise inverse cosine of the tensor. in-place opration

    Modifies the tensor.

    :return: None

    Example::

        a = np.arange(36).reshape(2,6,3).astype(np.float32)
        a =a/100
        A = QTensor(a,requires_grad = True)
        y = tensor.acos(A)

asin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.asin(t: pyvqnet.tensor.tensor.QTensor)

    Compute the element-wise inverse sine of the tensor.

    Returns a new tensor.

    :return: A QTensor

    Examples::

        t = tensor.arange(-1, 1, .5)
        u = tensor.asin(t)

atan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.atan(t: pyvqnet.tensor.tensor.QTensor)

    Compute the element-wise inverse tangent of the tensor.

    Returns a new tensor.

    :return: A QTensor

    Examples::

        t = tensor.arange(-1, 1, .5)
        u = Tensor.atan(t)

sin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.sin(t: pyvqnet.tensor.tensor.QTensor)

    Applies sin function to all the elements of the input tensor.


    :param t: 'QTensor' - input tensor
    :return:  QTensor

    Example::

        from vqnet.tensor import tensor
        from vqnet.tensor.tensor import QTensor
        t = QTensor([1, 2, 3])
        x = tensor.sin(t)

cos
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.cos(t: pyvqnet.tensor.tensor.QTensor)

    Applies cos function to all the elements of the input tensor.


    :parma t: 'QTensor' - input tensor
    :return:  QTensor

    Example::

        t = QTensor([1, 2, 3])
        x = tensor.cos(t)

tan 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.tan(t: pyvqnet.tensor.tensor.QTensor)

    Applies tan function to all the elements of the input tensor.


    :param t: 'QTensor' - input tensor
    :return:  QTensor

    Example::

        t = QTensor([1, 2, 3])
        x = tensor.tan(t)

tanh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.tanh(t: pyvqnet.tensor.tensor.QTensor)

    Applies tanh function to all the elements of the input tensor.

    :param t: 'QTensor' - input tensor
    :return:  QTensor

    Example::

        t = QTensor([1, 2, 3])
        x = tensor.tanh(t)

sinh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.sinh(t: pyvqnet.tensor.tensor.QTensor)

    Applies sinh function to all the elements of the input tensor.


    :param t: 'QTensor' - input tensor
    :return:  QTensor

    Example::

        t = QTensor([1, 2, 3])
        x = tensor.sinh(t)

cosh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.cosh(t: pyvqnet.tensor.tensor.QTensor)

    Applies cosh function to all the elements of the input tensor.


    :param t: 'QTensor' - input tensor
    :return:  QTensor

    Example::

        t = QTensor([1, 2, 3])
        x = tensor.cosh(t)

power
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.power(t1: pyvqnet.tensor.tensor.QTensor, t2: pyvqnet.tensor.tensor.QTensor)

    Raises first tensor to the power of second tensor.

    :param t1: 'QTensor' - first tensor
    :param t2: 'QTensor' - second tensor
    :return:  QTensor

    Example::

        t1 = QTensor([1, 4, 3])
        t2 = QTensor([2, 5, 6])
        x = tensor.power(t1, t2)

abs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.abs(t: pyvqnet.tensor.tensor.QTensor)

    Applies abs function to all the elements of the input tensor.

    :param t: 'QTensor' - input tensor
    :return:  QTensor

    Example::

        t = QTensor([1, -2, 3])
        x = tensor.abs(t)

log
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.log(t: pyvqnet.tensor.tensor.QTensor)

    Applies log (ln) function to all the elements of the input tensor.

    :param t: 'QTensor' - input tensor
    :return:  QTensor

    Example::

        t = QTensor([1, 2, 3])
        x = tensor.log(t)

sqrt
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.sqrt(t: pyvqnet.tensor.tensor.QTensor)

    Applies sqrt function to all the elements of the input tensor.


    :param t: 'QTensor' - input tensor
    :return:  QTensor

    Example::

        t = QTensor([1, 2, 3])
        x = tensor.sqrt(t)

square
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.square(t: pyvqnet.tensor.tensor.QTensor)

    Applies square function to all the elements of the input tensor.


    :parma t: 'QTensor' - input tensor
    :return:  QTensor

    Example::

        t = QTensor([1, 2, 3])
        x = tensor.square(t)

逻辑函数
--------------------------

maximum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.maximum(t1: pyvqnet.tensor.tensor.QTensor, t2: pyvqnet.tensor.tensor.QTensor)

    Element-wise maximum of two tensor.


    :param t1: 'QTensor' - first tensor
    :param t2: 'QTensor' - second tensor
    :return:  QTensor

    Example::

        t1 = QTensor([6, 4, 3])
        t2 = QTensor([2, 5, 7])
        x = tensor.maximum(t1, t2)

minimum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.minimum(t1: pyvqnet.tensor.tensor.QTensor, t2: pyvqnet.tensor.tensor.QTensor)

    Element-wise minimum of two tensor.


    :param t1: 'QTensor' - first tensor
    :param t2: 'QTensor' - second tensor
    :return:  QTensor

    Example::

        t1 = QTensor([6, 4, 3])
        t2 = QTensor([2, 5, 7])
        x = tensor.minimum(t1, t2)

min
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.min(t: pyvqnet.tensor.tensor.QTensor, axis=None, keepdims=False)

    Returns min elements of the input tensor alongside given axis.
    if axis == None, return the min value of all elements in tensor.

    :parma t: 'QTensor' - input tensor
    :param axis: 'int' - defaults to None
    :param keepdims: 'bool' - defaults to False
    :return: QTensor or float

    Example::

        t = QTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.min(t, axis=1, keepdims=True)

max
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.max(t: pyvqnet.tensor.tensor.QTensor, axis=None, keepdims=False)

    Returns max elements of the input tensor alongside given axis.

    :param t: 'QTensor' - input tensor
    :param axis: 'int' - defaults to None
    :param keepdims: 'bool' - defaults to False
    :return: QTensor or float

    Example::

        t = QTensor([[1, 2, 3], [4, 5, 6]])
        x = tensor.max(t, axis=1, keepdims=True)

clip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.clip(t: pyvqnet.tensor.tensor.QTensor, min_val, max_val)

    Clips input tensor to minimum and maximum value.

    :param t: 'QTensor' - input tensor
    :param min_val: 'float' - minimum value
    :param max_val: 'float' - maximum value
    :return:  QTensor

    Example::

        t = QTensor([2, 4, 6])
        x = tensor.clip(t, 3, 8)

where
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.where(condition: pyvqnet.tensor.tensor.QTensor, t1: Optional[pyvqnet.tensor.tensor.QTensor] = None, t2: Optional[pyvqnet.tensor.tensor.QTensor] = None)

    Return elements chosen from x or y depending on condition.

    :param condition: 'QTensor' - condition tensor
    :param t1: 'QTensor' - tensor from which to take elements if condition is met, defaults to None
    :param t2: 'QTensor' - tensor from which to take elements if condition is not met, defaults to None
    :return: QTensor

    Example::

        t1 = QTensor([1, 2, 3])
        t2 = QTensor([4, 5, 6])
        x = tensor.where(t1 < 2, t1, t2)

nonzero
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.nonzero(A)

    Returns a tensor containing the indices of nonzero elements.


    :param A: input tensor
    :return: A new tensor

    Examples::

        start = -5.0
        stop = 5.0
        num = 1
        t = tensor.arange(start, stop, num)
        t = tensor.nonzero(t)

isfinite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.isfinite(A)

    Test element-wise for finiteness (not infinity or not Not a Number).

    :param A: input QTensor
    :return: QTensor with each elements presents 1, if the tensor value is isfinite. else 0.

    Examples::

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isfinite(t)

isinf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.isinf(A)

    Test element-wise for positive or negative infinity.

    :param A: input QTensor
    :return: QTensor with each elements presents 1, if the tensor value is isinf. else 0.

    Examples::

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isinf(t)

isnan
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.isnan(A)

    Test element-wise for Nan.

    :param A: input QTensor
    :return: QTensor with each elements presents 1, if the tensor value is isnan. else 0.

    Examples::

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])

        flag = tensor.isnan(t)

isneginf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.isneginf(A)

    Test element-wise for negative infinity.

    :param A: a QTensor
    :return: QTensor with each elements presents 1, if the tensor value is isneginf. else 0.

    Examples::

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isneginf(t)

isposinf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.isposinf(A)

    Test element-wise for positive infinity.

    :param A: a QTensor
    :return: QTensor with each elements presents 1, if the tensor value is isposinf. else 0.

    Examples::

        t = QTensor([1, float('inf'), 2, float('-inf'), float('nan')])
        flag = tensor.isposinf(t)

logical_and
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.logical_and(A, B)

    Compute the truth value of ``A`` and ``B`` element-wise.if element is 0, it presents False,else True.

    :param A: a QTensor
    :param B: a QTensor
    :return: QTensor

    Examples::

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_and(a,b)

logical_or
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.logical_or(A, B)

    Compute the truth value of ``A or B`` element-wise.if element is 0, it presents False,else True.

    :param A: a QTensor
    :param B: a QTensor
    :return: QTensor

    Examples::

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_or(a,b)

logical_not
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.logical_not(A)

    Compute the truth value of ``not A`` element-wise.if element is 0, it presents False,else True.

    :param A: a QTensor
    :return: QTensor

    Examples::

        a = QTensor([0, 1, 10, 0])
        flag = tensor.logical_not(a)

logical_xor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.logical_xor(A, B)

    Compute the truth value of ``A xor B`` element-wise.if element is 0, it presents False,else True.

    :param A: a QTensor
    :param B: a QTensor
    :return: QTensor

    Examples::

        a = QTensor([0, 1, 10, 0])
        b = QTensor([4, 0, 1, 0])
        flag = tensor.logical_xor(a,b)

greater
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.greater(A, B)

    Return the truth value of ``this > A`` element-wise.


    :param A: a QTensor
    :param B: a QTensor
    :return: A boolean tensor that is True where input is greater than other and False elsewhere

    Examples::

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.greater(a,b)

greater_equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.greater_equal(A, B)

    Return the truth value of ``A >= B`` element-wise.

    :param A: a QTensor
    :param B: a QTensor
    :return: A boolean tensor that is True where input is greater than or equal to other and False elsewhere

    Examples::

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.greater_equal(a,b)

less
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.less(A, B)

    Return the truth value of ``A < B`` element-wise.


    :param A: a QTensor
    :param B: a QTensor
    :return: A boolean tensor that is True where input is less than other and False elsewhere

    Examples::

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.less(a,b)

less_equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.less_equal(A, B)

    Return the truth value of ``A <= B`` element-wise.


    :param A: a QTensor
    :param B: a QTensor
    :return: A boolean tensor that is True where input is less than or equal to other and False elsewhere

    Examples::

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.less_equal(a,b)

equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.equal(A, B)

    Return the truth value of ``B == A`` element-wise.


    :param A: a QTensor
    :param B: a QTensor
    :return: True if two tensors have the same size and elements, False otherwise.

    Examples::

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.equal(a,b)

not_equal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.not_equal(A, B)

    Return the truth value of ``B != A`` element-wise.


    :param A: a QTensor
    :param B: a QTensor
    :return: A boolean tensor that is True where input is not equal to other and False elsewhere

    Examples::

        a = QTensor([[1, 2], [3, 4]])
        b = QTensor([[1, 1], [4, 4]])
        flag = tensor.not_equal(a,b)

矩阵操作
--------------------------

select
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.select(t: pyvqnet.tensor.tensor.QTensor, index)

    Return QTensor in the tensor at the given axis. following operation get same result's value.
    
    :param t: input QTensor
    :param index: a string contains output dim  
    :return: QTensor

    Example::

        t = QTensor(np.arange(1,25).reshape(2,3,4))
        print(t)        
        indx = [":", "0", ":"]        
        t.requires_grad = True
        t.zero_grad()
        ts = tensor.select(t,indx)
        ts.backward(tensor.ones(ts.shape))

concatenate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.concatenate(args: list, axis=1)

    concatenate with channel, i.e. concatenate C of Tensor shape (N,C,H,W)

       :param args: tuple consist of Tensor
       :return: cat of tuple

    Example::

        x = QTensor([[1, 2, 3],[4,5,6]], requires_grad=True) 
        y = 1-x  
        x = tensor.concatenate((x,y),1)

stack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.stack(tensors: list, axis) → pyvqnet.tensor.tensor.QTensor

    Join a sequence of arrays along a new axis,return a new Tensor.

    :param tensors: list contains QTensors
    :param axis: stack axis
    :return: A QTensor

    Examples::

        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t11 = QTensor(a)
        t22 = QTensor(a)
        R, C = 3, 4
        a = np.arange(R * C).reshape(R, C).astype(np.float32)
        t33 = QTensor(a)
        rlt1 = tensor.stack([t11,t22,t33],2)

permute
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.permute(t: pyvqnet.tensor.tensor.QTensor, dim: list)

    Reverse or permute the axes of an array.if new_dims = None, revsers the dim.

    :param t: input QTensor
    :param new_dims: the new order of the dimensions (list of integers).
    :return: result tensor. 

    Examples::

        R, C = 3, 4
        a = np.arange(R * C).reshape([2,2,3]).astype(np.float32)
        t = QTensor(a)
        tt = tensor.permute(t,[2,0,1])

transpose
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.transpose(t: pyvqnet.tensor.tensor.QTensor, dim: list)

    Reverse or permute the axes of an array.if dim = None, revsers the dim.

    :param t: input QTensor
    :param dim: the new order of the dimensions (list of integers).
    :return: result tensor.

    Examples::

        R, C = 3, 4
        a = np.arange(R * C).reshape([2,2,3]).astype(np.float32)
        t = QTensor(a)
        tt = tensor.transpose(t,[2,0,1])

tile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.tile(t: pyvqnet.tensor.tensor.QTensor, reps: list)

    Construct an array by repeating tensors the number of times given by reps.

    If reps has length d, the result will have dimension of max(d, A.ndim).

    If A.ndim < d, A is promoted to be d-dimensional by prepending new axes.
    So a shape (3,) array is promoted to (1, 3) for 2-D replication, or shape (1, 1, 3) for 3-D replication.

    If this is not the desired behavior, promote A to d-dimensions manually before calling this function.

    If A.ndim > d, reps is promoted to A.ndim by pre-pending 1’s to it.
    Thus for an A of shape (2, 3, 4, 5), a reps of (2, 2) is treated as (1, 1, 2, 2).

    :param reps: the number of repetitions per dimension.
    :return: new tensor

    Examples::

        a = np.arange(24).reshape(4,6).astype(np.float32)
        A = QTensor(a)
        reps = [1,2,3,4,5]
        B = tensor.tile(A,reps)

squeeze
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.squeeze(t: pyvqnet.tensor.tensor.QTensor, axis: int = - 1)

    Remove axes of length one .

    :param axis: squeeze axis
    :return: A QTensor

    Examples::

        a = np.arange(6).reshape(1,6,1).astype(np.float32)
        A = QTensor(a)
        AA = tensor.squeeze(A,0)

unsqueeze
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.unsqueeze(t: pyvqnet.tensor.tensor.QTensor, axis: int = 0)

    Remove axes of length one .

    :param axis: squeeze axis
    :return: A QTensor

    Examples::

        a = np.arange(24).reshape(2,1,1,4,3).astype(np.float32)
        A = QTensor(a)
        AA = tensor.unsqueeze(A,1)

swapaxis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.swapaxis(t, axis1: int, axis2: int)

    Interchange two axes of an array.

    :param axis1: First axis.
    :param axis2:  Destination position for the original axis. These must also be unique
    :return: A QTensor

    Examples::

        a = np.arange(24).reshape(2,3,4).astype(np.float32)
        A = QTensor(a)
        AA = tensor.swapaxis(2,1)


flatten
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.flatten(t: pyvqnet.tensor.tensor.QTensor, start: int = 0, end: int = - 1)

    flatten tensor from dim start to dim end.

    :param t: 'QTensor' - input tensor
    :param start: 'int' - dim start
    :param end: 'int' - dim start
    :return:  QTensor

    Example::

        t = QTensor([1, 2, 3])
        x = tensor.flatten(t)

实用函数
-----------------------------


to_tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.tensor.tensor.to_tensor(x)

    Convert input parameter to Qtensor if it isn't already.

    :param x: 'QTensor-like' - input parameter
    :return: QTensor

    Example::

        from pyvqnet.tensor import tensor
        t = tensor.to_tensor(10.0)
