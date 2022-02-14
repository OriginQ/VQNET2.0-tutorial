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

.. py:class:: pyvqnet.nn.module.Module

    Base class for all neural network modules including quantum modules or classic modules.
    Your models should also be subclass of this class for autograd calculation.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::

        class Model(Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = pyvqnet.nn.Conv2d(1, 20, (5,5))
                self.conv2 = pyvqnet.nn.Conv2d(20, 20, (5,5))

            def forward(self, x):
                x = pyvqnet.nn.activation.relu(self.conv1(x))
                return pyvqnet.nn.activation.relu(self.conv2(x))

    Submodules assigned in this way will be registered

forward
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.nn.module.Module.forward(x, param_keys=None, circuits=None, func=None)

    Abstract method which performs forward pass.

    :param x: input QTensor
    :param param_keys: specific param keys for QNLP algorithm,default None.
    :param circuits: specific circuits from other code for QNLP algorithm,default None.
    :param func: specific convert function to qpanda circuits for QNLP algorithm,default None.
    :return: module output

state_dict 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.nn.module.Module.state_dict(destination=None, prefix='')

    Returns a dictionary containing a whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.

    :param destination: a dict where state will be stored
    :param prefix: the prefix for parameters and buffers used in this
        module

    :return: a dictionary containing a whole state of the module

    Example::

        module.state_dict().keys()
        ['bias', 'weight']

save_parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.utils.storage.save_parameters(obj, f: Union[str, os.PathLike])

    Saves model parmeters to a disk file.

    :param obj: saved OrderedDict from state_dict()
    :param f: a string or os.PathLike object containing a file name
    :return: None

    Example::

        class Net(Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")

            def forward(self, x):
                return super().forward(x)

        model = Net() 
        save_parameters( model.state_dict(),"tmp.model")

load_parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.utils.storage.load_parameters(f: Union[str, os.PathLike]) :

    Loads model paramters from a disk file.

    The model instance should be created first.

    :param f: a string or os.PathLike object containing a file name
    :return: saved OrderedDict for load_state_dict()

    Example::

        class Net(Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")

            def forward(self, x):
                return super().forward(x)

        model = Net()   
        model1 = Net()  # another Module object
        save_parameters( model.state_dict(),"tmp.model")
        model_para =  load_parameters("tmp.model")

        model1.load_state_dict(model_para))



经典神经网络层
-------------------------------

Conv1D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.conv.Conv1D(input_channels:int,output_channels:int,kernel_size:int ,stride:int= 1,padding:str="valid",use_bias:str = True,kernel_initializer = None,bias_initializer =None)

    1D Convolution module. Inputs to the conv module are of shape (batch_size, input_channels, height)

    :param input_channels: `int` - Number of input channels
    :param output_channels: `int` - Number of kernels
    :param kernel_size: `int` - Size of a single kernel. kernel shape = [input_channels,output_channels,kernel_size,1]
    :param stride: `int` - Stride, defaults to (1, 1)
    :param padding: `str` - Padding, defaults to "valid"
    :param use_bias: `bool` - if use bias, defaults to True
    :param kernel_initializer: `callable` - Defaults to _xavier_normal
    :param bias_initializer: `callable` - Defaults to _zeros
    :return: a Conv1D class

    .. note::
        ``padding='valid'`` is the same as no padding.

        out_length = ceil((input_size - (kerkel_size - 1) )/stride)

        ``padding='same'`` pads the input so the output has the shape as the input.

        out_length = ceil(input_size/stride)
    
    Example::
        
        b= 2
        ic =3
        oc = 2
        test_conv = Conv1D(ic,oc,3,2,"same",initializer.ones,initializer.ones)
        x0 = QTensor(np.arange(1,b*ic*5*5 +1).reshape([b,ic,25]),requires_grad=True)
        x = test_conv.forward(x0)
        print(x)

Conv2D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.conv.Conv2D(input_channels:int,output_channels:int,kernel_size:tuple,stride:tuple=(1, 1),padding="valid",use_bias = True,kernel_initializer=None,bias_initializer=None)

    Convolution module. Inputs to the conv module are of shape (batch_size, input_channels, height, width)

    :param input_channels: `int` - Number of input channels
    :param output_channels: `int` - Number of kernels
    :param kernel_size: `tuple` - Size of a single kernel.
    :param stride: `tuple` - Stride, defaults to (1, 1)
    :param padding: `str` - Padding, defaults to "valid"
    :param use_bias: `bool` - if use bias, defaults to True
    :param kernel_initializer: `callable` - Defaults to _xavier_normal
    :param bias_initializer: `callable` - Defaults to _zeros
    :return: a Conv2D class

    .. note::
        ``padding='valid'`` is the same as no padding.

        out_length = ceil((input_size - (kerkel_size - 1) )/stride)

        ``padding='same'`` pads the input so the output has the shape as the input.

        out_length = ceil(input_size/stride)

    Example::
        
        b= 2
        ic =3
        oc = 2
        test_conv = Conv2D(ic,oc,(3,3),(2,2),"same",initializer.ones,initializer.ones)
        x0 = QTensor(np.arange(1,b*ic*5*5+1).reshape([b,ic,5,5]),requires_grad=True)
        x = test_conv.forward(x0)
        print(x)

ConvT2D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.conv.ConvT2D(input_channels,output_channels,kernel_size,stride=(1, 1),padding="valid",kernel_initializer=None,bias_initializer=None)

    ConvTransposed module. Inputs to the convT module are of shape (batch_size, input_channels, height, width)

    :param input_channels: `int` - Number of input channels
    :param output_channels: `int` - Number of kernels
    :param kernel_size: `int` - Size of a single kernel. Each kernel is kernel_size x kernel_size
    :param stride: `tuple` - Stride, defaults to (1, 1)
    :param padding: `tuple` - Padding, defaults to (0, 0)
    :param kernel_initializer: `callable` - Defaults to xavier_normal
    :param bias_initializer: `callable` - Defaults to zeros
    :return: a ConvT2D class
    
    .. note::
        ``padding='valid'`` is the same as no padding.

        out_length = input_size*stride + (kerkel_size - stride)

        ``padding='same'`` pads the input so the output has the shape as the input.

        out_length = input_size*stride

    Example::

        test_conv = ConvT2D(3, 2, [3, 3], [1, 1], "valid", initializer.ones, initializer.ones)
        x = QTensor(np.arange(1, 1 * 3 * 5 * 5+1).reshape([1, 3, 5, 5]), requires_grad=True)
        y = test_conv.forward(x)
        print(y)


AvgPool1D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.pooling.AvgPool1D(kernel, stride, padding="valid",name="")

    Average pooling layer
    reference https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html#torch.nn.AvgPool1d

    :param kernel: size of the average pooling windows
    :param strides: factor by which to downscale
    :param padding: one of "none", "valid" or "same" 
    :param name: name of the output layer
    :return: AvgPool1D layer

    .. note::
        ``padding='valid'`` is the same as no padding.

        out_length = ceil((input_size - (kerkel_size - 1) )/stride)

        ``padding='same'`` pads the input so the output has the shape as the input.

        out_length = ceil(input_size/stride)
        
    Example::

        test_mp = AvgPool1D([2],[2],"same")
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,1,5,5]),requires_grad=True)

        y= test_mp.forward(x)

MaxPool1D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: MaxPool1D(kernel, stride, padding="valid",name="")

    Max pooling layer
    reference https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html#torch.nn.MaxPool1d

    :param kernel: size of the max pooling windows
    :param strides: factor by which to downscale
    :param padding: one of "none", "valid" or "same" 
    :param name: name of the output layer
    :return: MaxPool1D layer

    .. note::
    
        ``padding='valid'`` is the same as no padding.

        out_length = ceil((input_size - (kerkel_size - 1) )/stride)

        ``padding='same'`` pads the input so the output has the shape as the input.

        out_length = ceil(input_size/stride)

    Example::

        test_mp = MaxPool2D([2],[2],"same")
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,1,5,5]),requires_grad=True)

        y= test_mp.forward(x)

AvgPool2D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.pooling.AvgPool2D( kernel, stride, padding="valid",name="")

    Perform 2D average pooling.
    
    reference: https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html?highlight=avgpooling
    
    :param kernel: size of the average pooling windows
    :param strides: factors by which to downscale
    :param padding: one of  "valid" or "same"
    :param name: name of the output layer
    :return: AvgPool2D layer

    .. note::
        ``padding='valid'`` is the same as no padding.

        out_length = ceil((input_size - (kerkel_size - 1) )/stride)

        ``padding='same'`` pads the input so the output has the shape as the input.

        out_length = ceil(input_size/stride)

    Example::

        test_mp = AvgPool2D([2,2],[2,2],"same")
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,1,5,5]),requires_grad=True)

        y= test_mp.forward(x)

MaxPool2D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.pooling.MaxPool2D(kernel, stride, padding="valid",name="")

    Max pooling layer
    reference https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=pooling

    :param kernel: size of the max pooling windows
    :param strides: factor by which to downscale
    :param padding: one of "none", "valid" or "same" 
    :param name: name of the output layer
    :return: MaxPool2D layer

    .. note::
        ``padding='valid'`` is the same as no padding.

        out_length = ceil((input_size - (kerkel_size - 1) )/stride)

        ``padding='same'`` pads the input so the output has the shape as the input.

        out_length = ceil(input_size/stride)

    Example::

        test_mp = MaxPool2D([2,2],[2,2],"same")
        x= QTensor(np.array([0, 1, 0, 4, 5,
                                    2, 3, 2, 1, 3,
                                    4, 4, 0, 4, 3,
                                    2, 5, 2, 6, 4,
                                    1, 0, 0, 5, 7],dtype=float).reshape([1,1,5,5]),requires_grad=True)

        y= test_mp.forward(x)

Embedding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.embedding.Embedding(num_embeddings, embedding_dim, weight_initializer=xavier_normal, name: str = "")

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    
    :param num_embeddings: `int` - number of inputs features
    :param embedding_dim: `int` - number of output features
    :param weight_initializer: `callable` - defaults to normal
    :return: a Embedding class

    Example::

        vlayer = Embedding(300,3)
        x = QTensor(np.arange(1,151).reshape([2,3,5,5]))
        y = vlayer(x)
        


BatchNorm2d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.batch_norm.BatchNorm2d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5,beta_initializer=zeros, gamma_initializer=ones, name="")
    
    Applies Batch Normalization over a 4D input (B,C,H,W) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .
    
    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    where :math:`\gamma` and :math:`\beta` are learnable parameters.Also by default, during training this layer keeps running 
    estimates of its computed mean and variance, which are then used for normalization during evaluation. 
    The running estimates are kept with a default momentum of 0.1.

    :param channel_num: `int` - the number of input features channels
    :param momentum: `float` - momentum when calculation exponentially weighted average, defaults to 0.1
    :param beta_initializer: `callable` - defaults to zeros
    :param gamma_initializer: `callable` - defaults to ones
    :param epsilon: `float` - numerical stability constant, defaults to 1e-5
    :return: a BatchNorm2d class

    Example::

        b= 2
        ic =2
        test_conv = BatchNorm2d(ic)
        #set train mode 
        #test_conv.train()
        #set eval mode
        test_conv.eval()
        x = QTensor(np.arange(1,17).reshape([b,ic,4,1]),requires_grad=True)
        y = test_conv.forward(x)
    

BatchNorm1d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.batch_norm.BatchNorm1d(channel_num:int, momentum:float=0.1, epsilon:float = 1e-5, beta_initializer=zeros, gamma_initializer=ones, name="")

    Applies Batch Normalization over a 2D input (B,C) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .
    
    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    where :math:`\gamma` and :math:`\beta` are learnable parameters.Also by default, during training this layer keeps running 
    estimates of its computed mean and variance, which are then used for normalization during evaluation. 
    The running estimates are kept with a default momentum of 0.1.


    :param channel_num: `int` - the number of input features channels
    :param momentum: `float` - momentum when calculation exponentially weighted average, defaults to 0.1
    :param beta_initializer: `callable` - defaults to zeros
    :param gamma_initializer: `callable` - defaults to ones
    :param epsilon: `float` - numerical stability constant, defaults to 1e-5
    :return: a BatchNorm1d class

    Example::

        test_conv = BatchNorm1d(4)
        #set train mode 
        #test_conv.train()
        #set eval mode
        test_conv.eval()
        x = QTensor(np.arange(1,17).reshape([4,4]),requires_grad=True)
        y = test_conv.forward(x)

LayerNorm2d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.layer_norm.LayerNorm2d(norm_size:int, epsilon:float = 1e-5, name="")

    Applies Layer Normalization over a mini-batch of 4D inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last  `D` dimensions size.
    
    For input like (B,C,H,W), :attr:`norm_size` should equals to C * H * W.

    :param norm_size: `float` - normalize size，equals to C * H * W
    :param epsilon: `float` - numerical stability constant, defaults to 1e-5
    :return: a LayerNorm2d class

    Example::

        ic = 4
        test_conv = LayerNorm2d([8])
        x = QTensor(np.arange(1,17).reshape([2,2,4,1]),requires_grad=True)
        y = test_conv.forward(x)

LayerNorm1d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.layer_norm.LayerNorm1d(norm_size:int, epsilon:float = 1e-5, name="")
    
    Applies Layer Normalization over a mini-batch of 2D inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last dimensions size, where `norm_size`
    is the value  of :attr:`norm_size`. 

    :param norm_size: `float` - normalize size，equals to last dim
    :param epsilon: `float` - numerical stability constant, defaults to 1e-5
    :return: a LayerNorm1d class

    Example::

        test_conv = LayerNorm1d([4])
        x = QTensor(np.arange(1,17).reshape([4,4]),requires_grad=True)
        y = test_conv.forward(x)

Linear
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.linear.Linear(input_channels, output_channels, weight_initializer=None, bias_initializer=None,use_bias=True, name: str = "")

    Linear module (fully-connected layer).
    :math:`y = Ax + b`

    :param inputs: `int` - number of inputs features
    :param output: `int` - number of output features
    :param weight_initializer: `callable` - defaults to normal
    :param bias_initializer: `callable` - defaults to zeros
    :param use_bias: `bool` - defaults to True
    :return: a Linear class

    Example::

        c1 =2
        c2 = 3
        cin = 7
        cout = 5
        n = Linear(cin,cout,initializer.ones,initializer.ones)
        input = QTensor(np.arange(1,c1*c2*cin+1).reshape((c1,c2,cin)),requires_grad=True)
        y = n.forward(input)


Dropout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.dropout.Dropout(dropout_rate = 0.5)

    Dropout module.

    :param dropout_rate: `float` - probability that a neuron will be set to zero
    :return: a Dropout class

    Example::

        b = 2
        ic = 3
        from pyvqnet._core import Tensor as CoreTensor
        from pyvqnet.nn.dropout import Dropout
        x = QTensor(CoreTensor.range(-1*ic*5*5,(b-1)*ic*5*5-1).reshape([b,ic,5,5]),requires_grad=True)
        droplayer = Dropout(0.5)
        droplayer.train()
        y = droplayer(x)
        y.backward(QTensor(np.ones(y.shape)*2))
        droplayer.eval()
        y = droplayer(x)


损失函数层
----------------------------------

MeanSquaredError
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.loss.MeanSquaredError()

    Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. , then:

    .. math::
        \ell(x, y) =
            \operatorname{mean}(L)


    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The mean operation still operates over all the elements, and divides by :math:`n`.

    :return: a MeanSquaredError class

    Parameters for loss forward function:

    Target: :math:`(N, *)`, same shape as the input

    Input: :math:`(N, *)` where :math:`*` means, any number of additional dimensions
    

    Example::

            target = QTensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]], requires_grad=True)
            input = QTensor([[0.1, 0.05, 0.7, 0, 0.05, 0.1, 0, 0, 0, 0]], requires_grad=True)

            loss_result = loss.MeanSquaredError()
            result = loss_result(target, input)
            result.backward()

BinaryCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.loss.BinaryCrossEntropy()

    measures the Binary Cross Entropy between the target and the output:

    The unreduced loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    where :math:`N` is the batch size.

    .. math::
        \ell(x, y) = \operatorname{mean}(L)

    :return: a BinaryCrossEntropy class

    Parameters for loss forward function:

    Target: :math:`(N, *)`, same shape as the input

    Input: :math:`(N, *)` where :math:`*` means, any number of additional dimensions
    
    Example::

        output = QTensor([[0.3, 0.7, 0.2], [0.2, 0.3, 0.1]], requires_grad=True)
        target = QTensor([[0, 1, 0], [0, 0, 1]], requires_grad=True)

        loss_result = loss.BinaryCrossEntropy()
        result = loss_result(target, output)
        result.backward()
        print(result)
        print(output.requires_grad)
        print(output.grad)


CategoricalCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.loss.CategoricalCrossEntropy()

    This criterion combines LogSoftmax and NLLLoss in one single class.

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)
    
    :return: a CategoricalCrossEntropy class

    Parameters for loss forward function:

    Target: :math:`(N, *)`, same shape as the input

    Input: :math:`(N, *)` where :math:`*` means, any number of additional dimensions
    
    Example::

            output = QTensor([[1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5]], requires_grad=True)
            target = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], requires_grad=True)
            loss_result = loss.CategoricalCrossEntropy()
            result = loss_result(target, output)
            print(result)
            result.backward()
            print(output.grad)


SoftmaxCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: pyvqnet.nn.loss.SoftmaxCrossEntropy()

    This criterion combines LogSoftmax and NLLLoss in one single class with more numeral stablity.

    The loss can be described as:

    .. math::
        \text{loss}(x, class) = -\log\left(\frac{\exp(x[class])}{\sum_j \exp(x[j])}\right)
                       = -x[class] + \log\left(\sum_j \exp(x[j])\right)

    :return: a SoftmaxCrossEntropy class

    Parameters for loss forward function:

    Target: :math:`(N, *)`, same shape as the input

    Input: :math:`(N, *)` where :math:`*` means, any number of additional dimensions
    
    Example::

            output = QTensor([[1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5]], requires_grad=True)
            target = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], requires_grad=True)
            loss_result = loss.SoftmaxCrossEntropy()
            result = loss_result(target, output)
            print(result)
            result.backward()
            print(output.grad)


激活函数
----------------------------------


Activation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.nn.activation.Activation()

    Base class of activation. Specific activation functions inherit  this functions.

Sigmoid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.nn.activation.Sigmoid(name:str="")


    Apply a sigmoid activation function to the given layer.

    .. math::
        \text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}

    :param name: name of the output layer
    :return: sigmoid Activation layer

    Examples::

        layer = pyvqnet.nn.Sigmoid()

Softplus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.nn.activation.Softplus(name:str="")

    Apply the softplus activation function to the given layer.

    .. math::
        \text{Softplus}(x) = \log(1 + \exp(x))

    :param name: name of the output layer
    :return: softplus Activation layer

    Examples::

        layer = pyvqnet.nn.Softplus()

Softsign
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.nn.activation.Softsign(name:str="")

    Apply the softsign activation function to the given layer.

    .. math::
        \text{SoftSign}(x) = \frac{x}{ 1 + |x|}

    :param name: name of the output layer
    :return: softsign Activation layer

    Examples::

        layer = pyvqnet.nn.Softsign()


Softmax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.nn.activation.Softmax(name:str="")

    Apply a softmax activation function to the given layer.

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}


    :param axis: dimension on which to operate (-1 for last axis)
    :param name: name of the output layer
    :return: softmax Activation layer

    Examples::

        layer = pyvqnet.nn.Softmax()

HardSigmoid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.nn.activation.HardSigmoid(name:str="")

    Apply a hard sigmoid activation function to the given layer.

    .. math::
        \text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{ if } x \le -2.5, \\
            1 & \text{ if } x \ge +2.5, \\
            x / 5 + 1 / 2 & \text{otherwise}
        \end{cases}

    :param name: name of the output layer
    :return: hard sigmoid Activation layer

    Examples::

        layer = pyvqnet.nn.HardSigmoid()

ReLu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.nn.activation.ReLu(name:str="")

    Apply a rectified linear unit activation function to the given layer.

    .. math::
        \text{ReLu}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        0, & \text{ if } x \leq 0
        \end{cases}


    :param name: name of the output layer
    :return: ReLu Activation layer

    Examples::

        layer = pyvqnet.nn.ReLu() 


LeakyReLu
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.nn.activation.LeakyReLu(name:str="")

    Apply the leaky version of a rectified linear unit activation
        function to the given layer.

    .. math::
        \text{LeakyRelu}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \alpha * x, & \text{ otherwise }
        \end{cases}

    :param alpha: LeakyRelu coefficient, default: 0.01
    :param name: name of the output layer
    :return: leaky ReLu Activation layer

    Examples::

        layer = pyvqnet.nn.LeakyReLu() 


ELU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.nn.activation.ELU(name:str="")

    Apply the exponential linear unit activation function to the given layer.

    .. math::
        \text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (\exp(x) - 1), & \text{ if } x \leq 0
        \end{cases}

    :param alpha: Elu coefficient, default: 1.0
    :param name: name of the output layer
    :return: Elu Activation layer

    Examples::

        layer = pyvqnet.nn.ELU() 
         
Tanh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.nn.activation.Tanh(name:str="")

    Apply the hyperbolic tangent activation function to the given layer.

    .. math::
        \text{Tanh}(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    :param name: name of the output layer
    :return: hyperbolic tangent Activation layer

    Examples::

        layer = pyvqnet.nn.Tanh() 

优化器模块
----------------------------------


Optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.optim.optimizer.Optimizer( params, lr=0.01)

    Base class for all optimizers.

    :parma params: params of model which need to be optimized
    :param lr: learning_rate of model (default: 0.01)

adadelta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.optim.adadelta.Adadelta( params, lr=0.01, beta=0.99, epsilon=1e-8)

    ADADELTA: An Adaptive Learning Rate Method (https://arxiv.org/abs/1212.5701)


    :param params: params of model which need to be optimized
    :param lr: learning_rate of model (default: 0.01)
    :param beta: for computing a running average of squared gradients (default: 0.99)
    :param epsilon: term added to the denominator to improve numerical stability (default: 1e-8)
    :return: a Adadelta optimizer
    
    Example::

        from pyvqnet.optim import adadelta
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4))
        params = [param]
        opti = adadelta.Adadelta(params)
        
        for i in range(1,3):
            opti._step() 

adagrad
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.optim.adagrad.Adagrad( params, lr=0.01, epsilon=1e-8)

    Implements Adagrad algorithm.

    https://databricks.com/glossary/adagrad

    :param params: params of model which need to be optimized
    :param lr: learning_rate of model (default: 0.01)
    :param epsilon: term added to the denominator to improve numerical stability (default: 1e-8)
    :return: a Adagrad optimizer

    Example::

        from pyvqnet.optim import adagrad
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4))
        params = [param]
        opti = adagrad.Adagrad(params)
        
        for i in range(1,3):
            opti._step()  

adam
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.optim.adam.Adam( params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

    Adam: A Method for Stochastic Optimization (https://arxiv.org/abs/1412.6980)

    :param params: params of model which need to be optimized
    :param lr: learning_rate of model (default: 0.01)
    :param beta1: coefficients used for computing running averages of gradient and its square (default: 0.9)
    :param beta2: coefficients used for computing running averages of gradient and its square (default: 0.999)
    :param epsilon: term added to the denominator to improve numerical stability (default: 1e-8)
    :return: a Adam optimizer

    Example::

        from pyvqnet.optim import adam
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4))
        params = [param]
        opti = adam.Adam(params)
        
        for i in range(1,3):
            opti._step()

adamax
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.optim.adamax.Adamax(params, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

    Implements Adamax algorithm (a variant of Adam based on infinity norm).

    https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html?highlight=adamax#torch.optim.Adamax


    :param params: params of model which need to be optimized
    :param lr: learning_rate of model (default: 0.01)
    :param beta1: coefficients used for computing running averages of gradient and its square (default: 0.9)
    :param beta2: coefficients used for computing running averages of gradient and its square (default: 0.999)
    :param epsilon: term added to the denominator to improve numerical stability (default: 1e-8)
    :return: a Adamax optimizer

    Example::

        from pyvqnet.optim import adamax
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4))
        params = [param]
        opti = adamax.Adamax(params)
        
        for i in range(1,3):
            opti._step() 

rmsprop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.optim.rmsprop.RMSProp( params, lr=0.01, beta=0.99, epsilon=1e-8)
    
    Implements RMSprop algorithm.

    https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html

    :param params: params of model which need to be optimized
    :param lr: learning_rate of model (default: 0.01)
    :param beta: coefficients used for computing running averages of gradient and its square (default: 0.99)
    :param epsilon: term added to the denominator to improve numerical stability (default: 1e-8)
    :return: a RMSProp optimizer

    Example::

        from pyvqnet.optim import rmsprop
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4))
        params = [param]
        opti = rmsprop.RMSProp(params)
        
        for i in range(1,3):
            opti._step()    

sgd
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:class:: pyvqnet.optim.sgd.SGD(params, lr=0.01,momentum=0, nesterov=False)

    https://en.wikipedia.org/wiki/Stochastic_gradient_descent


    :param params: params of model which need to be optimized
    :param lr: learning_rate of model (default: 0.01)
    :param momentum: momentum factor (default: 0)
    :param nesterov: enables Nesterov momentum (default: False)
    :return: a SGD optimizer
    
    Example::

        from pyvqnet.optim import sgd
        w = np.arange(24).reshape(1,2,3,4).astype(np.float64)    
        param = QTensor(w)
        param.grad = QTensor(np.arange(24).reshape(1,2,3,4))
        params = [param]
        opti = sgd.SGD(params)
        
        for i in range(1,3):
            opti._step() 

rotosolve
^^^^^^^^^^^^^^
Rotosolve算法它允许相对于其他参数的固定值直接跳转到单个参数的最佳值，直接找到量子线路最佳参数的优化算法。

.. py:class:: pyvqnet.optim.rotosolve.Rotosolve(max_iter =50)

    Rotosolve: The rotosolve algorithm can be used to minimize a linear combination
    of quantum measurement expectation values. See the following paper:
    https://arxiv.org/abs/1903.12166, Ken M. Nakanishi.
    https://arxiv.org/abs/1905.09692, Mateusz Ostaszewski.

    :param max_iter: max number of iterations of the rotosolve update
    :return: a Rotosolve optimizer
    
    Example::

        from pyvqnet.optim.rotosolve import Rotosolve
        import pyqpanda as pq
        from pyvqnet.tensor.tensor import QTensor
        from pyvqnet.qnn.measure import expval
        machine = pq.CPUQVM()
        machine.init_qvm()
        nqbits = machine.qAlloc_many(2)

        def gen(param,generators,qbits,circuit):
            if generators == "X":
                circuit.insert(pq.RX(qbits,param))
            elif generators =="Y":
                circuit.insert(pq.RY(qbits,param))
            else:
                circuit.insert(pq.RZ(qbits,param))
        def circuits(params,generators,circuit):
            gen(params[0], generators[0], nqbits[0], circuit)
            gen(params[1], generators[1], nqbits[1], circuit)
            circuit.insert(pq.CNOT(nqbits[0], nqbits[1]))
            prog = pq.QProg()
            prog.insert(circuit)
            return prog

        def ansatz1(params:QTensor,generators):
            circuit = pq.QCircuit()
            params = params.getdata()
            prog = circuits(params,generators,circuit)
            return expval(machine,prog,{"Z0":1},nqbits), expval(machine,prog,{"Y1":1},nqbits)

        def ansatz2(params:QTensor,generators):
            circuit = pq.QCircuit()
            params = params.getdata()
            prog = circuits(params, generators, circuit)
            return expval(machine,prog,{"X0":1},nqbits)

        def loss(params):
            Z, Y = ansatz1(params,["X","Y"])
            X = ansatz2(params,["X","Y"])
            return 0.5 * Y + 0.8 * Z - 0.2 * X

        t = QTensor([0.3, 0.25])
        opt = Rotosolve(max_iter=5)

        costs_rotosolve = opt.minimize(t,loss)


.. figure:: ./images/rotosolve.png


