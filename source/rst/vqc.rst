
.. _vqc_api:

变分量子线路自动微分模拟
***********************************

VQNet基于自动微分算子构建以及一些常用量子逻辑门、量子线路以及测量方法,可使用自动微分模拟代替量子线路parameter-shift方法计算梯度。我们可以像其他 `Module` 一样,使用VQC算子构成复杂神经网络。在 `Module` 中需要定义虚拟机 `QMachine`,并且需要对machine中 `states` 根据输入的batchsize进行reset_states。请具体看下例:

.. code-block::

    from pyvqnet.nn import Module,Linear,ModuleList
    from pyvqnet.qnn.vqc.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ
    from pyvqnet.qnn.vqc import Probability,QMachine
    from pyvqnet import tensor

    class QM(Module):
        def __init__(self, name=""):
            super().__init__(name)
            self.linearx = Linear(4,2)
            self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                        entangle_gate="cnot",
                                        entangle_rules="linear",
                                        depth=2)
            #基于VQC的RZ 在0比特上
            self.encode1 = RZ(wires=0)
            #基于VQC的RZ 在1比特上
            self.encode2 = RZ(wires=1)
            #基于VQC的概率测量 在0,2比特上
            self.measure = Probability(wires=[0,2])
            #量子设备QMachine,使用4个比特。
            self.device = QMachine(4)
        def forward(self, x, *args, **kwargs):
            #必须要将states reset到与输入一样的batchsize。
            self.device.reset_states(x.shape[0])
            y = self.linearx(x)
            #将输入编码到RZ门上,注意输入必须是 [batchsize,1]的shape
            self.encode1(params = y[:, [0]],q_machine = self.device,)
            #将输入编码到RZ门上,注意输入必须是 [batchsize,1]的shape
            self.encode2(params = y[:, [1]],q_machine = self.device,)
            self.ansatz(q_machine =self.device)
            return self.measure(q_machine =self.device)

    bz =3
    inputx = tensor.arange(1.0,bz*4+1).reshape([bz,4])
    inputx.requires_grad= True
    #像其他Module一样定义
    qlayer = QM()
    #前传
    y = qlayer(inputx)
    #反传
    y.backward()
    print(y)


如果要使用一些带训练参数的变分量子线路逻辑门,而不止像上例一样将数据编码到线路上,可以参考下面例子:

.. code-block::

    from pyvqnet.nn import Module,Linear,ModuleList
    from pyvqnet.qnn.vqc.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ,rz,ry,cnot
    from pyvqnet.qnn.vqc import Probability,QMachine
    from pyvqnet import tensor

    class QM(Module):
        def __init__(self, name=""):
            super().__init__(name)
            self.linearx = Linear(4,2)
            self.ansatz = VQC_HardwareEfficientAnsatz(4, ["rx", "RY", "rz"],
                                        entangle_gate="cnot",
                                        entangle_rules="linear",
                                        depth=2)
            #基于VQC的RZ 在0比特上
            self.encode1 = RZ(wires=0)
            #基于VQC的RZ 在1比特上
            self.encode2 = RZ(wires=1)
            #设置RZ 有要训练参数has_params = True,需要训练trainable= True
            self.vqc = RZ(has_params = True,trainable = True,wires=1)
            #基于VQC的概率测量 在0,2比特上
            self.measure = Probability(wires=[0,2])
            #量子设备QMachine,使用4个比特。
            self.device = QMachine(4)
        def forward(self, x, *args, **kwargs):
            #必须要将states reset到与输入一样的batchsize。
            self.device.reset_states(x.shape[0])
            y = self.linearx(x)
            #将输入编码到RZ门上,注意输入必须是 [batchsize,1]的shape
            self.encode1(params = y[:, [0]],q_machine = self.device,)
            #将输入编码到RZ门上,注意输入必须是 [batchsize,1]的shape
            self.encode2(params = y[:, [1]],q_machine = self.device,)
            #使用RZ门构成的含参变分线路,会加入训练。
            self.vqc(q_machine =self.device)
            self.ansatz(q_machine =self.device)
            return self.measure(q_machine =self.device)

    bz =3
    inputx = tensor.arange(1.0,bz*4+1).reshape([bz,4])
    inputx.requires_grad= True
    #像其他Module一样定义
    qlayer = QM()
    #前传
    y = qlayer(inputx)
    #反传
    y.backward()
    print(y)


模拟器
=========================================

QMachine
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.QMachine(num_wires, dtype=pyvqnet.kcomplex64,grad_mode="",save_ir=False)

    变分量子计算的模拟器类,包含states属性为量子线路的statevectors。

    .. note::
        
        在每次运行一个完整的量子线路之前,必须使用 `pyvqnet.qnn.vqc.QMachine.reset_states(batchsize)` 将模拟器里面初态重新初始化,并且广播为
        (batchsize,*) 维度从而适应批量数据训练。

    :param num_wires: 量子比特数。
    :param dtype: 计算数据的数据类型。默认值是pyvqnet。kcomplex64,对应的参数精度为pyvqnet.kfloat32。
    :param grad_mode: 梯度计算模式,可为 "adjoint",默认值:"",使用自动微分模拟。
    :param save_ir: 设置为True时,将操作保存到originIR,默认值:False。

    :return: 输出QMachine。

    Example::
        
        from pyvqnet.qnn.vqc import QMachine
        qm = QMachine(4)

        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]

        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]


        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]

        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

    .. py:method:: reset_states(batchsize)

        将模拟器里面初态重新初始化,并且广播为
        (batchsize,[2]**num_qubits) 维度从而使得模拟器可以进行批量数据的模型计算。

        :param batchsize: 批处理的数量。


量子逻辑门接口
============================

i
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.i(q_machine, wires, params=None, use_dagger=False)

    对 ``q_machine`` 中的态矢(statevectors)作用量子逻辑门 I 。

    :param q_machine: 量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import i,QMachine
        qm  = QMachine(4)
        i(q_machine=qm, wires=1,)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]

        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]


        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]

        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]


I
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.I(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个I逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个I逻辑门类实例。

    Example::
        
        from pyvqnet.qnn.vqc import I,QMachine
        device = QMachine(4)
        layer = I(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

hadamard
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.hadamard(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 hadamard 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import hadamard,QMachine
        qm  = QMachine(4)
        hadamard(q_machine=qm, wires=1,)
        print(qm.states)
        # [[[[[0.7071068+0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]
        # 
        #    [[0.7071068+0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]]
        # 
        # 
        #   [[[0.       +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]
        # 
        #    [[0.       +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]]]]


Hadamard
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.Hadamard(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个Hadamard逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 Hadamard逻辑门类实例。

    Example::
        
        from pyvqnet.qnn.vqc import Hadamard,QMachine
        device = QMachine(4)
        layer = Hadamard(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)



t
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.t(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 t 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import t,QMachine
        qm  = QMachine(4)
        t(q_machine=qm, wires=1,)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

T
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.T(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个T逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 T逻辑门类实例。

    Example::
        
        from pyvqnet.qnn.vqc import T,QMachine
        device = QMachine(4)
        layer = T(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

s
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.s(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 s 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import s,QMachine
        qm  = QMachine(4)
        s(q_machine=qm, wires=1,)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]       
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

S
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.S(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个S逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 S逻辑门类实例。

    Example::
        
        from pyvqnet.qnn.vqc import S,QMachine
        device = QMachine(4)
        layer = S(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)


paulix
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.paulix(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 paulix 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import paulix,QMachine
        qm  = QMachine(4)
        paulix(q_machine=qm, wires=1,)
        print(qm.states)

        # [[[[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

PauliX
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.PauliX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliX逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliX逻辑门类实例。

    Example::
        
        from pyvqnet.qnn.vqc import PauliX,QMachine
        device = QMachine(4)
        layer = PauliX(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

pauliy
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.pauliy(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 pauliy 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import pauliy,QMachine
        qm  = QMachine(4)
        pauliy(q_machine=qm, wires=1,)
        print(qm.states)

        # [[[[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+1.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

PauliY
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.PauliY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliY逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliY逻辑门类实例。

    Example::
        
        from pyvqnet.qnn.vqc import PauliY,QMachine
        device = QMachine(4)
        layer = PauliY(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

pauliz
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.pauliz(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 pauliz 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import pauliz,QMachine
        qm  = QMachine(4)
        pauliz(q_machine=qm, wires=1,)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

PauliZ
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.PauliZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PauliZ逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PauliZ逻辑门类实例。

    Example::
        
        from pyvqnet.qnn.vqc import PauliZ,QMachine
        device = QMachine(4)
        layer = PauliZ(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

x1
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.x1(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 x1 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import x1,QMachine
        qm  = QMachine(4)
        x1(q_machine=qm, wires=1,)
        print(qm.states)

        # [[[[[0.7071068+0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       -0.7071068j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

X1
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.X1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个X1逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 X1逻辑门类实例。

    Example::
        
        from pyvqnet.qnn.vqc import X1,QMachine
        device = QMachine(4)
        layer = X1(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

rx
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.rx(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 rx 

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import rx,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        rx(q_machine=qm, wires=1,params=QTensor([0.5]))
        print(qm.states)

        # [[[[[0.9689124+0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       -0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]

RX
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RX逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RX逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import RX,QMachine
        device = QMachine(4)
        layer = RX(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

ry
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.ry(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 ry 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import ry,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        ry(q_machine=qm, wires=1,params=QTensor([0.5]),)
        print(qm.states)

        # [[[[[0.9689124+0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]
        # 
        #    [[0.247404 +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]]
        # 
        # 
        #   [[[0.       +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]
        # 
        #    [[0.       +0.j 0.       +0.j]
        #     [0.       +0.j 0.       +0.j]]]]]

RY
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RY逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RY逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import RY,QMachine
        device = QMachine(4)
        layer = RY(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

rz
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.rz(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 rz 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import rz,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        rz(q_machine=qm, wires=1,params=QTensor([0.5]),)
        print(qm.states)
        
        # [[[[[0.9689124-0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]

RZ
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZ逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RZ逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import RZ,QMachine
        device = QMachine(4)
        layer = RZ(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

crx
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.crx(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 crx 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::

        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.crx(q_machine=qm,wires=[0,2], params=QTensor([0.5]),)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]


CRX
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CRX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRX逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRX逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import CRX,QMachine
        device = QMachine(4)
        layer = CRX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

cry
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.cry(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 cry 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::

        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.cry(q_machine=qm,wires=[0,2], params=QTensor([0.5]))
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

CRY
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CRY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRY逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRY逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import CRY,QMachine
        device = QMachine(4)
        layer = CRY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

crz
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.crz(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 crz 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::

        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.crz(q_machine=qm,wires=[0,2], params=QTensor([0.5]))
        print(qm.states)
        
        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

CRZ
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CRZ逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CRZ逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import CRZ,QMachine
        device = QMachine(4)
        layer = CRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



u1
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.u1(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 u1 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import u1,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        u1(q_machine=qm, wires=1,params=QTensor([24.0]),)
        print(qm.states)

        # [[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]

U1
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.U1(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U1逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U1逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import U1,QMachine
        device = QMachine(4)
        layer = U1(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

u2
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.u2(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 u2 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import u2,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        u2(q_machine=qm, wires=1,params=QTensor([[24.0,-3]]),)
        print(qm.states)

        # [[[[[0.7071068+0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.2999398-0.6403406j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

U2
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.U2(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U2逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U2逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import U2,QMachine
        device = QMachine(4)
        layer = U2(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

u3
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.u3(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 u3 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import u3,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        u3(q_machine=qm, wires=1,params=QTensor([[24.0,-3,1]]),)
        print(qm.states)

        # [[[[[0.843854 +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.5312032+0.0757212j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

U3
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.U3(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个U3逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 U3逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import U3,QMachine
        device = QMachine(4)
        layer = U3(has_params= True, trainable= True, wires=0)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

cy
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.cy(q_machine, wires, params=None, use_dagger=False)

    对q_machine中的态矢作用量子逻辑门 cy 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。

    Example::

        from pyvqnet.qnn.vqc import cy,QMachine
        qm = QMachine(4)
        cy(q_machine=qm,wires=(1,0))
        print(qm.states)
        # [[[[[1.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]],


        #   [[[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]]]]


CY
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CY逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CY逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import CY,QMachine
        device = QMachine(4)
        layer = CY(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

cnot
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.cnot(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 cnot 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import cnot,QMachine
        qm  = QMachine(4)
        cnot(q_machine=qm,wires=[1,0],)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

CNOT
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CNOT(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CNOT逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CNOT逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import CNOT,QMachine
        device = QMachine(4)
        layer = CNOT(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

cr
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.cr(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 cr 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import cr,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        cr(q_machine=qm,wires=[1,0],params=QTensor([0.5]),)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

CR
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CR(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CR逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CR逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import CR,QMachine
        device = QMachine(4)
        layer = CR(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

swap
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.swap(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 swap 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import swap,QMachine
        qm  = QMachine(4)
        swap(q_machine=qm,wires=[1,0],)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]


SWAP
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.SWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SWAP逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 SWAP 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import SWAP,QMachine
        device = QMachine(4)
        layer = SWAP(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)


cswap
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.cswap(q_machine, wires, params=None, use_dagger=False)

    对q_machine中的态矢作用量子逻辑门 cswap 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::

        from pyvqnet.qnn.vqc import cswap,QMachine
        qm  = QMachine(4)
        cswap(q_machine=qm,wires=[1,0,3],)
        print(qm.states)
        # [[[[[1.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]],


        #   [[[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]]]]


CSWAP
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.CSWAP(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
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

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CSWAP 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import CSWAP,QMachine
        device = QMachine(4)
        layer = CSWAP(wires=[0,1,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)
        # [[[[[1.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]],


        #   [[[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]]],



        #  [[[[1.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]],


        #   [[[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]],

        #    [[0.+0.j,0.+0.j],
        #     [0.+0.j,0.+0.j]]]]]


iswap
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.iswap(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 iswap 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::

        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.iswap(q_machine=qm,wires=[0,1], params = QTensor([0.5]),)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

cz
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.cz(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 cz 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import cz,QMachine
        qm  = QMachine(4)
        cz(q_machine=qm,wires=[1,0],)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]


CZ
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.CZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个CZ逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 CZ 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import CZ,QMachine
        device = QMachine(4)
        layer = CZ(wires=[0,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

rxx
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.rxx(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 rxx 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import rxx,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        rxx(q_machine=qm,wires=[1,0],params=QTensor([0.2]),)
        print(qm.states)

        # [[[[[0.9950042+0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       -0.0998334j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

RXX
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RXX逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个RXX逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import RXX,QMachine
        device = QMachine(4)
        layer = RXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

ryy
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.ryy(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 ryy 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import ryy,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        ryy(q_machine=qm,wires=[1,0],params=QTensor([0.2]),)
        print(qm.states)

        # [[[[[0.9950042+0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.0998334j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

RYY
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RYY逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RYY 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import RYY,QMachine
        device = QMachine(4)
        layer = RYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

rzz
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.rzz(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 rzz 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import rzz,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        rzz(q_machine=qm,wires=[1,0],params=QTensor([0.2]),)
        print(qm.states)

        # [[[[[0.9950042-0.0998334j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]


RZZ
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.RZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZZ逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RZZ 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import RZZ,QMachine
        device = QMachine(4)
        layer = RZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

rzx
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.rzx(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 RZX 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import rzx,QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(4)
        rzx(q_machine=qm,wires=[1,0],params=QTensor([0.2]),)
        print(qm.states)

        # [[[[[0.9950042+0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]
        # 
        # 
        #   [[[0.       -0.0998334j 0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]
        # 
        #    [[0.       +0.j        0.       +0.j       ]
        #     [0.       +0.j        0.       +0.j       ]]]]]

RZX
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.RZX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个RZX逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 RZX 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import RZX,QMachine
        device = QMachine(4)
        layer = RZX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

toffoli
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.toffoli(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 toffoli 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
        
        from pyvqnet.qnn.vqc import toffoli,QMachine
        qm  = QMachine(4)
        toffoli(q_machine=qm,wires=[0,1,2],)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]


Toffoli
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.Toffoli(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个Toffoli逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 Toffoli 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import Toffoli,QMachine
        device = QMachine(4)
        layer = Toffoli(  wires=[0,2,1])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

isingxx
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.isingxx(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 isingxx 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.isingxx(q_machine=qm,wires=[0,1], params = QTensor([0.5]),)
        print(qm.states)

        # [[[[[0.9689124+0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       -0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]


IsingXX
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.IsingXX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingXX逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个IsingXX 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import IsingXX,QMachine
        device = QMachine(4)
        layer = IsingXX(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

isingyy
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.isingyy(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 isingyy 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.isingyy(q_machine=qm,wires=[0,1], params = QTensor([0.5]),)
        print(qm.states)

        # [[[[[0.9689124+0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]


IsingYY
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.IsingYY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingYY逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingYY 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import IsingYY,QMachine
        device = QMachine(4)
        layer = IsingYY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

isingzz
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.isingzz(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 isingzz 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.isingzz(q_machine=qm,wires=[0,1], params = QTensor([0.5]),)
        print(qm.states)

        # [[[[[0.9689124-0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]

IsingZZ
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.IsingZZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingZZ逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingZZ 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import IsingZZ,QMachine
        device = QMachine(4)
        layer = IsingZZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

isingxy
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.isingxy(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 isingxy 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.isingxy(q_machine=qm,wires=[0,1], params = QTensor([0.5]),)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

IsingXY
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.IsingXY(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个IsingXY逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 IsingXY 逻辑门类实例。
    
    Example::

        from pyvqnet.qnn.vqc import IsingXY,QMachine
        device = QMachine(4)
        layer = IsingXY(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

phaseshift
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.phaseshift(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 phaseshift 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.phaseshift(q_machine=qm,wires=[0], params = QTensor([0.5]),)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

PhaseShift
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.PhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个PhaseShift逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 PhaseShift 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import PhaseShift,QMachine
        device = QMachine(4)
        layer = PhaseShift(has_params= True, trainable= True, wires=1)
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)

multirz
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.multirz(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 multirz 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.multirz(q_machine=qm,wires=[0, 1], params = QTensor([0.5]),)
        print(qm.states)

        # [[[[[0.9689124-0.247404j 0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]
        # 
        # 
        #   [[[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]
        # 
        #    [[0.       +0.j       0.       +0.j      ]
        #     [0.       +0.j       0.       +0.j      ]]]]]

MultiRZ
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.MultiRZ(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个MultiRZ逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个 MultiRZ 逻辑门类实例。

    Example::

        from pyvqnet.qnn.vqc import MultiRZ,QMachine
        device = QMachine(4)
        layer = MultiRZ(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)
        
sdg
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.sdg(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 sdg 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.sdg(q_machine=qm,wires=[0],)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]


SDG
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.SDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个SDG逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个SDG逻辑门类实例。

    Example::
        
        from pyvqnet.qnn.vqc import SDG,QMachine
        device = QMachine(4)
        layer = SDG(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

tdg
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.tdg(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 tdg 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.tdg(q_machine=qm,wires=[0],)
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

TDG
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.TDG(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个TDG逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个TDG逻辑门类实例。

    Example::
        
        from pyvqnet.qnn.vqc import TDG,QMachine
        device = QMachine(4)
        layer = TDG(wires=0)
        batchsize = 1
        device.reset_states(1)
        layer(q_machine = device)
        print(device.states)

controlledphaseshift
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.controlledphaseshift(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 controlledphaseshift 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        for i in range(4):
            vqc.hadamard(q_machine=qm, wires=i)
        vqc.controlledphaseshift(q_machine=qm,params=QTensor([0.5]),wires=[0,1],)
        print(qm.states)

        # [[[[[0.25     +0.j        0.25     +0.j       ]
        #     [0.25     +0.j        0.25     +0.j       ]]
        # 
        #    [[0.25     +0.j        0.25     +0.j       ]
        #     [0.25     +0.j        0.25     +0.j       ]]]
        # 
        # 
        #   [[[0.25     +0.j        0.25     +0.j       ]
        #     [0.25     +0.j        0.25     +0.j       ]]
        # 
        #    [[0.2193956+0.1198564j 0.2193956+0.1198564j]
        #     [0.2193956+0.1198564j 0.2193956+0.1198564j]]]]]


ControlledPhaseShift
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.ControlledPhaseShift(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False)
    
    定义一个ControlledPhaseShift逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :return: 一个ControlledPhaseShift。

    Example::

        from pyvqnet.qnn.vqc import ControlledPhaseShift,QMachine
        device = QMachine(4)
        layer = ControlledPhaseShift(has_params= True, trainable= True, wires=[0,2])
        batchsize = 2
        device.reset_states(batchsize)
        layer(q_machine = device)
        print(device.states)



multicontrolledx
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.multicontrolledx(q_machine, wires, params=None, use_dagger=False,control_values=None)
    
    对q_machine中的态矢作用量子逻辑门 multicontrolledx 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    :param control_values: 控制值,默认为None,当比特位为1时控制。


    Example::
 


        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        for i in range(4):
            vqc.hadamard(q_machine=qm, wires=i)
        vqc.phaseshift(q_machine=qm,wires=[0], params = QTensor([0.5]))
        vqc.phaseshift(q_machine=qm,wires=[1], params = QTensor([2]))
        vqc.phaseshift(q_machine=qm,wires=[3], params = QTensor([3]))
        vqc.multicontrolledx(qm, wires=[0, 1, 3, 2])
        print(qm.states)

        # [[[[[ 0.25     +0.j       ,-0.2474981+0.03528j  ],
        #     [ 0.25     +0.j       ,-0.2474981+0.03528j  ]],

        #    [[-0.1040367+0.2273243j, 0.0709155-0.239731j ],
        #     [-0.1040367+0.2273243j, 0.0709155-0.239731j ]]],


        #   [[[ 0.2193956+0.1198564j,-0.2341141-0.0876958j],
        #     [ 0.2193956+0.1198564j,-0.2341141-0.0876958j]],

        #    [[-0.2002859+0.149618j , 0.1771674-0.176385j ],
        #     [-0.2002859+0.149618j , 0.1771674-0.176385j ]]]]]


MultiControlledX
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.MultiControlledX(has_params: bool = False,trainable: bool = False,init_params=None,wires=None,dtype=pyvqnet.kcomplex64,use_dagger=False,control_values=None)
    
    定义一个MultiControlledX逻辑门类 。

    :param has_params:  是否具有参数,例如RX,RY等门需要设置为True,不含参数的需要设置为False,默认为False。
    :param trainable: 是否自带含待训练参数,如果该层使用外部输入数据构建逻辑门矩阵,设置为False,如果待训练参数需要从该层初始化,则为True,默认为False。
    :param init_params: 初始化参数,用来编码经典数据QTensor,默认为None,如果为p个参数的含参逻辑门,入参的数据维度需要为[1,p]或者[p]。
    :param wires: 线路作用的比特索引,默认为None。
    :param dtype: 逻辑门内部矩阵的数据精度,可以设置为pyvqnet.kcomplex64,或pyvqnet.kcomplex128,分别对应float输入或者double入参。
    :param use_dagger: 是否使用该门的转置共轭版本,默认为False。
    :param control_values: 控制值,默认为None,当比特位为1时控制。

    :return: 一个 MultiControlledX 逻辑门实例。

    Example::

        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor,kcomplex64

        qm = QMachine(4,dtype=kcomplex64)
        qm.reset_states(2)
        for i in range(4):
            vqc.hadamard(q_machine=qm, wires=i)
        vqc.isingzz(q_machine=qm, params=QTensor([0.25]), wires=[1,0])
        vqc.double_excitation(q_machine=qm, params=QTensor([0.55]), wires=[0,1,2,3])

        mcx = vqc.MultiControlledX( 
                        init_params=None,
                        wires=[2,3,0,1],
                        dtype=kcomplex64,
                        use_dagger=False,control_values=[1,0,0])
        y = mcx(q_machine = qm)
        print(qm.states)
        """
        [[[[[0.2480494-0.0311687j,0.2480494-0.0311687j],
            [0.2480494+0.0311687j,0.1713719-0.0215338j]],

        [[0.2480494+0.0311687j,0.2480494+0.0311687j],
            [0.2480494-0.0311687j,0.2480494+0.0311687j]]],


        [[[0.2480494+0.0311687j,0.2480494+0.0311687j],
            [0.2480494+0.0311687j,0.2480494+0.0311687j]],

        [[0.306086 -0.0384613j,0.2480494-0.0311687j],
            [0.2480494-0.0311687j,0.2480494-0.0311687j]]]],



        [[[[0.2480494-0.0311687j,0.2480494-0.0311687j],
            [0.2480494+0.0311687j,0.1713719-0.0215338j]],

        [[0.2480494+0.0311687j,0.2480494+0.0311687j],
            [0.2480494-0.0311687j,0.2480494+0.0311687j]]],


        [[[0.2480494+0.0311687j,0.2480494+0.0311687j],
            [0.2480494+0.0311687j,0.2480494+0.0311687j]],

        [[0.306086 -0.0384613j,0.2480494-0.0311687j],
            [0.2480494-0.0311687j,0.2480494-0.0311687j]]]]]
        """

single_excitation
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.single_excitation(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 single_excitation 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        vqc.single_excitation(q_machine=qm, wires=[0, 1], params=QTensor([0.5]))
        print(qm.states)

        # [[[[[1.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]
        # 
        # 
        #   [[[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]
        # 
        #    [[0.+0.j 0.+0.j]
        #     [0.+0.j 0.+0.j]]]]]

double_excitation
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.double_excitation(q_machine, wires, params=None, use_dagger=False)
    
    对q_machine中的态矢作用量子逻辑门 double_excitation 。

    :param q_machine:  量子虚拟机设备。
    :param wires: 量子比特索引。
    :param params: 参数矩阵,默认为None,对于含p个参数的逻辑门操作函数,入参的维度需要为[1,p],或[p]。
    :param use_dagger: 是否共轭转置,默认为False。
    

    Example::
    
        from pyvqnet.qnn.vqc import QMachine
        import pyvqnet.qnn.vqc as vqc
        from pyvqnet.tensor import QTensor
        qm = QMachine(4)
        for i in range(4):
            vqc.hadamard(q_machine=qm, wires=i)
        vqc.isingzz(q_machine=qm, params=QTensor([0.55]), wires=[1,0])
        vqc.double_excitation(q_machine=qm, params=QTensor([0.55]), wires=[0,1,2,3])
        print(qm.states)

        # [[[[[0.2406063-0.0678867j 0.2406063-0.0678867j]
        #     [0.2406063-0.0678867j 0.1662296-0.0469015j]]
        # 
        #    [[0.2406063+0.0678867j 0.2406063+0.0678867j]
        #     [0.2406063+0.0678867j 0.2406063+0.0678867j]]]
        # 
        # 
        #   [[[0.2406063+0.0678867j 0.2406063+0.0678867j]
        #     [0.2406063+0.0678867j 0.2406063+0.0678867j]]
        # 
        #    [[0.2969014-0.0837703j 0.2406063-0.0678867j]
        #     [0.2406063-0.0678867j 0.2406063-0.0678867j]]]]]  


测量接口
===================


VQC_Purity
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_Purity(state, qubits_idx, num_wires)

    从态矢中计算特定量子比特 ``qubits_idx`` 上的纯度。

    .. math::
        \gamma = \text{Tr}(\rho^2)

    式中 :math:`\rho` 为密度矩阵。标准化量子态的纯度满足 :math:`\frac{1}{d} \leq \gamma \leq 1` ,
    其中 :math:`d` 是希尔伯特空间的维数。
    纯态的纯度是1。

    :param state: 从pyqpanda get_qstate()获取的量子态
    :param qubits_idx: 要计算纯度的量子比特位索引
    :param num_wires: 量子比特数

    :return: 对应比特位置上的纯度。

    .. note::
        
        该函数结果一般为[b,len(qubits_idx)],其中b为 q_machine.reset_states(b)的批处理数量b。

    Example::

        from pyvqnet.qnn.vqc import VQC_Purity, rx, ry, cnot, QMachine
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.7, 0.4], [1.7, 2.4]], requires_grad=True)
        qm = QMachine(3)
        qm.reset_states(2)
        rx(q_machine=qm, wires=0, params=x[:, [0]])
        ry(q_machine=qm, wires=1, params=x[:, [1]])
        ry(q_machine=qm, wires=2, params=x[:, [1]])
        cnot(q_machine=qm, wires=[0, 1])
        cnot(q_machine=qm, wires=[2, 1])
        y = VQC_Purity(qm.states, [0, 1], num_wires=3)
        y.backward()
        print(y)

        # [0.9356751 0.875957]

VQC_VarMeasure
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_VarMeasure(q_machine, obs)

    提供的可观察量 ``obs`` 的方差。

    :param q_machine: 从pyqpanda get_qstate()获取的量子态
    :param obs: 测量观测量,当前支持Hadamard,I,PauliX,PauliY,PauliZ 几种Observable.

    :return: 计算可观测量方差。

    .. note::

        测量结果一般为[b,1],其中b为 q_machine.reset_states(b)的批处理数量b。

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc import VQC_VarMeasure, rx, cnot, hadamard, QMachine,PauliY
        x = QTensor([[0.5]], requires_grad=True)
        qm = QMachine(3)
        rx(q_machine=qm, wires=0, params=x)
        var_result = VQC_VarMeasure(q_machine= qm, obs=PauliY(wires=0))
        var_result.backward()
        print(var_result)

        # [[0.7701511]]


VQC_DensityMatrixFromQstate
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_DensityMatrixFromQstate(state, indices)

    计算量子态在一组特定量子比特上的密度矩阵。

    :param state: 一维列表状态向量。 这个列表的大小应该是 ``(2**N,)`` 对于量子比特个数 ``N`` ,qstate 应该从 000 ->111 开始。
    :param indices: 所考虑子系统中的量子比特索引列表。
    :return: 大小为“(b, 2**len(indices), 2**len(indices))”的密度矩阵,其中b为 q_machine.reset_states(b)的批处理数量b。


    Example::

        from pyvqnet.qnn.vqc import VQC_DensityMatrixFromQstate,rx,ry,cnot,QMachine
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.7,0.4],[1.7,2.4]],requires_grad=True)

        qm = QMachine(3)
        qm.reset_states(2)
        rx(q_machine=qm,wires=0,params=x[:,[0]])
        ry(q_machine=qm,wires=1,params=x[:,[1]])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,1])
        cnot(q_machine=qm,wires=[2, 1])
        y = VQC_DensityMatrixFromQstate(qm.states,[0,1])
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
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.Probability(wires, name="")

    一个计算量子线路在特定比特上概率测量的Module类。


    :param wires: 测量比特的索引,列表、元组或者整数。
    :param name: 模块的名字,默认:""。
    :return: 一个概率测量的类。

    .. py:method:: forward(q_machine)

        进行概率测量计算

        :param q_machine: 作用的量子态矢模拟器
        :return: 概率测量结果

    .. note::

        使用该类进行计算的概率测量结果一般为[b,len(wires)],其中b为 q_machine.reset_states(b)的批处理数量b。

    Example::

        from pyvqnet.qnn.vqc import Probability,rx,ry,cnot,QMachine,rz
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

        # [[1.0000002 0.       ]
        #  [1.0000002 0.       ]]


MeasureAll
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.MeasureAll(obs, name="")

    计算量子线路的测量结果,支持输入观测量 ``obs`` 为由观测量 `observables`、作用比特 `wires` 、系数 `coefficient` 键值对构成的字典,或者键值对字典列表。
    例如:

    {\'wires\': [0,  1], \'observables\': [\'x\', \'i\'],\'coefficient\':[0.23,-3.5]}
     
    {\'X0\': 0.23}
     
    [{\'wires\': [0, 2, 3],\'observables\': [\'X\', \'Y\', \'Z\'],\'coefficient\': [1, 0.5, 0.4]}, {\'wires\': [0, 1, 2],\'observables\': [\'X\', \'Y\', \'Z\'],\'coefficient\': [1, 0.5, 0.4]}]
    
    [{\'X1 Z2 I0\':4,\'Z1 Z0\':3},\{'wires\': [0,  1], \'observables\': [\'x\', \'i\'],\'coefficient\':[0.23,-3.5]}]

    {\'X1 Z2 I0\':4,\'Z1 Z0\':3}

    :param obs: 测量观测量,可以是单个观测量类包括,或者由观测量、作用比特、系数键值对构成的字典,或者键值对字典列表。
    :param name: 模块的名字,默认:""。

    .. py:method:: forward(q_machine)
        
        进行测量操作

        :param q_machine: 作用的量子态矢模拟器
        :return: 测量结果,QTensor。

    .. note::

        如果 ``obs`` 是列表,使用该类进行计算的测量结果一般为[b,obs的列表长度],其中b为 q_machine.reset_states(b)的批处理数量b。
        如果 ``obs`` 是字典,使用该类进行计算的测量结果一般为[b,1],其中b为 q_machine.reset_states(b)的批处理数量b。
    

    Example::

        from pyvqnet.qnn.vqc import MeasureAll,rx,ry,cnot,QMachine,rz
        from pyvqnet.tensor import QTensor
        from pyvqnet import kfloat64
        x = QTensor([[0.56, 0.1],[0.56, 0.1]],requires_grad=True)

        # 4 qubits
        qm = QMachine(4)
        # batch size = 2
        qm.reset_states(2)

        rz(q_machine=qm,wires=0,params=x[:,[0]])
        rz(q_machine=qm,wires=1,params=x[:,[0]])
        cnot(q_machine=qm,wires=[0,1])
        ry(q_machine=qm,wires=2,params=x[:,[1]])
        cnot(q_machine=qm,wires=[0,2])
        rz(q_machine=qm,wires=3,params=x[:,[1]])
        
        # list of 2 observables
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
        
        #return QTensor of [batch_size,len(obs)]
        print(y)

        # [[0.4000001 0.3980018]
        #  [0.4000001 0.3980018]]


Samples
----------------------------

.. py:class:: pyvqnet.qnn.vqc.Samples(wires=None, obs=None, shots = 1,name="")

    获取特定线路 ``wires`` 上的带有 ``shots`` 的观测量 ``obs`` 结果。

    :param wires: 样本量子比特索引。默认值:None,根据运行时使用模拟器的所有比特。
    :param obs: 该值只能设为None。
    :param shots: 样本重复次数,默认值:1。
    :param name: 此模块的名称,默认值:“”。
    :return: 一个测量方法类


    .. py:method:: forward(q_machine)

        进行采样操作。

        :param q_machine: 作用的量子态矢模拟器
        :return: 测量结果,QTensor。

    .. note::

        使用该类进行计算的测量结果一般为[b,shots,len(wires)],其中b为 q_machine.reset_states(b)的批处理数量b。

    Example::

        from pyvqnet.qnn.vqc import Samples,rx,ry,cnot,QMachine,rz
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
        """
        [[[0,0,0],
        [0,1,0],
        [0,0,0]],

        [[0,1,0],
        [0,0,0],
        [0,1,0]]]
        """


SparseHamiltonian
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.SparseHamiltonian(obs, name="")

    计算观测量 ``obs`` 的稀疏哈密顿量,例如 {"observables":H,"wires":[0,2,3]}。

    :param obs: 稀疏观测量,使用 `tensor.dense_to_csr()` 函数获取稠密函数的稀疏格式。
    :param name: 模块的名字,默认:""。
    :return: 一个测量方法类

    .. py:method:: forward(q_machine)

        进行稀疏哈密顿量测量。

        :param q_machine: 作用的量子态矢模拟器
        :return: 测量结果,QTensor。

    .. note::

        使用该类进行计算的测量结果一般为[b,1],其中b为 q_machine.reset_states(b)的批处理数量b。

    Example::

        import pyvqnet
        pyvqnet.utils.set_random_seed(42)
        from pyvqnet import tensor
        from pyvqnet.nn import Module
        from pyvqnet.qnn.vqc import QMachine,CRX,PauliX,paulix,crx,SparseHamiltonian
        H = tensor.QTensor(
        [[ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
        0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
        0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
        0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
        0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
        0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
        0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
        1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,
        0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,
        0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,
        0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
        0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,],
        [ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
        0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,],
        [ 0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
        0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,],
        [ 0.+0.j,  0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
        0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,],
        [ 0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
        0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,],
        [-1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
        0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,]],dtype=pyvqnet.kcomplex64)
        cpu_csr = tensor.dense_to_csr(H)
        class QModel(Module):
            def __init__(self, num_wires, dtype,grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires)
                self.measure = SparseHamiltonian(obs = {"observables":cpu_csr, "wires":[2, 1, 3, 5]})


            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                paulix(q_machine=self.qm, wires= 0)
                paulix(q_machine=self.qm, wires = 2)
                crx(q_machine=self.qm,wires=[0, 1],params=tensor.full((x.shape[0],1),0.1,dtype=pyvqnet.kcomplex64))
                crx(q_machine=self.qm,wires=[2, 3],params=tensor.full((x.shape[0],1),0.2,dtype=pyvqnet.kcomplex64))
                crx(q_machine=self.qm,wires=[1, 2],params=tensor.full((x.shape[0],1),0.3,dtype=pyvqnet.kcomplex64))
                crx(q_machine=self.qm,wires=[2, 4],params=tensor.full((x.shape[0],1),0.3,dtype=pyvqnet.kcomplex64))
                crx(q_machine=self.qm,wires=[5, 3],params=tensor.full((x.shape[0],1),0.3,dtype=pyvqnet.kcomplex64))
                
                rlt = self.measure(q_machine=self.qm)
                return rlt

        model = QModel(6,pyvqnet.kcomplex64)
        y = model(tensor.ones([1,1]))

        print(y)
        #[0.]


HermitianExpval
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.HermitianExpval(obs, name="")

    计算量子线路某个厄密特观测量 ``obs`` 的期望。

    :param obs: 厄密特观测量。
    :param name: 模块的名字,默认:""。
    :return: 期望结果,QTensor。

    .. py:method:: forward(q_machine)

        进行厄密特量测量。

        :param q_machine: 作用的量子态矢模拟器
        :return: 测量结果,QTensor。

    .. note::

        使用该类进行计算的测量结果一般为[b,1],其中b为 q_machine.reset_states(b)的批处理数量b。


    Example::


        from pyvqnet.qnn.vqc import qcircuit
        from pyvqnet.qnn.vqc import QMachine, RX, RY, CNOT, PauliX, qmatrix, PauliZ, VQC_RotCircuit,HermitianExpval
        from pyvqnet.tensor import QTensor, tensor
        import pyvqnet
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

                qcircuit.rx(q_machine=self.qm, wires=0, params=x[:, [1]])
                qcircuit.ry(q_machine=self.qm, wires=1, params=x[:, [0]])
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


        # [[5.3798223],
        #  [7.1294155],
        #  [0.7028297]]


常用量子变分线路模板
=======================================

VQC_HardwareEfficientAnsatz
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_HardwareEfficientAnsatz(n_qubits,single_rot_gate_list,entangle_gate="CNOT",entangle_rules='linear',depth=1,initial=None,dtype=None)

    一个含可训练参数的变分量子线路类,实现了论文介绍的Hardware Efficient Ansatz的实现: `Hardware-efficient Variational Quantum Eigensolver for Small Molecules <https://arxiv.org/pdf/1704.05018.pdf>`__ 。

    :param n_qubits: 量子比特数。
    :param single_rot_gate_list: 单个量子比特旋转门列表由一个或多个作用于每个量子比特的旋转门构成。目前支持 Rx、Ry、Rz。
    :param entangle_gate: 非参数化纠缠门。支持 CNOT、CZ。默认值: CNOT。
    :param entangle_rules: 纠缠门在电路中的使用方式。"linear" 表示纠缠门将作用于每个相邻的量子比特。'all' 表示纠缠门将作用于任意两个量子比特。默认值: "linear"。
    :param depth: 假设的深度, 默认值: 1。
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数, 默认值: None, 此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用 float32。
    :return: 一个 VQC_HardwareEfficientAnsatz 实例。



    Example::

        from pyvqnet.nn import Module,Linear,ModuleList
        from pyvqnet.qnn.vqc.qcircuit import VQC_HardwareEfficientAnsatz,RZZ,RZ
        from pyvqnet.qnn.vqc import Probability,QMachine
        from pyvqnet import tensor

        class QM(Module):
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

        # [[0.3075959 0.2315064 0.2491432 0.2117545]
        #  [0.3075958 0.2315062 0.2491433 0.2117546]
        #  [0.3075958 0.2315062 0.2491432 0.2117545]]

VQC_BasicEntanglerTemplate
-----------------------------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_BasicEntanglerTemplate(num_layer=1, num_qubits=1, rotation="RX", initial=None, dtype=None)

    由每个量子位上的单参数单量子位旋转组成的层,后跟一个闭合链或环组合的多个CNOT 门构成的类。

    CNOT 门环将每个量子位与其邻居连接起来,最后一个量子位被认为是第一个量子位的邻居。

    :param num_layer: 量子比特线路层数。
    :param num_qubits: 量子比特数,默认为1。
    :param rotation: 使用单参数单量子比特门,``RX`` 被用作默认值。
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数, 默认值: None, 此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用 float32。
    :return: 返回一个含可训练参数的VQC_BasicEntanglerTemplate实例。

    Example::

        from pyvqnet.nn import Module, Linear, ModuleList
        from pyvqnet.qnn.vqc.qcircuit import VQC_BasicEntanglerTemplate, RZZ, RZ
        from pyvqnet.qnn.vqc import Probability, QMachine
        from pyvqnet import tensor


        class QM(Module):
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

        # [[1.0000002 0.        0.        0.       ]]


VQC_StronglyEntanglingTemplate
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_StronglyEntanglingTemplate(num_layers=1, num_qubits=1, rotation = "RX", initial = None, dtype: = None)

    由单个量子比特旋转和纠缠器组成的层,参考 `circuit-centric classifier design <https://arxiv.org/abs/1804.00633>`__ .

 
    :param num_layers: 重复层数,默认值: 1。
    :param num_qubits: 量子比特数,默认值: 1。
    :param rotation: 要使用的单参数单量子比特门,默认值: `RX`
    :param initial: 使用initial 初始化所有其中参数逻辑门的参数,默认值: None,此模块将随机初始化参数。
    :param dtype: 参数的数据类型,默认值: None,使用 float32。
    :return: VQC_BasicEntanglerTemplate 实例


    Example::

        from pyvqnet.nn import Module
        from pyvqnet.qnn.vqc.qcircuit import VQC_StronglyEntanglingTemplate
        from pyvqnet.qnn.vqc import Probability, QMachine
        from pyvqnet import tensor


        class QM(Module):
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

        # [[0.3745951 0.154298  0.059156  0.4119509]]


VQC_QuantumEmbedding
---------------------------------------------------------------


.. py:class:: pyvqnet.qnn.vqc.VQC_QuantumEmbedding(qubits, machine, num_repetitions_input, depth_input, num_unitary_layers, num_repetitions,initial = None,dtype = None,name= "")

    使用 RZ,RY,RZ 创建变分量子电路,将经典数据编码为量子态。
    参考 `Quantum embeddings for machine learning <https://arxiv.org/abs/2001.03622>`_。

 
    :param num_repetitions_input: 子模块中输入编码的重复次数。
    :paramdepth_input: 输入维数。
    :param num_unitary_layers: 变分量子门的重复次数。
    :param num_repetitions: 子模块的重复次数。
    :param initial: 参数初始化值,默认为None
    :param dtype: 参数的类型,默认 None,使用float32.
    :param name: 类的名字

    Example::

        from pyvqnet.nn import Module
        from pyvqnet.qnn.vqc.qcircuit import VQC_QuantumEmbedding
        from pyvqnet.qnn.vqc import  QMachine,MeasureAll
        from pyvqnet import tensor
        import pyvqnet
        depth_input = 2
        num_repetitions = 2
        num_repetitions_input = 2
        num_unitary_layers = 2
        nq = depth_input * num_repetitions_input
        bz = 12

        class QM(Module):
            def __init__(self, name=""):
                super().__init__(name)

                self.ansatz = VQC_QuantumEmbedding(num_repetitions_input, depth_input,
                                                num_unitary_layers,
                                                num_repetitions, initial=tensor.full([1],12.0),dtype=pyvqnet.kfloat64)

                self.measure = MeasureAll(obs = {f"Z{nq-1}":1})
                self.device = QMachine(nq,dtype=pyvqnet.kcomplex128)

            def forward(self, x, *args, **kwargs):
                self.device.reset_states(x.shape[0])
                self.ansatz(x,q_machine=self.device)
                return self.measure(q_machine=self.device)

        inputx = tensor.arange(1.0, bz * depth_input + 1,
                                dtype=pyvqnet.kfloat64).reshape([bz, depth_input])
        qlayer = QM()
        y = qlayer(inputx)
        y.backward()
        print(y)

        # [[-0.2539548]
        #  [-0.1604787]
        #  [ 0.1492931]
        #  [-0.1711956]
        #  [-0.1577133]
        #  [ 0.1396999]
        #  [ 0.016864 ]
        #  [-0.0893069]
        #  [ 0.1897014]
        #  [ 0.0941301]
        #  [ 0.0550722]
        #  [ 0.2408579]]

ExpressiveEntanglingAnsatz
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.ExpressiveEntanglingAnsatz(type: int, num_wires: int, depth: int, dtype=None, name: str = "")

    论文 `Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms <https://arxiv.org/pdf/1905.10876.pdf>`_ 中的 19 种不同的ansatz。

    :param type: 电路类型从 1 到 19,共19种线路。
    :param num_wires: 量子比特数。
    :param depth: 电路深度。
    :param dtype: 参数的数据类型, 默认值: None, 使用 float32。
    :param name: 名字,默认"".

    :return:
        一个 ExpressiveEntanglingAnsatz 实例

    Example::

        from pyvqnet.qnn.vqc  import *
        import pyvqnet
        pyvqnet.utils.set_random_seed(42)
        from pyvqnet.nn import Module
        class QModel(Module):
            def __init__(self, num_wires, dtype,grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype,grad_mode=grad_mode,save_ir=True)
                self.c1 = ExpressiveEntanglingAnsatz(13,3,2)
                self.measure = MeasureAll(obs = {
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

        #input_x = tensor.broadcast_to(input_x,[2,3])

        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex64)

        batch_y = qunatum_model(input_x)
        z = vqc_to_originir_list(qunatum_model)
        for zi in z:
            print(zi)
        batch_y.backward()
        print(batch_y)



VQC_BasisEmbedding
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_BasisEmbedding(basis_state,q_machine)

    将二进制特征 ``basis_state`` 编码为 ``q_machine`` 的基态。

    例如, 对于 ``basis_state=([0, 1, 1])``, 在量子系统下其基态为 :math:`|011 \rangle`。

    :param basis_state:  ``(n)`` 大小的二进制输入。
    :param q_machine: 量子虚拟机设备。

    Example::
        
        from pyvqnet.qnn.vqc import VQC_BasisEmbedding,QMachine
        qm  = QMachine(3)
        VQC_BasisEmbedding(basis_state=[1,1,0],q_machine=qm)
        print(qm.states)

        # [[[[0.+0.j 0.+0.j]
        #    [0.+0.j 0.+0.j]]
        # 
        #   [[0.+0.j 0.+0.j]
        #    [1.+0.j 0.+0.j]]]]


VQC_AngleEmbedding
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_AngleEmbedding(input_feat, wires, q_machine: pyvqnet.qnn.vqc.QMachine, rotation: str = "X")

    将 :math:`N` 特征编码到 ``q_machine`` :math:`n` 量子比特的旋转角度中, 其中 :math:`N \leq n`。

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

        from pyvqnet.qnn.vqc import VQC_AngleEmbedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(2)
        VQC_AngleEmbedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='X')
        
        print(qm.states)
        # [[[ 0.398068 +0.j         0.       -0.2174655j]
        #   [ 0.       -0.7821081j -0.4272676+0.j       ]]]

        VQC_AngleEmbedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Y')

        print(qm.states)
        # [[[-0.0240995+0.6589843j  0.4207355+0.2476033j]
        #   [ 0.4042482-0.2184162j  0.       -0.3401631j]]]

        VQC_AngleEmbedding(QTensor([2.2, 1]), [0, 1], q_machine=qm, rotation='Z')

        print(qm.states)

        # [[[0.659407 +0.0048471j 0.4870554-0.0332093j]
        #   [0.4569675+0.047989j  0.340018 +0.0099326j]]]

VQC_AmplitudeEmbedding
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_AmplitudeEmbeddingCircuit(input_feature, q_machine)

    将 :math:`2^n` 特征编码为 ``q_machine``  :math:`n` 量子比特的振幅向量。

    :param input_feature: 表示参数的numpy数组。
    :param q_machine: 量子虚拟机设备。
    

    Example::

        from pyvqnet.qnn.vqc import VQC_AmplitudeEmbedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        VQC_AmplitudeEmbedding(QTensor([3.2,-2,-2,0.3,12,0.1,2,-1]), q_machine=qm)
        print(qm.states)

        # [[[[ 0.2473717+0.j -0.1546073+0.j]
        #    [-0.1546073+0.j  0.0231911+0.j]]
        # 
        #   [[ 0.9276441+0.j  0.0077304+0.j]
        #    [ 0.1546073+0.j -0.0773037+0.j]]]]

VQC_IQPEmbedding
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_IQPEmbedding(input_feat, q_machine: pyvqnet.qnn.vqc.QMachine, rep: int = 1)

    使用IQP线路的对角门将 :math:`n` 特征编码为 ``q_machine``  :math:`n` 量子比特。

    编码是由 `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_ 提出。

    通过指定 ``rep`` ,可以重复基本IQP线路。

    :param input_feat: 表示参数的数组。
    :param q_machine: 量子虚拟机设备。
    :param rep: 重复量子线路块次数,默认次数为1。
    

    Example::

        from pyvqnet.qnn.vqc import VQC_IQPEmbedding, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        VQC_IQPEmbedding(QTensor([3.2,-2,-2]), q_machine=qm)
        print(qm.states)        
        
        # [[[[ 0.0309356-0.3521973j  0.3256442+0.1376801j]
        #    [ 0.3256442+0.1376801j  0.2983474+0.1897071j]]
        # 
        #   [[ 0.0309356+0.3521973j -0.3170519-0.1564546j]
        #    [-0.3170519-0.1564546j -0.2310978-0.2675701j]]]]


VQC_RotCircuit
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_RotCircuit(q_machine, wire, params)

    在 ``q_machine`` 的比特 ``wire`` 上使用 ``params`` 进行单量子比特旋转操作。

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.


    :param q_machine: 量子虚拟机设备。
    :param wire: 量子比特索引。
    :param params: 表示参数  :math:`[\phi, \theta, \omega]`。
    

    Example::

        from pyvqnet.qnn.vqc import VQC_RotCircuit, QMachine
        from pyvqnet.tensor import QTensor
        qm  = QMachine(3)
        VQC_RotCircuit(q_machine=qm, wire=[1],params=QTensor([2.0,1.5,2.1]))
        print(qm.states)

        # [[[[-0.3373617-0.6492732j  0.       +0.j       ]
        #    [ 0.6807868-0.0340677j  0.       +0.j       ]]
        # 
        #   [[ 0.       +0.j         0.       +0.j       ]
        #    [ 0.       +0.j         0.       +0.j       ]]]]

VQC_CRotCircuit
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_CRotCircuit(para,control_qubits,rot_wire,q_machine)

    在 ``q_machine`` 的比特 ``rot_wire`` 以及受控比特 ``control_qubits`` 上使用 ``params`` 进行受控Rot操作符。

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

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.qcircuit import VQC_CRotCircuit
        from pyvqnet.qnn.vqc import QMachine, MeasureAll
        p = QTensor([2, 3, 4.0])
        qm = QMachine(2)
        VQC_CRotCircuit(p, 0, 1, qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.9999999]]




VQC_Controlled_Hadamard
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_Controlled_Hadamard(wires, q_machine)

    在 ``q_machine`` 的比特 ``wire`` 上使用受控Hadamard逻辑门

    .. math:: CH = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
            0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
        \end{bmatrix}.

    :param wires: 量子比特索引列表, 第一位是控制比特, 列表长度为2。
    :param q_machine: 量子虚拟机设备。
    

    Examples::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.qcircuit import VQC_Controlled_Hadamard
        from pyvqnet.qnn.vqc import QMachine, MeasureAll
        p = QTensor([0.2, 3, 4.0])

        qm = QMachine(3)

        VQC_Controlled_Hadamard([1, 0], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[1.]]

VQC_CCZ
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_CCZ(wires, q_machine)

    在 ``q_machine`` 的比特 ``wire`` 上使用受控-受控-Z (controlled-controlled-Z) 逻辑门。

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

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.qcircuit import VQC_CCZ
        from pyvqnet.qnn.vqc import QMachine, MeasureAll
        p = QTensor([0.2, 3, 4.0])

        qm = QMachine(3)

        VQC_CCZ([1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.9999999]]


VQC_FermionicSingleExcitation
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_FermionicSingleExcitation(weight, wires, q_machine)

    在 ``q_machine`` 的比特 ``wire`` 上使用由 ``weight`` 构成的泡利矩阵的张量积求幂的耦合簇单激励算子。矩阵形式下式给出:

    .. math::

        \hat{U}_{pr}(\theta) = \mathrm{exp} \{ \theta_{pr} (\hat{c}_p^\dagger \hat{c}_r
        -\mathrm{H.c.}) \},

    :param weight:  量子比特p上的参数, 只有一个元素.
    :param wires: 表示区间[r, p]中的量子比特索引子集。最小长度必须为2。第一索引值被解释为r,最后一个索引值被解释为p。
                中间的索引被CNOT门作用,以计算量子位集的奇偶校验。
    :param q_machine: 量子虚拟机设备。

    

    Examples::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.qcircuit import VQC_FermionicSingleExcitation
        from pyvqnet.qnn.vqc import QMachine, MeasureAll
        qm = QMachine(3)
        p0 = QTensor([0.5])

        VQC_FermionicSingleExcitation(p0, [1, 0, 2], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.9999998]]


VQC_FermionicDoubleExcitation
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_FermionicDoubleExcitation(weight, wires1, wires2, q_machine)

    在 ``q_machine`` 的比特 ``wire1`` , ``wire2`` 上使用由 ``weight`` 对泡利矩阵的张量积求幂的耦合聚类双激励算子,矩阵形式由下式给出:

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

    :param weight: 可变参数
    :param wires1: 代表的量子比特的索引列表区间 [s, r] 中占据量子比特的子集。第一个索引被解释为 s,最后一索引被解释为 r。 CNOT 门对中间的索引进行操作,以计算一组量子位的奇偶性。
    :param wires2: 代表的量子比特的索引列表区间 [q, p] 中占据量子比特的子集。第一根索引被解释为 q,最后一索引被解释为 p。 CNOT 门对中间的索引进行操作,以计算一组量子位的奇偶性。
    :param q_machine: 量子虚拟机设备。

    

    Examples::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.vqc.qcircuit import VQC_FermionicDoubleExcitation
        from pyvqnet.qnn.vqc import QMachine, MeasureAll
        qm = QMachine(5)
        p0 = QTensor([0.5])

        VQC_FermionicDoubleExcitation(p0, [0, 1], [2, 3], qm)
        m = MeasureAll(obs={"Z0": 1})
        exp = m(q_machine=qm)
        print(exp)
        
        # [[0.9999998]]

VQC_UCCSD
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_UCCSD(weights, wires, s_wires, d_wires, init_state, q_machine)

    在 ``q_machine`` 上实现酉耦合簇单激发和双激发拟设(UCCSD)。UCCSD 是 VQE 拟设,通常用于运行量子化学模拟。

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

        from pyvqnet.qnn.vqc import VQC_UCCSD, QMachine, MeasureAll
        from pyvqnet.tensor import QTensor
        p0 = QTensor([2, 0.5, -0.2, 0.3, -2, 1, 3, 0])
        s_wires = [[0, 1, 2], [0, 1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4, 5]]
        d_wires = [[[0, 1], [2, 3]], [[0, 1], [2, 3, 4, 5]], [[0, 1], [3, 4]],
                [[0, 1], [4, 5]]]
        qm = QMachine(6)

        VQC_UCCSD(p0, range(6), s_wires, d_wires, QTensor([1.0, 1, 0, 0, 0, 0]), qm)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)

        # [[0.963802]]

VQC_ZFeatureMap
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_ZFeatureMap(input_feat, q_machine: pyvqnet.qnn.vqc.QMachine, data_map_func=None, rep: int = 2)

    在 ``q_machine`` 上运行一阶泡利 Z 演化电路。

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

        from pyvqnet.qnn.vqc import VQC_ZFeatureMap, QMachine,hadamard
        from pyvqnet.tensor import QTensor
        qm = QMachine(3)
        for i in range(3):
            hadamard(q_machine=qm, wires=[i])
        VQC_ZFeatureMap(input_feat=QTensor([[0.1,0.2,0.3]]),q_machine = qm)
        print(qm.states)
        
        # [[[[0.3535534+0.j        0.2918002+0.1996312j]
        #    [0.3256442+0.1376801j 0.1910257+0.2975049j]]
        # 
        #   [[0.3465058+0.0702402j 0.246323 +0.2536236j]
        #    [0.2918002+0.1996312j 0.1281128+0.3295255j]]]]

VQC_ZZFeatureMap
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_ZZFeatureMap(input_feat, q_machine: pyvqnet.qnn.vqc.QMachine, data_map_func=None, entanglement: Union[str, List[List[int]],Callable[[int], List[int]]] = "full",rep: int = 2)

    在 ``q_machine`` 上运行二阶 Pauli-Z 演化电路。

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

        from pyvqnet.qnn.vqc import VQC_ZZFeatureMap, QMachine
        from pyvqnet.tensor import QTensor
        qm = QMachine(3)
        VQC_ZZFeatureMap(q_machine=qm, input_feat=QTensor([[0.1,0.2,0.3]]))
        print(qm.states)

        # [[[[-0.4234843-0.0480578j -0.144067 +0.1220178j]
        #    [-0.0800646+0.0484439j -0.5512857-0.2947832j]]
        # 
        #   [[ 0.0084012-0.0050071j -0.2593993-0.2717131j]
        #    [-0.1961917-0.3470543j  0.2786197+0.0732045j]]]]

VQC_AllSinglesDoubles
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_AllSinglesDoubles(weights, q_machine: pyvqnet.qnn.vqc.QMachine, hf_state, wires, singles=None, doubles=None)

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

        from pyvqnet.qnn.vqc import VQC_AllSinglesDoubles, QMachine
        from pyvqnet.tensor import QTensor
        qubits = 4
        qm = QMachine(qubits)

        VQC_AllSinglesDoubles(q_machine=qm, weights=QTensor([0.55, 0.11, 0.53]), 
                              hf_state = QTensor([1,1,0,0]), singles=[[0, 2], [1, 3]], doubles=[[0, 1, 2, 3]], wires=[0,1,2,3])
        print(qm.states)
        
        # [ 0.        +0.j  0.        +0.j  0.        +0.j -0.23728043+0.j
        #   0.        +0.j  0.        +0.j -0.27552837+0.j  0.        +0.j
        #   0.        +0.j -0.12207296+0.j  0.        +0.j  0.        +0.j
        #   0.9235152 +0.j  0.        +0.j  0.        +0.j  0.        +0.j]


VQC_BasisRotation
---------------------------------------------------------------


.. py:function:: pyvqnet.qnn.vqc.VQC_BasisRotation(q_machine: pyvqnet.qnn.vqc.QMachine, wires, unitary_matrix: QTensor, check=False)

    在 ``q_machine`` 上实现一个电路,执行精确的单体基础旋转。

    :class:`~.vqc.VQC_BasisRotation` 执行以下由 `arXiv:1711.04789 <https://arxiv.org/abs/1711.04789>`_\ 中给出的单粒子费米子确定的酉变换 :math:`U(u)`
    
    .. math::

        U(u) = \exp{\left( \sum_{pq} \left[\log u \right]_{pq} (a_p^\dagger a_q - a_q^\dagger a_p) \right)}.
    
    :math:`U(u)` 通过使用论文 `Optica, 3, 1460 (2016) <https://opg.optica.org/optica/fulltext.cfm?uri=optica-3-12-1460&id=355743>`_\ 中给出的方案。
    

    :param q_machine: 量子虚拟机。
    :param wires: 作用的量子位。
    :param unitary_matrix: 指定基础变换的矩阵。
    :param check: 检测 `unitary_matrix` 是否为酉矩阵。

    Example::

        from pyvqnet.qnn.vqc import VQC_BasisRotation, QMachine, hadamard, isingzz
        from pyvqnet.tensor import QTensor
        import numpy as np
        V = np.array([[0.73678+0.27511j, -0.5095 +0.10704j, -0.06847+0.32515j],
                      [0.73678+0.27511j, -0.5095 +0.10704j, -0.06847+0.32515j],
                      [-0.21271+0.34938j, -0.38853+0.36497j,  0.61467-0.41317j]])

        eigen_vals, eigen_vecs = np.linalg.eigh(V)
        umat = eigen_vecs.T
        wires = range(len(umat))
        
        qm = QMachine(len(umat))

        for i in range(len(umat)):
            hadamard(q_machine=qm, wires=i)
        isingzz(q_machine=qm, params=QTensor([0.55]), wires=[0,2])
        VQC_BasisRotation(q_machine=qm, wires=wires,unitary_matrix=QTensor(umat,dtype=qm.state.dtype))
        
        print(qm.states)
        
        # [[[[ 0.3402686-0.0960063j  0.4140436-0.3069579j]
        #    [ 0.1206574+0.1982292j  0.5662895-0.0949503j]]
        # 
        #   [[-0.1715559-0.1614315j  0.1624039-0.0598041j]
        #    [ 0.0608986-0.1078906j -0.305845 +0.1773662j]]]]

VQC_QuantumPoolingCircuit
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.VQC_QuantumPoolingCircuit(ignored_wires, sinks_wires, params, q_machine)

    在 ``q_machine`` 上对数据进行降采样的量子电路。为了减少电路中的量子位数量,首先在系统中创建成对的量子位。在最初配对所有量子位之后,将广义2量子位酉元应用于每一对量子位上。并在应用这两个量子位酉元之后,在神经网络的其余部分忽略每对量子位中的一个量子位。

    :param sources_wires: 将被忽略的源量子位索引。
    :param sinks_wires: 将保留的目标量子位索引。
    :param params: 输入参数,形状应该为[ len(ignored_wires) + len(sinks_wires) // 2 * 3]。
    :param q_machine: 量子虚拟机设备。


    Examples:: 

        from pyvqnet.qnn.vqc import VQC_QuantumPoolingCircuit, QMachine, MeasureAll
        from pyvqnet import tensor
        p = tensor.full([6], 0.35)
        qm = QMachine(4)
        VQC_QuantumPoolingCircuit(q_machine=qm,
                                ignored_wires=[0, 1],
                                sinks_wires=[2, 3],
                                params=p)
        m = MeasureAll(obs={"Z1": 1})
        exp = m(q_machine=qm)
        print(exp)


vqc_qft_add_to_register
-------------------------------------

.. py:function:: pyvqnet.qnn.vqc.vqc_qft_add_to_register(q_machine, m, k)

    在 ``q_machine`` 进行如下操作:将无符号整数 `m` 编码到量子比特中,然后将 `k` 加到此量子比特上。

    .. math:: \text{Sum(k)}\vert m \rangle = \vert m + k \rangle.

    实现此幺正运算的过程如下:
    (1). 通过 将 QFT 应用于 :math:`\vert m \rangle` 状态,将状态从计算基础转换为傅里叶基础。
    (2). 使用 :math:`R_Z` 门将 :math:`j` 个量子比特旋转角度 :math:`\frac{2k\pi}{2^{j}}`,从而得到新相 :math:`\frac{2(m + k)\pi}{2^{j}}`。
    (3). 应用 QFT 逆返回计算基础并得到 :math:`m+k`。

    :param q_machine: 用于模拟的量子机。
    :param m: 嵌入寄存器的经典整数。
    :param k: 添加到寄存器的经典整数。

    :retrun: 返回目标和的二进制表示。

    .. note::

        请注意 ``q_machine`` 使用的比特数量需要足够使用X基态编码结果和的二进制值。

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine,Samples, vqc_qft_add_to_register
        dev = QMachine(4)
        vqc_qft_add_to_register(dev,3, 7)
        ma = Samples()
        y = ma(q_machine=dev)
        print(y)
        #[[1,0,1,0]]


vqc_qft_add_two_register
-------------------------------------

.. py:function:: vqc_qft_add_two_register(q_machine, m, k, wires_m, wires_k, wires_solution)

    在 ``q_machine`` 进行如下操作: 将两个量子比特wires_m, wires_k中编码的无符号整数m, k进行加法,结果存在wires_solution比特上。

    .. math:: \text{Sum}_2\vert m \rangle \vert k \rangle \vert 0 \rangle = \vert m \rangle \vert k \rangle \vert m+k \rangle

    在这种情况下,我们可以将第三个寄存器(最初位于 :math:`0`)理解为一个计数器,它将计算出 :math:`m` 和 :math:`k` 加起来的单位数。二进制分解将使这变得简单。如果我们有 :math:`\vert m \rangle = \vert \overline{q_0q_1q_2} \rangle`,则如果 :math:`q_2 = 1`,则我们必须将 :math:`1` 添加到计数器,否则不添加任何内容。一般来说,如果 :math:`i`-th 量子位处于 :math:`\vert 1 \rangle` 状态,则我们应该添加 :math:`2^{n-i-1}` 个单位,否则添加 0。

    :param q_machine: 用于模拟的量子机。
    :param m: 嵌入寄存器中的经典整数作为 lhs。
    :param k: 嵌入到寄存器中的经典整数作为 rhs。
    :param wires_m: 要编码 m 的量子比特的索引。
    :param wires_k: 要编码 k 的量子比特的索引。
    :param wires_solution: 要编码解决方案的量子比特的索引。

    :retrun: 返回目标和的二进制表示。


    .. note::

        ``wires_m`` 使用的比特数量需要足够使用X基态编码 `m` 的二进制值。
        ``wires_k`` 使用的比特数量需要足够使用X基态编码 `k` 的二进制值。
        ``wires_solution`` 使用的比特数量需要足够使用X基态编码结果的二进制值。

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine,Samples, vqc_qft_add_two_register
        wires_m = [0, 1, 2]           # qubits needed to encode m
        wires_k = [3, 4, 5]           # qubits needed to encode k
        wires_solution = [6, 7, 8, 9, 10]  # qubits needed to encode the solution

        wires_m = [0, 1, 2]             # qubits needed to encode m
        wires_k = [3, 4, 5]             # qubits needed to encode k
        wires_solution = [6, 7, 8, 9]   # qubits needed to encode the solution
        dev = QMachine(len(wires_m) + len(wires_k) + len(wires_solution))

        vqc_qft_add_two_register(dev,3, 7, wires_m, wires_k, wires_solution)

        ma = Samples(wires=wires_solution)
        y = ma(q_machine=dev)
        print(y)


vqc_qft_mul
-------------------------------------

.. py:function:: vqc_qft_mul(q_machine, m, k, wires_m, wires_k, wires_solution)

    在 ``q_machine`` 进行如下操作: 将两个量子比特wires_m, wires_k中编码的数值m, k, 进行乘法,结果存在wires_solution比特上。

    .. math:: \text{Mul}\vert m \rangle \vert k \rangle \vert 0 \rangle = \vert m \rangle \vert k \rangle \vert m\cdot k \rangle

    :param q_machine: 用于模拟的量子机。
    :param m: 嵌入寄存器中的经典整数,作为左侧。
    :param k: 嵌入寄存器中的经典整数,作为右侧。
    :param wires_m: 要编码 m 的量子比特索引。
    :param wires_k: 要编码 k 的量子比特索引。
    :param wires_solution: 要编码解决方案的量子比特索引。

    :retrun: 返回目标乘积的二进制表示。

    .. note::

        ``wires_m`` 使用的比特数量需要足够使用X基态编码 `m` 的二进制值。
        ``wires_k`` 使用的比特数量需要足够使用X基态编码 `k` 的二进制值。
        ``wires_solution`` 使用的比特数量需要足够使用X基态编码结果的二进制值。

    Example::

        import numpy as np
        from pyvqnet.qnn.vqc import QMachine,Samples, vqc_qft_mul
        wires_m = [0, 1, 2]           # qubits needed to encode m
        wires_k = [3, 4, 5]           # qubits needed to encode k
        wires_solution = [6, 7, 8, 9, 10]  # qubits needed to encode the solution
        
        dev = QMachine(len(wires_m) + len(wires_k) + len(wires_solution))

        vqc_qft_mul(dev,3, 7, wires_m, wires_k, wires_solution)


        ma = Samples(wires=wires_solution)
        y = ma(q_machine=dev)
        print(y)
        #[[1,0,1,0,1]]

VQC_FABLE
--------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_FABLE(wires)

    使用快速近似块编码方法构建基于 VQC 的 QCircuit。对于特定结构的矩阵 [`arXiv:2205.00081 <https://arxiv.org/abs/2205.00081>`_], FABLE 方法可以简化块编码电路而不降低准确性。

    :param wires: 运算符作用的 qlist 索引。

    :return: 返回一个基于VQC的FABLE类实例。

    Examples::

        from pyvqnet.qnn.vqc import VQC_FABLE
        from pyvqnet.qnn.vqc import QMachine
        from pyvqnet.dtype import float_dtype_to_complex_dtype
        import numpy as np
        from pyvqnet import QTensor
        
        A = QTensor(np.array([[0.1, 0.2 ], [0.3, 0.4 ]]) )
        qf = VQC_FABLE(list(range(3)))
        qm = QMachine(3,dtype=float_dtype_to_complex_dtype(A.dtype))
        qm.reset_states(1)
        z1 = qf(qm,A,0.001)
 
        """
        [[[[0.05     +0.j,0.15     +0.j],
        [0.05     +0.j,0.15     +0.j]],

        [[0.4974937+0.j,0.4769696+0.j],
        [0.4974937+0.j,0.4769696+0.j]]]]
        """


VQC_LCU
--------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_LCU(wires)

    使用线性组合单元 (LCU) 构建基于 VQC 的 QCircuit,`通过量子比特化进行哈密顿模拟 <https://arxiv.org/abs/1610.06546>`_。
    输入 dtype 可以是 kfloat32、kfloat64、kcomplex64、kcomplex128
    输入应为 Hermitian。

    :param wires: 运算符作用的 qlist 索引,可能需要辅助量子位。
    :param check_hermitian: 检查输入是否为 Hermitian,默认值:True。

    Examples::

        from pyvqnet.qnn.vqc import VQC_LCU
        from pyvqnet.qnn.vqc import QMachine
        from pyvqnet.dtype import float_dtype_to_complex_dtype,kfloat64

        from pyvqnet import QTensor

        A = QTensor([[0.25,0,0,0.75],[0,-0.25,0.75,0],[0,0.75,0.25,0],[0.75,0,0,-0.25]],device=1001,dtype=kfloat64)
        qf = VQC_LCU(list(range(3)))
        qm = QMachine(3,dtype=float_dtype_to_complex_dtype(A.dtype))
        qm.reset_states(2)
        z1 = qf(qm,A)
        print(z1)
        """
        [[[[ 0.25     +0.j, 0.       +0.j],
        [ 0.       +0.j, 0.75     +0.j]],

        [[-0.4330127+0.j, 0.       +0.j],
        [ 0.       +0.j, 0.4330127+0.j]]],


        [[[ 0.25     +0.j, 0.       +0.j],
        [ 0.       +0.j, 0.75     +0.j]],

        [[-0.4330127+0.j, 0.       +0.j],
        [ 0.       +0.j, 0.4330127+0.j]]]]
        <QTensor [2, 2, 2, 2] DEV_CPU kcomplex128>
        """


VQC_QSVT
--------------------

.. py:class:: pyvqnet.qnn.vqc.VQC_QSVT(A, angles, wires)

    实现
    `quantum singular value transformation <https://arxiv.org/abs/1806.01838>`__ (QSVT) 线路.


    :param A: 要编码的一般 :math:`(n \times m)` 矩阵。
    :param angles:  构建QSVT多项式的偏移系数。
    :param wires: A 作用于的量子比特索引。

    Example::

        from pyvqnet import DEV_GPU
        from pyvqnet.qnn.vqc import QMachine,VQC_QSVT
        from pyvqnet.dtype import float_dtype_to_complex_dtype,kfloat64
        import numpy as np
        from pyvqnet import QTensor

        A = QTensor([[0.1, 0.2], [0.3, 0.4]])
        angles = QTensor([0.1, 0.2, 0.3])
        qm = QMachine(4,dtype=float_dtype_to_complex_dtype(A.dtype))
        qm.reset_states(1)
        qf = VQC_QSVT(A,angles,wires=[2,1,3])
        z1 = qf(qm)
        print(z1)
        """
        [[[[[ 0.9645935+0.2352667j,-0.0216623+0.0512362j],
        [-0.0062613+0.0308878j,-0.0199871+0.0985996j]],

        [[ 0.       +0.j       , 0.       +0.j       ],
            [ 0.       +0.j       , 0.       +0.j       ]]],


        [[[ 0.       +0.j       , 0.       +0.j       ],
            [ 0.       +0.j       , 0.       +0.j       ]],

        [[ 0.       +0.j       , 0.       +0.j       ],
            [ 0.       +0.j       , 0.       +0.j       ]]]]]
        """

量子机器学习模型接口
=====================


Quanvolution
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qcnn.Quanvolution(params_shape, strides=(1, 1), kernel_initializer=quantum_uniform, machine_type_or_cloud_token: str = "cpu")


    基于《Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits》（https://arxiv.org/abs/1904.04767）实现的量子卷积，用变分量子电路替换经典的卷积滤波器，从而得到具有量子卷积滤波器的量子卷积神经网络。

    :param params_shape: 参数的形状，应为二维。
    :param strides: 切片窗口的步长，默认为 (1,1)。
    :param kernel_initializer: 参数的卷积核初始化器。
    :param machine_type_or_cloud_token: 机器类型字符串或 Qcloud 令牌，默认为“cpu”。

    Examples::

        from pyvqnet.qnn.qcnn import Quanvolution
        import pyvqnet.tensor as tensor
        qlayer = Quanvolution([4,2],(3,3))

        x = tensor.arange(1,25*25*3+1).reshape([3,1,25,25])

        y = qlayer(x)

        print(y.shape)

        y.backward()

        print(qlayer.m_para)
        print(qlayer.m_para.grad)
        #[3, 4, 8, 8]

        #[4.0270405,4.3587413,2.4935627,2.8155506,0.3314773,0.8889271,3.7357519, 0.9196261]
        #<Parameter [8] DEV_CPU kfloat32>

        #[ -0.2364242, -0.6942478, -8.445061 , -0.0558891, -0.       ,-49.498577 ,40.339344 , 40.339344 ]
        #<QTensor [8] DEV_CPU kfloat32>


QDRL
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qdrl_vqc.QDRL(nq)


    基于《Data re-uploading for a universal quantum classifier》（https://arxiv.org/abs/1907.02085）实现的量子数据重上传(Quantum Data Re-upLoading,QDRL)算法是一种将量子电路与经典神经网络相结合的量子数据重上传模型。

    :param nq: 量子电路中所使用的量子比特（qubits的数量）。这决定了模型将要处理的量子系统的规模。


    Example::

        import numpy as np
        from pyvqnet.dtype import kcomplex64
        from pyvqnet.qnn.qdrl_vqc import QDRL
        import pyvqnet.tensor as tensor

        # Set the number of quantum bits (qubits)
        nq = 1

        # Initialize the model
        model = QDRL(nq)

        # Create an example input (assume the input is a (batch_size, 3) shaped data)
        # Suppose we have a batch_size of 4 and each input has 3 features
        x_input = tensor.QTensor(np.random.randn(4, 3), dtype=kcomplex64)

        # Pass the input through the model
        output = model(x_input)

        output.backward()

        # Output the result
        print("Model output:")
        print(output)


QGRU
------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qgru.QGRU(para_num, num_of_qubits,input_size,hidden_size,batch_first=True)

    基于量子量子变分线路实现的GRU（门控循环单元），通过利用量子电路来进行状态更新和记忆保持。

    :param para_num: 量子电路中的参数数量。
    :param num_of_qubits: 量子比特的数量。
    :param input_size: 输入数据的特征维度。
    :param hidden_size: 隐藏单元的维度。
    :param batch_first:  输入的第一维是否为批次数量。

    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.qgru import QGRU
        from pyvqnet.dtype import kfloat32
        # Example usage
        if __name__ == "__main__":
            # Set parameters
            para_num = 8
            num_of_qubits = 8
            input_size = 4
            hidden_size = 4
            batch_size = 1
            seq_length = 1
            # Create QGRU model
            qgru = QGRU(para_num, num_of_qubits, input_size, hidden_size, batch_first=True)

            # Create input data
            x = tensor.QTensor(np.random.randn(batch_size, seq_length, input_size), dtype=kfloat32)

            # Call the model
            output, h_t = qgru(x)
            output.backward()

            print("Output shape:", output.shape)  # Output shape
            print("h_t shape:", h_t.shape)  # Final hidden state shape

QLinear
--------------------------------------------------------------
.. py:class:: pyvqnet.qnn.qlinear.QLinear(nput_channels, output_channels, machine: str = "")

    QLinear 实现了一种量子全连接算法。首先将数据编码到量子态,然后通过量子线路进行演化操作和测量得到最终的全连接结果。

    :param input_channels: 输入通道数。
    :param output_channels: 输出通道数。
    :param machine:  使用的虚拟机,默认使用CPU模拟。

    Example::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.qlinear import QLinear
        params = [[0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
        [1.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
        [1.37454012, 1.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
        [1.37454012, 1.95071431, 1.73199394, 1.59865848, 0.15601864, 0.15599452]]
        m = QLinear(6, 2)
        input = QTensor(params, requires_grad=True)
        output = m(input)
        output.backward()
        print(output)

        #[
        #[0.0568473, 0.1264389],
        #[0.1524036, 0.1264389],
        #[0.1524036, 0.1442845],
        #[0.1524036, 0.1442845]
        #]



QLSTM
-------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qlstm.QLSTM(para_num, num_of_qubits,input_size, hidden_size,batch_first=True)


    QLSTM (Quantum Long Short-Term Memory) 是一种结合了量子计算和经典LSTM的混合模型，旨在利用量子计算的并行性和经典LSTM的记忆能力来处理序列数据。

    :param para_num: 量子电路中的参数数量。
    :param num_of_qubits: 量子比特的数量。
    :param input_size: 输入数据的特征维度。
    :param hidden_size: 隐藏单元的维度。
    :param batch_first: 输入的第一维是否为批次数量。


    Examnple::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.qlstm import QLSTM
        from pyvqnet.dtype import *
        if __name__ == "__main__":
            para_num = 4
            num_of_qubits = 4
            input_size = 4
            hidden_size = 20
            batch_size = 3
            seq_length = 5
            qlstm = QLSTM(para_num, num_of_qubits, input_size, hidden_size, batch_first=True)
            x = tensor.QTensor(np.random.randn(batch_size, seq_length, input_size), dtype=kfloat32)

            output, (h_t, c_t) = qlstm(x)

            print("Output shape:", output.shape)
            print("h_t shape:", h_t.shape)
            print("c_t shape:", c_t.shape)

QMLPModel
--------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qmlp.qmlp.QMLPModel(input_channels: int,output_channels: int,num_qubits: int, kernel: _size_type,stride: _size_type,padding: _padding_type = "valid",weight_initializer: Union[Callable, None] = None,bias_initializer: Union[Callable, None] = None,use_bias: bool = True,dtype: Union[int, None] = None)


    QMLPModel是基于《QMLP: An Error-Tolerant Nonlinear Quantum MLP Architecture using Parameterized Two-Qubit Gates》（https://arxiv.org/abs/2206.01345）实现的量子启发式神经网络，QMLPModel将量子电路与经典神经网络操作（如池化和全连接层）相结合。它旨在处理量子数据，并通过量子操作和经典层提取相关特征。

    :param input_channels: 输入特征的数量。
    :param output_channels: 输出特征的数量。
    :param num_qubits: 量子比特的数量。
    :param kernel: 平均池化窗口的大小。
    :param stride: 下采样的步长因子。
    :param padding: 填充方式，可选“valid”或“same”。
    :param weight_initializer: 权重初始化器，默认为正态分布。
    :param bias_initializer: 偏置初始化器，默认为零初始化。
    :param use_bias: 是否使用偏置，默认为True。
    :param dtype: 默认为None，使用默认数据类型。


    Example::

        import numpy as np
        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.qmlp.qmlp import QMLPModel
        from pyvqnet.dtype import *

        input_channels = 16
        output_channels = 10
        num_qubits = 4
        kernel = (2, 2)
        stride = (2, 2)
        padding = "valid"
        batch_size = 8

        model = QMLPModel(input_channels=num_qubits,
                          output_channels=output_channels,
                          num_qubits=num_qubits,
                          kernel=kernel,
                          stride=stride,
                          padding=padding)

        x = tensor.QTensor(np.random.randn(batch_size, input_channels, 32, 32),dtype=kfloat32)

        output = model(x)

        print("Output shape:", output.shape)


QRLModel
-------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qrl.QRLModel(num_qubits, n_layers)

    使用变分量子电路的量子深度强化学习模型。

    :param num_qubits: 量子电路中所使用的量子比特的数量。
    :param n_layers: 变分量子电路中的层数。


    Example::

        from pyvqnet.tensor import tensor, QTensor
        from pyvqnet.qnn.qrl import QRLModel

        num_qubits = 4
        model = QRLModel(num_qubits=num_qubits, n_layers=2)

        batch_size = 3
        x = QTensor([[1.1, 0.3, 1.2, 0.6], [0.2, 1.1, 0, 1.1], [1.3, 1.3, 0.3, 0.3]])
        output = model(x)
        output.backward()

        print("Model output:", output)


QRNN
--------------------------------------------------------------

.. py:class:: pyvqnet.qnn.qrnn.QRNN(para_num, num_of_qubits=4,input_size=100,hidden_size=100,batch_first=True)


    QRNN（Quantum Recurrent Neural Network）是一种量子循环神经网络，旨在处理序列数据并捕捉序列中的长期依赖关系。


    :param para_num: 量子电路中的参数数量。
    :param num_of_qubits: 量子比特的数量。
    :param input_size: 输入数据的特征维度。
    :param hidden_size: 隐藏单元的维度。
    :param batch_first: 输入的第一维是否为批次数量,默认为True。

    Example::

        from pyvqnet.dtype import kfloat32
        from pyvqnet.qnn.qrnn import QRNN
        from pyvqnet.tensor import tensor, QTensor
        import numpy as np

        if __name__ == "__main__":
            para_num = 8
            num_of_qubits = 8
            input_size = 4
            hidden_size = 4
            batch_size = 1
            seq_length = 1
            qrnn = QRNN(para_num, num_of_qubits, input_size, hidden_size, batch_first=True)

            x = tensor.QTensor(np.random.randn(batch_size, seq_length, input_size), dtype=kfloat32)

            output, h_t = qrnn(x)

            print("Output shape:", output.shape)
            print("h_t shape:", h_t.shape)


TTOLayer
----------------------------------------------------------------

.. py:class:: pyvqnet.qnn.ttolayer.TTOLayer(inp_modes,out_modes,mat_ranks,biases_initializer=tensor.zeros)


    基于《Compressing deep neural networks by matrix product operators》（https://arxiv.org/abs/1904.06194）实现的TTOLayer对输入张量进行分解，从而实现对高维数据的高效表示。该层允许在秩约束条件下学习张量分解，相比于传统的全连接层，能够降低计算复杂度和内存使用。


    :param inp_modes: 输入张量的维度。
    :param out_modes: 输出张量的维度。
    :param mat_ranks: 张量分解中张量核（分解秩）的秩。
    :param biases_initializer: 偏置的初始化函数。

    Example::

        from pyvqnet.tensor import tensor
        import numpy as np
        from pyvqnet.dtype import kfloat32
        inp_modes = [4, 5]
        out_modes = [4, 5]
        mat_ranks = [1, 3, 1]
        tto_layer = TTOLayer(inp_modes, out_modes, mat_ranks)

        batch_size = 2
        len = 4
        embed_size = 5
        inp = tensor.QTensor(np.random.randn(batch_size, len, embed_size), dtype=kfloat32)

        output = tto_layer(inp)

        print("Input shape:", inp.shape)
        print("Output shape:", output.shape)




 
其他函数
=====================



QuantumLayerAdjoint
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.QuantumLayerAdjoint(general_module: pyvqnet.nn.Module,q_machine, name="")


    使用伴随矩阵方式进行梯度计算的可自动微分的QuantumLayer层,参考  `Efficient calculation of gradients in classical simulations of variational quantum algorithms <https://arxiv.org/abs/2009.02823>`_ 。

    :param general_module: 一个仅使用 ``pyvqnet.qnn.vqc`` 下量子线路接口搭建的 `pyvqnet.nn.Module` 实例。
    :param use_qpanda: 是否使用qpanda线路进行前传,默认:False。
    :param name: 该层名字,默认为""。
    :return: 返回一个 QuantumLayerAdjoint 类实例。

    .. note::

        general_module 的 QMachine 应设置 grad_method = "adjoint".

        当前支持由如下含参逻辑门 `RX`, `RY`, `RZ`, `PhaseShift`, `RXX`, `RYY`, `RZZ`, `RZX`, `U1`, `U2`, `U3` 以及其他不含参逻辑门构成的变分线路。

    Example::

        import pyvqnet
        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import QuantumLayerAdjoint, QMachine, RX, RY, CNOT, PauliX, qmatrix, PauliZ, T, MeasureAll, RZ, VQC_RotCircuit, VQC_HardwareEfficientAnsatz
        from pyvqnet.utils import set_random_seed

        set_random_seed(42)
        class QModel(pyvqnet.nn.Module):
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
                self.measure = MeasureAll(obs = {
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

        input_x = tensor.broadcast_to(input_x, [4, 3])

        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=6,
                            dtype=pyvqnet.kcomplex64,
                            grad_mode="adjoint")

        adjoint_model = QuantumLayerAdjoint(qunatum_model,q_machine=qunatum_model.qm)

        batch_y = adjoint_model(input_x)
        batch_y.backward()
        print(batch_y)
        # [[-0.0778451],
        #  [-0.0778451],
        #  [-0.0778451],
        #  [-0.0778451]]


QuantumLayerES
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.QuantumLayerES(general_module: nn.Module, q_machine: pyvqnet.qnn.vqc.QMachine, name="", sigma = np.pi / 24)


    根据进化策略进行梯度计算的可自动微分的QuantumLayer层,参考  `Learning to learn with an evolutionary strategy Learning to learn with an evolutionary strategy <https://arxiv.org/abs/2310.17402>`_ 。

    :param general_module: 一个仅使用 ``pyvqnet.qnn.vqc`` 下量子线路接口搭建的 `pyvqnet.nn.QModule` 实例。
    :param q_machine: 来自general_module中定义的QMachine。
    :param name: 该层名字,默认为""。
    :param sigma: 多元正态分布的采样方差.

    .. note::

        general_module 的 QMachine 应设置 grad_method = "ES".

        当前支持由如下含参逻辑门 `RX`, `RY`, `RZ`, `PhaseShift`, `RXX`, `RYY`, `RZZ`, `RZX`, `U1`, `U2`, `U3` 以及其他不含参逻辑门构成的变分线路。

    Example::

        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import QuantumLayerES, QMachine, RX, RY, CNOT, T, MeasureAll, RZ, VQC_HardwareEfficientAnsatz
        import pyvqnet
        from pyvqnet.utils import set_random_seed

        set_random_seed(42)

        class QModel(pyvqnet.nn.Module):
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
                self.measure = MeasureAll(obs = {
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

        input_x = tensor.broadcast_to(input_x, [4, 3])

        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=6,
                            dtype=pyvqnet.kcomplex64,
                            grad_mode="ES")

        ES_model = QuantumLayerES(qunatum_model, qunatum_model.qm)

        batch_y = ES_model(input_x)
        batch_y.backward()
        print(batch_y)
        # [[-0.0778451],
        #  [-0.0778451],
        #  [-0.0778451],
        #  [-0.0778451]]



DataParallelVQCAdjointLayer
---------------------------------------------------------------

.. py:class:: pyvqnet.distributed.DataParallelVQCAdjointLayer(Comm_OP, vqc_module, name="")


    使用数据并行对数据批次大小创建 vqc 使用伴随梯度计算。其中 ``vqc_module`` 必须为 ``QuantumLayerAdjoint`` 类型的VQC模块.
    如果我们使用 N 个节点来运行此模块,
    在每个节点中, `batch_size/N` 数据向前运行变分量子线路 计算梯度。

    :param Comm_OP: 设置分布式环境的通信控制器。
    :param vqc_module: 带有 forward() 的 QuantumLayerAdjoint类型的VQC模块,确保qmachine 已正确设置。
    :param name: 模块的名称。默认值为空字符串。
    :return: 可以计算量子电路的模块。


    Example::

        #mpirun -n 2 python test.py

        import sys
        sys.path.insert(0,"../../")
        from pyvqnet.distributed import CommController,DataParallelVQCAdjointLayer,\
        get_local_rank

        from pyvqnet.qnn import *
        from pyvqnet.qnn.vqc import *
        import pyvqnet
        from pyvqnet.nn import Module, Linear
        from pyvqnet.device import DEV_GPU_0

        bsize = 100


        class QModel(Module):
            def __init__(self, num_wires, dtype, grad_mode="adjoint"):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype, grad_mode=grad_mode)
                self.rx_layer = RX(has_params=True, trainable=False, wires=0)
                self.ry_layer = RY(has_params=True, trainable=False, wires=1)
                self.rz_layer = RZ(has_params=True, trainable=False, wires=1)
                self.u1 = U1(has_params=True, trainable=True, wires=[2])
                self.u2 = U2(has_params=True, trainable=True, wires=[3])
                self.u3 = U3(has_params=True, trainable=True, wires=[1])
                self.i = I(wires=[3])
                self.s = S(wires=[3])
                self.x1 = X1(wires=[3])
                self.y1 = Y1(wires=[3])
                self.z1 = Z1(wires=[3])
                self.x = PauliX(wires=[3])
                self.y = PauliY(wires=[3])
                self.z = PauliZ(wires=[3])
                self.swap = SWAP(wires=[2, 3])
                self.cz = CZ(wires=[2, 3])
                self.cr = CR(has_params=True, trainable=True, wires=[2, 3])
                self.rxx = RXX(has_params=True, trainable=True, wires=[2, 3])
                self.rzz = RYY(has_params=True, trainable=True, wires=[2, 3])
                self.ryy = RZZ(has_params=True, trainable=True, wires=[2, 3])
                self.rzx = RZX(has_params=True, trainable=False, wires=[2, 3])
                self.toffoli = Toffoli(wires=[2, 3, 4], use_dagger=True)

                self.h = Hadamard(wires=[1])

                self.iSWAP = iSWAP(wires=[0, 2])
                self.tlayer = T(wires=1)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs={'Z0': 2})

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.i(q_machine=self.qm)
                self.s(q_machine=self.qm)
                self.swap(q_machine=self.qm)
                self.cz(q_machine=self.qm)
                self.x(q_machine=self.qm)
                self.x1(q_machine=self.qm)
                self.y(q_machine=self.qm)
                self.y1(q_machine=self.qm)
                self.z(q_machine=self.qm)
                self.z1(q_machine=self.qm)
                self.ryy(q_machine=self.qm)
                self.rxx(q_machine=self.qm)
                self.rzz(q_machine=self.qm)
                self.rzx(q_machine=self.qm, params=x[:, [1]])

                self.u1(q_machine=self.qm)
                self.u2(q_machine=self.qm)
                self.u3(q_machine=self.qm)
                self.rx_layer(params=x[:, [0]], q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                self.h(q_machine=self.qm)
                self.iSWAP(q_machine=self.qm)
                self.ry_layer(params=x[:, [1]], q_machine=self.qm)
                self.tlayer(q_machine=self.qm)
                self.rz_layer(params=x[:, [2]], q_machine=self.qm)
                self.toffoli(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)

                return rlt


        pyvqnet.utils.set_random_seed(42)

        Comm_OP = CommController("mpi")

        input_x = tensor.QTensor([[0.1, 0.2, 0.3]])
        input_x = tensor.broadcast_to(input_x, [bsize, 3])
        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=6, dtype=pyvqnet.kcomplex64)

        l = DataParallelVQCAdjointLayer(
            Comm_OP,
            qunatum_model,
        )

        y = l(input_x)

        y.backward()    



DataParallelVQCLayer
---------------------------------------------------------------

.. py:class:: pyvqnet.distributed.DataParallelVQCLayer(Comm_OP, vqc_module, name="")


    使用数据并行对数据批次大小创建 vqc 使用自动微分计算。 
    如果我们使用 N 个节点来运行此模块,
    在每个节点中, `batch_size/N` 数据向前运行变分量子线路 计算梯度。

    :param Comm_OP: 设置分布式环境的通信控制器。
    :param vqc_module: 带有 forward() 的 VQC模块,确保qmachine 已正确设置。
    :param name: 模块的名称。默认值为空字符串。
    :return: 可以计算量子电路的模块。


    Example::

        #mpirun -n 2 python xxx.py

        import pyvqnet.backends

        from pyvqnet.qnn.vqc import QMachine, cnot, rx, rz, ry, MeasureAll
        from pyvqnet.tensor import tensor

        from pyvqnet.distributed import CommController, DataParallelVQCLayer

        from pyvqnet.qnn import *
        from pyvqnet.qnn.vqc import *
        import pyvqnet
        from pyvqnet.nn import Module, Linear
        from pyvqnet.device import DEV_GPU_0


        class QModel(Module):

            def __init__(self, num_wires, num_layer, dtype, grad_mode=""):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype, grad_mode=grad_mode)

                self.measure = MeasureAll(obs=PauliX)
                self.n = num_wires
                self.l = num_layer

            def forward(self, param, *args, **kwargs):
                n = self.n
                l = self.l
                qm = self.qm
                qm.reset_states(param.shape[0])
                j = 0

                for j in range(l):
                    cnot(qm, wires=[j, (j + 1) % l])
                    for i in range(n):
                        rx(qm, i, param[:, 3 * n * j + i])
                    for i in range(n):
                        rz(qm, i, param[:, 3 * n * j + i + n], i)
                    for i in range(n):
                        rx(qm, i, param[:, 3 * n * j + i + 2 * n], i)

                y = self.measure(qm)
                return y


        n = 4
        b = 4
        l = 2

        input = tensor.ones([b, 3 * n * l])

        Comm = CommController("mpi")
        
        input.requires_grad = True
        qunatum_model = QModel(num_wires=n, num_layer=l, dtype=pyvqnet.kcomplex64)
        
        layer = qunatum_model

        layer = DataParallelVQCLayer(
            Comm,
            qunatum_model,
        )
        y = layer(input)
        y.backward()


vqc_to_originir_list
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.vqc_to_originir_list(vqc_model: pyvqnet.nn.Module)

    将 VQNet vqc 模块转换为 `originIR <https://qpanda-tutorial.readthedocs.io/zh/latest/QProgToOriginIR.html#id2>`_ 。

    vqc_model 应在此函数之前运行前向函数以获取输入数据。
    如果输入数据是批量数据。 对于每个输入,它将返回多个 IR 字符串。

    :param vqc_model: VQNet vqc 模块,应该先向前运行。

    :return: originIR 字符串或 originIR 字符串列表。

    Example::

        import pyvqnet
        import pyvqnet.tensor as tensor
        from pyvqnet.qnn.vqc import *
        from pyvqnet.nn import Module
        from pyvqnet.utils import set_random_seed
        set_random_seed(42)
        class QModel(Module):
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
                self.y1 = Y1(wires=[3])
                self.z1 = Z1(wires=[3])
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
                #self.rz_layer2 = RZ(has_params=True, trainable=True, wires=1)
                self.h =Hadamard(wires=[1])
                self.rot = VQC_HardwareEfficientAnsatz(6, ["rx", "RY", "rz"],
                                                    entangle_gate="cnot",
                                                    entangle_rules="linear",
                                                    depth=5)
                
                self.iSWAP = iSWAP(True,True,wires=[0,2])
                self.tlayer = T(wires=1)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs = {
                    'wires': [1],
                    'observables': ['x'],
                    'coefficient': [1]
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.i(q_machine=self.qm)
                self.s(q_machine=self.qm)
                self.swap(q_machine=self.qm)
                self.cz(q_machine=self.qm)
                self.x(q_machine=self.qm)
                self.x1(q_machine=self.qm)
                self.y(q_machine=self.qm)
                self.y1(q_machine=self.qm)
                self.z(q_machine=self.qm)
                self.z1(q_machine=self.qm)
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
                self.iSWAP(q_machine=self.qm)
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

        batch_y = qunatum_model(input_x)
        batch_y.backward()
        ll = vqc_to_originir_list(qunatum_model)
        from pyqpanda import CPUQVM,convert_originir_str_to_qprog,convert_qprog_to_originir
        for l in ll :
            print(l)

            machine = CPUQVM()
            machine.init_qvm()
            prog, qv, cv = convert_originir_str_to_qprog(l, machine)
            print(machine.prob_run_dict(prog,qv))

        # QINIT 6
        # CREG 6
        # I q[3]
        # S q[3]
        # SWAP q[2],q[3]
        # CZ q[2],q[3]
        # X q[3]
        # X1 q[3]
        # Y q[3]
        # Y1 q[3]
        # Z q[3]
        # Z1 q[3]
        # RZZ q[2],q[3],(4.484121322631836)
        # RXX q[2],q[3],(5.302337169647217)
        # RYY q[2],q[3],(3.470323085784912)
        # RZX q[2],q[3],(0.20000000298023224)
        # CR q[2],q[3],(5.467088222503662)
        # U1 q[2],(6.254805088043213)
        # U2 q[3],(1.261604905128479,0.9901542067527771)
        # U3 q[1],(5.290454387664795,6.182775020599365,1.1797741651535034)
        # RX q[0],(0.10000000149011612)
        # CNOT q[0],q[1]
        # H q[1]
        # ISWAPTHETA q[0],q[2],(0.6857681274414062)
        # RY q[1],(0.20000000298023224)
        # T q[1]
        # RZ q[1],(0.30000001192092896)
        # DAGGER
        # TOFFOLI q[2],q[3],q[4]
        # ENDDAGGER

        # {'000000': 0.006448949346548678, '000001': 0.004089870964118778, '000010': 0.1660891289303212, '000011': 0.08520414851665635, '000100': 0.0048503036661063, '000101': 8.679196482917438e-05, '000110': 0.14379026566368325, '000111': 0.0005079553597106437, '001000': 0.0023774056959510325, '001001': 0.008241263544544148, '001010': 0.06122877075562884, '001011': 0.1984226195587807, '001100': 0.0, '001101': 0.0, '001110': 0.0, '001111': 0.0, '010000': 0.0, '010001': 0.0, '010010': 0.0, '010011': 0.0, '010100': 0.0, '010101': 0.0, '010110': 0.0, '010111': 0.0, '011000': 0.0, '011001': 0.0, '011010': 0.0, '011011': 0.0, '011100': 0.011362100696548312, '011101': 0.00019143557058348747, '011110': 0.3059886012103368, '011111': 0.0011203885556518832, '100000': 0.0, '100001': 0.0, '100010': 0.0, '100011': 0.0, '100100': 0.0, '100101': 0.0, '100110': 0.0, '100111': 0.0, '101000': 0.0, '101001': 0.0, '101010': 0.0, '101011': 0.0, '101100': 0.0, '101101': 0.0, '101110': 0.0, '101111': 0.0, '110000': 0.0, '110001': 0.0, '110010': 0.0, '110011': 0.0, '110100': 0.0, '110101': 0.0, '110110': 0.0, '110111': 0.0, '111000': 0.0, '111001': 0.0, '111010': 0.0, '111011': 0.0, '111100': 0.0, '111101': 0.0, '111110': 0.0, '111111': 0.0}
        # QINIT 6
        # CREG 6
        # I q[3]
        # S q[3]
        # SWAP q[2],q[3]
        # CZ q[2],q[3]
        # X q[3]
        # X1 q[3]
        # Y q[3]
        # Y1 q[3]
        # Z q[3]
        # Z1 q[3]
        # RZZ q[2],q[3],(4.484121322631836)
        # RXX q[2],q[3],(5.302337169647217)
        # RYY q[2],q[3],(3.470323085784912)
        # RZX q[2],q[3],(0.20000000298023224)
        # CR q[2],q[3],(5.467088222503662)
        # U1 q[2],(6.254805088043213)
        # U2 q[3],(1.261604905128479,0.9901542067527771)
        # U3 q[1],(5.290454387664795,6.182775020599365,1.1797741651535034)
        # RX q[0],(0.10000000149011612)
        # CNOT q[0],q[1]
        # H q[1]
        # ISWAPTHETA q[0],q[2],(0.6857681274414062)
        # RY q[1],(0.20000000298023224)
        # T q[1]
        # RZ q[1],(0.30000001192092896)
        # DAGGER
        # TOFFOLI q[2],q[3],q[4]
        # ENDDAGGER

        # {'000000': 0.006448949346548678, '000001': 0.004089870964118778, '000010': 0.1660891289303212, '000011': 0.08520414851665635, '000100': 0.0048503036661063, '000101': 8.679196482917438e-05, '000110': 0.14379026566368325, '000111': 0.0005079553597106437, '001000': 0.0023774056959510325, '001001': 0.008241263544544148, '001010': 0.06122877075562884, '001011': 0.1984226195587807, '001100': 0.0, '001101': 0.0, '001110': 0.0, '001111': 0.0, '010000': 0.0, '010001': 0.0, '010010': 0.0, '010011': 0.0, '010100': 0.0, '010101': 0.0, '010110': 0.0, '010111': 0.0, '011000': 0.0, '011001': 0.0, '011010': 0.0, '011011': 0.0, '011100': 0.011362100696548312, '011101': 0.00019143557058348747, '011110': 0.3059886012103368, '011111': 0.0011203885556518832, '100000': 0.0, '100001': 0.0, '100010': 0.0, '100011': 0.0, '100100': 0.0, '100101': 0.0, '100110': 0.0, '100111': 0.0, '101000': 0.0, '101001': 0.0, '101010': 0.0, '101011': 0.0, '101100': 0.0, '101101': 0.0, '101110': 0.0, '101111': 0.0, '110000': 0.0, '110001': 0.0, '110010': 0.0, '110011': 0.0, '110100': 0.0, '110101': 0.0, '110110': 0.0, '110111': 0.0, '111000': 0.0, '111001': 0.0, '111010': 0.0, '111011': 0.0, '111100': 0.0, '111101': 0.0, '111110': 0.0, '111111': 0.0}

originir_to_vqc
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.originir_to_vqc(originir, tmp="code_tmp.py", verbose=False)

    将 originIR 解析为 vqc 模型代码。
    代码创建一个没有 `Measure` 的变分量子线路 `pyvqnet.nn.Module` ,返回是量子态的态矢形式,如 [b,2,...,2]。
    该函数将在“./origin_ir_gen_code/” + tmp +“.py”中生成一个定义对应VQNet模型的代码文件。
    
    :param originir: 原始IR。
    :param tmp: 代码文件名,默认 ``code_tmp.py``。
    :param verbose: 如果显示生成代码,默认 = False
    :return:
        生成可运行代码。

    Example::

        from pyvqnet.qnn.vqc import originir_to_vqc
        ss = "QINIT 3\nCREG 3\nH q[1]"
        
        Z = originir_to_vqc(ss,verbose=True)

        exec(Z)
        m =Exported_Model()
        print(m(2))

        # from pyvqnet.nn import Module
        # from pyvqnet.tensor import QTensor
        # from pyvqnet.qnn.vqc import *
        # class Exported_Model(Module):
        #         def __init__(self, name=""):
        #                 super().__init__(name)

        #                 self.q_machine = QMachine(num_wires=3)
        #                 self.H_0 = Hadamard(wires=1, use_dagger = False)

        #         def forward(self, x, *args, **kwargs):
        #                 x = self.H_0(q_machine=self.q_machine)
        #                 return self.q_machine.states

        # [[[[0.7071068+0.j 0.       +0.j]
        #    [0.7071068+0.j 0.       +0.j]]

        #   [[0.       +0.j 0.       +0.j]
        #    [0.       +0.j 0.       +0.j]]]]


model_summary
---------------------------------------------------------------

.. py:function:: pyvqnet.model_summary(vqc_module)

    打印在 vqc_module 中注册的经典层和量子门运算符的信息。
    
    :param vqc_module: vqc 模块
    :return:
         摘要字符串


    Example::

        from pyvqnet.qnn.vqc import QMachine, RX, RY, CNOT, PauliX, qmatrix, PauliZ,MeasureAll
        from pyvqnet.tensor import QTensor, tensor,kcomplex64
        import pyvqnet
        from pyvqnet.nn import LSTM,Linear
        from pyvqnet import model_summary
        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)
                self.rx_layer1 = RX(has_params=True,
                                    trainable=True,
                                    wires=1,
                                    init_params=tensor.QTensor([0.5]))
                self.ry_layer2 = RY(has_params=True,
                                    trainable=True,
                                    wires=0,
                                    init_params=tensor.QTensor([-0.5]))
                self.xlayer = PauliX(wires=0)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs = PauliZ)
                self.linear = Linear(24,2)
                self.lstm =LSTM(23,5)
            def forward(self, x, *args, **kwargs):
                return super().forward(x, *args, **kwargs)
        Z = QModel(4,kcomplex64)

        print(model_summary(Z))
        # ###################QModel Summary#######################

        # classic layers: {'Linear': 1, 'LSTM': 1}
        # total classic parameters: 650

        # =========================================
        # qubits num: 4
        # gates: {'RX': 1, 'RY': 1, 'PauliX': 1, 'CNOT': 1}
        # total quantum gates: 4
        # total quantum parameter gates: 2
        # total quantum parameters: 2
        # #########################################################


QNG
---------------------------------------------------------------

.. py:class:: pyvqnet.qnn.vqc.qng.QNG(qmodel, stepsize=0.01)

    `量子自然梯度法(Quantum Nature Gradient) <https://arxiv.org/abs/1909.02108>`_ 借鉴经典自然梯度法的概念 `Amari (1998) <https://www.mitpressjournals.org/doi/abs/10.1162/089976698300017746>`__ ,
    我们改为将优化问题视为给定输入的可能输出值的概率分布(即,最大似然估计),则更好的方法是在分布
    空间中执行梯度下降,它相对于参数化是无量纲和不变的. 因此,无论参数化如何,每个优化步骤总是会为每个参数选择最佳步长。
    在量子机器学习任务中,量子态空间拥有一个独特的不变度量张量,称为 Fubini-Study 度量张量 :math:`g_{ij}`。
    该张量将量子线路参数空间中的最速下降转换为分布空间中的最速下降。
    量子自然梯度的公式如下:

    .. math:: \theta_{t+1} = \theta_t - \eta g^{+}(\theta_t)\nabla \mathcal{L}(\theta),

    其中 :math:`g^{+}` 是伪逆。

    ``wrapper_calculate_qng`` 是需要加到待计算量子自然梯度的模型的forward函数的装饰器。仅对模型注册的 `Parameter` 类型的参数优化。

    :param qmodel: 量子变分线路模型,需要使用 `wrapper_calculate_qng` 作为forward函数的装饰器。
    :param stepsize: 梯度下降法的步长,默认0.01。

    .. note::

        仅在非批处理数据上进行了测试。
        仅支持纯变分量子电路。
        step() 将更新输入和参数的梯度。
        step() 仅会更新模型参数的数值。

    Example::

        from pyvqnet.qnn.vqc import QMachine, RX, RY, RZ, CNOT, rz, PauliX, qmatrix, PauliZ, Probability, rx, ry, MeasureAll, U2
        from pyvqnet.tensor import QTensor, tensor
        import pyvqnet
        import numpy as np
        from pyvqnet.qnn.vqc import wrapper_calculate_qng

        class QModel(pyvqnet.nn.Module):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)
                self.rz_layer1 = RZ(has_params=True, trainable=False, wires=0)
                self.rz_layer2 = RZ(has_params=True, trainable=False, wires=1)
                self.u2_layer1 = U2(has_params=True, trainable=False, wires=0)
                self.l_train1 = RY(has_params=True, trainable=True, wires=1)
                self.l_train1.params.init_from_tensor(
                    QTensor([333], dtype=pyvqnet.kfloat32))
                self.l_train2 = RX(has_params=True, trainable=True, wires=2)
                self.l_train2.params.init_from_tensor(
                    QTensor([4444], dtype=pyvqnet.kfloat32))
                self.xlayer = PauliX(wires=0)
                self.cnot01 = CNOT(wires=[0, 1])
                self.cnot12 = CNOT(wires=[1, 2])
                self.measure = MeasureAll(obs={'Y0': 1})

            @wrapper_calculate_qng
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                ry(q_machine=self.qm, wires=0, params=np.pi / 4)
                ry(q_machine=self.qm, wires=1, params=np.pi / 3)
                ry(q_machine=self.qm, wires=2, params=np.pi / 7)
                self.rz_layer1(q_machine=self.qm, params=x[:, [0]])
                self.rz_layer2(q_machine=self.qm, params=x[:, [1]])

                self.u2_layer1(q_machine=self.qm, params=x[:, [3, 4]])  #

                self.cnot01(q_machine=self.qm)
                self.cnot12(q_machine=self.qm)
                ry(q_machine=self.qm, wires=0, params=np.pi / 7)

                self.l_train1(q_machine=self.qm)
                self.l_train2(q_machine=self.qm)
                #rx(q_machine=self.qm, wires=2, params=x[:, [3]])
                rz(q_machine=self.qm, wires=1, params=x[:, [2]])
                ry(q_machine=self.qm, wires=0, params=np.pi / 7)
                rz(q_machine=self.qm, wires=1, params=x[:, [2]])

                self.cnot01(q_machine=self.qm)
                self.cnot12(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)
                return rlt


        qmodel = QModel(3, pyvqnet.kcomplex64)

        x = QTensor([[1111.0, 2222, 444, 55, 666]])

        qng = pyvqnet.qnn.vqc.QNG(qmodel,0.01)

        qng.step(x)

        print(qmodel.parameters())
        #[[[333.0084]], [[4443.9985]]]


wrapper_single_qubit_op_fuse
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.wrapper_single_qubit_op_fuse(f)

    一个用于将单比特运算融合到 Rot 运算中的装饰器。

    .. note::

        f 是模块的前向函数,需要运行一次模型的前向函数才能生效。
        此处定义的模型继承自 ``pyvqnet.qnn.vqc.QModule``,该类是 `pyvqnet.nn.Module` 的子类。

    Example::

        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import QMachine, Operation, apply_unitary_bmm
        from pyvqnet import kcomplex128
        from pyvqnet.tensor import adjoint
        import numpy as np
        from pyvqnet.qnn.vqc import single_qubit_ops_fuse, wrapper_single_qubit_op_fuse, QModule,op_history_summary
        from pyvqnet.qnn.vqc import QMachine, RX, RY, CNOT, PauliX, qmatrix, PauliZ, T, MeasureAll, RZ
        from pyvqnet.tensor import QTensor, tensor
        import pyvqnet
        import numpy as np
        from pyvqnet.utils import set_random_seed


        set_random_seed(42)

        class QModel(QModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)
                self.rx_layer = RX(has_params=True, trainable=False, wires=0, dtype=dtype)
                self.ry_layer = RY(has_params=True, trainable=False, wires=1, dtype=dtype)
                self.rz_layer = RZ(has_params=True, trainable=False, wires=1, dtype=dtype)
                self.rz_layer2 = RZ(has_params=True, trainable=False, wires=1, dtype=dtype)
                self.tlayer = T(wires=1)
                self.cnot = CNOT(wires=[0, 1])
                self.measure = MeasureAll(obs={
                    'wires': [1],
                    'observables': ['x'],
                    'coefficient': [1]
                })

            @wrapper_single_qubit_op_fuse
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])

                self.rx_layer(params=x[:, [0]], q_machine=self.qm)
                self.cnot(q_machine=self.qm)
                self.ry_layer(params=x[:, [1]], q_machine=self.qm)
                self.tlayer(q_machine=self.qm)
                self.rz_layer(params=x[:, [2]], q_machine=self.qm)
                self.rz_layer2(params=x[:, [3]], q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)

                return rlt

        input_x = tensor.QTensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]],
                                dtype=pyvqnet.kfloat64)

        input_xt = tensor.tile(input_x, (100, 1))
        input_xt.requires_grad = True

        qunatum_model = QModel(num_wires=2, dtype=pyvqnet.kcomplex128)
        batch_y = qunatum_model(input_xt)
        print(op_history_summary(qunatum_model.qm.op_history))


        # ###################Summary#######################
        # qubits num: 2
        # gates: {'rot': 2, 'cnot': 1}
        # total gates: 3
        # total parameter gates: 2
        # total parameters: 6
        # #################################################


wrapper_commute_controlled
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.wrapper_commute_controlled(f, direction = "right")

    装饰器用于进行受控门交换
    这是一个量子变换,用于将可交换的门移动到受控操作的控制比特和目标比特之前。
    控制比特两侧的对角门不会影响受控门的结果；因此,我们可以将所有作用在第一个比特上的单比特门一起推到右边(如果需要,可以进行融合)。
    类似地,X 门与 CNOT 和 Toffoli 的目标比特可交换(PauliY 与 CRY 也是如此)。
    我们可以使用此变换将单比特门尽可能推到受控操作的深处。

    .. note::

        f 是模块的前向函数,需要运行一次模型的前向函数才能生效。
        此处定义的模型继承自 ``pyvqnet.qnn.vqc.QModule``,该类是 `pyvqnet.nn.Module` 的子类。

    :param f: 前向函数。
    :param direction: 移动单比特门的方向,可选值为 "left" 或 "right",默认为 "right"。


    Example::

        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import QMachine
        from pyvqnet import kcomplex128
        from pyvqnet.tensor import adjoint
        import numpy as np
        from pyvqnet.qnn.vqc import wrapper_commute_controlled, pauliy, QModule,op_history_summary

        from pyvqnet.qnn.vqc import QMachine, RX, RY, CNOT, S, CRY, PauliZ, PauliX, T, MeasureAll, RZ, CZ, PhaseShift, Toffoli, cnot, cry, toffoli
        from pyvqnet.tensor import QTensor, tensor
        import pyvqnet
        import numpy as np
        from pyvqnet.utils import set_random_seed
        from pyvqnet.qnn import expval, QuantumLayerV2
        import time
        from functools import partial
        set_random_seed(42)

        class QModel(QModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)

                self.cz = CZ(wires=[0, 2])
                self.paulix = PauliX(wires=2)
                self.s = S(wires=0)
                self.ps = PhaseShift(has_params=True, trainable= True, wires=0, dtype=dtype)
                self.t = T(wires=0)
                self.rz = RZ(has_params=True, wires=1, dtype=dtype)
                self.measure = MeasureAll(obs={
                    'wires': [0],
                    'observables': ['z'],
                    'coefficient': [1]
                })

            @partial(wrapper_commute_controlled, direction="left")
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.cz(q_machine=self.qm)
                self.paulix(q_machine=self.qm)
                self.s(q_machine=self.qm)
                cnot(q_machine=self.qm, wires=[0, 1])
                pauliy(q_machine=self.qm, wires=1)
                cry(q_machine=self.qm, params=1 / 2, wires=[0, 1])
                self.ps(q_machine=self.qm)
                toffoli(q_machine=self.qm, wires=[0, 1, 2])
                self.t(q_machine=self.qm)
                self.rz(q_machine=self.qm)
                rlt = self.measure(q_machine=self.qm)

                return rlt

        import pyvqnet
        import pyvqnet.tensor as tensor
        input_x = tensor.QTensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]],
                                    dtype=pyvqnet.kfloat64)

        input_xt = tensor.tile(input_x, (100, 1))
        input_xt.requires_grad = True

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex128)

        batch_y = qunatum_model(input_xt)
        for d in qunatum_model.qm.op_history:
            name = d["name"]
            wires = d["wires"]
            p = d["params"]
            print(f"name: {name} wires: {wires}, params = {p}")


        # name: s wires: (0,), params = None
        # name: phaseshift wires: (0,), params = [[4.744782]]
        # name: t wires: (0,), params = None
        # name: cz wires: (0, 2), params = None
        # name: paulix wires: (2,), params = None
        # name: cnot wires: (0, 1), params = None
        # name: pauliy wires: (1,), params = None
        # name: cry wires: (0, 1), params = [[0.5]]
        # name: rz wires: (1,), params = [[4.7447823]]
        # name: toffoli wires: (0, 1, 2), params = None
        # name: MeasureAll wires: [0], params = None


wrapper_merge_rotations
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.wrapper_merge_rotations(f)


    合并相同类型的旋转门的装饰器,包括 "rx"、"ry"、"rz"、"phaseshift"、"crx"、"cry"、"crz"、"controlledphaseshift"、"isingxx"、
    "isingyy"、"isingzz"、"rot"。

    .. note::

        f 是模块的前向函数,需要运行一次模型的前向函数才能生效。
        此处定义的模型继承自 ``pyvqnet.qnn.vqc.QModule``,该类是 `pyvqnet.nn.Module` 的子类。

    :param f: 前向函数。

    Example::

        import pyvqnet
        from pyvqnet.tensor import tensor

        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import QMachine,op_history_summary
        from pyvqnet import kcomplex128
        from pyvqnet.tensor import adjoint
        import numpy as np


        from pyvqnet.qnn.vqc import *
        from pyvqnet.qnn.vqc import QModule
        from pyvqnet.tensor import QTensor, tensor
        import pyvqnet
        import numpy as np
        from pyvqnet.utils import set_random_seed

        set_random_seed(42)

        class QModel(QModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)

                self.measure = MeasureAll(obs={
                    'wires': [0],
                    'observables': ['z'],
                    'coefficient': [1]
                })

            @wrapper_merge_rotations
            def forward(self, x, *args, **kwargs):

                self.qm.reset_states(x.shape[0])
                
                rx(q_machine=self.qm, params=x[:, [1]], wires=(0, ))
                rx(q_machine=self.qm, params=x[:, [1]], wires=(0, ))
                rx(q_machine=self.qm, params=x[:, [1]], wires=(0, ))
                rot(q_machine=self.qm, params=x, wires=(1, ), use_dagger=True)
                rot(q_machine=self.qm, params=x, wires=(1, ), use_dagger=True)
                isingxy(q_machine=self.qm, params=x[:, [2]], wires=(0, 1))
                isingxy(q_machine=self.qm, params=x[:, [0]], wires=(0, 1))
                cnot(q_machine=self.qm, wires=[1, 2])
                ry(q_machine=self.qm, params=x[:, [1]], wires=(1, ))
                hadamard(q_machine=self.qm, wires=(2, ))
                crz(q_machine=self.qm, params=x[:, [2]], wires=(2, 0))
                ry(q_machine=self.qm, params=-x[:, [1]], wires=1)
                return self.measure(q_machine=self.qm)


        input_x = tensor.QTensor([[1, 2, 3], [1, 2, 3]], dtype=pyvqnet.kfloat64)

        input_x.requires_grad = True

        qunatum_model = QModel(num_wires=3, dtype=pyvqnet.kcomplex128)
        qunatum_model.use_merge_rotations = True
        batch_y = qunatum_model(input_x)
        print(op_history_summary(qunatum_model.qm.op_history))
        # ###################Summary#######################
        # qubits num: 3
        # gates: {'rx': 1, 'rot': 1, 'isingxy': 2, 'cnot': 1, 'hadamard': 1, 'crz': 1}
        # total gates: 7
        # total parameter gates: 5
        # total parameters: 7
        # #################################################



wrapper_compile
---------------------------------------------------------------

.. py:function:: pyvqnet.qnn.vqc.wrapper_compile(f,compile_rules=[commute_controlled_right, merge_rotations, single_qubit_ops_fuse])

    使用编译规则来优化 QModule 的电路。

    .. note::

        f 是模块的前向函数,需要运行一次模型的前向函数才能生效。
        此处定义的模型继承自 `pyvqnet.qnn.vqc.QModule`,该类是 `pyvqnet.nn.Module` 的子类。

    :param f: 前向函数。

    Example::

        from functools import partial

        from pyvqnet.qnn.vqc import op_history_summary
        from pyvqnet.qnn.vqc import QModule
        from pyvqnet import tensor
        from pyvqnet.qnn.vqc import QMachine, wrapper_compile

        from pyvqnet.qnn.vqc import pauliy

        from pyvqnet.qnn.vqc import QMachine, ry,rz, ControlledPhaseShift, \
            rx, S, rot, isingxy,CSWAP, PauliX, T, MeasureAll, RZ, CZ, PhaseShift, u3, cnot, cry, toffoli, cy
        from pyvqnet.tensor import QTensor, tensor
        import pyvqnet

        class QModel_before(QModule):
            def __init__(self, num_wires, dtype):
                super(QModel_before, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)
                self.qm.set_save_op_history_flag(True)
                self.cswap = CSWAP(wires=(0, 2, 1))
                self.cz = CZ(wires=[0, 2])

                self.paulix = PauliX(wires=2)

                self.s = S(wires=0)

                self.ps = PhaseShift(has_params=True,
                                        trainable=True,
                                        wires=0,
                                        dtype=dtype)

                self.cps = ControlledPhaseShift(has_params=True,
                                                trainable=True,
                                                wires=(1, 0),
                                                dtype=dtype)
                self.t = T(wires=0)
                self.rz = RZ(has_params=True, wires=1, dtype=dtype)

                self.measure = MeasureAll(obs={
                    'wires': [0],
                    'observables': ['z'],
                    'coefficient': [1]
                })

            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.cz(q_machine=self.qm)
                self.paulix(q_machine=self.qm)
                rx(q_machine=self.qm,wires=1,params = x[:,[0]])
                ry(q_machine=self.qm,wires=1,params = x[:,[1]])
                rz(q_machine=self.qm,wires=1,params = x[:,[2]])
                rot(q_machine=self.qm, params=x[:, 0:3], wires=(1, ), use_dagger=True)
                rot(q_machine=self.qm, params=x[:, 1:4], wires=(1, ), use_dagger=True)
                isingxy(q_machine=self.qm, params=x[:, [2]], wires=(0, 1))
                u3(q_machine=self.qm, params=x[:, 0:3], wires=1)
                self.s(q_machine=self.qm)
                self.cswap(q_machine=self.qm)
                cnot(q_machine=self.qm, wires=[0, 1])
                ry(q_machine=self.qm,wires=2,params = x[:,[1]])
                pauliy(q_machine=self.qm, wires=1)
                cry(q_machine=self.qm, params=1 / 2, wires=[0, 1])
                self.ps(q_machine=self.qm)
                self.cps(q_machine=self.qm)
                ry(q_machine=self.qm,wires=2,params = x[:,[1]])
                rz(q_machine=self.qm,wires=2,params = x[:,[2]])
                toffoli(q_machine=self.qm, wires=[0, 1, 2])
                self.t(q_machine=self.qm)

                cy(q_machine=self.qm, wires=(2, 1))
                ry(q_machine=self.qm,wires=1,params = x[:,[1]])
                self.rz(q_machine=self.qm)

                rlt = self.measure(q_machine=self.qm)

                return rlt
        class QModel(QModule):
            def __init__(self, num_wires, dtype):
                super(QModel, self).__init__()

                self._num_wires = num_wires
                self._dtype = dtype
                self.qm = QMachine(num_wires, dtype=dtype)

                self.cswap = CSWAP(wires=(0, 2, 1))
                self.cz = CZ(wires=[0, 2])

                self.paulix = PauliX(wires=2)

                self.s = S(wires=0)

                self.ps = PhaseShift(has_params=True,
                                        trainable=True,
                                        wires=0,
                                        dtype=dtype)

                self.cps = ControlledPhaseShift(has_params=True,
                                                trainable=True,
                                                wires=(1, 0),
                                                dtype=dtype)
                self.t = T(wires=0)
                self.rz = RZ(has_params=True, wires=1, dtype=dtype)

                self.measure = MeasureAll(obs={
                    'wires': [0],
                    'observables': ['z'],
                    'coefficient': [1]
                })

            @partial(wrapper_compile)
            def forward(self, x, *args, **kwargs):
                self.qm.reset_states(x.shape[0])
                self.cz(q_machine=self.qm)
                self.paulix(q_machine=self.qm)
                rx(q_machine=self.qm,wires=1,params = x[:,[0]])
                ry(q_machine=self.qm,wires=1,params = x[:,[1]])
                rz(q_machine=self.qm,wires=1,params = x[:,[2]])
                rot(q_machine=self.qm, params=x[:, 0:3], wires=(1, ), use_dagger=True)
                rot(q_machine=self.qm, params=x[:, 1:4], wires=(1, ), use_dagger=True)
                isingxy(q_machine=self.qm, params=x[:, [2]], wires=(0, 1))
                u3(q_machine=self.qm, params=x[:, 0:3], wires=1)
                self.s(q_machine=self.qm)
                self.cswap(q_machine=self.qm)
                cnot(q_machine=self.qm, wires=[0, 1])
                ry(q_machine=self.qm,wires=2,params = x[:,[1]])
                pauliy(q_machine=self.qm, wires=1)
                cry(q_machine=self.qm, params=1 / 2, wires=[0, 1])
                self.ps(q_machine=self.qm)
                self.cps(q_machine=self.qm)
                ry(q_machine=self.qm,wires=2,params = x[:,[1]])
                rz(q_machine=self.qm,wires=2,params = x[:,[2]])
                toffoli(q_machine=self.qm, wires=[0, 1, 2])
                self.t(q_machine=self.qm)

                cy(q_machine=self.qm, wires=(2, 1))
                ry(q_machine=self.qm,wires=1,params = x[:,[1]])
                self.rz(q_machine=self.qm)

                rlt = self.measure(q_machine=self.qm)

                return rlt

        import pyvqnet
        import pyvqnet.tensor as tensor
        input_x = tensor.QTensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]],
                                    dtype=pyvqnet.kfloat64)

        input_x.requires_grad = True
        num_wires = 3
        qunatum_model = QModel(num_wires=num_wires, dtype=pyvqnet.kcomplex128)
        qunatum_model_before = QModel_before(num_wires=num_wires, dtype=pyvqnet.kcomplex128)

        batch_y = qunatum_model(input_x)
        batch_y = qunatum_model_before(input_x)

        flatten_oph_names = []

        print("before")

        print(op_history_summary(qunatum_model_before.qm.op_history))
        flatten_oph_names = []
        for d in qunatum_model.compiled_op_historys:
                if "compile" in d.keys():
                    oph = d["op_history"]
                    for i in oph:
                        n = i["name"]
                        w = i["wires"]
                        p = i["params"]
                        flatten_oph_names.append({"name":n,"wires":w, "params": p})
        print("after")
        print(op_history_summary(qunatum_model.qm.op_history))


        # ###################Summary#######################
        # qubits num: 3
        # gates: {'cz': 1, 'paulix': 1, 'rx': 1, 'ry': 4, 'rz': 3, 'rot': 2, 'isingxy': 1, 'u3': 1, 's': 1, 'cswap': 1, 'cnot': 1, 'pauliy': 1, 'cry': 1, 'phaseshift': 1, 'controlledphaseshift': 1, 'toffoli': 1, 't': 1, 'cy': 1}
        # total gates: 24
        # total parameter gates: 15
        # total parameters: 21
        # #################################################
            
        # after


        # ###################Summary#######################
        # qubits num: 3
        # gates: {'cz': 1, 'rot': 7, 'isingxy': 1, 'u3': 1, 'cswap': 1, 'cnot': 1, 'cry': 1, 'controlledphaseshift': 1, 'toffoli': 1, 'cy': 1}
        # total gates: 16
        # total parameter gates: 11
        # total parameters: 27
        # #################################################
