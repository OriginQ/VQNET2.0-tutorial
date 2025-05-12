常见问题
================

**问: VQNet有哪些特性**

答: VQNet是基于本源量子pyQPanda开发的量子机器学习工具集。VQNet提供了丰富、易用的经典神经网络计算模块接口,可以方便地进行机器学习的优化,
模型定义方式与主流机器学习框架一致,降低了用户学习成本。
同时,基于本源量子研发的高性能量子模拟器pyQPanda,VQNet在个人笔记本电脑上也能支持大数量量子比特的运算。
最后,VQNet还有丰富的 :doc:`./qml_demo` 供大家参考和学习。

**问: 如何使用VQNet进行量子机器学习模型的训练** 

答: 量子机器学习算法中有一类是基于量子变分线路构建可微的量子机器学习模型。
VQNet可以使用梯度下降法对这类量子机器学习模型进行训练。一般步骤如下: 首先在本地计算机上,用户可以通过pyQPanda的 ``CPUQVM()`` 构建虚拟机,并结合VQNet中提供的接口构建量子、量子经典混合模型 ``Module`` ; 其次,调用 ``Module`` 的 ``forward()`` 可按用户定义的运行方式进行量子线路模拟以及经典神经网络前向运算；
当调用 ``Module`` 的 ``backward()`` 用户构建的模型可以像PyTorch等经典机器学习框架一样进行自动微分,计算量子变分线路以及经典计算层中的参数梯度；最后结合优化器的 ``step()`` 功能进行参数的优化。

在VQNet中,我们使用 `parameter-shift <https://arxiv.org/abs/1803.00745>`_ 计算量子变分线路的梯度。用户可使用
VQNet提供的 ``QuantumLayer`` 以及 ``QpandaQCircuitVQCLayerLite`` 类已经封装了量子变分线路的自动微分,用户仅需按一定格式定义量子变分线路作为参数构建以上类即可。
具体可参考本文档相关接口以及示例代码。

**问: 在Windows中,安装VQNet遇到错误: “importError: DLL load failed while importing _core: 找不到指定的模块。”**

答: 用户在Windows上可能需要安装VC++ 运行时库。
可参考 https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170 安装合适的运行库。
此外,VQNet当前仅支持python3.8, 3.9, 3.10 版本,故请确认你的python版本。

**问: 如何调用本源量子云以及量子芯片进行计算**

答: 使用pyQPanda的云端资源,可以在量子线路模拟中使用高性能计算机集群或真实的量子计算机,用云计算的方式替代本地量子线路模拟。在VQNet中,若用户使用 ``QuantumLayerV2`` 构建变分量子线路模块,可使用pyQPanda的云服务接口 ``QCloud()`` 接口
代替本地全振幅模拟器 ``CPUQVM()``,测量函数也进行相应修改,具体参见 `QCloudServer <https://pyqpanda-toturial.readthedocs.io/zh/latest/QCloudServer.html>`_ 。
若用户使用 ``QuantumLayer`` 构建变分量子线路模块,使用 `machine_type_or_cloud_token` 参数输入申请的QCloud token,该模块可内部构建一个云端虚拟机。


**问: 为什么我定义的模型参数在训练时候不更新**

答: 构建VQNet的模型需要保证其中所使用的所有模块是可微分。当模型某个模块无法计算梯度,则会导致该模块以及之前的模块无法使用链式法则计算梯度。
若用户自定义一个量子变分线路,请使用VQNet提供的 ``QuantumLayer`` 以及 ``QpandaQCircuitVQCLayerLite`` 接口。对于经典机器学习模块,需要使用 :doc:`./QTensor` 以及 :doc:`./nn` 定义的接口,这些接口封装了梯度计算的函数,VQNet可以进行自动微分。

若用户想在 `Module` 中使用包含多个模块的列表作为子模块,请不要使用python自带的List,需要使用 pyvqnet.nn.module.ModuleList 代替 List。这样,子模块的训练参数可以被注册到整个模型中,可以进行自动微分训练。以下是例子: 

    Example::

        from pyvqnet.tensor import *
        from pyvqnet.nn import Module,Linear,ModuleList
        from pyvqnet.qnn import ProbsMeasure,QuantumLayer
        import pyqpanda as pq
        def pqctest (input,param,qubits,cubits,m_machine):
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
            #print(circuit)

            prog = pq.QProg()
            prog.insert(circuit)

            rlt_prob = ProbsMeasure([0,2],prog,m_machine,qubits)
            return rlt_prob


        class M(Module):
            def __init__(self):
                super(M, self).__init__()
                #应该使用ModuleList构建
                self.pqc2 = ModuleList([QuantumLayer(pqctest,3,"cpu",4,1), Linear(4,1)
                ])
                #直接使用list 是无法保存pqc3中的参数的。
                #self.pqc3 = [QuantumLayer(pqctest,3,"cpu",4,1), Linear(4,1)
                #]
            def forward(self, x, *args, **kwargs):
                y = self.pqc2[0](x)  + self.pqc2[1](x)
                return y

        mm = M()
        print(mm.state_dict().keys())

**问: 为什么原先的代码在2.0.7及以后版本无法运行**

答: 自v2.0.7版本中,我们为QTensor增加了不同数据类型,dtype属性,并参照pytorch对输入进行了限制。例如:  Emedding层输入需要为kint64,CategoricalCrossEntropy, SoftmaxCrossEntropy, NLL_Loss, CrossEntropyLoss 的标签需要为kint64。
你可以使用 `astype()` 接口进行类型转化为指定数据类型,或使用对应的数据类型numpy数组初始化QTensor。
