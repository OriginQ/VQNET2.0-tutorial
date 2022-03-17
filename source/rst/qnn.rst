量子机器学习模块
==================================

量子计算层
----------------------------------

.. _QuantumLayer:

QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

QuantumLayer是一个支持量子含参线路作为参数的自动求导模块的封装类。用户定义一个函数作为参数 ``qprog_with_measure`` ，该函数需要包含pyQPanda定义的量子线路：一般包含量子线路的编码线路，演化线路和测量操作。
该类可以嵌入量子经典混合机器学习模型，通过经典的梯度下降法，使得量子经典混合模型的目标函数或损失函数最小。
用户可通过参数 ``diff_method`` 指定 ``QuantumLayer`` 层中量子线路参数的梯度计算方式，``QuantumLayer`` 当前支持有限差分法 ``finite_diff`` 以及 ``parameter-shift`` 方法。

有限差分法是估算函数梯度最传统和最常用的数值方法之一。主要思想是用差分代替偏导数：

.. math::

    f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}


若使用 ``parameter-shift`` 方法，我们使用如下目标函数：

.. math:: O(\theta)=\left\langle 0\left|U^{\dagger}(\theta) H U(\theta)\right| 0\right\rangle

理论上可以通过 ``parameter-shift`` 这一更精确的方法计算量子线路中参数对哈密顿量的梯度：

.. math::

    \nabla O(\theta)=
    \frac{1}{2}\left[O\left(\theta+\frac{\pi}{2}\right)-O\left(\theta-\frac{\pi}{2}\right)\right]

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayer(qprog_with_measure,para_num,machine_type_or_cloud_token,num_of_qubits:int,num_of_cbits:int = 1,diff_method:str = "parameter_shift",delta:float = 0.01)

	Abstract Calculation module for Variational Quantum Layer. It simulate a parameterized quantum circuit and get the measurement result. It inherits from Module,so that it can calculate gradients of circuits parameters,and trains Variational Quantum Circuits model or embeds Variational Quantum Circuits into hybird Quantum and Classic model.

    :param qprog_with_measure: callable quantum circuits functions ,cosntructed by qpanda
    :param para_num: `int` - Number of parameter
    :param machine_type_or_cloud_token: qpanda machine type or pyQPANDA QCLOUD token : https://pyqpanda-toturial.readthedocs.io/zh/latest/Realchip.html
    :param num_of_qubits: num of qubits
    :param num_of_cbits: num of classic bits
    :param diff_method: 'parameter_shift' or 'finite_diff'
    :param delta:  delta for diff
    :return: a module can calculate quantum circuits .

    .. note::
        qprog_with_measure is quantum circuits function defined in pyQPanda :https://pyqpanda-toturial.readthedocs.io/zh/latest/QCircuit.html.
        
        This function should contains following parameters,otherwise it can not run properly in QuantumLayer.

        qprog_with_measure (input,param,qubits,cbits,m_machine)
        
            `input`: array_like input 1-dim classic data
            
            `param`: array_like input 1-dim quantum circuit's parameters
            
            `qubits`: qubits allocated by QuantumLayer
            
            `cbits`: cbits allocated by QuantumLayer.if your circuits does not use cbits,you should also reserve this parameter.
            
            `m_machine`: simulator created by QuantumLayer

    Example::

        import pyqpanda as pq
        from pyvqnet.qnn.measure import ProbsMeasure
        from pyvqnet.qnn.quantumlayer import QuantumLayer
        import numpy as np 
        from pyvqnet.tensor import QTensor
        def pqctest (input,param,qubits,cbits,m_machine):
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
            # pauli_dict  = {'Z0 X1':10,'Y2':-0.543}
            rlt_prob = ProbsMeasure([0,2],prog,m_machine,qubits)
            return rlt_prob

        pqc = QuantumLayer(pqctest,3,"cpu",4,1)
        #classic data as input       
        input = QTensor([[1,2,3,4],[40,22,2,3],[33,3,25,2]] )
        #forward circuits
        rlt = pqc(input)
        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #backward circuits
        rlt.backward(grad)
        print(rlt)
        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]


QuantumLayerV2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如您更加熟悉pyQPanda语法，可以使用QuantumLayerV2，自定义量子比特 ``qubits`` ,经典比特 ``cbits`` ,后端模拟器 ``machine`` 加入QuantumLayerV2的参数 ``qprog_with_measure`` 函数中。

.. py:class:: pyvqnet.qnn.quantumlayer.QuantumLayerV2(qprog_with_measure,para_num,diff_method:str = "parameter_shift",delta:float = 0.01)

	Abstract Calculation module for Variational Quantum Layer. It simulate a parameterized quantum circuit and get the measurement result. It inherits from Module,so that it can calculate gradients of circuits parameters,and trains Variational Quantum Circuits model or embeds Variational Quantum Circuits into hybird Quantum and Classic model.

    To use this module,you need to create your quantum virtual machine and allocate qubits and cbits by yourself.
    
    :param qprog_with_measure: callable quantum circuits functions ,cosntructed by qpanda
    :param para_num: `int` - Number of parameter
    :param diff_method: 'parameter_shift' or 'finite_diff'
    :param delta:  delta for diff
    :return: a module can calculate quantum circuits .

    .. note::
        qprog_with_measure is quantum circuits function defined in pyQPanda :https://pyqpanda-toturial.readthedocs.io/zh/latest/QCircuit.html.
        
        This function should contains following parameters,otherwise it can not run properly in QuantumLayerV2.

        Compare to QuantumLayer.you should allocate qubits and simulator: https://pyqpanda-toturial.readthedocs.io/zh/latest/QuantumMachine.html,

        you may also need to allocate cbits if qprog_with_measure needs quantum measure: https://pyqpanda-toturial.readthedocs.io/zh/latest/Measure.html
        
        qprog_with_measure (input,param)
        
        `input`: array_like input 1-dim classic data
        
        `param`: array_like input 1-dim quantum circuit's parameters
        

    Example::

        import pyqpanda as pq
        from pyvqnet.qnn.measure import ProbsMeasure
        from pyvqnet.qnn.quantumlayer import QuantumLayerV2
        import numpy as np
        from pyvqnet.tensor import QTensor
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
            #print(circuit)

            prog = pq.QProg()    
            prog.insert(circuit)    
            rlt_prob = ProbsMeasure([0,2],prog,m_machine,qubits)
            return rlt_prob


        pqc = QuantumLayerV2(pqctest,3)

        #classic data as input       
        input = QTensor([[1,2,3,4],[4,2,2,3],[3,3,2,2]] )

        #forward circuits
        rlt = pqc(input)

        grad =  QTensor(np.ones(rlt.data.shape)*1000)
        #backward circuits
        rlt.backward(grad)
        print(rlt)

        # [
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000],
        # [0.2500000, 0.2500000, 0.2500000, 0.2500000]
        # ]
        


NoiseQuantumLayer
^^^^^^^^^^^^^^^^^^^

在真实的量子计算机中，受制于量子比特自身的物理特性，常常存在不可避免的计算误差。为了能在量子虚拟机中更好的模拟这种误差，VQNet同样支持含噪声量子虚拟机。含噪声量子虚拟机的模拟更贴近真实的量子计算机，我们可以自定义支持的逻辑门类型，自定义逻辑门支持的噪声模型。
现有可支持的量子噪声模型依据QPanda中定义，具体参考链接 `QPANDA2 <https://pyqpanda-toturial.readthedocs.io/zh/latest/NoiseQVM.html>`_ 中的介绍。

使用 NoiseQuantumLayer 定义一个量子线路自动微分类,该类支持QPanda噪声虚拟机。用户定义一个函数作为参数 ``qprog_with_measure`` ，该函数需要包含pyQPanda定义的量子线路，同样需要传入一个参数 ``noise_set_config``,使用pyQPanda接口，设置噪声模型。

.. py:class:: pyvqnet.qnn.quantumlayer.NoiseQuantumLayer(qprog_with_measure,para_num,machine_type,num_of_qubits:int,num_of_cbits:int=1,diff_method:str= "parameter_shift",delta:float=0.01,noise_set_config = None)

	Abstract Calculation module for Variational Quantum Layer. It simulate a parameterized quantum circuit and get the measurement result. It inherits from Module,so that it can calculate gradients of circuits parameters,and trains Variational Quantum Circuits model or embeds Variational Quantum Circuits into hybird Quantum and Classic model.

    This layer can use noise model in quantum circuits.

    :param qprog_with_measure: callable quantum circuits functions ,cosntructed by qpanda
    :param para_num: `int` - Number of parameters
    :param machine_type: qpanda machine type
    :param num_of_qubits: num of quantum bits
    :param num_of_cbits: num of classic bits
    :param diff_method: 'parameter_shift' or 'finite_diff'
    :param delta:  delta for diff
    :param noise_set_config: noise set function
    :return: a module can calculate quantum circuits with noise model.
    
    .. note::
        qprog_with_measure is quantum circuits function defined in pyQPanda :https://pyqpanda-toturial.readthedocs.io/zh/latest/QCircuit.html.
        
        This function should contains following parameters,otherwise it can not run properly in NoiseQuantumLayer.
        
        qprog_with_measure (input,param,qubits,cbits,m_machine)
        
            `input`: array_like input 1-dim classic data
            
            `param`: array_like input 1-dim quantum circuit's parameters
            
            `qubits`: qubits allocated by NoiseQuantumLayer
            
            `cbits`: cbits allocated by NoiseQuantumLayer.if your circuits does not use cbits,you should also reserve this parameter.
            
            `m_machine`: simulator created by NoiseQuantumLayer

    Example::

        import pyqpanda as pq
        from pyvqnet.qnn.measure import ProbsMeasure
        from pyvqnet.qnn.quantumlayer import NoiseQuantumLayer
        import numpy as np
        from pyqpanda import * 
        from pyvqnet.tensor import QTensor
        def circuit(weights,param,qubits,cbits,machine):

            circuit = pq.QCircuit()

            circuit.insert(pq.H(qubits[0]))
            circuit.insert(pq.RY(qubits[0], weights[0]))
            circuit.insert(pq.RY(qubits[0], param[0]))
            prog = pq.QProg()
            prog.insert(circuit)
            prog << measure_all(qubits, cbits)

            result = machine.run_with_configuration(prog, cbits, 100)

            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            # Compute probabilities for each state
            probabilities = counts / 100
            # Get state expectation
            expectation = np.sum(states * probabilities)
            return expectation

        def default_noise_config(qvm,q):

            p = 0.01
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_X_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_Y_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_Z_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RX_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RY_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RZ_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RY_GATE, p)
            qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.HADAMARD_GATE, p)
            qves =[]
            for i in range(len(q)-1):
                qves.append([q[i],q[i+1]])#
            qves.append([q[len(q)-1],q[0]])
            qvm.set_noise_model(NoiseModel.DAMPING_KRAUS_OPERATOR, GateType.CNOT_GATE, p, qves)

            return qvm

        qvc = NoiseQuantumLayer(circuit,24,"noise",1,1,diff_method= "parameter_shift", delta=0.01,noise_set_config = default_noise_config)
        input = QTensor([
            [0.0000000, 1.0000000, 1.0000000, 1.0000000],

            [0.0000000, 0.0000000, 1.0000000, 1.0000000],

            [1.0000000, 0.0000000, 1.0000000, 1.0000000]
            ] )
        rlt = qvc(input)
        grad =  QTensor(np.ones(rlt.data.shape)*1000)

        rlt.backward(grad)
        print(qvc.m_para.grad)

        #[1195.0000000, 105.0000000, 70.0000000, 0.0000000, 
        # 45.0000000, -45.0000000, 50.0000000, 15.0000000, 
        # -80.0000000, 50.0000000, 10.0000000, -30.0000000, 
        # 10.0000000, 60.0000000, 75.0000000, -110.0000000, 
        # 55.0000000, 45.0000000, 25.0000000, 5.0000000, 
        # 5.0000000, 50.0000000, -25.0000000, -15.0000000]

下面给出一个 ``noise_set_config`` 的例子，这里使得 ``RX`` , ``RY`` , ``RZ`` , ``X`` , ``Y`` , ``Z`` , ``H`` 等逻辑门加入了 p = 0.01 的 BITFLIP_KRAUS_OPERATOR噪声模型。

.. code-block::

	def noise_set_config(qvm,q):

		p = 0.01
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_X_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_Y_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.PAULI_Z_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RX_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RY_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RZ_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.RY_GATE, p)
		qvm.set_noise_model(NoiseModel.BITFLIP_KRAUS_OPERATOR, GateType.HADAMARD_GATE, p)
		qves =[]
		for i in range(len(q)-1):
			qves.append([q[i],q[i+1]])#
		qves.append([q[len(q)-1],q[0]])
		qvm.set_noise_model(NoiseModel.DAMPING_KRAUS_OPERATOR, GateType.CNOT_GATE, p, qves)

		return qvm
		
VQCLayer
^^^^^^^^^^^^^^^^^^^^^^^^

基于pyQPanda的可变量子线路VariationalQuantumCircuit，VQNet提供了抽象量子计算层 ``VQCLayer`` 。用户只需要定义一个类 ``VQC_wrapper`` ，
其中定义相应的量子线路逻辑门和测量函数即可基于pyQPanda的 ``VariationalQuantumCircuit`` ，进行机器学习模型的构建。

在 ``VQC_wrapper`` 中，用户使用普通逻辑门函数 ``build_common_circuits`` 构建模型中线路结构变化的子线路，使用VQG在 ``build_vqc_circuits`` 构建结构不变，参数变化的子线路。使用
``run`` 函数定义线路运行方式以及测量。

.. py:class:: pyvqnet.qnn.quantumlayer.VQC_wrapper

    ``VQC_wrapper`` is a abstract class help to run VariationalQuantumCircuit on VQNet.

    ``build_common_circuits`` function contains circuits may be varaible according to the input.

    ``build_vqc_circuits`` function contains VQC circuits with trainable weights.

    ``run`` function contains run function for VQC.
    
    Example::

        import pyqpanda as pq
        from pyqpanda import *
        from pyvqnet.qnn.quantumlayer import VQCLayer,VQC_wrapper
        class QVC_demo(VQC_wrapper):
            
            def __init__(self):
                super(QVC_demo, self).__init__()


            def build_common_circuits(self,input,qlists,):
                qc = pq.QCircuit()
                for i in range(len(qlists)):
                    if input[i]==1:
                        qc.insert(pq.X(qlists[i]))
                return qc
                
            def build_vqc_circuits(self,input,weights,machine,qlists,clists):

                def get_cnot(qubits):
                    vqc = VariationalQuantumCircuit()
                    for i in range(len(qubits)-1):
                        vqc.insert(pq.VariationalQuantumGate_CNOT(qubits[i],qubits[i+1]))
                    vqc.insert(pq.VariationalQuantumGate_CNOT(qubits[len(qubits)-1],qubits[0]))
                    return vqc

                def build_circult(weights, xx, qubits,vqc):
                    
                    def Rot(weights_j, qubits):
                        vqc = VariationalQuantumCircuit()
                        
                        vqc.insert(pq.VariationalQuantumGate_RZ(qubits, weights_j[0]))
                        vqc.insert(pq.VariationalQuantumGate_RY(qubits, weights_j[1]))
                        vqc.insert(pq.VariationalQuantumGate_RZ(qubits, weights_j[2]))
                        return vqc

                    #2,4,3
                    for i in range(2):
                        
                        weights_i = weights[i,:,:]
                        for j in range(len(qubits)):
                            weights_j = weights_i[j]
                            vqc.insert(Rot(weights_j,qubits[j]))
                        cnots = get_cnot(qubits)  
                        vqc.insert(cnots) 

                    vqc.insert(pq.VariationalQuantumGate_Z(qubits[0]))#pauli z(0)

                    return vqc
                
                weights = weights.reshape([2,4,3])
                vqc = VariationalQuantumCircuit()
                return build_circult(weights, input,qlists,vqc)

将该实例化对象 ``VQC_wrapper`` 作为参数传入 ``VQCLayer``

.. py:class:: pyvqnet.qnn.quantumlayer.VQCLayer(vqc_wrapper,para_num,machine_type_or_cloud_token,num_of_qubits:int,num_of_cbits:int = 1,diff_method:str = "parameter_shift",delta:float = 0.01)

    Abstract Calculation module for Variational Quantum Circuits in pyQPanda.Please reference to :https://pyqpanda-toturial.readthedocs.io/zh/latest/VQG.html.
    
    :param vqc_wrapper: VQC_wrapper class
    :param para_num: `int` - Number of parameter
    :param diff_method: 'parameter_shift' or 'finite_diff'
    :param delta:  delta for diff
    :return: a module can calculate VQC quantum circuits .

    Example::

        import pyqpanda as pq
        from pyqpanda import *
        from pyvqnet.qnn.quantumlayer import VQCLayer,VQC_wrapper

        class QVC_demo(VQC_wrapper):
            
            def __init__(self):
                super(QVC_demo, self).__init__()


            def build_common_circuits(self,input,qlists,):
                qc = pq.QCircuit()
                for i in range(len(qlists)):
                    if input[i]==1:
                        qc.insert(pq.X(qlists[i]))
                return qc
                
            def build_vqc_circuits(self,input,weights,machine,qlists,clists):

                def get_cnot(qubits):
                    vqc = VariationalQuantumCircuit()
                    for i in range(len(qubits)-1):
                        vqc.insert(pq.VariationalQuantumGate_CNOT(qubits[i],qubits[i+1]))
                    vqc.insert(pq.VariationalQuantumGate_CNOT(qubits[len(qubits)-1],qubits[0]))
                    return vqc

                def build_circult(weights, xx, qubits,vqc):
                    
                    def Rot(weights_j, qubits):
                        vqc = VariationalQuantumCircuit()
                        
                        vqc.insert(pq.VariationalQuantumGate_RZ(qubits, weights_j[0]))
                        vqc.insert(pq.VariationalQuantumGate_RY(qubits, weights_j[1]))
                        vqc.insert(pq.VariationalQuantumGate_RZ(qubits, weights_j[2]))
                        return vqc

                    #2,4,3
                    for i in range(2):
                        
                        weights_i = weights[i,:,:]
                        for j in range(len(qubits)):
                            weights_j = weights_i[j]
                            vqc.insert(Rot(weights_j,qubits[j]))
                        cnots = get_cnot(qubits)  
                        vqc.insert(cnots) 

                    vqc.insert(pq.VariationalQuantumGate_Z(qubits[0]))#pauli z(0)

                    return vqc
                
                weights = weights.reshape([2,4,3])
                vqc = VariationalQuantumCircuit()
                return build_circult(weights, input,qlists,vqc)
            
            def run(self,vqc,input,machine,qlists,clists):

                prog = QProg()
                vqc_all = VariationalQuantumCircuit()
                # add encode circuits
                vqc_all.insert(self.build_common_circuits(input,qlists))
                vqc_all.insert(vqc)
                qcir = vqc_all.feed()
                prog.insert(qcir)
                #print(pq.convert_qprog_to_originir(prog, machine))
                prob = machine.prob_run_dict(prog, qlists[0], -1)
                prob = list(prob.values())
            
                return prob

        qvc_vqc = QVC_demo()
        VQCLayer(qvc_vqc,24,"cpu",4)

Qconv
^^^^^^^^^^^^^^^^^^^^^^^^

Qconv是一种量子卷积算法接口。
量子卷积操作采用量子线路对经典数据进行卷积操作，其无需计算乘法和加法操作，只需将数据编码到量子态，然后通过量子线路进行演化操作和测量得到最终的卷积结果。
根据卷积核的范围中的输入数据数量申请相同数量的量子比特，然后构建量子线路进行计算。

.. image:: ./images/qcnn.png

其量子线路由每个qubit上首先插入 :math:`RY` , :math:`RZ` 门进行编码，接着在任意两个qubit上使用 :math:`Z` 以及 :math:`U3` 进行信息纠缠和交换。下图为4qubits的例子

.. image:: ./images/qcnn_cir.png

.. py:class:: pyvqnet.qnn.qcnn.qconv.QConv(input_channels,output_channels,quantum_number,stride=(1, 1),padding=(0, 0),kernel_initializer=normal,machine:str = "cpu"))

	Quantum Convolution module. Replace Conv2D kernal with quantum circuits.Inputs to the conv module are of shape (batch_size, input_channels, height, width).reference `Samuel et al. (2020) <https://arxiv.org/abs/2012.12177>`_.

    :param input_channels: `int` - Number of input channels
    :param output_channels: `int` - Number of output channels
    :param quantum_number: `int` - Size of a single kernel. 
    :param stride: `tuple` - Stride, defaults to (1, 1)
    :param padding: `tuple` - Padding, defaults to (0, 0)
    :param kernel_initializer: `callable` - Defaults to normal
    :param machine: `str` - cpu simulation 
    :return: a quantum cnn class
    
    Example::

        from pyvqnet.tensor import tensor
        from pyvqnet.qnn.qcnn.qconv import QConv
        x = tensor.ones([1,3,4,4])
        layer = QConv(input_channels=3, output_channels=2, quantum_number=4, stride=(2, 2))
        y = layer(x)
        print(y)

        # [
        # [[[-0.0889078, -0.0889078],
        #  [-0.0889078, -0.0889078]],
        # [[0.7992646, 0.7992646],
        #  [0.7992646, 0.7992646]]]
        # ]

QLinear
^^^^^^^^^^

QLinear 实现了一种量子全连接算法。首先将数据编码到量子态，然后通过量子线路进行演化操作和测量得到最终的全连接结果。

.. image:: ./images/qlinear_cir.png

.. py:class:: pyvqnet.qnn.qlinear.QLinear(input_channels,output_channels,machine: str = "cpu"))

    Quantum Linear module. Inputs to the linear module are of shape (input_channels, output_channels)

    :param input_channels: `int` - Number of input channels
    :param output_channels: `int` - Number of output channels
    :param machine: `str` - cpu simulation
    :return: a quantum linear layer

    Exmaple::

        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.qlinear import QLinear
        params = [[0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452], 
        [1.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
        [1.37454012, 1.95071431, 0.73199394, 0.59865848, 0.15601864, 0.15599452],
        [1.37454012, 1.95071431, 1.73199394, 1.59865848, 0.15601864, 0.15599452]]
        m = QLinear(32, 2)
        input = QTensor(params, requires_grad=True)
        output = m(input)
        output.backward()
        print(output)

        # [
        # [0.0590705, 0.1264583],
        # [0.5157100, 0.1264583],
        # [0.5157100, 0.1443066],
        # [0.9382225, 0.1443066]
        # ]

Compatiblelayer
^^^^^^^^^^^^^^^^^

VQNet不仅可以支持 ``QPANDA`` 的量子线路，同时可以支持其他量子计算框架(例如 ``Cirq``, ``Qiskit`` 等）的量子线路作为VQNet混合量子经典优化的量子计算部分。
VQNet提供了自动微分的量子线路运算接口 ``Compatiblelayer`` 。构建 ``Compatiblelayer`` 的参数中需要传入一个类，其中定义了第三方库量子线路 ，以及其运行和测量函数 ``run`` 。
使用 ``Compatiblelayer`` ,量子线路的输入以及参数的自动微分就可交由VQNet进行实现。
VQNet提供了一个示例使用qiskit线路进行训练。

.. py:class:: pyvqnet.qnn.utils.compatible_layer.Compatiblelayer(para_num)

    An abstract wrapper to use other framework's quantum circuits(such as Qiskit `qiskit.QuantumCircuit`, TFQ `cirq.Circuit`) to forward and backward in the form of vqnet.
    Your should define the quantums circuits in the forward() and backward() functions.

    `pyvqnet.qnn.utils.qikitlayer.QiskitLayer` is an implementation of using Qiskit's circuits to run in vqnet.

        

    :param para_num: `int` - Number of parameters.
    :return: a quantum layer

在VQNet中，我们同样可以使用IBM的qiskit量子计算库进行量子机器学习任务。

VQNet实现了自动微分的qiskit量子线路运算类 ``QiskitLayer``,其继承于 ``Compatiblelayer``, ``Compatiblelayer`` 是用来兼容其他框架线路到VQNet的类。构建 ``QiskitLayer`` 的参数中需要传入一个类，其中定义了qiksit量子线路 ``qiskit.QuantumCircuit`` ，以及其运行和测量函数 ``run`` 。``run`` 函数需要把输入的数值绑定到qiskit的量子线路上。使用 ``QiskitLayer`` ,量子线路的输入以及参数的自动微分就可交由vqnet进行实现。

参照上面的混合量子经典机器学习示例HQCNN进行MNIST分类任务,我们用qiskit构建了一个类 ``QISKIT_VQC`` 。其中 ``self._circuit`` 变量为量子线路。``self.input`` 为可变输入参数。在 ``run`` 函数中，需要使用qiskit的 ``assign_parameters`` 绑定参数并使用 ``self.backend.run`` 运行。

.. note:: 如线路中还有其他可变参数，需要定义新的参数例如 ``self.theta = [qiskit.circuit.Parameter('theta1'),qiskit.circuit.Parameter('theta2')]`` ，并在 ``run`` 函数中，需要使用qiskit的 ``assign_parameters`` 绑定参数。

.. code-block::

	import sys,os

	import pyvqnet.tensor as tensor
	from pyvqnet.nn.linear import Linear
	from pyvqnet.nn.conv import Conv2D
	from pyvqnet.nn import activation as F
	from pyvqnet.nn.pooling import MaxPool2D
	from pyvqnet.nn.module import Module
	from pyvqnet.optim.adam import Adam
	import numpy as np
	from pyvqnet.nn.loss import CategoricalCrossEntropy

	from pyvqnet.tensor import QTensor
	import random

	random.seed(1234)

	import qiskit
	simulator = qiskit.Aer.get_backend('aer_simulator')

	from pyvqnet.qnn.utils.qiskitlayer import QiskitLayer
	from pyvqnet.data.data import data_generator
	import time
	import datetime

	import matplotlib
	try:
		matplotlib.use('TkAgg')
	except:
		pass
	import matplotlib.pyplot as plt
    try:
        import urllib.request
    except ImportError:
        raise ImportError('You should use Python 3.x')
    import os.path
    import gzip

    url_base = 'http://yann.lecun.com/exdb/mnist/'
    key_file = {
        'train_img':'train-images-idx3-ubyte.gz',
        'train_label':'train-labels-idx1-ubyte.gz',
        'test_img':'t10k-images-idx3-ubyte.gz',
        'test_label':'t10k-labels-idx1-ubyte.gz'
    }



    def _download(dataset_dir,file_name):
        file_path = dataset_dir + "/" + file_name
        
        if os.path.exists(file_path):
            with gzip.GzipFile(file_path) as f:
                file_path_ungz = file_path[:-3].replace('\\', '/')
                if not os.path.exists(file_path_ungz):
                    open(file_path_ungz,"wb").write(f.read())
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(url_base + file_name, file_path)
        if os.path.exists(file_path):
                with gzip.GzipFile(file_path) as f:
                    file_path_ungz = file_path[:-3].replace('\\', '/')
    
                    if not os.path.exists(file_path_ungz):
                        open(file_path_ungz,"wb").write(f.read())
        print("Done")
        
    def download_mnist(dataset_dir):
        for v in key_file.values():
        _download(dataset_dir,v)

	class QISKIT_VQC:
		"""
		This class provides a simple interface for interaction
		with the quantum circuit
		"""

		def __init__(self, n_qubits, backend, shots):
			# --- Circuit definition ---
			self._circuit = qiskit.QuantumCircuit(n_qubits)

			all_qubits = [i for i in range(n_qubits)]
			self.input = [qiskit.circuit.Parameter('input')]

			self._circuit.h(all_qubits)
			self._circuit.barrier()
			self._circuit.ry(self.input[0], all_qubits)

			self._circuit.measure_all()
			# ---------------------------

			self.backend = backend
			self.shots = shots

		def run(self,x):

			params = dict(zip(self.input, x))
			c1 = self._circuit.assign_parameters(params)

			job = self.backend.run(c1,shots=self.shots)
			result = job.result().get_counts()

			counts = np.array(list(result.values()))
			states = np.array(list(result.keys())).astype(float)

			# Compute probabilities for each state
			probabilities = counts / self.shots
			# Get state expectation
			expectation = np.sum(states * probabilities)

			return expectation

接下来就是使用vqnet定义模型以及训练流程了，使用 ``QiskitLayer`` 把qiskit线路加入vqnet的模型中。

.. code-block::

	#define qiskit circuits class
	circuit = QISKIT_VQC(1, simulator, 100)

	class Net(Module):
		def __init__(self):
			super(Net, self).__init__()
		
			self.conv1 = Conv2D(input_channels=1, output_channels=6, kernel_size=(5, 5), stride=(1, 1), padding="valid")
			self.maxpool1 = MaxPool2D([2, 2], [2, 2], padding="valid")
			self.conv2 = Conv2D(input_channels=6, output_channels=16, kernel_size=(5, 5), stride=(1, 1), padding="valid")
			self.maxpool2 = MaxPool2D([2, 2], [2, 2], padding="valid")
			self.fc1 = Linear(input_channels=256, output_channels=64)
			self.fc2 = Linear(input_channels=64, output_channels=1)
			self.hybrid = QiskitLayer(circuit,0)
			self.fc3 = Linear(input_channels=1, output_channels=2)

		def forward(self, x):
			x = F.ReLu()(self.conv1(x))  # 1 6 24 24
			x = self.maxpool1(x)
			x = F.ReLu()(self.conv2(x))  # 1 16 8 8
			x = self.maxpool2(x)
			x = tensor.flatten(x, 1)  # view(1, -1)  # 1 256
			x = F.ReLu()(self.fc1(x))  # 1 64
			x = self.fc2(x)    # 1 1
			x = self.hybrid(x)
			x = self.fc3(x)
			return x



    def load_mnist(dataset="training_data", digits=np.arange(2), path="./"):         # 下载数据
        import os, struct
        from array import array as pyarray
        download_mnist(path)
        if dataset == "training_data":
            fname_image = os.path.join(path, 'train-images-idx3-ubyte').replace('\\', '/')
            fname_label = os.path.join(path, 'train-labels-idx1-ubyte').replace('\\', '/')
        elif dataset == "testing_data":
            fname_image = os.path.join(path, 't10k-images-idx3-ubyte').replace('\\', '/')
            fname_label = os.path.join(path, 't10k-labels-idx1-ubyte').replace('\\', '/')
		else:
			raise ValueError("dataset must be 'training_data' or 'testing_data'")

		flbl = open(fname_label, 'rb')
		magic_nr, size = struct.unpack(">II", flbl.read(8))
		lbl = pyarray("b", flbl.read())
		flbl.close()

		fimg = open(fname_image, 'rb')
		magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
		img = pyarray("B", fimg.read())
		fimg.close()

		ind = [k for k in range(size) if lbl[k] in digits]
		N = len(ind)
		images = np.zeros((N, rows, cols))
		labels = np.zeros((N, 1), dtype=int)
		for i in range(len(ind)):
			images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
			labels[i] = lbl[ind[i]]

		return images, labels

	def data_select(train_num, test_num):
		x_train, y_train = load_mnist("training_data")  # 下载训练数据
		x_test, y_test = load_mnist("testing_data")

		# Train Leaving only labels 0 and 1
		idx_train = np.append(np.where(y_train == 0)[0][:train_num],
						np.where(y_train == 1)[0][:train_num])

		x_train = x_train[idx_train]
		y_train = y_train[idx_train]
		x_train = x_train / 255
		y_train = np.eye(2)[y_train].reshape(-1, 2)

		# Test Leaving only labels 0 and 1
		idx_test = np.append(np.where(y_test == 0)[0][:test_num],
						np.where(y_test == 1)[0][:test_num])

		x_test = x_test[idx_test]
		y_test = y_test[idx_test]
		x_test = x_test / 255
		y_test = np.eye(2)[y_test].reshape(-1, 2)
		return x_train, y_train, x_test, y_test

	if __name__=="__main__":
		x_train, y_train, x_test, y_test = data_select(100, 50)

		model = Net()
		optimizer = Adam(model.parameters(), lr=0.001)
		loss_func = CategoricalCrossEntropy()

		epochs = 20
		loss_list = []
		model.train()
		start = time.time()
		start_init = datetime.datetime.now()
		for epoch in range(1, epochs):
			total_loss = []
			for x, y in data_generator(x_train, y_train, batch_size=1, shuffle=True):
				start_time = datetime.datetime.now()
				x = x.reshape(-1, 1, 28, 28)

				optimizer.zero_grad()
				# Forward pass
				output = model(x)

				# Calculating loss
				loss = loss_func(y, output)  # target output
				loss_np = np.array(loss.data)
				# Backward pass
				loss.backward()
				# Optimize the weights
				optimizer._step()
				end_time = datetime.datetime.now()
				total_loss.append(loss_np)

			loss_list.append(np.sum(total_loss) / len(total_loss))

			print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))

		end_init = datetime.datetime.now()
		print("run all epochs time {}".format((end_init - start_init).seconds))
		end = time.time()
		print(end - start)

		plt.plot(loss_list)
		plt.title('VQNet NN Training')
		plt.xlabel('Training Iterations')
		plt.ylabel('Loss')
		plt.show()


		model.eval()
		correct = 0
		total_loss1 = []
		n_eval = 0
		for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
			x = x.reshape(-1, 1, 28, 28)
			output = model(x)
			loss = loss_func(y, output)
			loss_data = np.array(loss.data)
			np_output = np.array(output.data, copy=False)
			mask = (np_output.argmax(1) == y.argmax(1))
			correct += np.sum(np.array(mask))
			n_eval += 1
			# total_loss1.append(loss_data)
		print(f"Eval Accuracy: {correct / n_eval}")


		n_samples_show = 6
		count = 0
		fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
		model.eval()
		for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
			if count == n_samples_show:
				break
			x = x.reshape(-1, 1, 28, 28)
			output = model(x)
			pred = QTensor.argmax(output, [1])
			axes[count].imshow(x[0].squeeze(), cmap='gray')
			axes[count].set_xticks([])
			axes[count].set_yticks([])
			axes[count].set_title('Predicted {}'.format(np.array(pred.data)))
			count += 1
		plt.show()


.. note:: 以上示例在如下qiskit版本中验证qiskit: 0.31.0 , qiskit-aer: 0.9.1 , qiskit-aqua: 0.9.5 , qiskit-ibmq-provider: 0.17.0 , qiskit-ignis: 0.6.0 , qiskit-terra: 0.18.3 。

训练集上Loss情况

.. figure:: ./images/qiskit_hqcnn_train_loss.png

测试集上运行分类情况

.. figure:: ./images/qiskit_eval_test.png
     

量子逻辑门
----------------------------------

处理量子比特的方式就是量子逻辑门。 使用量子逻辑门，我们有意识的使量子态发生演化。量子逻辑门是构成量子算法的基础。


基本量子逻辑门
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在VQNet中，我们使用本源量子自研的 `pyQPanda <https://pyqpanda-toturial.readthedocs.io/zh/latest/>`_ 的各个逻辑门搭建量子线路，进行量子模拟。
当前pyQPanda支持的逻辑门可参考pyQPanda `量子逻辑门 <https://pyqpanda-toturial.readthedocs.io/zh/latest/>`_ 部分的定义。
此外VQNet还封装了部分在量子机器学习中常用的量子逻辑门组合：


BasicEmbeddingCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.template.BasicEmbeddingCircuit(input_feat,qlist)

    Encodes n binary features into a basis state of n qubits.

    For example, for ``features=([0, 1, 1])``, the quantum system will be
    prepared in state :math:`|011 \rangle`.

    :param input_feat: binary input of shape ``(n)``
    :param qlist: qubit lists that the template acts on
    :return: quantum circuits

    Example::
        
        import numpy as np
        import pyqpanda as pq
        from pyvqnet.qnn.template import BasicEmbeddingCircuit
        input_feat = np.array([0,1,1]).reshape([3])
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)

        qlist = m_machine.qAlloc_many(3)
        circuit = BasicEmbeddingCircuit(input_feat,qlist)
        print(circuit)

        #           ┌─┐
        # q_0:  |0>─┤X├
        #           ├─┤
        # q_1:  |0>─┤X├
        #           └─┘

AngleEmbeddingCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.template.AngleEmbeddingCircuit(input_feat,qubits,rotation:str='X')

    Encodes :math:`N` features into the rotation angles of :math:`n` qubits, where :math:`N \leq n`.

    The rotations can be chosen as either : 'X' , 'Y' , 'Z', as defined by the ``rotation`` parameter:

    * ``rotation='X'`` uses the features as angles of RX rotations

    * ``rotation='Y'`` uses the features as angles of RY rotations

    * ``rotation='Z'`` uses the features as angles of RZ rotations

    The length of ``features`` has to be smaller or equal to the number of qubits. If there are fewer entries in
    ``features`` than qubits, the circuit does not Applies the remaining rotation gates.
    
    :param input_feat: numpy array which represents paramters
    :param qubits: qubits allocated by pyQPanda
    :param rotation: use what rotation ,default 'X'
    :return: quantum circuits

    Example::

        import numpy as np
        import pyqpanda as pq
        from pyvqnet.qnn.template import AngleEmbeddingCircuit
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        m_qlist = m_machine.qAlloc_many(2)
        m_clist = m_machine.cAlloc_many(2)
        m_prog = pq.QProg()

        input_feat = np.array([2.2, 1])
        C = AngleEmbeddingCircuit(input_feat,m_qlist,'X')
        print(C)
        C = AngleEmbeddingCircuit(input_feat,m_qlist,'Y')
        print(C)
        C = AngleEmbeddingCircuit(input_feat,m_qlist,'Z')
        print(C)
        pq.destroy_quantum_machine(m_machine)

        #           ┌────────────┐
        # q_0:  |0>─┤RX(2.200000)├
        #           ├────────────┤
        # q_1:  |0>─┤RX(1.000000)├
        #           └────────────┘



        #           ┌────────────┐
        # q_0:  |0>─┤RY(2.200000)├
        #           ├────────────┤
        # q_1:  |0>─┤RY(1.000000)├
        #           └────────────┘



        #           ┌────────────┐
        # q_0:  |0>─┤RZ(2.200000)├
        #           ├────────────┤
        # q_1:  |0>─┤RZ(1.000000)├
        #           └────────────┘

AmplitudeEmbeddingCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.template.AmplitudeEmbeddingCircuit(input_feat,qubits)

	Encodes :math:`2^n` features into the amplitude vector of :math:`n` qubits.To represent a valid quantum state vector, the L2-norm of ``features`` must be one.

    :param input_feat: numpy array which represents paramters
    :param qubits: qubits list allocated by pyQPanda
    :return: quantum circuits

    Example::

        import numpy as np
        import pyqpanda as pq
        from pyvqnet.qnn.template import AmplitudeEmbeddingCircuit
        input_feat = np.array([2.2, 1, 4.5, 3.7])
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        m_qlist = m_machine.qAlloc_many(2)
        m_clist = m_machine.cAlloc_many(2)
        m_prog = pq.QProg()
        cir = AmplitudeEmbeddingCircuit(input_feat,m_qlist)
        print(cir)
        pq.destroy_quantum_machine(m_machine)

        #                              ┌────────────┐     ┌────────────┐
        # q_0:  |0>─────────────── ─── ┤RY(0.853255)├ ─── ┤RY(1.376290)├
        #           ┌────────────┐ ┌─┐ └──────┬─────┘ ┌─┐ └──────┬─────┘
        # q_1:  |0>─┤RY(2.355174)├ ┤X├ ───────■────── ┤X├ ───────■──────
        #           └────────────┘ └─┘                └─┘

IQPEmbeddingCircuits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.template.IQPEmbeddingCircuits(input_feat,qubits,rep:int = 1)

	Encodes :math:`n` features into :math:`n` qubits using diagonal gates of an IQP circuit.

    The embedding was proposed by `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_.

    The basic IQP circuit can be repeated by specifying ``n_repeats``. 

    :param input_feat: numpy array which represents paramters
    :param qubits: qubits list allocated by pyQPanda
    :param rep: repeat circuits block
    :return: quantum circuits

    Example::

        import numpy as np
        import pyqpanda as pq
        from pyvqnet.qnn.template import IQPEmbeddingCircuits
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        input_feat = np.arange(1,100)
        qlist = m_machine.qAlloc_many(3)
        circuit = IQPEmbeddingCircuits(input_feat,qlist,rep = 1)
        print(circuit)

        #           ┌─┐ ┌────────────┐
        # q_0:  |0>─┤H├ ┤RZ(1.000000)├ ───■── ────────────── ───■── ───■── ────────────── ───■── ────── ────────────── ──────
        #           ├─┤ ├────────────┤ ┌──┴─┐ ┌────────────┐ ┌──┴─┐    │                     │
        # q_1:  |0>─┤H├ ┤RZ(2.000000)├ ┤CNOT├ ┤RZ(2.000000)├ ┤CNOT├ ───┼── ────────────── ───┼── ───■── ────────────── ───■──
        #           ├─┤ ├────────────┤ └────┘ └────────────┘ └────┘ ┌──┴─┐ ┌────────────┐ ┌──┴─┐ ┌──┴─┐ ┌────────────┐ ┌──┴─┐
        # q_2:  |0>─┤H├ ┤RZ(3.000000)├ ────── ────────────── ────── ┤CNOT├ ┤RZ(3.000000)├ ┤CNOT├ ┤CNOT├ ┤RZ(3.000000)├ ┤CNOT├
        #           └─┘ └────────────┘                              └────┘ └────────────┘ └────┘ └────┘ └────────────┘ └────┘


RotCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.template.RotCircuit(para,qubits)

	Arbitrary single qubit rotation.Number of qlist should be 1,and number of parameters should be 3

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.


    :param para: numpy array which represents paramters [\phi, \theta, \omega]
    :param qubits: qubits allocated by pyQPanda,only accepted single qubits.
    :return: quantum circuits

    Example::

        import numpy as np
        import pyqpanda as pq
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.template import RotCircuit
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        m_clist = m_machine.cAlloc_many(2)
        m_prog = pq.QProg()
        m_qlist = m_machine.qAlloc_many(1)
        param = np.array([3,4,5])
        c = RotCircuit(QTensor(param),m_qlist)
        print(c)
        pq.destroy_quantum_machine(m_machine)

        #           ┌────────────┐ ┌────────────┐ ┌────────────┐
        # q_0:  |0>─┤RZ(5.000000)├ ┤RY(4.000000)├ ┤RZ(3.000000)├
        #           └────────────┘ └────────────┘ └────────────┘


CRotCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.template.CRotCircuit(para,control_qubits,rot_qubits)

	The controlled-Rot operator	

    .. math:: CR(\phi, \theta, \omega) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
            0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.
    
    :param para: numpy array which represents paramters [\phi, \theta, \omega]
    :param control_qubits: control qubit allocated by pyQPanda,number of qubit should be 1.
    :param rot_qubits: Rot qubit allocated by pyQPanda,number of qubit should be 1.
    :return: quantum circuits

    Example::

        import numpy as np
        import pyqpanda as pq
        from pyvqnet.tensor import QTensor
        from pyvqnet.qnn.template import CRotCircuit
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        m_clist = m_machine.cAlloc_many(2)
        m_prog = pq.QProg()
        m_qlist = m_machine.qAlloc_many(1)
        param = np.array([3,4,5])
        control_qlist = m_machine.qAlloc_many(1)
        c = CRotCircuit(QTensor(param),control_qlist,m_qlist)
        print(c)
        pq.destroy_quantum_machine(m_machine)

        #           ┌────────────┐ ┌────────────┐ ┌────────────┐
        # q_0:  |0>─┤RZ(5.000000)├ ┤RY(4.000000)├ ┤RZ(3.000000)├
        #           └──────┬─────┘ └──────┬─────┘ └──────┬─────┘
        # q_1:  |0>────────■────── ───────■────── ───────■──────


CSWAPcircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.template.CSWAPcircuit(qubits)

    The controlled-swap circuit

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

    .. note:: The first qubits provided corresponds to the **control qubit**.

    :param qubits: list of qubits allocated by pyQPanda the first qubits is control qubit. length of qlists have to be 3.
    :return: quantum circuits

    Example::

        from pyvqnet.qnn.template import CSWAPcircuit
        import pyqpanda as pq
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)

        m_qlist = m_machine.qAlloc_many(3)

        c = CSWAPcircuit([m_qlist[1],m_qlist[2],m_qlist[0]])
        print(c)
        pq.destroy_quantum_machine(m_machine)

        # q_0:  |0>─X─
        #           │
        # q_1:  |0>─■─
        #           │
        # q_2:  |0>─X─

对量子线路进行测量
----------------------------------

expval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.measure.expval(machine,prog,pauli_str_dict,qubits)

	Expectation value of the supplied Hamiltonian observables 
    
    if the observables are :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I`,
    then ``Hamiltonian`` ``dict`` would be ``{{'Z0, X1':0.7} ,{'Z1':0.2}}`` .

    expval api only supports on QPanda CPUQVM now.Please checks  https://pyqpanda-toturial.readthedocs.io/zh/latest/index.html for alternative api.
    
    :param machine: machine created by qpanda
    :param prog: quantum program created by qpanda
    :param pauli_str_dict: Hamiltonian observables 
    :param qubits: qubits allocated by pyQPanda
    :return: expectation
               

    Example::

        import pyqpanda as pq
        from pyvqnet.qnn.measure import expval
        input = [0.56, 0.1]
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        m_prog = pq.QProg()
        m_qlist = m_machine.qAlloc_many(3)
        cir = pq.QCircuit()
        cir.insert(pq.RZ(m_qlist[0],input[0]))
        cir.insert(pq.CNOT(m_qlist[0],m_qlist[1]))
        cir.insert(pq.RY(m_qlist[1],input[1]))
        cir.insert(pq.CNOT(m_qlist[0],m_qlist[2]))
        m_prog.insert(cir)    
        pauli_dict  = {'Z0 X1':10,'Y2':-0.543}
        exp2 = expval(m_machine,m_prog,pauli_dict,m_qlist)
        print(exp2)
        pq.destroy_quantum_machine(m_machine)
        #0.9983341664682731

QuantumMeasure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.measure.QuantumMeasure(measure_qubits:list,prog,machine,qubits,slots:int = 1000)
	
    Calculates circuits quantum measurement. Return the normalized result of the measurements obtained by the Monte Carlo method.
    
    Please checks  https://pyqpanda-toturial.readthedocs.io/zh/latest/Measure.html?highlight=measure_all for alternative api.
    
    QuantumMeasure api only supports on QPanda CPUQVM or QCloud now.

    :param measure_qubits: list contains measure qubits index.
    :param prog: quantum program from qpanda
    :param machine: quantum virtual machine allocated by pyQPanda
    :param qubits: qubits allocated by pyQPanda
    :param slots: measure time,default 1000
    :return: returns the normalized result of the measurements obtained by the Monte Carlo method.

    Example::

        from pyvqnet.qnn.measure import QuantumMeasure
        import pyqpanda as pq
        input = [0.56,0.1]
        measure_qubits = [0,2]
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        m_prog = pq.QProg()
        m_qlist = m_machine.qAlloc_many(3)

        cir = pq.QCircuit()
        cir.insert(pq.RZ(m_qlist[0],input[0]))
        cir.insert(pq.CNOT(m_qlist[0],m_qlist[1]))
        cir.insert(pq.RY(m_qlist[1],input[1]))
        cir.insert(pq.CNOT(m_qlist[0],m_qlist[2]))
        cir.insert(pq.H(m_qlist[0]))
        cir.insert(pq.H(m_qlist[1]))
        cir.insert(pq.H(m_qlist[2]))

        m_prog.insert(cir)    
        rlt_quant = QuantumMeasure(measure_qubits,m_prog,m_machine,m_qlist)
        print(rlt_quant)
        #[0.25, 0.264, 0.257, 0.229]

ProbsMeasure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: pyvqnet.qnn.measure.ProbsMeasure(measure_qubits:list,prog,machine,qubits)

	Calculates circuits probabilities measurement.
    
    Please checks https://pyqpanda-toturial.readthedocs.io/zh/latest/PMeasure.html for alternative api.

    ProbsMeasure api only supports on QPanda CPUQVM or QCloud now.

    :param measure_qubits: list contains measure qubits index.
    :param prog: quantum program from qpanda
    :param machine: quantum virtual machine allocated by pyQPanda
    :param qubits: qubit allocated by pyQPanda
    :return: prob of measure qubits in lexicographic order.

    Example::

        from pyvqnet.qnn.measure import ProbsMeasure
        import pyqpanda as pq

        input = [0.56,0.1]
        measure_qubits = [0,2]
        m_machine = pq.init_quantum_machine(pq.QMachineType.CPU)
        m_prog = pq.QProg()
        m_qlist = m_machine.qAlloc_many(3)

        cir = pq.QCircuit()
        cir.insert(pq.RZ(m_qlist[0],input[0]))
        cir.insert(pq.CNOT(m_qlist[0],m_qlist[1]))
        cir.insert(pq.RY(m_qlist[1],input[1]))
        cir.insert(pq.CNOT(m_qlist[0],m_qlist[2]))
        cir.insert(pq.H(m_qlist[0]))
        cir.insert(pq.H(m_qlist[1]))
        cir.insert(pq.H(m_qlist[2]))

        m_prog.insert(cir)    

        rlt_prob = ProbsMeasure([0,2],m_prog,m_machine,m_qlist)
        print(rlt_prob)
        #[0.2499999999999947, 0.2499999999999947, 0.2499999999999947, 0.2499999999999947]




