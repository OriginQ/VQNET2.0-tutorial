量子机器学习模块
==================================

量子计算层
----------------------------------

.. _QuantumLayer:

QuantumLayer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

QuantumLayer是一个支持量子含参线路作为参数的自动求导模块的封装类。用户定义一个函数作为参数 ``qprog_with_meansure`` ，该函数需要包含pyQPanda定义的量子线路：一般包含量子线路的编码线路，演化线路和测量操作。
该类可以嵌入量子经典混合机器学习模型，通过经典的梯度下降法，使得量子经典混合模型的目标函数或损失函数最小。
用户可通过参数 ``diff_method`` 指定 ``QuantumLayer`` 层中量子线路参数的梯度计算方式，``QuantumLayer`` 当前支持有限差分法 ``finite_diff`` 以及 ``parameter-shift`` 方法。

有限差分法是估算函数梯度最传统和最常用的数值方法之一。主要思想是用差分代替偏导数：

.. math::

    f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}


parameter-shift方法，我们使用如下目标函数：

.. math:: O(\theta)=\left\langle 0\left|U^{\dagger}(\theta) H U(\theta)\right| 0\right\rangle

理论上可以通过 ``parameter-shift`` 这一更精确的方法计算量子线路中参数对哈密顿量的梯度：

.. math::

    \nabla O(\theta)=
    \frac{1}{2}\left[O\left(\theta+\frac{\pi}{2}\right)-O\left(\theta-\frac{\pi}{2}\right)\right]

.. autoclass:: pyvqnet.qnn.quantumlayer.QuantumLayer


QuantumLayerV2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

如您更加熟悉pyQPanda语法，可以使用QuantumLayerV2，自定义量子比特 ``qubits`` ,经典比特 ``cubits`` ,后端模拟器 ``machine`` 加入QuantumLayerV2的参数 ``qprog_with_meansure`` 函数中。

.. autoclass:: pyvqnet.qnn.quantumlayer.QuantumLayerV2

NoiseQuantumLayer
^^^^^^^^^^^^^^^^^^^

在真实的量子计算机中，受制于量子比特自身的物理特性，常常存在不可避免的计算误差。为了能在量子虚拟机中更好的模拟这种误差，VQNet同样支持含噪声量子虚拟机。含噪声量子虚拟机的模拟更贴近真实的量子计算机，我们可以自定义支持的逻辑门类型，自定义逻辑门支持的噪声模型。
现有可支持的量子噪声模型依据QPanda中定义，具体参考链接 `QPANDA2 <https://pyqpanda-toturial.readthedocs.io/zh/latest/NoiseQVM.html>`_ 中的介绍。

使用 NoiseQuantumLayer 定义一个量子线路自动微分类。用户定义一个函数作为参数 ``qprog_with_meansure`` ，该函数需要包含pyQPanda定义的量子线路，同样需要传入一个参数 ``noise_set_config``,使用pyQPanda接口，设置噪声模型。

.. autoclass:: pyvqnet.qnn.quantumlayer.NoiseQuantumLayer

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
其中定义相应的量子线路逻辑门和测量函数即可基于pyQPanda的VariationalQuantumCircuit，进行机器学习模型的构建。

在 `VQC_wrapper` 中，用户使用普通逻辑门函数 `build_common_circuits` 构建模型中线路结构变化的子线路，使用VQG在 `build_vqc_circuits` 构建结构不变，参数变化的子线路。使用
`run` 函数定义线路运行方式以及测量。

.. autoclass:: pyvqnet.qnn.quantumlayer.VQC_wrapper

将该实例化对象 `VQC_wrapper` 作为参数传入 `VQCLayer`

.. autoclass:: pyvqnet.qnn.quantumlayer.VQCLayer



Qconv
^^^^^^^^^^^^^^^^^^^^^^^^

Qconv是一种量子卷积算法接口。
量子卷积操作采用量子线路对经典数据进行卷积操作，其无需计算乘法和加法操作，只需将数据编码到量子态，然后通过量子线路进行衍化操作和测量得到最终的卷积结果。
根据卷积核的范围中的输入数据数量申请相同数量的量子比特，然后构建量子线路进行计算。

.. image:: ./images/qcnn.png

其量子线路由每个qubit上首先插入 :math:`RY` , :math:`RZ` 门进行编码，接着在任意两个qubit上使用 :math:`Z` 以及 :math:`U3` 进行信息纠缠和交换。下图为4qubits的例子

.. image:: ./images/qcnn_cir.png

.. autoclass:: pyvqnet.qnn.qcnn.qconv.QConv

QLinear
^^^^^^^^^^

QLinear 实现了一种量子全连接算法。首先将数据编码到量子态，然后通过量子线路进行衍化操作和测量得到最终的全连接结果。

.. image:: ./images/qlinear_cir.png

.. autoclass:: pyvqnet.qnn.qlinear.qlinear.QLinear


Compatiblelayer
^^^^^^^^^^^^^^^^^

VQNet不仅可以支持 ``QPANDA`` 的量子线路，同时可以支持其他量子计算框架(例如 ``Cirq``, ``Qiskit`` 等）的量子线路作为VQNet混合量子经典优化的量子计算部分。
VQNet提供了自动微分的量子线路运算接口 ``Compatiblelayer`` 。构建 ``Compatiblelayer`` 的参数中需要传入一个类，其中定义了第三方库量子线路 ，以及其运行和测量函数 ``run`` 。
使用 ``Compatiblelayer`` ,量子线路的输入以及参数的自动微分就可交由VQNet进行实现。
VQNet提供了一个示例使用qiskit线路: :ref:`my-reference-label`.

.. autoclass:: pyvqnet.qnn.utils.compatible_layer.Compatiblelayer


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

.. autofunction:: pyvqnet.qnn.template.BasicEmbeddingCircuit

AngleEmbeddingCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.template.AngleEmbeddingCircuit

AmplitudeEmbeddingCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.template.AmplitudeEmbeddingCircuit

IQPEmbeddingCircuits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.template.IQPEmbeddingCircuits

RotCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.template.RotCircuit

CRotCircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.template.CRotCircuit

CSWAPcircuit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.template.CSWAPcircuit

对量子线路进行测量
----------------------------------

expval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.measure.expval

QuantumMeasure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.measure.QuantumMeasure

ProbsMeasure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyvqnet.qnn.measure.ProbsMeasure




