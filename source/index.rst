.. VQNet documentation master file, created by
   sphinx-quickstart on Tue Jul 27 15:25:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VQNet
=================================


**一种功能齐全，运行高效的量子软件开发工具包**

VQNet是由本源量子开发的量子机器学习计算框架，它可以用于构建、运行和优化量子算法。
本使用文档是VQNet的api以及示例文档。英文版本可见： `VQNet API DOC. <https://vqnet20-tutorial-en.readthedocs.io/en/latest/>`_ 


**什么是量子机器学习？**

量子机器学习是一个探索量子计算和机器学习思想相互作用的研究领域。
例如，我们可能想知道量子计算机是否可以加快训练或评估机器学习模型所需的时间。另一方面，我们可以利用机器学习技术来帮助我们发现量子纠错码、估计量子系统的特性或开发新的量子算法。

**量子计算机作为人工智能加速器**
 
机器可以学习的极限一直由我们运行算法的计算机硬件定义——例如，现代神经网络深度学习的成功是由并行 GPU 集群实现的。
量子机器学习通过一种全新类型的计算设备——量子计算机，扩展了机器学习的硬件池。量子计算机的信息处理依赖于被称为量子理论的完全不同的物理定律。
在现代观点中，量子计算机可以像神经网络一样使用和训练。我们可以系统地调整例如电磁场强度或激光脉冲频率这种物理参数来构建可变量子线路。
使用这种可变量子线路电路将图像编码为设备的物理状态并进行测量，可用于对图像内容进行分类。

**更大的视野：可微分编程**

但我们的目标不仅仅是使用量子计算机来解决机器学习问题。量子电路是可微的，量子计算机本身可以计算在给定任务中变得更好所需的控制参数的变化。
可微分编程是深度学习的基础，可微分编程不仅仅是深度学习：它是一种编程范式，其中算法不是手工编码的，而是自动学习的。`VQNet` 同样基于可微分编程的思想实现。
训练量子计算机的意义比量子机器学习更加深远。可训练的量子电路可用于其他领域，如量子化学或量子优化。它可以提升多种应用，例如量子算法的设计、量子纠错方案的发现以及物理系统的理解。

**VQNet特点**

•	统一性。支持混合量子和经典神经网络模型的训练，支持量子计算机与经典计算机等多种硬件上的模型运行。	 
•	实用性。使用python作为前端语言，接口友好、支持自动微分。
•	高效性。设计统一架构，使用本源量子的QPanda量子计算库，以及自带的经典计算层提高量子机器学习的效率。
•	应用丰富。提供丰富量子机器学习示例快速上手。


.. toctree::
    :caption: 安装介绍
    :maxdepth: 2

    rst/install.rst

.. toctree::
    :caption: 上手实例
    :maxdepth: 2


    rst/qml_demo.rst

.. toctree::
    :caption: 接口介绍
    :maxdepth: 2

    rst/QTensor.rst
    rst/nn.rst
    rst/qnn.rst
    rst/utils.rst



.. toctree::
    :caption: 其他
    :maxdepth: 2

    rst/xtensor.rst
    rst/xtensor_nn.rst
    rst/FAQ.rst
    rst/CHANGELOG.rst





