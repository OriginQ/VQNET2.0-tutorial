.. VQNet documentation master file, created by
   sphinx-quickstart on Tue Jul 27 15:25:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VQNet
=================================


 
VQNet 是本源量子自主研发的量子机器学习计算框架，以参数化量子线路（Parameterized Quantum Circuit, PQC）为核心计算原语，将其作为可微算子嵌入经典神经网络，依托自动微分机制实现混合量子-经典模型的端到端构建与训练，并支持调用本源量子计算机与量子云服务完成线路模拟与真实芯片实验。框架提供覆盖张量计算、经典神经网络、量子线路模拟与自动微分、分布式训练以及真实硬件部署的完整接口体系。

本使用文档是VQNet的 API 以及示例文档。英文版本可见:  `VQNet API DOC. <https://vqnet20-tutorial-en.readthedocs.io/en/latest/>`_


VQNet 的核心特点
-----------------

多平台兼容性与跨环境支持
~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet 支持用户在多种硬件和操作系统环境中进行量子与经典机器学习的研究与开发。无论是使用 CPU、GPU 进行量子计算模拟，还是通过本源量子云服务调用真实量子芯片，VQNet 都能提供无缝支持。目前，VQNet 已兼容 Windows (x86)、Linux (x86) 和 macOS (ARM) 系统的 Python 3.10～Python 3.14 版本。

完善的接口设计与易用性
~~~~~~~~~~~~~~~~~~~~~~~

VQNet 采用 Python 作为前端语言，提供类似于主流神经网络训练框架的编程接口，并通过 ``pyvqnet.backends.set_backend`` 在 ``pyvqnet`` （原生）与 ``torch`` 两种计算后端之间选择。两种后端的接口相互独立，同一模型需在单一后端下运行，用户可依据生态需求选择。这些接口覆盖了从经典机器学习到量子机器学习的完整开发流程，并且将持续更新。

- **原生后端**：以 ``pyvqnet`` 计算后端运行。张量库 ``pyvqnet.tensor`` 提供 100+ 常用计算接口并支持自动微分；神经网络模块 ``pyvqnet.nn`` 涵盖卷积、池化、循环神经网络、Transformer 注意力、多种归一化与优化器等 50+ 接口；量子线路模块 ``pyvqnet.qnn.vqc`` 提供状态向量模拟 ``QMachine`` 、张量网络后端 ``tn.native`` ，以及 QLSTM、QRNN、QMLP、Quanvolution 等量子-经典融合层；大模型算子（RoPE、SwiGLU、融合 MoE、缩放 Softmax、top-k/top-p/min-p 采样等）亦由原生后端提供。
- **torch 后端**：面向需要与 PyTorch 生态协作的场景，以 ``torch`` 计算后端运行，提供与原生接口对应的 torch 版本——经典层 ``pyvqnet.nn.torch`` （ ``TorchModule`` 、 ``Linear`` 、 ``Conv2D`` 等），量子线路 ``pyvqnet.qnn.vqc.sv.torch`` 与张量网络 ``pyvqnet.qnn.vqc.tn.torch`` ，可与 ``torch.nn`` 模块互相嵌套并参与 ``torch`` 自动微分；大模型微调损失族 ``pyvqnet.torch.trl`` （SFT、DPO、PPO、GRPO、Reward）同样基于该后端，可配合 peft/LlamaFactory 实现基于量子线路的大模型微调。

高效的计算性能与扩展能力
~~~~~~~~~~~~~~~~~~~~~~~~~

- **双量子模拟后端**：提供基于状态向量演化的 ``QMachine`` 接口（50+ 量子逻辑门、``MeasureAll`` 、 ``Probability`` 、 ``Samples`` 等测量族、 ``QuantumLayerAdjoint`` 伴随梯度接口，以及 UCCSD、QSVT、硬件高效拟设等线路模板）；同时提供面向较大比特规模线路的张量网络/矩阵乘积态后端 ``pyvqnet.qnn.vqc.tn`` ，支持 ``native`` 与 ``torch`` 两套实现。框架内置旋转门合并、受控门交换等线路编译优化 pass，并支持 OriginIR 线路导入导出。
- **真实量子芯片实验支持**：对于需要真实量子芯片实验的用户，VQNet 集成了 本源 pyqpanda3 接口，并结合本源司南的高效调度能力，可实现快速的量子线路模拟计算和真实芯片运行；pq3 系列量子层支持异步批量云端提交，打通从模拟到真实硬件的迁移路径。
- **本地计算优化**：对于本地计算需求，VQNet 提供基于 CPU 或 GPU 的量子机器学习编程接口，利用自动微分技术进行量子变分线路梯度计算。相比传统参数移位法，梯度计算效率提升明显，性能测试可见 :ref:`benchmarks` 。针对 RX、RY、RZ、CNOT、测量等高频操作提供融合 CUDA 算子，降低大规模参数化线路的训练与模拟耗时。
- **分布式计算支持**：VQNet 内置基于 gloo/NCCL 集合通信的多进程分布式运行时（ ``vqnetrun`` 启动器），支持数据并行、张量并行（ ``ColumnParallelLinear`` / ``RowParallelLinear`` ）、流水线并行及混合并行策略（ ``ParallelTrainingWrapper`` ），并提供 ``DistributedQMachine`` / ``DistQuantumLayerAdjoint`` 支撑多进程、多节点环境下的量子线路梯度计算与混合量子-经典模型训练。

丰富的应用场景与示例支持
~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet 不仅是一个强大的开发工具，还在公司内部多个项目中得到了广泛应用，包括 电力优化、医疗数据分析、图像处理 等领域。为了帮助用户快速上手，VQNet 在官网和 API 在线文档中提供了涵盖从基础教程（变分量子线路、量子机器学习示例）到高级应用（分布式混合训练、量子大模型微调）的多种场景与示例。这些资源让用户能够轻松理解如何利用 VQNet 解决实际问题，并快速构建自己的量子机器学习应用。



.. toctree::
    :caption: 安装介绍
    :maxdepth: 2

    rst/install.rst

.. toctree::
    :caption: 上手实例
    :maxdepth: 2

    rst/vqc_demo.rst
    rst/qml_demo.rst

.. toctree::
    :caption: 经典神经网络接口介绍
    :maxdepth: 2

    rst/QTensor.rst
    rst/nn.rst
    rst/utils.rst

.. toctree::
    :caption: 使用pyqpanda3的量子神经网络接口
    :maxdepth: 2


    rst/qnn_pq3.rst

.. toctree::
    :caption: 量子神经网络自动微分模拟接口
    :maxdepth: 2

    rst/vqc.rst


.. toctree::
    :caption: 量子大模型微调
    :maxdepth: 2

    rst/llm.rst

.. toctree::
    :caption: 其他
    :maxdepth: 2
    
    rst/vqnet_dist.rst
    rst/torch_api.rst
    rst/FAQ.rst
    rst/CHANGELOG.rst





