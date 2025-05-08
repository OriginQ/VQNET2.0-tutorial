.. VQNet documentation master file, created by
   sphinx-quickstart on Tue Jul 27 15:25:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

VQNet
=================================


 
VQNet 是由本源量子自主研发的量子机器学习计算框架，专注于为用户提供一个灵活、高效的开发平台，支持构建和运行混合量子-经典机器学习算法，并调用本源量子芯片进行变分量子线路训练。
本使用文档是VQNet的api以及示例文档。英文版本可见:  `VQNet API DOC. <https://vqnet20-tutorial-en.readthedocs.io/en/latest/>`_ 


VQNet 的核心特点
-----------------

多平台兼容性与跨环境支持
~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet 支持用户在多种硬件和操作系统环境中进行量子机器学习的研究与开发。无论是使用 CPU、GPU 进行量子计算模拟，还是通过 本源量子云服务 调用真实量子芯片，VQNet 都能提供无缝支持。目前，VQNet 已兼容 Windows、Linux 和 macOS 系统的python3.9，python3.10，python3.11版本。

完善的接口设计与易用性
~~~~~~~~~~~~~~~~~~~~~~~

VQNet 采用 Python 作为前端语言，提供类似于 PyTorch 的函数接口，并可自由选择多种计算后端实现经典量子机器学习模型的自动微分功能。框架内置了：100+ 常用 Tensor 计算接口、100+ 量子变分线路计算接口、50+ 经典神经网络接口。这些接口覆盖了从经典机器学习到量子机器学习的完整开发流程，并且将持续更新。

高效的计算性能与扩展能力
~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet 在性能优化方面表现出色：

- **真实量子芯片实验支持**：对于需要真实量子芯片实验的用户，VQNet 集成了 本源 pyQPanda 接口，并结合本源司南的高效调度能力，可实现快速的量子线路模拟计算和真实芯片运行。
- **本地计算优化**：对于本地计算需求，VQNet 提供基于 CPU 或 GPU 的量子机器学习编程接口，利用自动微分技术进行量子变分线路梯度计算，相比传统参数漂移法（如 Qiskit）速度提升明显。
- **分布式计算支持**：VQNet 支持基于 MPI 的分布式计算，可在多个节点上实现训练大规模混合量子-经典神经网络模型的功能。

丰富的应用场景与示例支持
~~~~~~~~~~~~~~~~~~~~~~~~~

VQNet 不仅是一个强大的开发工具，还在公司内部多个项目中得到了广泛应用，包括 电力优化、医疗数据分析、图像处理 等领域。为了帮助用户快速上手，VQNet 在官网和 API 在线文档中提供了涵盖从基础教程到高级应用的多种场景。这些资源让用户能够轻松理解如何利用 VQNet 解决实际问题，并快速构建自己的量子机器学习应用。



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
    :caption: 使用pyqpanda的量子神经网络接口
    :maxdepth: 2

    rst/qnn.rst
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

    rst/torch_api.rst
    rst/FAQ.rst
    rst/CHANGELOG.rst





