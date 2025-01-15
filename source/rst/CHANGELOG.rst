
VQNet Changelog
######################


[v2.16.0] - 2025-1-15
***************************

Added
===================

- 增加使用pyqpanda3进行量子线路计算的接口。


[v2.15.0] - 2024-11-19
***************************

Added
===================

- 增加 `pyvqnet.backends.set_backend()` 接口，在用户安装 `torch` 时候，可使用 `torch` 进行QTensor的矩阵计算，变分量子线路计算，具体见文档 :ref:`torch_api` 。
- 增加 `pyvqnet.nn.torch` 下继承于 `torch.nn.Module` 的神经网络接口，变分量子线路神经接口等，具体见文档 :ref:`torch_api` 。

Changed
===================
- diag接口修改。
- 修改all_gather实现与torch.distributed.all_gather一致。
- 修改 `QTensor` 最大支持30维度数据。
- 修改分布式功能所需 `mpi4py` 需要4.0.1版本以上

Fixed
===================
- 部分随机数实现由于omp无法固定种子。
- 修复分布式启动的部分bug。


[v2.14.0] - 2024-09-30
***************************

Added
===================

- 增加 `VQC_LCU`, `VQC_FABLE`, `VQC_QSVT` 的block-encoding算法, 以及qpanda算法实现 `QPANDA_QSVT`, `QPANDA_LCU`, `QPANDA_FABLE` 接口.
- 增加整数加到量子比特上 `vqc_qft_add_to_register`, 两个量子比特上的数加法 `vqc_qft_add_two_register`，两个量子比特上的数乘法的 `vqc_qft_mul`.
- 增加混合qpanda与vqc的训练模块 `HybirdVQCQpandaQVMLayer`.
- 增加 `einsum`, `moveaxis`, `eigh`, `dignoal` 等接口实现.
- 增加分布式计算中张量并行计算功能 `ColumnParallelLinear`, `RowParallelLinear`.
- 增加分布式计算中Zero stage-1 功能 `ZeroModelInitial`.
- `QuantumBatchAsyncQcloudLayer` 指定 diff_method == "random_coordinate_descent" 时候不会使用PSR而是随机对量子参数选择一个进行梯度计算.

Changed
===================
- 删除了xtensor部分。
- api文档进行部分修改。区分了基于自动微分的量子机器学习示例以及基于qpanda的机器学习示例,区别基于自动微分的量子机器学习接口以及基于qpanda的机器学习示例接口。
- `matmul` 支持1d@1d,2d@1d,1d@2d。
- 增加了一些量子计算层别名: `QpandaQCircuitVQCLayer`` = `QuantumLayer` , `QpandaQCircuitVQCLayerLite` = `QuantumLayerV2`, `QpandaQProgVQCLayer` = `QuantumLayerV3`.

Fixed
===================
- 修改分布式计算功能中底层通信接口 `allreduce`, `allgather`, `reduce`, `broadcast` , 添加对 `core.Tensor` 数据通信支持
- 解决随机数生成的bug。
- 解决了 VQC的 `RXX`, `RYY`, `RZZ`, `RZX` 转换成originIR的错误。


[v2.13.0] - 2024-07-30
***************************

Added
===================

- 增加 `no_grad`, `GroupNorm`, `Interpolate`, `contiguous`, `QuantumLayerV3`, `fuse_model`, `SDPA`, `quantum_fisher` 接口。
- 为解决量子机器学习过程中出现的过拟合问题添加量子Dropout样例.

Changed
===================

- `BatchNorm`, `LayerNorm`, `GroupNorm` 增加affine接口。
- `diag` 接口在2d输入时候现在返回对角线上的1d输出,与torch一致。
- slice,permute等操作会尝试使用view方式返回共享内存的QTensor。
- 所有接口支持非contiguous的输入。
- `Adam` 支持 weight_decay 参数。

Fixed
===================
- 修改 VQC 部分逻辑门分解函数的错误。
- 修复部分函数的内存泄露问题。
- 修复 `QuantumLayerMultiProcess` 不支持GPU输入的问题。
- 修改 `Linear` 的默认参数初始化话方式


[v2.12.0] - 2024-05-01
***************************

Added
===================

- 添加流水线并行PipelineParallelTrainingWrapper接口。
- 添加 `Gelu`, `DropPath`, `binomial`, `adamW` 接口。
- 添加 `QuantumBatchAsyncQcloudLayer` 支持pyqpanda的本地虚拟机模拟计算。
- 添加 xtensor的 `QuantumBatchAsyncQcloudLayer` 支持pyqpanda的本地虚拟机模拟计算以及真机计算。
- 使得QTensor 可以被deepcopy以及pickle。
- 添加分布式计算启动命令 `vqnetrun`, 使用分布式计算接口时使用。
- 添加ES梯度计算方法真机接口 `QuantumBatchAsyncQcloudLayerES` 支持pyqpanda的本地虚拟机模拟计算以及真机计算。
- 添加在分布式计算中支持QTensor的数据通信接口 `allreduce`, `reduce`, `broadcast`, `allgather`, `send`, `recv` 等。

Changed
===================

- 安装包新加入依赖 "Pillow", "hjson", linux系统下安装包添加新依赖 "psutil"。 "cloudpickle"。
- 优化softmax以及tranpose在GPU下运行速度。
- 使用cuda11.8编译。
- 整合了基于cpu、gpu下的分布式计算接口。

Fixed
===================
- 降低Linux-GPU版本启动时候的显存消耗。
- 修复select以及power函数的内存泄露问题。
- 删除了cpu、gpu下基于reduce方法的模型参数以及梯度更新方法 `nccl_average_parameters_reduce`, `nccl_average_grad_reduce`。

[v2.11.0] - 2024-03-01
***************************

Added
===================

- 添加新的 `QNG` （量子自然梯度）API 和演示。
- 添加量子电路优化，例如 `wrapper_single_qubit_op_fuse` , `wrapper_commute_controlled` , `wrapper_merge_rotations` api 和 demo。
- 添加 `CY`, `SparseHamiltonian` , `HermitianExpval` 。
- 添加 `is_csr`、 `is_dense`、 `dense_to_csr` 、 `csr_to_dense` 。
- 添加 `QuantumBatchAsyncQcloudLayer` 支持pyqpanda的QCloud真实芯片计算， `expval_qcloud`。
- 添加基于NCCL的单节点下多GPU分布式计算数据并行模型训练的相关接口实现 `nccl_average_parameters_allreduce`, `nccl_average_parameters_reduce`, `nccl_average_grad_allreduce`, `nccl_average_grad_reduce` 以及控制NCCL初始化以及相关操作的类 `NCCL_api`。
- 添加量子线路进化策略梯度计算方法 `QuantumLayerES` 接口。

Changed
===================

- 将 `VQC_CSWAP` 电路重构为 `CSWAP`。
- 删除旧的 QNG 文档。
- 从 `pyvqnet.qnn.vqc` 中删除函数和类无用的 `num_wires` 参数。
- 重构 `MeasureAll`, `Probability` api。
- 为 `QuantumMeasure` 增加qtype参数。

Fixed
===================
- 将 `QuantumMeasure` 的 slots 改为 shots。

[v2.10.0] - 2023-12-30
***************************

Added
===========
- 增加了pyvqnet.qnn.vqc下的新接口:IsingXX、IsingXY、IsingYY、IsingZZ、SDG、TDG、PhaseShift、MutliRZ、MultiCnot、MultixCnot、ControlledPhaseShift、SingleExcitation、DoubleExcitation、VQC_AllSinglesDoubles,ExpressiveEntanglingAnsatz等；
- 支持adjoint梯度计算的pyvqnet.qnn.vqc.QuantumLayerAdjoint接口;
- 支持originIR与VQC相互转换的功能;
- 支持统计VQC模型中的经典和量子模块信息;
- 增加量子经典神经网络混合模型下的两个案例：基于小样本的量子卷积神经网络模型、用于手写数字识别的量子核函数模型;
- 增加对arm芯片Mac的支持;


[v2.9.0] - 2023-11-15
***************************

Added
===========
- 增加了xtensor接口定义，支持经典神经网络模块自动并行和CPU/GPU多后端，包含对多维数组的常用数学，逻辑，矩阵计算，以及常见的经典神经网络层，优化器等150余个接口。

Changed
===========
- 从本版本开始，版本号从2.0.8 升级为2.9.0。
- 自本版本开始，软件包上传到 https://pypi.originqc.com.cn， 使用 ``pip install pyvqnet --index-url https://pypi.originqc.com.cn`` 安装。

[v2.0.8] - 2023-09-26
***************************

Added
===========
- 增加了现有接口支持complex128、complex64、double、float、uint8、int8、bool、int16、int32、int64等类型计算。
- Linux版本支持gpu下的计算,需要cuda11.7版本cudatoolkit以及nvidia驱动。
- 基于vqc的基础逻辑门：Hadamard、CNOT、I、RX、RY、PauliZ、PauliX、PauliY、S、RZ、RXX、RYY、RZZ、RZX、X1、Y1、Z1、U1、U2、U3、T、SWAP、P、TOFFOLI、CZ、CR。
- 基于vqc的组合量子线路：VQC_HardwareEfficientAnsatz、VQC_BasicEntanglerTemplate、VQC_StronglyEntanglingTemplate、VQC_QuantumEmbedding、VQC_RotCircuit、VQC_CRotCircuit、VQC_CSWAPcircuit、VQC_Controlled_Hadamard、VQC_CCZ、VQC_FermionicSingleExcitation、VQC_FermionicDoubleExcitation、VQC_UCCSD、VQC_QuantumPoolingCircuit、VQC_BasisEmbedding、VQC_AngleEmbedding、VQC_AmplitudeEmbedding、VQC_IQPEmbedding。
- 基于vqc的测量方法：VQC_Purity、VQC_VarMeasure、VQC_DensityMatrixFromQstate、Probability、MeasureAll。


[v2.0.7] - 2023-07-03
***************************

Added
===========
- 经典神经网络，增加kron，gather,scatter,broadcast_to接口。
- 增加对不同数据精度支持：数据类型dtype支持kbool,kuint8,kint8,kint16,kint32,kint64,kfloat32,kfloat64,kcomplex64,kcomplex128.分别代表C++的 bool,uint8_t,int8_t,int16_t,int32_t,int64_t,float,double,complex<float>,complex<double>.
- 支持python 3.8，3.9，3.10三个版本。

Changed
===========
- QTensor 以及Module类的init函数增加 `dtype` 参数。对QTensor索引、 部分神经网络层的输入进行了类型限制。
- 量子神经网络，由于MacOS兼容性问题，去掉了Mnist_Dataset，CIFAR10_Dataset接口。

[v2.0.6] - 2023-02-22
***************************


Added
===========

- 经典神经网络，增加接口：multinomial,pixel_shuffle,pixel_unshuffle,为QTensor增加numel，增加CPU动态内存池功能，为Parameter增加init_from_tensor接口。
- 经典神经网络，增加接口：Dynamic_LSTM,Dynamic_RNN,Dynamic_GRU。
- 经典神经网络，增加接口：pad_sequence,pad_packed_sequence,pack_pad_sequence。
- 量子神经网络，增加接口：CCZ,Controlled_Hadamard,FermionicSingleExcitation,UCCSD,QuantumPoolingCircuit,
- 量子神经网络，增加接口：Quantum_Embedding,Mnist_Dataset,CIFAR10_Dataset,grad，Purity。
- 量子神经网络，增加示例：基于梯度裁剪，quanvolution,量子线路表达力，贫瘠高原，量子强化学习QDRL。

Changed
===========

- API文档，重构内容结构，增加 `量子机器学习研究` 模块，将 `VQNet2ONNX模块` 改为 `其他函数` 。



Fixed
===========

- 经典神经网络，解决相同随机种子跨平台产生不同正态分布的问题。
- 量子神经网络，实现expval，ProbMeasure，QuantumMeasure 对QPanda GPU虚拟机的支持。


[v2.0.5] - 2022-12-25
***************************


Added
===========

- 经典神经网络，增加log_softmax实现，增加模型转ONNX的接口export_model函数。
- 经典神经网络，支持当前已有的绝大多数经典神经网络模块转换为ONNX，详情参考API文档 “VQNet2ONNX模块”。
- 量子神经网络，增加VarMeasure,MeasurePauliSum,Quantum_Embedding,SPSA等接口
- 量子神经网络，增加LinearGNN,ConvGNN,ConvGNN，QMLP,量子自然梯度，量子随机parameter-shift算法，DoublySGD算法等。


Changed
===========

- 经典神经网络，为BN1d,BN2d接口增加维度检查。

Fixed
===========

- 解决maxpooling参数检查的bug。
- 解决[::-1]的切片bug。


[v2.0.4] - 2022-09-20
***************************


Added
===========

- 经典神经网络，增加LayernormNd实现，支持多维数据layernorm计算。
- 经典神经网络，增加CrossEntropyLoss以及NLL_Loss损失函数计算接口，支持1维~N维输入。
- 量子神经网络，增加常用线路模板：HardwareEfficientAnsatz,StronglyEntanglingTemplate,BasicEntanglerTemplate。
- 量子神经网络，增加计算量子比特子系统互信息的Mutal_info接口、Von Neumann 熵VB_Entropy、密度矩阵DensityMatrixFromQstate。
- 量子神经网络，增加量子感知器算法例子QuantumNeuron，增加量子傅里叶级数算法例子。
- 量子神经网络，增加支持多进程加速运行量子线路的接口QuantumLayerMultiProcess。

Changed
===========

- 经典神经网络，支持组卷积参数group，空洞卷积dilation_rate，任意数值padding作为一维卷积Conv1d、二维卷积Conv2d、反卷积ConvT2d的参数。
- 在相同维度的数据跳过广播操作，减少不必要运行逻辑。

Fixed
===========

- 解决stack函数在部分参数下计算错误的问题。


[v2.0.3] - 2022-07-15
***************************


Added
===========

- 增加支持stack,双向的循环神经网络接口：RNN, LSTM, GRU
- 增加常用计算性能指标的接口：MSE,RMSE, MAE, R_Square, precision_recall_f1_2_score, precision_recall_f1_Multi_scoreprecision_recall_f1_N_score, auc_calculate
- 增加量子核SVM的算法示例

Changed
===========

- 加快QTensor数据过多时候的print速度
- Windows和linux下使用openmp加速运算。

Fixed
===========

- 解决部分python import方式无法导入的问题
- 解决批归一化BN层重复计算的问题
- 解决QTensor.reshape,transpose接口无法计算梯度的bug
- 为tensor.power接口增加入参形状判断

[v2.0.2] - 2022-05-15
***************************

Added
===========

- 增加topK, argtoK
- 增加cumsum
- 增加masked_fill
- 增加triu,tril
- 增加QGAN生成随机分布的示例

Changed
===========

- 支持高级切片索引和普通切片索引
- matmul支持3D,4D张量运算
- 修改HardSigmoid函数实现

Fixed
===========

- 解决卷积，批归一化，反卷积，池化层等层没有缓存内部变量，导致一次前传后多次反传时计算梯度的问题
- 修正QLinear层的实现和示例
- 解决MAC在conda环境中导入VQNet时候 Image not load的问题。




[v2.0.1] - 2022-03-30
***************************


Added
===========

- 增加基本数据结构QTensor接口100余个，包括创建函数，逻辑函数，数学函数，矩阵操作。
- 增加基本神经网络网络函数14个，包括卷积，反卷积，池化等。
- 增加损失函数4个，包括MSE,BCE,CCE,SCE等。
- 增加激活函数10个，包括ReLu，Sigmoid，ELU等。
- 增加优化器6个，包括SGD,RMSPROP,ADAM等。
- 增加机器学习示例：QVC,QDRL,Q-KMEANS,QUnet，HQCNN，VSQL,量子自编码器。
- 增加量子机器学习层：QuantumLayer，NoiseQuantumLayer。