
VQNet Changelog
================

[v2.0.9] - 2023-09-08
----------------------

Added
^^^^^^^^^^^
- 增加了xtensor接口定义，支持算子自动并行和CPU/GPU多后端，包含对多维数组的常用数学，逻辑，矩阵计算，以及常见的经典神经网络层，优化器等150余个接口。
- 增加了分布式计算接口定义，支持cpu上单节点多进程、多节点多进程进行模型训练，包括初始化，通信，数据切分接口。


[v2.0.8] - 2023-07-26
----------------------

Added
^^^^^^^^^^^
- 增加了现有接口支持complex128、complex64、double、float、uint8、int8、bool、int16、int32、int64等类型计算（Linux下GPU,cuda11.7）。
- 基于vqc的基础逻辑门：Hadamard、CNOT、I、RX、RY、PauliZ、PauliX、PauliY、S、RZ、RXX、RYY、RZZ、RZX、X1、Y1、Z1、U1、U2、U3、T、SWAP、P、TOFFOLI、CZ、CR、ISWAP。
- 基于vqc的组合量子线路：VQC_HardwareEfficientAnsatz、VQC_BasicEntanglerTemplate、VQC_StronglyEntanglingTemplate、VQC_QuantumEmbedding、VQC_RotCircuit、VQC_CRotCircuit、VQC_CSWAPcircuit、VQC_Controlled_Hadamard、VQC_CCZ、VQC_FermionicSingleExcitation、VQC_FermionicDoubleExcitation、VQC_UCCSD、VQC_QuantumPoolingCircuit、VQC_BasisEmbedding、VQC_AngleEmbedding、VQC_AmplitudeEmbedding、VQC_IQPEmbedding。
- 基于vqc的测量方法：VQC_Purity、VQC_VarMeasure、VQC_DensityMatrixFromQstate、Probability、MeasureAll。


[v2.0.7] - 2023-07-03
----------------------

Added
^^^^^^^^^^^
- 经典神经网络，增加kron，gather,scatter,broadcast_to接口。
- 增加对不同数据精度支持：数据类型dtype支持kbool,kuint8,kint8,kint16,kint32,kint64,kfloat32,kfloat64,kcomplex64,kcomplex128.分别代表C++的 bool,uint8_t,int8_t,int16_t,int32_t,int64_t,float,double,complex<float>,complex<double>.
- 支持python 3.8，3.9，3.10三个版本。

Changed
^^^^^^^^^^^
- QTensor 以及Module类的init函数增加 `dtype` 参数。对QTenor索引、 部分神经网络层的输入进行了类型限制。
- 量子神经网络，由于MacOS兼容性问题，去掉了Mnist_Dataset，CIFAR10_Dataset接口。

[v2.0.6] - 2023-02-22
----------------------


Added
^^^^^^^^^^^

- 经典神经网络，增加接口：multinomial,pixel_shuffle,pixel_unshuffle,为QTensor增加numel，增加CPU动态内存池功能，为Parameter增加init_from_tensor接口。
- 经典神经网络，增加接口：Dynamic_LSTM,Dynamic_RNN,Dynamic_GRU。
- 经典神经网络，增加接口：pad_sequence,pad_packed_sequence,pack_pad_sequence。
- 量子神经网络，增加接口：CCZ,Controlled_Hadamard,FermionicSingleExcitation,UCCSD,QuantumPoolingCircuit,
- 量子神经网络，增加接口：Quantum_Embedding,Mnist_Dataset,CIFAR10_Dataset,grad，Purity。
- 量子神经网络，增加示例：基于梯度裁剪，quanvolution,量子线路表达力，贫瘠高原，量子强化学习QDRL。

Changed
^^^^^^^^^^^

- API文档，重构内容结构，增加 `量子机器学习研究` 模块，将 `VQNet2ONNX模块` 改为 `其他函数` 。



Fixed
^^^^^^^^^^^

- 经典神经网络，解决相同随机种子跨平台产生不同正态分布的问题。
- 量子神经网络，实现expval，ProbMeasure，QuantumMeasure 对QPanda GPU虚拟机的支持。


[v2.0.5] - 2022-12-25
----------------------


Added
^^^^^^^^^^^

- 经典神经网络，增加log_softmax实现，增加模型转ONNX的接口export_model函数。
- 经典神经网络，支持当前已有的绝大多数经典神经网络模块转换为ONNX，详情参考API文档 “VQNet2ONNX模块”。
- 量子神经网络，增加VarMeasure,MeasurePauliSum,Quantum_Embedding,SPSA等接口
- 量子神经网络，增加LinearGNN,ConvGNN,ConvGNN，QMLP,量子自然梯度，量子随机parameter-shift算法，DoublySGD算法等。


Changed
^^^^^^^^^^^

- 经典神经网络，为BN1d,BN2d接口增加维度检查。

Fixed
^^^^^^^^^^^

- 解决maxpooling参数检查的bug。
- 解决[::-1]的切片bug。


[v2.0.4] - 2022-09-20
----------------------


Added
^^^^^^^^^^^

- 经典神经网络，增加LayernormNd实现，支持多维数据layernorm计算。
- 经典神经网络，增加CrossEntropyLoss以及NLL_Loss损失函数计算接口，支持1维~N维输入。
- 量子神经网络，增加常用线路模板：HardwareEfficientAnsatz,StronglyEntanglingTemplate,BasicEntanglerTemplate。
- 量子神经网络，增加计算量子比特子系统互信息的Mutal_info接口、Von Neumann 熵VB_Entropy、密度矩阵DensityMatrixFromQstate。
- 量子神经网络，增加量子感知器算法例子QuantumNeuron，增加量子傅里叶级数算法例子。
- 量子神经网络，增加支持多进程加速运行量子线路的接口QuantumLayerMultiProcess。

Changed
^^^^^^^^^^^

- 经典神经网络，支持组卷积参数group，空洞卷积dilation_rate，任意数值padding作为一维卷积Conv1d、二维卷积Conv2d、反卷积ConvT2d的参数。
- 在相同维度的数据跳过广播操作，减少不必要运行逻辑。

Fixed
^^^^^^^^^^^

- 解决stack函数在部分参数下计算错误的问题。


[v2.0.3] - 2022-07-15
----------------------


Added
^^^^^^^^^^^

- 增加支持stack,双向的循环神经网络接口：RNN, LSTM, GRU
- 增加常用计算性能指标的接口：MSE,RMSE, MAE, R_Square, precision_recall_f1_2_score, precision_recall_f1_Multi_scoreprecision_recall_f1_N_score, auc_calculate
- 增加量子核SVM的算法示例

Changed
^^^^^^^^^^^

- 加快QTensor数据过多时候的print速度
- Windows和linux下使用openmp加速运算。

Fixed
^^^^^^^^^^^

- 解决部分python import方式无法导入的问题
- 解决批归一化BN层重复计算的问题
- 解决QTensor.reshape,transpose接口无法计算梯度的bug
- 为tensor.power接口增加入参形状判断

[v2.0.2] - 2022-05-15
-----------------------


Added
^^^^^^^^^^^

- 增加topK, argtoK
- 增加cumsum
- 增加masked_fill
- 增加triu,tril
- 增加QGAN生成随机分布的示例

Changed
^^^^^^^^^^^

- 支持高级切片索引和普通切片索引
- matmul支持3D,4D张量运算
- 修改HardSigmoid函数实现

Fixed
^^^^^^^^^^^

- 解决卷积，批归一化，反卷积，池化层等层没有缓存内部变量，导致一次前传后多次反传时计算梯度的问题
- 修正QLinear层的实现和示例
- 解决MAC在conda环境中导入VQNet时候 Image not load的问题。




[v2.0.1] - 2022-03-30
----------------------------


Added
^^^^^^^^^^^

- 增加基本数据结构QTenor接口100余个，包括创建函数，逻辑函数，数学函数，矩阵操作。
- 增加基本神经网络网络函数14个，包括卷积，反卷积，池化等。
- 增加损失函数4个，包括MSE,BCE,CCE,SCE等。
- 增加激活函数10个，包括ReLu，Sigmoid，ELU等。
- 增加优化器6个，包括SGD,RMSPROP,ADAM等。
- 增加机器学习示例：QVC,QDRL,Q-KMEANS,QUnet，HQCNN，VSQL,量子自编码器。
- 增加量子机器学习层：QuantumLayer，NoiseQuantumLayer。